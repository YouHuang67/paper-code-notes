---
tags:
  - CUDA
  - Flash Attention
  - Sparse Attention
---

# Python 入口与元数据展开

**源码**:

- [wrapper.py:L41-L397](src/wrapper_py.md#__codelineno-0-41)
- [metadata.py:L22-L88](src/metadata_py.md#__codelineno-0-22)
- [backend.py:L28-L50](src/backend_py.md#__codelineno-0-28)

这一层是整套实现里最值得仔细读的部分，因为 variable block 语义基本都在这里被消化掉了。后面的 runtime/JIT/C++ 层更像是“把翻译结果接进既有 FA2 paged prefill 体系”。

## 入口对象：wrapper 只缓存状态，不实现算法

公开入口只有一个：

```python
from variable_block_attn import VariableBlockSparseAttentionWrapper
```

真正有信息量的是 [`wrapper.py:L74-L111`](src/wrapper_py.md#__codelineno-0-74) 的构造函数：

```python
@flashinfer_api
def __init__(
    self,
    float_workspace_buffer: torch.Tensor,
    backend: str = "auto",
) -> None:
    '''
    float workspace: split-k 中间结果与 merge 临时缓冲
    int workspace: request/tile/merge 等调度数组
    '''
    self._float_workspace_buffer = float_workspace_buffer
    self.device = float_workspace_buffer.device
    self._workspace_size = (
        float_workspace_buffer.numel() * float_workspace_buffer.element_size()
    )
    self._int_workspace_buffer = torch.empty(
        (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
    )
    self._pin_memory_int_workspace_buffer = torch.empty(
        self._int_workspace_buffer.shape,
        dtype=torch.uint8,
        pin_memory=True,
        device="cpu",
    )

    '''
    plan() 完成后常驻的 paged prefill metadata
    '''
    self._kv_layout = "NHD"
    self._qo_indptr: Optional[torch.Tensor] = None
    self._paged_kv_indptr_buf: Optional[torch.Tensor] = None
    self._paged_kv_indices_buf: Optional[torch.Tensor] = None
    self._paged_kv_last_page_len: Optional[torch.Tensor] = None
    self._backend = backend
```

这里有两个阅读点：

- 当前代码已经没有旧版本里那种 `_kv_lens_buffer` 常驻状态，wrapper 缓存的就是 workspace、layout 和 plan 后元数据
- wrapper 本身并不“持有一套 attention 算法”，它只是缓存翻译结果和 runtime 句柄

也就是说，这个对象更像“variable block -> paged prefill”的状态容器，而不是完整计算引擎。

## `plan()`：先翻译元数据，再接入 batch prefill `plan`

[`wrapper.py:L131-L252`](src/wrapper_py.md#__codelineno-0-131) 是整个 Python 层的主线：

```python
q_data_type = canonicalize_torch_dtype(q_data_type)
if kv_data_type is None:
    kv_data_type = q_data_type
kv_data_type = canonicalize_torch_dtype(kv_data_type)
self._o_dtype = q_data_type

if logits_soft_cap is None:
    logits_soft_cap = 0.0

num_blocks_row = block_row_sz.shape[-1]
num_blocks_col = block_col_sz.shape[-1]

'''
第一段：把 block 级描述翻译成 paged prefill 需要的 token 级 metadata
'''
qo_indptr = build_qo_indptr(block_row_sz)
qo_indptr_host = qo_indptr.to("cpu", non_blocking=non_blocking)
last_block_len = build_last_page_len(
    block_mask_map,
    num_blocks_row,
    num_kv_heads,
)

kv_indptr, kv_indices = block_mask_map_to_expanded_indices(
    block_mask_map,
    block_col_sz,
)
kv_indptr_host = kv_indptr.to("cpu", non_blocking=non_blocking)
kv_indices_host = kv_indices.to("cpu", non_blocking=non_blocking)

self._qo_indptr = qo_indptr.to(self.device, non_blocking=non_blocking)
self._paged_kv_indptr_buf = kv_indptr.to(self.device, non_blocking=non_blocking)
self._paged_kv_indices_buf = kv_indices.to(self.device, non_blocking=non_blocking)
self._paged_kv_last_page_len = last_block_len.to(self.device, non_blocking=non_blocking)
torch.cuda.synchronize()

'''
第二段：backend 解析、模块获取、进入标准 batch prefill plan
'''
self._backend = resolve_attention_backend(...)
self._cached_module = get_batch_prefill_module(self._backend, *get_module_args)

kv_lens_arr_host = kv_indptr_host[1:] - kv_indptr_host[:-1]
args = [
    self._float_workspace_buffer,
    self._int_workspace_buffer,
    self._pin_memory_int_workspace_buffer,
    qo_indptr_host,
    kv_indptr_host,
    kv_lens_arr_host,
    qo_indptr_host[-1].item(),
    num_blocks_row * num_kv_heads,
    num_qo_heads // num_kv_heads,
    1,
    1,
    False,
    head_dim,
    head_dim,
    causal,
    -1,
]
if self._backend == "fa2":
    args.append(-1)
    args.append(False)
    args.append(0)
self._plan_info = self._cached_module.plan(*args)
```

这段代码可以拆成两个阶段理解。

第一阶段是 variable block 特有逻辑：

- `build_qo_indptr(block_row_sz)`
- `build_last_page_len(...)`
- `block_mask_map_to_expanded_indices(block_mask_map, block_col_sz)`

第二阶段已经是标准 FlashInfer batch prefill 流程：

- backend 选择
- JIT 模块获取
- `module.plan(...)`

所以 `plan()` 这个名字有点误导。它真正先做的是“语义翻译”，然后才把翻译结果送进底层 `plan`。

## `page_size=1`：整套设计的关键压缩点

在 [`wrapper.py:L230-L252`](src/wrapper_py.md#__codelineno-0-230) 里有两个决定全局设计的常量：

```python
args = [
    ...,
    num_qo_heads // num_kv_heads,
    1,
    1,
    False,
    head_dim,
    head_dim,
    ...
]
```

同时 [`metadata.py:L32-L42`](src/metadata_py.md#__codelineno-0-32) 里 `last_page_len` 被固定为 1：

```python
def build_last_page_len(
    block_mask_map: torch.Tensor,
    num_blocks_row: int,
    num_kv_heads: int,
) -> torch.Tensor:
    return torch.full(
        (num_blocks_row * num_kv_heads,),
        1,
        dtype=torch.int32,
        device=block_mask_map.device,
    )
```

这意味着当前实现明确采用：

- `page_size = 1`
- 每个 token 视作一个 page
- `last_page_len` 恒等于 1

于是“变长 block”问题被转写成了“token 级 page table”问题。底层 kernel 不需要知道 block 的原始尺寸，只需要知道每一行 query 最终对应哪些 token 索引。

## `metadata.py`：真正的语义核心

[`metadata.py:L22-L88`](src/metadata_py.md#__codelineno-0-22) 只有几十行，但它完成了最关键的 block-to-token 展开：

```python
def build_qo_indptr(block_row_sz: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=block_row_sz.device),
            torch.cumsum(block_row_sz.flatten(), dim=0, dtype=torch.int32),
        ],
        dim=0,
    )


def block_mask_map_to_expanded_indices(
    block_mask_map: torch.Tensor,  # [H, R, C]
    block_col_sz: torch.Tensor,    # [H, C]
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = block_mask_map.device
    dtype_i = torch.int32

    '''
    每个 (head, row) 下，先算出它总共会展开出多少个 KV token
    '''
    row_lengths = (block_mask_map * block_col_sz[:, None, :]).sum(-1)  # [H, R]
    kv_indptr = torch.cat(
        [
            torch.zeros(1, dtype=dtype_i, device=device),
            torch.cumsum(row_lengths.flatten(), 0),
        ],
        dim=0,
    )

    '''
    每个列块在本 head 内的 token 起始偏移
    '''
    col_offset = torch.cumsum(block_col_sz.to(dtype_i), 1) - block_col_sz
    head_len = block_col_sz.sum(1, dtype=dtype_i)
    head_offset = torch.cumsum(head_len, 0) - head_len

    '''
    把选中的 (h, r, c) 逐块展开成连续 token 下标
    '''
    h_idx, _, c_idx = block_mask_map.nonzero(as_tuple=True)
    lengths = block_col_sz[h_idx, c_idx].to(dtype_i)
    base = head_offset[h_idx] + col_offset[h_idx, c_idx]

    cum = torch.cumsum(lengths, 0)
    starts = torch.repeat_interleave(cum - lengths, lengths)
    offsets_within = torch.arange(cum[-1], device=device) - starts
    kv_indices = torch.repeat_interleave(base, lengths) + offsets_within

    return kv_indptr.to(dtype=dtype_i, device=device), kv_indices.to(
        dtype=dtype_i, device=device
    )
```

它做了两件事。

第一件事是为 Q 侧构造 CSR 风格的分段：

- `block_row_sz.flatten()` 把 `(kv_head, row_block)` 拍平
- `cumsum` 得到每一段 query token 的起止边界
- `qo_indptr` 最终表达的是“第几个 `(head, row)` 对应哪一段 query token”

第二件事是把 KV 侧的块稀疏图展开成 token 索引：

- `row_lengths` 算每个 `(head, row)` 最终选中多少个 token
- `kv_indptr` 记录每一行的 token 段边界
- `col_offset + head_offset` 找到每个列块在扁平 KV 空间里的起点
- `repeat_interleave + offsets_within` 把变长列块逐 token 展开

这里的结果已经完全是 token-level 表达，不再保留 “block size” 这个概念。也正因为如此，后续层可以把它当作普通 paged KV metadata 使用。

## `run()`：只做布局改写，然后进入 `paged_run`

`run()` 的核心在 [`wrapper.py:L284-L394`](src/wrapper_py.md#__codelineno-0-284)：

```python
if enable_pdl is None:
    enable_pdl = device_support_pdl(q.device)

if logits_soft_cap is None:
    logits_soft_cap = 0.0
if sm_scale is None:
    sm_scale = 1.0 / math.sqrt(q.size(-1))
if rope_scale is None:
    rope_scale = 1.0
if rope_theta is None:
    rope_theta = 1e4

'''
Q 改成 batch prefill 期望的 grouped-query 视图
K/V 改成 page size = 1 的 paged cache 视图
'''
q = einops.rearrange(
    q,
    "(num_kv_heads gqa_group_size) qo_len head_dim -> (num_kv_heads qo_len) gqa_group_size head_dim",
    num_kv_heads=self._num_kv_heads,
).contiguous()
k = einops.rearrange(
    k,
    "num_kv_heads kv_len head_dim -> (num_kv_heads kv_len) 1 1 head_dim",
).contiguous()
v = einops.rearrange(
    v,
    "num_kv_heads kv_len head_dim -> (num_kv_heads kv_len) 1 1 head_dim",
).contiguous()

'''
把 plan() 阶段缓存的 metadata 全部交给 paged_run
'''
self._cached_module.paged_run(
    self._float_workspace_buffer,
    self._int_workspace_buffer,
    self._plan_info,
    q,
    k,
    v,
    self._qo_indptr,
    self._paged_kv_indptr_buf,
    self._paged_kv_indices_buf,
    self._paged_kv_last_page_len,
    out,
    lse,
    self._mask_mode,
    TensorLayout[self._kv_layout].value,
    -1,
    enable_pdl,
    None,
    None,
    None,
    None,
    None,
    None,
    logits_soft_cap,
    sm_scale,
    None,
    None,
    None,
    rope_scale,
    rope_theta,
    0,
    self._workspace_size,
)
```

这一步要点也很明确：

- `q` 被重排成按 `num_kv_heads` 分组的 GQA 视图
- `k/v` 被重排成 `(token, 1, 1, dim)`，正好对应 `page_size=1`
- `run()` 并不重新计算 metadata，只消费 `plan()` 阶段缓存的结果

因此 `run()` 更像是“布局适配 + 参数透传”，而不是另一个有独立语义的阶段。

## backend 边界：接口兼容，但执行路径收敛

虽然 runtime/JIT 细节放在下一篇，这里先看一下 [`backend.py:L28-L50`](src/backend_py.md#__codelineno-0-28)：

```python
def resolve_attention_backend(...):
    if backend == "auto":
        backend = determine_attention_backend(...)

    if backend == "fa3":
        raise NotImplementedError(_FA3_NOT_SUPPORTED_MSG)
    if backend != "fa2":
        raise ValueError(f"Unsupported backend: {backend}")
    return backend
```

这段逻辑和 `plan()` 紧密相关，因为它解释了为什么当前 wrapper 看起来支持三种 backend，但真正 plan 下去只会落到 `fa2`：

- `auto` 仍然保留上游同名接口
- 运行期如果检测到 `fa3` 能力，也不会继续走下去
- 当前闭环只允许 `fa2`

这正是“对外保留 `auto/fa2/fa3`，内部只保留 `fa2` 最小闭环”的第一现场。

## 小结

Python 层的职责可以压缩成三步：

1. 把 variable block 稀疏图翻译成 token 级 `qo_indptr / kv_indptr / kv_indices`
2. 用 `page_size=1` 把问题改写成普通 paged prefill
3. 在 `run()` 中把 Q/K/V 重排成底层 ABI 期待的布局

后面的 runtime、JIT、C++、CUDA 都是在消费这一层已经翻译好的结果。
