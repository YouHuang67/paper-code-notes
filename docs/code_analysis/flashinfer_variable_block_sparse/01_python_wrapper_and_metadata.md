---
tags:
  - CUDA
  - Flash Attention
  - Sparse Attention
---

# Python Wrapper 与 Metadata

**核心文件**:

- [`wrapper.py`](src/wrapper_py.md)
- [`metadata.py`](src/metadata_py.md)
- [`backend.py`](src/backend_py.md)

## 入口对象缓存了什么

这套实现只有一个公开入口：

```python
from variable_block_attn import VariableBlockSparseAttentionWrapper
```

对应的导出非常薄，只是把类从 [`wrapper.py`](src/wrapper_py.md) 重新暴露出来：

```python
from .wrapper import VariableBlockSparseAttentionWrapper

__all__ = ["VariableBlockSparseAttentionWrapper"]
```

真正值得看的，是 `VariableBlockSparseAttentionWrapper.__init__` 和 `plan()`。

下面这段代码直接决定了 wrapper 在生命周期里缓存哪些状态：

```python
@flashinfer_api
def __init__(
    self,
    float_workspace_buffer: torch.Tensor,
    backend: str = "auto",
) -> None:
    self._float_workspace_buffer = float_workspace_buffer
    self.device = float_workspace_buffer.device
    self._workspace_size = (
        float_workspace_buffer.numel() * float_workspace_buffer.element_size()
    )
    self._int_workspace_buffer = torch.empty(
        (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
    )
    self._kv_lens_buffer = torch.empty(
        (32768,), dtype=torch.int32, device=self.device
    )
    self._pin_memory_int_workspace_buffer = torch.empty(
        self._int_workspace_buffer.shape,
        dtype=torch.uint8,
        pin_memory=True,
        device="cpu",
    )
    self._kv_layout = "NHD"
    self._qo_indptr: Optional[torch.Tensor] = None
    self._paged_kv_indptr_buf: Optional[torch.Tensor] = None
    self._paged_kv_indices_buf: Optional[torch.Tensor] = None
    self._paged_kv_last_page_len: Optional[torch.Tensor] = None
    self._backend = backend
```

这里可以直接读出四层状态：

- `float_workspace_buffer`
  - 给 split-k 的中间结果和 merge 使用
- `int_workspace_buffer`
  - 给 request/tile/merge 等调度索引使用
- `pin_memory_int_workspace_buffer`
  - 给 host-device 规划过程使用
- `*_indptr / *_indices / *_last_page_len`
  - `plan()` 完成后常驻的 paged prefill 元数据

注意这里的重点不是“wrapper 自己能算 attention”，而是“wrapper 只是缓存翻译后的元数据和 runtime 句柄”。真正的调度初始化发生在 `plan()`，真正的算子调用发生在 `run()`。

## `plan()` 的本质不是计划，而是语义翻译

`plan()` 的完整关键路径如下：

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
self._paged_kv_indices_buf = kv_indices.to(
    self.device,
    non_blocking=non_blocking,
)
self._paged_kv_last_page_len = last_block_len.to(
    self.device,
    non_blocking=non_blocking,
)
torch.cuda.synchronize()
self._mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value

self._backend = resolve_attention_backend(
    self._backend,
    self.device,
    PosEncodingMode[pos_encoding_mode].value,
    use_fp16_qk_reduction,
    self._mask_mode == MaskMode.CUSTOM.value,
    q_data_type,
    kv_data_type,
)

get_module_args = (
    q_data_type,
    kv_data_type,
    self._o_dtype,
    kv_indptr_host.dtype,
    head_dim,
    head_dim,
    PosEncodingMode[pos_encoding_mode].value,
    False,
    logits_soft_cap > 0,
    use_fp16_qk_reduction,
)
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

这段代码里面，真正“variable block 特有”的事情只有前半段：

- `build_qo_indptr(block_row_sz)`
- `build_last_page_len(...)`
- `block_mask_map_to_expanded_indices(block_mask_map, block_col_sz)`

后半段已经是标准 FlashInfer paged prefill 体系：

- backend 选择
- batch prefill 模块获取
- `module.plan(...)`

所以工程上最值得记住的一句话是：

> variable block 的本体不在 CUDA kernel，而在元数据翻译层。

## 为什么 `page_size=1` 是整个设计的关键

在 `plan()` 里有两个非常不起眼、但决定整套设计的常量：

```python
last_block_len = build_last_page_len(...)
...
args = [
    ...
    1,
    1,
    False,
    ...
]
```

而 `metadata.py` 对 `last_page_len` 的实现是：

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

这说明当前实现明确选择了：

- `page_size = 1`
- 每个 token 都被视作一个 page
- `last_page_len` 自然恒为 1

这一步把“变长 block”问题，转成了“token 级 page table”问题。于是底层 kernel 不需要知道什么叫 variable block，它只需要知道：

- 第 `i` 行 query 对应哪些 KV token
- 这些 KV token 在扁平化 cache 里的下标是什么

## `metadata.py` 是真正的语义核心

`metadata.py` 只有 88 行，但这是整套实现里最关键的文件。

完整核心逻辑如下：

```python
def build_qo_indptr(block_row_sz: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=block_row_sz.device),
            torch.cumsum(block_row_sz.flatten(), dim=0, dtype=torch.int32),
        ],
        dim=0,
    )


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


def block_mask_map_to_expanded_indices(
    block_mask_map: torch.Tensor,
    block_col_sz: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = block_mask_map.device
    dtype_i = torch.int32

    row_lengths = (block_mask_map * block_col_sz[:, None, :]).sum(-1)
    kv_indptr = torch.cat(
        [
            torch.zeros(1, dtype=dtype_i, device=device),
            torch.cumsum(row_lengths.flatten(), 0),
        ],
        dim=0,
    )

    col_offset = torch.cumsum(block_col_sz.to(dtype_i), 1) - block_col_sz
    head_len = block_col_sz.sum(1, dtype=dtype_i)
    head_offset = torch.cumsum(head_len, 0) - head_len

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

这段代码完成了两层翻译。

### 第一层：Q 侧分段

`build_qo_indptr` 很直白：

- `block_row_sz` 先 flatten
- 再做 `cumsum`
- 得到 `[H * R + 1]` 长度的前缀和

它表达的是：

- 第 0 个 `(kv_head, row_block)` 的 query token 范围
- 第 1 个 `(kv_head, row_block)` 的 query token 范围
- ...

也就是把“按块组织”的 query 行，压平成 paged prefill 认识的 CSR 风格 row indptr。

### 第二层：KV 侧从 block 稀疏图到 token 级索引

`block_mask_map_to_expanded_indices` 才是最重要的一步。

它先算：

```python
row_lengths = (block_mask_map * block_col_sz[:, None, :]).sum(-1)
```

含义是：

- `block_mask_map[h, r, c] = 1` 表示第 `h` 个 KV head 的第 `r` 个 query block 需要第 `c` 个 KV block
- 而 `block_col_sz[h, c]` 是这个 KV block 真实包含的 token 数
- 所以按列求和后，得到每个 `(h, r)` 最终访问多少个 KV token

接着它构造：

- `col_offset`
  - 每个列块在所属 head 内的起始 token 偏移
- `head_offset`
  - 每个 head 在全局扁平 KV 空间中的基址

然后用：

```python
h_idx, _, c_idx = block_mask_map.nonzero(as_tuple=True)
```

把所有被选中的 `(head, row, col)` 抽出来，转成：

- `lengths`
  - 这个被选中的列块有多长
- `base`
  - 这个被选中的列块在全局扁平 KV 里的起始 token 下标

最后通过：

```python
cum = torch.cumsum(lengths, 0)
starts = torch.repeat_interleave(cum - lengths, lengths)
offsets_within = torch.arange(cum[-1], device=device) - starts
kv_indices = torch.repeat_interleave(base, lengths) + offsets_within
```

把 block 级别的选择，完全展开成 token 级别的连续索引。

这一步之后，底层 runtime 看到的已经不是：

- 第 `r` 行选中了哪些 block

而是：

- 第 `r` 行应该访问扁平 KV 序列里的哪些 token 下标

也因此，这里的输出实际上就是一个 token 级 CSR：

- `kv_indptr`
  - 每个 `(head, row_block)` 对应的 token 段边界
- `kv_indices`
  - 每个 token 的真实 KV 全局下标

## `run()` 只是协议重排，不再重新理解 variable block

`run()` 的关键逻辑如下：

```python
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

这里最重要的是理解：

- `run()` 不再做 variable block 语义分析
- 它只负责把 `q/k/v` 排成底层 ABI 期望的形状
- 然后把 `plan()` 缓存好的元数据直接交给 `paged_run`

也就是说，热路径上最重要的不是“重新做图分析”，而是“避免任何多余逻辑”。

### 重排后的形状为什么长这样

Q 被重排成：

```text
[num_qo_heads, qo_len, d]
-> [num_kv_heads * qo_len, gqa_group_size, d]
```

K/V 被重排成：

```text
[num_kv_heads, kv_len, d]
-> [num_kv_heads * kv_len, 1, 1, d]
```

这本质上是在做两件事：

- 把 head 维部分压到“批次/行”语义里
- 把底层 `PagedParams` 需要的 paged cache 布局准备出来

从这里也能看出，底层 runtime 并不知道“原来你有个 variable block mask”。它只知道：

- query 行如何分段
- 每行对应哪些 paged KV token

## Python 层的最终结论

把 [`wrapper.py`](src/wrapper_py.md) 和 [`metadata.py`](src/metadata_py.md) 连起来看，这套实现最关键的工程思想其实很朴素：

1. 在 `plan()` 阶段一次性完成 block 语义到 token 语义的翻译
2. 把翻译结果缓存成 paged prefill 能直接消费的 `indptr + indices`
3. 在 `run()` 热路径里完全避免重新理解稀疏结构
4. 尽量把后续执行交回成熟的 FA2 paged prefill runtime

这就是为什么这套代码虽然名叫 variable block sparse attention，但它最值得分析的部分其实不是 kernel，而是 metadata 展开。

