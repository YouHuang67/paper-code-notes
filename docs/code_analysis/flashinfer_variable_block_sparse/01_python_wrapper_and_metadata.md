---
tags:
  - CUDA
  - Flash Attention
  - Sparse Attention
---

# 初始化与 plan 前准备

**源码**:

- [wrapper.py:L41-L397](src/wrapper_py.md#__codelineno-0-41)
- [metadata.py:L22-L88](src/metadata_py.md#__codelineno-0-22)
- [backend.py:L28-L50](src/backend_py.md#__codelineno-0-28)

这一章只讲执行链路里最前面的部分，也就是：

1. wrapper 构造时缓存了什么状态
2. `plan()` 前半段怎样把 variable block 描述翻译成 token 级 metadata
3. backend 在这里是怎样被解析出来的

也就是说，这一章先停在“底层模块和调度真正启动之前”。

## 1. 先把 shape 和逻辑单位说清楚

后面所有代码解释都围绕下面这些符号展开：

- `H_kv = num_kv_heads`
- `H_qo = num_qo_heads`
- `G = H_qo / H_kv`：每个 KV head 对应的 GQA group size
- `R = num_blocks_row`
- `C = num_blocks_col`
- `D = head_dim`
- `qo_len`：外部 `q` 的 token 长度
- `kv_len`：外部 `k/v` 的 token 长度

这一章里最关键的输入张量 shape 是：

- `block_mask_map`: `[H_kv, R, C]`
- `block_row_sz`: `[H_kv, R]`
- `block_col_sz`: `[H_kv, C]`
- 外部 `q`: `[H_qo, qo_len, D]`
- 外部 `k/v`: `[H_kv, kv_len, D]`

最重要的“视角切换”是：

- `block_*` 系列仍然是 **block 视角**
- `qo_indptr / paged_kv_indptr / paged_kv_indices` 已经变成 **token 视角**

并且，这套实现里还有一个必须提前记住的概念：

> 一个 **逻辑 request** = 一个 `(kv_head, row_block)` 对。

所以后面很多一维数组的长度其实不是原始 batch size，而是：

```text
num_logical_requests = H_kv * R
```

## 2. `__init__()`：wrapper 初始化时到底缓存了什么

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
    第一组状态：workspace
    - float workspace: 后面给 split-k 中间结果 / merge 临时缓冲使用
    - int workspace:   后面给 request/tile/merge 等整型调度数组使用
    - pinned CPU buf:  后面给 plan 阶段的 host <-> device 协作用
    '''
    self._float_workspace_buffer = float_workspace_buffer          # [workspace_bytes] 1D raw buffer, CUDA
    self.device = float_workspace_buffer.device                    # cuda:x
    self._workspace_size = (
        float_workspace_buffer.numel() * float_workspace_buffer.element_size()
    )                                                              # int，总 workspace 字节数
    self._int_workspace_buffer = torch.empty(
        (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
    )                                                              # [8MiB] uint8 raw buffer, CUDA
    self._pin_memory_int_workspace_buffer = torch.empty(
        self._int_workspace_buffer.shape,
        dtype=torch.uint8,
        pin_memory=True,
        device="cpu",
    )                                                              # [8MiB] uint8 raw buffer, pinned CPU

    '''
    第二组状态：plan() 完成后常驻的 paged prefill metadata
    这些成员在 __init__ 时都还是 None，只是先占位
    '''
    self._kv_layout = "NHD"                                        # paged KV layout，后面 run() 直接透传
    self._qo_indptr: Optional[torch.Tensor] = None                 # [H_kv * R + 1]
    self._paged_kv_indptr_buf: Optional[torch.Tensor] = None       # [H_kv * R + 1]
    self._paged_kv_indices_buf: Optional[torch.Tensor] = None      # [nnz_tokens]
    self._paged_kv_last_page_len: Optional[torch.Tensor] = None    # [H_kv * R]

    '''
    第三组状态：执行路径选择
    '''
    self._backend = backend                                        # "auto" / "fa2" / "fa3"
```

这里有三个很容易误解的点：

1. `float_workspace_buffer` 名字里虽然有 `float`，但外部调用时通常直接传一个 1D `torch.uint8` byte buffer。
   它叫这个名字，是因为后面某些偏移会被 reinterpret 成浮点临时区；不是说 Python 侧一定要传 float Tensor。

2. `_int_workspace_buffer` / `_pin_memory_int_workspace_buffer` 都不是“有固定结构的 Tensor”。
   它们本质上是后面 planner 和 runtime 会切片复用的原始内存池。

3. `_qo_indptr` / `_paged_kv_*` 这些成员在 `__init__` 阶段都还不存在真实内容。
   只有 `plan()` 跑完以后，它们才会变成真正可执行的 metadata。

4. `_kv_layout = "NHD"` 不是随手写的字符串。
   在当前 paged KV 视角里，它对应的物理布局可以理解成：
   `[num_pages, page_size, num_heads, head_dim]`。
   因为后面 `run()` 会把 `k/v` 重排成 `[H_kv * kv_len, 1, 1, D]`，
   所以这里实际落到的就是：
   - `num_pages = H_kv * kv_len`
   - `page_size = 1`
   - `num_heads = 1`
   - `head_dim = D`

所以 `VariableBlockSparseAttentionWrapper` 的角色不是“自己实现一套 attention 算法”，而是：

- 持有 workspace
- 持有 plan 后的 metadata
- 持有后续 runtime 模块句柄

它更像“variable block -> paged prefill”的状态容器。

## 3. 为什么这里会出现 `H_kv * R` 这个长度

`plan()` 之后最常出现的长度是：

```text
H_kv * R
```

因为这套实现把每个 `(kv_head, row_block)` 都看成一个单独的逻辑 request。

如果用一个很小的例子：

```text
H_kv = 2
R    = 3
```

那么逻辑 request 顺序会被拍平成：

```text
(h0, r0), (h0, r1), (h0, r2), (h1, r0), (h1, r1), (h1, r2)
```

于是：

- `qo_indptr.shape = [H_kv * R + 1] = [7]`
- `paged_kv_indptr.shape = [H_kv * R + 1] = [7]`
- `paged_kv_last_page_len.shape = [H_kv * R] = [6]`

这一步非常关键，因为它决定了后面的 planner 和 kernel 不再按“原始 batch”理解数据，而是按“逻辑 request 序列”理解数据。

## 4. `plan()` 前半段：先把 block 视角翻译成 token 视角

[`wrapper.py:L131-L252`](src/wrapper_py.md#__codelineno-0-131) 是整个 Python 层的主线，但这一章只看它的前半段：

```python
q_data_type = canonicalize_torch_dtype(q_data_type)               # 统一 q dtype 表达
if kv_data_type is None:                                          # 默认 Q/KV 同 dtype
    kv_data_type = q_data_type
kv_data_type = canonicalize_torch_dtype(kv_data_type)
self._o_dtype = q_data_type                                       # 输出 dtype 跟 q 一致

if logits_soft_cap is None:                                       # None -> 0.0，表示不启用 soft cap
    logits_soft_cap = 0.0

num_blocks_row = block_row_sz.shape[-1]                           # R
num_blocks_col = block_col_sz.shape[-1]                           # C

'''
第一阶段：把 block 级描述翻译成 paged prefill 需要的 token 级 metadata
'''
qo_indptr = build_qo_indptr(block_row_sz)                         # [H_kv * R + 1]
qo_indptr_host = qo_indptr.to("cpu", non_blocking=non_blocking)   # host mirror，给 C++ plan 用
last_block_len = build_last_page_len(
    block_mask_map,
    num_blocks_row,
    num_kv_heads,
)                                                                 # [H_kv * R]，这里恒为 1

kv_indptr, kv_indices = block_mask_map_to_expanded_indices(
    block_mask_map,
    block_col_sz,
)                                                                 # kv_indptr:[H_kv * R + 1], kv_indices:[nnz_tokens]
kv_indptr_host = kv_indptr.to("cpu", non_blocking=non_blocking)   # host mirror
kv_indices_host = kv_indices.to("cpu", non_blocking=non_blocking) # host mirror

self._qo_indptr = qo_indptr.to(self.device, non_blocking=non_blocking)
self._paged_kv_indptr_buf = kv_indptr.to(self.device, non_blocking=non_blocking)
self._paged_kv_indices_buf = kv_indices.to(                       # [nnz_tokens]
    self.device,
    non_blocking=non_blocking,
)
self._paged_kv_last_page_len = last_block_len.to(                 # [H_kv * R]
    self.device,
    non_blocking=non_blocking,
)
torch.cuda.synchronize()                                          # 后面马上会读 host mirror，先同步

'''
第二阶段：先把执行分支收敛清楚
模块获取与真正的 plan 生成放到下一章
'''
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
```

这一段代码从执行顺序上可以拆成两步：

### 第一步：把 block 描述翻译成 token 级 metadata

真正属于 variable block 语义的，是这三件事：

- `build_qo_indptr(block_row_sz)`
- `build_last_page_len(...)`
- `block_mask_map_to_expanded_indices(block_mask_map, block_col_sz)`

这三步之后，Python 层已经得到了底层 paged prefill 真正会消费的：

- query 侧 CSR 分段：`qo_indptr`
- KV 侧 CSR 分段：`paged_kv_indptr`
- KV 侧 token 索引：`paged_kv_indices`
- 每行最后一页长度：`paged_kv_last_page_len`

### 第二步：把 backend 解析清楚

这一章先停在：

- metadata 已经准备好
- `backend` 已经从 `auto` 收敛为真正可执行的分支

但还没有进入：

- JIT 模块生成
- `module.plan(...)`
- C++ `BatchPrefillWithKVCachePlan(...)`

这些属于下一章。

## 5. `page_size=1`：为什么它是整个设计的关键压缩点

这一组实现最关键的设计选择，就是：

```text
page_size = 1
```

这不是一个细节，而是把“variable block”翻译到底层 paged prefill 的核心桥梁。

在 [`metadata.py:L32-L42`](src/metadata_py.md#__codelineno-0-32) 里，`last_page_len` 被直接固定成了 1：

```python
def build_last_page_len(
    block_mask_map: torch.Tensor,
    num_blocks_row: int,
    num_kv_heads: int,
) -> torch.Tensor:
    return torch.full(
        (num_blocks_row * num_kv_heads,),                          # [H_kv * R]
        1,                                                         # 每行最后一页长度固定为 1
        dtype=torch.int32,
        device=block_mask_map.device,
    )
```

它意味着：

- 每个 token 都被视作一个 page
- 不需要再为“页内还剩几个 token”维护复杂逻辑
- 变长 block 问题被转写成了“token 索引选择”问题

所以后面底层 kernel 根本不需要知道：

- 某个块原来长 64
- 某个块原来长 192

它只需要知道：

- 这一行 query 要看哪些 token
- 这些 token 在 paged KV 里的页号是什么

## 6. `metadata.py`：真正把 block 稀疏图拆成 token 索引

[`metadata.py:L22-L88`](src/metadata_py.md#__codelineno-0-22) 虽然只有几十行，但它实际上完成了整个 variable block 语义最核心的翻译：

```python
def build_qo_indptr(block_row_sz: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=block_row_sz.device),   # [1]
            torch.cumsum(block_row_sz.flatten(), dim=0, dtype=torch.int32),  # [H_kv * R]
        ],
        dim=0,
    )                                                                        # [H_kv * R + 1]


def block_mask_map_to_expanded_indices(
    block_mask_map: torch.Tensor,  # [H_kv, R, C]
    block_col_sz: torch.Tensor,    # [H_kv, C]
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = block_mask_map.device
    dtype_i = torch.int32

    '''
    第一段：每个 (head, row) 最终会展开出多少个 KV token
    '''
    row_lengths = (block_mask_map * block_col_sz[:, None, :]).sum(-1)        # [H_kv, R]
    kv_indptr = torch.cat(
        [
            torch.zeros(1, dtype=dtype_i, device=device),                     # [1]
            torch.cumsum(row_lengths.flatten(), 0),                           # [H_kv * R]
        ],
        dim=0,
    )                                                                        # [H_kv * R + 1]

    '''
    第二段：每个列块在各自 head 内部的 token 起点
    '''
    col_offset = torch.cumsum(block_col_sz.to(dtype_i), 1) - block_col_sz    # [H_kv, C]
    head_len = block_col_sz.sum(1, dtype=dtype_i)                             # [H_kv]
    head_offset = torch.cumsum(head_len, 0) - head_len                        # [H_kv]

    '''
    第三段：把所有被选中的 (head,row,col) 列块展开成 token 索引
    '''
    h_idx, _, c_idx = block_mask_map.nonzero(as_tuple=True)                   # [N]
    lengths = block_col_sz[h_idx, c_idx].to(dtype_i)                         # [N]
    base = head_offset[h_idx] + col_offset[h_idx, c_idx]                     # [N]

    cum = torch.cumsum(lengths, 0)                                            # [N]
    starts = torch.repeat_interleave(cum - lengths, lengths)                 # [nnz_tokens]
    offsets_within = torch.arange(cum[-1], device=device) - starts           # [nnz_tokens]
    kv_indices = torch.repeat_interleave(base, lengths) + offsets_within      # [nnz_tokens]

    return kv_indptr.to(dtype=dtype_i, device=device), kv_indices.to(
        dtype=dtype_i, device=device
    )
```

这段代码做了两件不同层级的事情：

### 第一件事：给 Q 侧建立 CSR 分段

`build_qo_indptr` 的含义非常直接：

- `block_row_sz` 还在 block 视角：每个 row block 有多少个 query token
- `flatten()` 把 `(kv_head, row_block)` 顺序拍平成逻辑 request 序列
- `cumsum()` 把“每段长度”变成“每段起止边界”

例如：

```text
block_row_sz = [[2, 3]]
```

那么：

```text
qo_indptr = [0, 2, 5]
```

表示：

- 第 0 个逻辑 request 使用 query token `[0:2)`
- 第 1 个逻辑 request 使用 query token `[2:5)`

### 第二件事：给 KV 侧建立 token 索引

`block_mask_map_to_expanded_indices` 做的是更核心的那一步：

- `block_mask_map` 只决定“哪些列块要看”
- `block_col_sz` 决定“每个列块实际有多少个 token”
- `kv_indices` 最终给出的已经不是 block 编号，而是底层 runtime 真正要读取的 token 编号

也就是说，到了 `kv_indices` 这里，block 的概念已经被拆平了。

如果把它翻译成人话：

1. 先用 `row_lengths` 算出每个 `(head,row)` 一共会看到多少个 KV token。
2. 再用 `col_offset + head_offset` 算出每个被选中列块在全局扁平 KV 空间里的起点。
3. 最后把每个变长列块逐 token 展开，生成真正的一维 token 索引数组。

所以后面的 runtime/JIT/C++ 根本不关心“块长是什么”，它们只关心：

- 每一行 query 对应哪些 token
- 这些 token 在扁平 paged KV 结构里的编号是什么

## 7. backend 边界：接口兼容，但执行路径收敛

虽然 runtime/JIT 细节放在下一章，这里先把 backend 边界钉清楚。见 [`backend.py:L28-L50`](src/backend_py.md#__codelineno-0-28)：

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

这段逻辑和 `plan()` 前半段紧密相关，因为它解释了为什么当前 wrapper 看起来支持三种 backend，但真正继续执行时只会落到 `fa2`：

- `auto` 仍然保留上游同名接口
- 运行期即使检测到 `fa3` 能力，也不会继续走
- 当前闭环只允许 `fa2`

这正是“对外保留 `auto/fa2/fa3`，内部只保留 `fa2` 最小闭环”的第一现场。

## 8. 这一章结束时，系统里已经有什么

当这一章讲完时，执行链路停在下面这个状态：

- wrapper 已经持有 float/int workspace
- variable block 稀疏图已经被翻译成：
  - `qo_indptr`
  - `paged_kv_indptr`
  - `paged_kv_indices`
  - `paged_kv_last_page_len`
- backend 已经被解析为当前真正可执行的 `fa2`

但还没有发生两件事：

- 还没展开 batch prefill 模块是怎样生成/加载出来的
- 还没展开 C++ `plan` 和后续 `run` 的执行细节

这两部分分别在后两章展开。

## 小结

这一章按执行顺序做了四件事：

1. 构造 wrapper，准备 workspace 和生命周期状态。
2. 在 `plan()` 前半段，把 variable block 稀疏图翻译成 token 级 metadata。
3. 用 `page_size=1` 把问题改写成普通 paged prefill。
4. 在 Python 层完成 backend 解析，为后面的模块准备和调度生成铺路。

后面的 runtime、JIT、C++、CUDA 都是在消费这一章已经准备好的结果。
