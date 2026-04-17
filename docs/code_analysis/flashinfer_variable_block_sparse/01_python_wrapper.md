---
tags:
  - CUDA
  - CUTLASS
  - Sparse Attention
  - Flash Attention
---

# Python Wrapper 实现

**源码位置**: [VariableBlockSparseAttentionWrapper](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/sparse.py#L649-L1168)

## 类结构

`VariableBlockSparseAttentionWrapper` 是一个无状态计算封装，核心状态在 `plan()` 阶段生成并缓存：

```python
class VariableBlockSparseAttentionWrapper:
    def __init__(self, float_workspace_buffer, backend="auto"):
        self._float_workspace_buffer = float_workspace_buffer       # 128MB workspace
        self._int_workspace_buffer = torch.empty((8*1024*1024,), dtype=torch.uint8, ...)
        self._qo_indptr = None           # plan 后填充
        self._paged_kv_indptr_buf = None # plan 后填充
        self._paged_kv_indices_buf = None # plan 后填充
        self._paged_kv_last_page_len = None # plan 后填充
        self._backend = backend          # "auto" / "fa2" / "fa3"
```

## plan() 阶段

### 输入

```python
def plan(self,
    block_mask_map: torch.Tensor,  # [num_kv_heads, MB, NB] bool
    block_row_sz: torch.Tensor,    # [num_kv_heads, MB] int
    block_col_sz: torch.Tensor,    # [num_kv_heads, NB] int
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    ...
)
```

- `block_mask_map[h, i, j] = True` 表示第 h 个 KV head 的第 i 个 Q block 需要与第 j 个 K block 计算注意力
- `block_row_sz[h, i]` 是第 h 个 head 的第 i 个 Q block 包含的 token 数
- `block_col_sz[h, j]` 是第 h 个 head 的第 j 个 K block 包含的 token 数
- 每个 KV head 可以有完全不同的稀疏模式和 block 大小

### 步骤 1：构建 Q 侧分段索引

```python
'''
把 block_row_sz 展平后做 cumsum，得到每个 (head, q_block) 在扁平序列中的起止位置。
flatten 的顺序是 head-major：先 head 0 的所有 Q block，再 head 1 的所有 Q block，...
'''
qo_indptr = torch.cat([
    torch.zeros(1, dtype=torch.int32, device=block_row_sz.device),
    torch.cumsum(block_row_sz.flatten(), dim=0, dtype=torch.int32),
], dim=0)  # [num_kv_heads * MB + 1]
```

### 步骤 2：page_size = 1 的设计

```python
last_block_len = torch.full(
    (num_blocks_row * num_kv_heads,), 1,
    dtype=torch.int32, device=block_mask_map.device,
)  # We use page_size == 1 for variable length support
```

这是整个映射的关键。在标准 paged attention 中，KV cache 被划分为固定大小的 page（如 page_size=16），kernel 通过 page table 间接寻址。将 page_size 设为 1 意味着：

- 每个 token 就是一个独立的 page
- page table（`kv_indices`）退化为 token 级索引数组
- kernel 可以按任意粒度访问 KV token，不受 page 边界约束

这样，无论 block 大小是 32、96 还是 200，都可以精确表示——只需要在 `kv_indices` 中列出该 block 对应的所有 token 索引即可。

### 步骤 3：block 级稀疏图 → token 级索引

这是 `plan()` 中最核心的数据变换，由内部函数 [`_block_mask_map_to_expanded_indices()`](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/sparse.py#L861-L906) 完成。下面结合一个具体数值例子逐步拆解。

**例子设定**：1 个 KV head（H=1），3 个 Q block（R=3），4 个 K block（C=4）

```
block_col_sz = [[3, 1, 2, 2]]           # K block 大小：3, 1, 2, 2（共 8 个 K token）
block_row_sz = [[2, 3, 1]]              # Q block 大小：2, 3, 1（共 6 个 Q token）

block_mask_map = [[[0, 0, 1, 0],        # Q0 只看 K2
                   [1, 0, 0, 1],        # Q1 看 K0 和 K3
                   [0, 1, 1, 0]]]       # Q2 看 K1 和 K2
```

#### 3.1 `row_lengths` → `kv_indptr`：每个 Q block 需要看多少 KV token

```python
row_lengths = (block_mask_map * block_col_sz[:, None, :]).sum(-1)  # [H, R]
kv_indptr = torch.cat([
    torch.zeros(1, dtype=torch.int32, device=device),
    torch.cumsum(row_lengths.flatten(), 0),
], dim=0)
```

`block_col_sz[:, None, :]` 广播为 `[1,1,4]`，与 `block_mask_map [1,3,4]` 逐元素相乘——未激活的列被置零，激活的列保留其 token 数。对最后一维求和得到每行的 KV token 总数：

```
row_lengths = [[2, 5, 3]]
  Q0: K2(size=2)       → 2
  Q1: K0(3) + K3(2)    → 5
  Q2: K1(1) + K2(2)    → 3
```

flatten 后 cumsum 加前导 0，得到 CSR indptr：

```
kv_indptr = [0, 2, 7, 10]
```

含义：Q0 的 KV token 在 `kv_indices[0:2]`，Q1 在 `[2:7]`，Q2 在 `[7:10]`。

#### 3.2 `col_offset` 与 `head_offset`：建立 token 地址映射

```python
'''
col_offset: 每个 K block 在其 head 的 token 序列中的起始位置
head_offset: 多 head 场景下，每个 head 在全局 token 序列中的起始位置
'''
col_offset = torch.cumsum(block_col_sz.to(torch.int32), 1) - block_col_sz  # [H, C]
head_len = block_col_sz.sum(1, dtype=torch.int32)          # [H]
head_offset = torch.cumsum(head_len, 0) - head_len         # [H]
```

`cumsum - self` 是一个常见的 exclusive prefix sum 技巧：

```
cumsum([3,1,2,2]) = [3, 4, 6, 8]
col_offset         = [3-3, 4-1, 6-2, 8-2] = [0, 3, 4, 6]
```

这就是把 4 个变长 K block 拼成一条连续 token 序列时的偏移表：K0 从 token 0 开始，K1 从 3，K2 从 4，K3 从 6。

`head_offset` 在多 head 场景下累加各 head 的 token 总数。单 head 时 `head_offset = [0]`。

#### 3.3 `nonzero` → `lengths` + `base`：提取激活 block 的元信息

```python
'''
nonzero 按行优先顺序返回所有激活 (h,r,c) 三元组。
行优先保证同一 Q row 的激活 block 连续排列，与 kv_indptr 分段对应。
'''
h_idx, r_idx, c_idx = block_mask_map.nonzero(as_tuple=True)
lengths = block_col_sz[h_idx, c_idx].to(torch.int32)       # 每个激活 block 的 token 数
base = head_offset[h_idx] + col_offset[h_idx, c_idx]       # 每个激活 block 的全局起始 token
```

```
nonzero 结果（行优先）:
  (0,0,2), (0,1,0), (0,1,3), (0,2,1), (0,2,2)

lengths = [2, 3, 2, 1, 2]    # 对应 K block 的 size
base    = [4, 0, 6, 3, 4]    # 对应 col_offset 查表
```

| 激活 block | K block | lengths | base（head_offset + col_offset） |
|---|---|---|---|
| (0,0,2) | K2 | 2 | 0 + 4 = 4 |
| (0,1,0) | K0 | 3 | 0 + 0 = 0 |
| (0,1,3) | K3 | 2 | 0 + 6 = 6 |
| (0,2,1) | K1 | 1 | 0 + 3 = 3 |
| (0,2,2) | K2 | 2 | 0 + 4 = 4 |

#### 3.4 token 级展开：`repeat_interleave` 变长段展开技巧

这一步将 5 个变长 block（`lengths=[2,3,2,1,2]`，`base=[4,0,6,3,4]`）展开为 10 个 token 索引。这是整个函数最精巧的部分。

```python
'''
变长段展开的向量化实现：
1. cumsum(lengths) → 每段结尾位置
2. repeat_interleave(段起始, lengths) → 每个 token 所属段的起始
3. arange(total) - starts → 段内偏移
4. repeat_interleave(base, lengths) + offsets → 全局 token 索引
'''
cum = torch.cumsum(lengths, 0)
starts = torch.repeat_interleave(cum - lengths, lengths)
offsets_within = torch.arange(cum[-1], device=device) - starts
kv_indices = torch.repeat_interleave(base, lengths) + offsets_within
```

逐步推导：

**`cum`** = cumsum(`[2,3,2,1,2]`) = `[2, 5, 7, 8, 10]`：每个 block 在展开序列中的结束位置。

**`cum - lengths`** = `[0, 2, 5, 7, 8]`：每个 block 的起始位置。

**`starts`** = `repeat_interleave([0,2,5,7,8], [2,3,2,1,2])`：

```
block 0 (len=2): [0, 0]
block 1 (len=3): [2, 2, 2]
block 2 (len=2): [5, 5]
block 3 (len=1): [7]
block 4 (len=2): [8, 8]
→ starts = [0, 0, 2, 2, 2, 5, 5, 7, 8, 8]
```

**`offsets_within`** = `arange(10) - starts`：

```
[0,1,2,3,4,5,6,7,8,9] - [0,0,2,2,2,5,5,7,8,8] = [0,1,0,1,2,0,1,0,0,1]
```

每个 token 在其 block 内的局部偏移。

**`kv_indices`** = `repeat_interleave(base, lengths) + offsets_within`：

```
base 展开: [4,4, 0,0,0, 6,6, 3, 4,4]
+ offset:  [0,1, 0,1,2, 0,1, 0, 0,1]
= 结果:    [4,5, 0,1,2, 6,7, 3, 4,5]
```

#### 3.5 结果验证

```
kv_indptr  = [0, 2, 7, 10]
kv_indices = [4, 5, 0, 1, 2, 6, 7, 3, 4, 5]
```

配合 `kv_indptr` 分段读取：

- Q0 → `kv_indices[0:2] = [4, 5]` → K2 的 token 4,5 ✓
- Q1 → `kv_indices[2:7] = [0, 1, 2, 6, 7]` → K0 的 token 0,1,2 + K3 的 token 6,7 ✓
- Q2 → `kv_indices[7:10] = [3, 4, 5]` → K1 的 token 3 + K2 的 token 4,5 ✓

这就是 paged attention 看到的输入：对于第 i 个 Q block，它的 KV 范围是 `kv_indices[kv_indptr[i]:kv_indptr[i+1]]`。配合 `page_size=1`，每个索引直接指向一个 KV token 的全局位置。

### 步骤 4：获取 kernel module 并 plan

```python
'''
获取 JIT 编译的 batch prefill module（根据 backend/dtype/head_dim 等参数选择）
'''
self._cached_module = get_batch_prefill_module(self._backend, *get_module_args)

'''
调用 module 的 plan 函数，传入 CSR 索引结构。
注意 page_size=1，num_kv_heads=1（因为 head 维度已折叠进 batch）
'''
self._plan_info = self._cached_module.plan(
    self._float_workspace_buffer,
    self._int_workspace_buffer,
    self._pin_memory_int_workspace_buffer,
    qo_indptr_host,
    kv_indptr_host,
    kv_lens_arr_host,
    qo_indptr_host[-1].item(),           # total_num_rows
    num_blocks_row * num_kv_heads,        # batch_size
    num_qo_heads // num_kv_heads,         # gqa_group_size
    1,                                     # num_kv_heads（已折叠）
    1,                                     # page_size
    False,                                 # is_cuda_graph_enabled
    head_dim, head_dim,
    causal,
    -1,                                    # window_left
)
```

## run() 阶段

### Q/K/V reshape

run() 的输入是 `[num_qo_heads, qo_len, head_dim]` 格式。为了配合 paged attention 的 batch 处理，需要把 KV head 维度折叠进 batch：

```python
'''
Q reshape: 把 GQA group 维度提取出来
[num_kv_heads * gqa_group_size, qo_len, D]
→ [num_kv_heads * qo_len, gqa_group_size, D]
'''
q = einops.rearrange(q,
    "(num_kv_heads gqa_group_size) qo_len head_dim "
    "-> (num_kv_heads qo_len) gqa_group_size head_dim",
    num_kv_heads=self._num_kv_heads,
).contiguous()

'''
K/V reshape: 每个 token 变成一个独立的 "page"
[num_kv_heads, kv_len, D] → [num_kv_heads * kv_len, 1, 1, D]
                              ^^^^^^^^^^^^^^^^^^^^^^^^^
                              num_pages=kv_len*H, page_size=1, num_kv_heads=1, D
'''
k = einops.rearrange(k,
    "num_kv_heads kv_len head_dim -> (num_kv_heads kv_len) 1 1 head_dim",
).contiguous()
v = einops.rearrange(v,
    "num_kv_heads kv_len head_dim -> (num_kv_heads kv_len) 1 1 head_dim",
).contiguous()
```

### paged_run 执行

```python
self._cached_module.paged_run(
    self._float_workspace_buffer,
    self._int_workspace_buffer,
    self._plan_info,
    q, k, v,
    self._qo_indptr,               # Q 分段
    self._paged_kv_indptr_buf,     # KV 分段（CSR indptr）
    self._paged_kv_indices_buf,    # KV token 索引（CSR indices）
    self._paged_kv_last_page_len,  # 全 1（page_size=1）
    out, lse,
    self._mask_mode,               # NON_CAUSAL 或 CAUSAL
    TensorLayout["NHD"].value,
    -1,                            # window_left
    enable_pdl,
    ...
    logits_soft_cap, sm_scale,
    ...
    rope_scale, rope_theta,
    0,                             # token_pos_in_items_len
    self._workspace_size,
)
```

### 输出 reshape

```python
'''
[num_kv_heads * qo_len, gqa_group_size, D]
→ [num_kv_heads * gqa_group_size, qo_len, D]
恢复回 [num_qo_heads, qo_len, D] 格式
'''
out = einops.rearrange(out,
    "(num_kv_heads qo_len) gqa_group_size head_dim "
    "-> (num_kv_heads gqa_group_size) qo_len head_dim",
    num_kv_heads=self._num_kv_heads,
).contiguous()
```

## GQA 支持的实现方式

标准 GQA 中，`num_qo_heads = num_kv_heads × gqa_group_size`。wrapper 的处理方式是：

1. `plan()` 中 `batch_size = num_blocks_row × num_kv_heads`，`num_qo_heads = gqa_group_size`，`num_kv_heads = 1`
2. 即把每个 KV head 视为独立的 batch item，同一组内的多个 Q head 作为 GQA group

这样 kernel 内部的 GQA 机制自动处理同组 Q head 共享 K/V 的逻辑，无需 wrapper 额外处理。

## 小结

`VariableBlockSparseAttentionWrapper` 的设计精髓在于**不引入新的 kernel**，而是通过索引变换把 variable block sparse 问题映射到已有的 paged attention 基础设施上：

- `page_size = 1` 消除了 page 边界约束
- `_block_mask_map_to_expanded_indices` 把 block 级稀疏图展开为 token 级 CSR 索引
- KV head 折叠进 batch 维度，让每个 head 拥有独立的稀疏模式
- 所有计算复用 batch prefill module 的 `paged_run`

开销在于 plan() 阶段的索引展开（纯 Python/PyTorch tensor 操作），但相比 attention 主计算通常可忽略。
