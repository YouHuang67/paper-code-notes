---
tags:
  - Triton
  - Sparse Attention
  - Bitonic Sort
---
# Top-k 选择机制

本节分析 NSA 的核心创新：Online Top-k 选择，避免物化完整注意力矩阵。

## 1. 选择策略

基于压缩注意力的分数，选择 top-k 个最重要的 block。重要性分数定义为 GQA 组内的注意力概率之和：

$$
\text{score}_c = \sum_{g=1}^{G} p_{t,c}^{(g)}, \quad p_{t,c} = \frac{\exp(s_{t,c})}{\sum_{j} \exp(s_{t,j})}
$$

特殊处理：当前 token 所在的 block 强制设为最高分（确保局部信息不丢失）。

## 2. Bitonic Sort

**源码位置**: [utils.py#L1-L90](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/utils.py)

Bitonic Sort 是一种适合 GPU 并行的排序算法，复杂度 $O(n \log^2 n)$。

### 2.1 基本原理

Bitonic 序列：先单调递增后单调递减（或反之）的序列。

核心操作是 **compare-and-swap**：比较两个元素，根据排序方向交换。

### 2.2 Compare-and-Swap 实现

```python
@triton.jit
def _compare_and_swap(
    x,                      # 待排序的值
    ids,                    # 对应的索引
    flip,                   # 排序方向
    i: tl.constexpr,        # 当前阶段
    n_dims: tl.constexpr,   # 总维度 log2(n)
):
    '''
    Bitonic sort 的核心操作：compare-and-swap
    将数组 reshape 成 [outer, 2, inner] 形状
    比较相邻的 left/right 对，根据 flip 方向决定是否交换
    '''
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = tl.reshape(x, shape)

    # 分离 left 和 right
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)

    # 同样处理索引
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)

    # compare-and-swap
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    # 条件交换：(left > right) XOR flip
    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))

    return ret.to(x.dtype, bitcast=True), new_ids
```

### 2.3 Bitonic Merge

```python
@triton.jit
def _bitonic_merge(
    x,
    ids,
    stage: tl.constexpr,    # 当前 merge 阶段
    order: tl.constexpr,    # 排序方向: 2=交替, True=升序, False=降序
    n_dims: tl.constexpr,
):
    '''
    Bitonic merge: 将两个 bitonic 序列合并成一个排序序列
    stage: 当前处理的子序列大小 (2^stage)
    order: 2 表示交替方向（构建 bitonic 序列），True/False 表示最终排序方向
    '''
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)

    # flip 控制每个子序列的排序方向
    if order == 2:
        shape: tl.constexpr = [n_outer * 2**(n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order

    # 执行 stage 轮 compare-and-swap
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)

    return x, ids
```

## 3. Online Top-k 内核

**源码位置**: [parallel.py#L339-L458](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L339-L458)

### 3.1 内核签名

```python
@triton.jit
def parallel_nsa_kernel_topk(
    q,              # Query
    k,              # Compressed Key
    lse,            # 预计算的 LSE（可选）
    scale,
    block_indices,  # 输出: top-k block 索引
    offsets,
    token_indices,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,    # top-k 的 k 值
    BC: tl.constexpr,   # 每次处理的 chunk 数
    BS: tl.constexpr,   # block size
    BK: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
):
```

### 3.2 LSE 计算（可选）

如果没有预计算 LSE，先通过 online softmax 计算：

```python
if lse is not None:
    b_lse = tl.load(lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G))
else:
    '''
    与压缩注意力相同的 online softmax
    计算 LSE = log(sum(exp(s - max))) + max
    '''
    b_m = tl.full([G], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([G], dtype=tl.float32)

    for i_c in range(0, NC, BC):
        o_c = i_c + tl.arange(0, BC)
        p_k = tl.make_block_ptr(k + (boc * H + i_h) * K, (K, TC), (1, H*K),
                                (0, i_c), (BK, BC), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where((o_c < NC)[None, :], b_s, float('-inf'))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = tl.exp(b_mp - b_m)
        b_p = tl.exp(b_s - b_m[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)

    if NC == 0:
        b_lse = tl.zeros([G], dtype=tl.float32)
    else:
        b_lse = b_m + tl.log(b_acc)
```

### 3.3 Online Top-k 选择

核心创新：边迭代边维护 top-k，避免存储所有分数。

```python
'''
Online top-k 选择
维护一个大小为 BC 的 buffer，迭代过程中不断合并新的候选
BC = 2 * S，前半部分存 top-k 结果，后半部分存当前 batch 的候选
'''
# [BC] 存储重要性分数和对应索引
b_i = tl.full([BC], -1, dtype=tl.float32)   # importance scores
o_i = tl.zeros([BC], dtype=tl.int32)         # block indices
m_i = tl.arange(0, BC) < BC//2               # mask: 前半部分

for i_c in range(0, i_t // BS + 1, BC):
    o_c = i_c + tl.arange(0, BC)

    # 加载 K 并计算注意力分数
    p_k = tl.make_block_ptr(k + (boc * H + i_h) * K, (K, TC), (1, H*K),
                            (0, i_c), (BK, BC), (0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_s = tl.dot(b_q, b_k)                              # [G, BC]

    # causal mask: 只考虑当前 token 之前的 block
    b_s = tl.where((i_t // BS > o_c)[None, :], b_s, float('-inf'))

    # 计算重要性分数
    # 当前 block 设为 1.0（最高优先级），其他用 softmax 概率
    b_p = tl.where((i_t // BS == o_c)[None, :], float(1.0),
                   tl.exp(b_s - b_lse[:, None]))        # [G, BC]
    b_i, b_ip = tl.sum(b_p, 0), b_i                     # [BC] GQA 组内求和
    o_i, o_ip = tl.where(o_c <= i_t // BS, o_c + 1, 0), o_i

    # Bitonic sort 排序当前 batch
    n_dims: tl.constexpr = tl.standard._log2(b_i.shape[0])
    for i in tl.static_range(1, n_dims):
        b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), i, 2, n_dims)

    # 与历史 top-k 合并
    if i_c != 0:
        # 降序排列当前 batch
        b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, False, n_dims)

        # 合并: 前半部分取历史 top-k，后半部分取当前 batch
        b_i_new = b_ip * m_i + b_i * (1 - m_i)
        o_i_new = o_ip * m_i + o_i * (1 - m_i)

        # 对合并后的序列排序（升序），top-k 在前半部分
        b_i, o_i = _bitonic_merge(b_i_new, o_i_new.to(tl.int32), n_dims, True, n_dims)
    else:
        b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, True, n_dims)
```

### 3.4 提取 Top-k 结果

```python
'''
从排序后的 buffer 中提取 top-S 个索引
buffer 布局: [BC] = [S, S] (两组各 S 个)
取第一组的 top-S
'''
m_top = tl.arange(0, BC//S) == 0                        # [BC//S] 选择第一组
b_top = tl.sum(m_top[:, None] * tl.reshape(o_i - 1, [BC//S, S]), 0)  # [S]

p_b = tl.make_block_ptr(block_indices + (bos + i_t) * H*S, (H*S,), (1,),
                        (i_h * S,), (S,), (0,))
tl.store(p_b, b_top.to(p_b.dtype.element_ty))
```

## 4. 设计要点

### 4.1 避免物化

传统做法需要存储所有 $O(T \cdot C)$ 个注意力分数再排序，内存开销大。
Online top-k 只维护 $O(S)$ 大小的 buffer，边计算边筛选。

### 4.2 Bitonic Sort 的优势

- **完全并行**: 所有 compare-and-swap 可同时执行
- **无分支**: 通过 XOR 实现条件交换，适合 GPU
- **固定模式**: 访存模式编译期确定，便于优化

### 4.3 GQA 聚合

重要性分数通过 `tl.sum(b_p, 0)` 在 G 个 Q heads 上聚合，确保 KV head 级别的一致选择。
