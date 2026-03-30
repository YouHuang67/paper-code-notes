---
tags:
  - Triton
  - Sparse Attention
---
# 选择注意力实现

本节分析 NSA 的选择注意力（Selected Attention），对 top-k 选中的 block 执行精细注意力计算。

## 1. 算法概述

给定 top-k 选择的 block indices $\mathcal{I}_t$，对原始 KV 执行稀疏注意力：

$$
\mathbf{o}_{\text{slc},t} = \sum_{c \in \mathcal{I}_t} \sum_{i \in \text{block}_c} \text{softmax}\left(\frac{\mathbf{q}_t \cdot \mathbf{K}_{\mathcal{I}_t}^\top}{\sqrt{d}}\right)_i \mathbf{v}_i
$$

## 2. 前向内核

**源码位置**: [parallel.py#L472-L556](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L472-L556)

### 2.1 内核签名

```python
@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None,
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor),
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BS', 'BK', 'BV'],
)
@triton.jit
def parallel_nsa_fwd_kernel(
    q,              # Query: [B, T, HQ, K]
    k,              # Key: [B, T, H, K]
    v,              # Value: [B, T, H, V]
    o,              # Output: [B, T, HQ, V]
    lse,            # Log-Sum-Exp: [B, T, HQ]
    scale,
    block_indices,  # 选中的 block 索引: [B, T, H, S]
    block_counts,   # 每个 token 选中的 block 数（可选）
    offsets,
    token_indices,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,    # 最大选择数
    BS: tl.constexpr,   # block size
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr
):
```

### 2.2 核心计算逻辑

```python
'''
Grid 配置: (T, NV, B*H)
每个 thread block 处理一个 token 的选择注意力
'''
i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
i_b, i_h = i_bh // H, i_bh % H

# 变长序列支持
if USE_OFFSETS:
    i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), \
               tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
    bos, eos = tl.load(offsets + i_n).to(tl.int32), \
               tl.load(offsets + i_n + 1).to(tl.int32)
    T = eos - bos
else:
    bos, eos = i_b * T, i_b * T + T

# 定位 K, V 和 block_indices
k += (bos * H + i_h) * K
v += (bos * H + i_h) * V
block_indices += (bos + i_t) * H*S + i_h * S

# 获取当前 token 的选择数量
if USE_BLOCK_COUNTS:
    NS = tl.load(block_counts + (bos + i_t) * H + i_h)
else:
    NS = S

'''
加载 Q 并缩放
'''
p_q = tl.make_block_ptr(q + (bos + i_t) * HQ*K, (HQ, K), (K, 1),
                        (i_h * G, 0), (G, BK), (1, 0))
b_q = tl.load(p_q, boundary_check=(0, 1))
b_q = (b_q * scale).to(b_q.dtype)  # [G, BK]

'''
遍历选中的 block，执行 online softmax
关键：block 访问是非连续的（稀疏）
'''
b_o = tl.zeros([G, BV], dtype=tl.float32)
b_m = tl.full([G], float('-inf'), dtype=tl.float32)
b_acc = tl.zeros([G], dtype=tl.float32)

for i in range(NS):
    # 加载当前选中的 block 起始位置
    i_s = tl.load(block_indices + i).to(tl.int32) * BS

    # 有效性检查：block 必须在当前 token 之前
    if i_s <= i_t and i_s >= 0:
        # 加载原始 K, V（非压缩）
        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))  # [BK, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))  # [BS, BV]

        # 计算注意力分数
        b_s = tl.dot(b_q, b_k)  # [G, BS]

        # 因果 mask：block 内部也需要 mask
        b_s = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s, float('-inf'))

        # Online softmax 更新
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = tl.exp(b_mp - b_m)
        b_p = tl.exp(b_s - b_m[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

# 归一化
b_o = b_o / b_acc[:, None]
b_m += tl.log(b_acc)

# 存储输出
p_o = tl.make_block_ptr(o + (bos + i_t) * HQ*V, (HQ, V), (V, 1),
                        (i_h * G, i_v * BV), (G, BV), (1, 0))
tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
p_lse = lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
tl.store(p_lse, b_m.to(p_lse.dtype.element_ty))
```

## 3. 反向传播

反向传播分为两个内核：dQ 和 dKV。

### 3.1 dQ 计算

**源码位置**: [parallel.py#L617-L711](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L617-L711)

```python
'''
dQ 计算：遍历选中的 block，累积梯度
与前向类似，但使用 LSE 而非 online softmax
'''
b_dq = tl.zeros([G, BK], dtype=tl.float32)

for i in range(NS):
    i_s = tl.load(block_indices + i).to(tl.int32) * BS

    if i_s <= i_t and i_s >= 0:
        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))  # [BK, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))  # [BV, BS]

        # 重新计算注意力概率（使用保存的 LSE）
        b_s = tl.dot(b_q, b_k)                              # [G, BS]
        b_p = tl.exp(b_s - b_lse[:, None])                  # [G, BS]
        b_p = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_p, 0)

        # 计算 dS = P * (dP - delta)
        b_dp = tl.dot(b_do, b_v)                            # [G, BV] @ [BV, BS] -> [G, BS]
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])

        # dQ += dS @ K^T
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))  # [G, BS] @ [BS, BK] -> [G, BK]

b_dq *= scale
```

### 3.2 Block Mask 生成

**源码位置**: [parallel.py#L558-L584](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L558-L584)

dKV 需要知道每个 K/V block 被哪些 token 选中，通过 block mask 实现：

```python
@triton.jit
def parallel_nsa_kernel_mask(
    block_indices,
    block_counts,
    block_mask,         # 输出: [B, T, H, NS] 的 bool mask
    T: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    NS: tl.constexpr,   # 总 block 数
    USE_BLOCK_COUNTS: tl.constexpr
):
    '''
    生成 block mask，标记每个 (token, block) 对是否参与计算
    '''
    i_t, i_b, i_hs = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_s = i_hs // S, i_hs % S

    # 加载当前 token 选择的第 i_s 个 block
    b_i = tl.load(block_indices + i_b * T * H * S + i_t * H * S + i_h * S + i_s)

    # 判断是否有效
    if USE_BLOCK_COUNTS:
        b_m = b_i * BS <= i_t and i_s < tl.load(block_counts + i_b * T * H + i_t * H + i_h)
    else:
        b_m = b_i * BS <= i_t

    # 存储到对应 block 位置
    if b_i < NS and b_i >= 0:
        tl.store(block_mask + i_b * T * H * NS + i_t * H * NS + i_h * NS + b_i,
                 b_m.to(block_mask.dtype.element_ty))
```

### 3.3 dKV 计算

**源码位置**: [parallel.py#L724-L802](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L724-L802)

```python
'''
dKV 计算：遍历所有 token，累积对当前 K/V block 的梯度
利用 block_mask 跳过未选中的 token
'''
b_dk = tl.zeros([BS, BK], dtype=tl.float32)
b_dv = tl.zeros([BS, BV], dtype=tl.float32)

for i in range(i_s * BS, T):
    # 检查当前 token 是否选中了这个 block
    b_m = tl.load(block_mask + (bos + i) * H*M + i_h * M + i_s)

    if b_m:
        # 加载 Q 和 dO
        p_q = tl.make_block_ptr(q + (bos + i) * HQ*K, (HQ, K), (K, 1),
                                (i_h * G, 0), (G, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)

        p_do = tl.make_block_ptr(do + (bos + i) * HQ*V, (HQ, V), (V, 1),
                                 (i_h * G, i_v * BV), (G, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_lse = tl.load(lse + (bos + i) * HQ + i_h * G + tl.arange(0, G))
        b_delta = tl.load(delta + (bos + i) * HQ + i_h * G + tl.arange(0, G))

        # 重新计算注意力
        b_s = tl.dot(b_k, tl.trans(b_q))                    # [BS, G]
        b_p = tl.exp(b_s - b_lse[None, :])                  # [BS, G]
        b_p = tl.where((i >= (i_s * BS + tl.arange(0, BS)))[:, None], b_p, 0)

        # dV += P^T @ dO
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)            # [BS, G] @ [G, BV] -> [BS, BV]

        # dK += dS^T @ Q
        b_dp = tl.dot(b_v, tl.trans(b_do))                  # [BS, BV] @ [BV, G] -> [BS, G]
        b_ds = b_p * (b_dp - b_delta[None, :])              # [BS, G]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)             # [BS, G] @ [G, BK] -> [BS, BK]
```

## 4. 设计要点

### 4.1 稀疏访存模式

与压缩注意力不同，选择注意力的 K/V 访存是非连续的：

- 每个 token 访问不同的 block 集合
- Block indices 动态确定
- 使用标量循环（`for i in range(NS)`）而非向量化

### 4.2 Variable Block Counts

支持每个 token 选择不同数量的 block：

```python
if USE_BLOCK_COUNTS:
    NS = tl.load(block_counts + (bos + i_t) * H + i_h)
else:
    NS = S  # 固定数量
```

### 4.3 Block 内因果 Mask

选中的 block 内部也需要因果 mask：

```python
# i_s: block 起始位置
# i_t: 当前 token 位置
# 只有 i_s + j <= i_t 的位置参与计算
b_s = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s, float('-inf'))
```

这确保即使选中了当前 token 所在的 block，也不会看到未来信息。
