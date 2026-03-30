---
tags:
  - Triton
  - Sparse Attention
---
# 压缩注意力实现

本节分析 NSA 的压缩注意力（Compression Attention）实现，包括 mean pooling 压缩和 Triton 内核。

## 1. Mean Pooling 压缩

**源码位置**: [naive.py#L13-L26](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/naive.py#L13-L26)

将 KV 序列按固定 block size 进行均值池化，降低序列长度：

$$
\bar{\mathbf{k}}_c = \frac{1}{B} \sum_{i=(c-1)B+1}^{cB} \mathbf{k}_i
$$

```python
@torch.compile
def compression(
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int
) -> torch.Tensor:
    '''
    Mean pooling 压缩 K, V
    输入: k, v shape [B, T, H, D]
    输出: k_cmp, v_cmp shape [B, num_block, H, D]
    '''
    B, T, H = k.shape[:3]
    num_block = math.ceil(T / block_size)

    # padding 到 block_size 的整数倍
    if k.shape[1] % block_size != 0:
        k = F.pad(k, (0, 0, 0, 0, 0, num_block * block_size - T))
        v = F.pad(v, (0, 0, 0, 0, 0, num_block * block_size - T))

    # reshape 并沿 block 维度取均值
    k_cmp = k.view(B, num_block, block_size, H, -1).mean(dim=2)  # [B, C, H, D]
    v_cmp = v.view(B, num_block, block_size, H, -1).mean(dim=2)  # [B, C, H, D]
    return k_cmp, v_cmp
```

!!! note "实际调用"
    在 `parallel_nsa` 函数中使用 `fla.ops.utils.mean_pooling`：
    ```python
    k_cmp, v_cmp = mean_pooling(k, block_size, cu_seqlens), mean_pooling(v, block_size, cu_seqlens)
    ```

## 2. 压缩注意力前向内核

**源码位置**: [parallel.py#L38-L127](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L38-L127)

### 2.1 内核签名与参数

```python
@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BS', 'BK', 'BV'],
)
@triton.jit
def parallel_nsa_compression_fwd_kernel(
    q,              # Query: [B, T, HQ, K]
    k,              # Compressed Key: [B*C, H, K] (C = num_blocks)
    v,              # Compressed Value: [B*C, H, V]
    o,              # Output: [B, T, HQ, V]
    lse,            # Log-Sum-Exp: [B, T, HQ]
    scale,          # softmax scale = 1/sqrt(d)
    offsets,        # variable-length offsets
    token_indices,  # 2D token indices for varlen
    chunk_offsets,  # chunk offsets for varlen
    T,              # sequence length
    H: tl.constexpr,    # num KV heads
    HQ: tl.constexpr,   # num Q heads
    G: tl.constexpr,    # group size = HQ / H
    K: tl.constexpr,    # head dim for K
    V: tl.constexpr,    # head dim for V
    BC: tl.constexpr,   # block size for chunks iteration
    BS: tl.constexpr,   # compression block size
    BK: tl.constexpr,   # tile size for K dim
    BV: tl.constexpr,   # tile size for V dim
    USE_OFFSETS: tl.constexpr,
):
```

### 2.2 Grid 与索引计算

```python
'''
Grid 配置: (T, NV, B*H)
- i_t: token position
- i_v: V 维度分块索引
- i_bh: batch * head 联合索引
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
    boc = tl.load(chunk_offsets + i_n).to(tl.int32)  # 当前序列的 chunk 起始位置
else:
    bos, eos = i_b * T, i_b * T + T
    boc = i_b * tl.cdiv(T, BS)  # batch 内的 chunk 起始位置
```

### 2.3 核心计算逻辑

```python
'''
加载 Q block 并缩放
- GQA: 每个 KV head 对应 G 个 Q heads
- Q shape: [G, BK]
'''
p_q = tl.make_block_ptr(
    q + (bos + i_t) * HQ*K,         # base pointer
    (HQ, K),                         # shape
    (K, 1),                          # strides
    (i_h * G, 0),                    # offsets: 选择对应的 G 个 Q heads
    (G, BK),                         # block shape
    (1, 0)                           # order
)
b_q = tl.load(p_q, boundary_check=(0, 1))
b_q = (b_q * scale).to(b_q.dtype)   # [G, BK]

# 压缩块总数与需要迭代的块数
TC = tl.cdiv(T, BS)                 # total compression blocks
NC = (i_t + 1) // BS                # causal: 只看当前 token 之前完整的 blocks

'''
Online Softmax 迭代
遍历所有可见的压缩块，累积计算注意力输出
'''
b_o = tl.zeros([G, BV], dtype=tl.float32)
b_m = tl.full([G], float('-inf'), dtype=tl.float32)
b_acc = tl.zeros([G], dtype=tl.float32)

for i_c in range(0, NC, BC):
    o_c = i_c + tl.arange(0, BC)

    # 加载压缩后的 K block: [BK, BC]
    p_k = tl.make_block_ptr(
        k + (boc * H + i_h) * K,     # 定位到正确的 head
        (K, TC),                      # shape: [K, total_chunks]
        (1, H*K),                     # strides: chunk 维度步长为 H*K
        (0, i_c),                     # offsets
        (BK, BC),                     # block shape
        (0, 1)                        # order
    )
    # 加载压缩后的 V block: [BC, BV]
    p_v = tl.make_block_ptr(
        v + (boc * H + i_h) * V,
        (TC, V),
        (H*V, 1),
        (i_c, i_v * BV),
        (BC, BV),
        (1, 0)
    )
    b_k = tl.load(p_k, boundary_check=(0, 1))  # [BK, BC]
    b_v = tl.load(p_v, boundary_check=(0, 1))  # [BC, BV]

    # 计算注意力分数: [G, BC]
    b_s = tl.dot(b_q, b_k)
    b_s = tl.where((o_c < NC)[None, :], b_s, float('-inf'))  # causal mask

    # Online softmax 更新
    b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m  # [G] 新旧 max
    b_r = tl.exp(b_mp - b_m)                          # [G] rescale factor
    b_p = tl.exp(b_s - b_m[:, None])                  # [G, BC] attention probs
    b_acc = b_acc * b_r + tl.sum(b_p, 1)              # [G] cumulative sum
    b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)  # [G, BV]

# 归一化并存储
if NC == 0:
    b_lse = tl.zeros([G], dtype=tl.float32)
else:
    b_o = b_o / b_acc[:, None]
    b_lse = b_m + tl.log(b_acc)

p_o = tl.make_block_ptr(o + (bos + i_t) * HQ*V, (HQ, V), (V, 1),
                        (i_h * G, i_v * BV), (G, BV), (1, 0))
tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
if i_v == 0:
    tl.store(lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G),
             b_lse.to(lse.dtype.element_ty))
```

## 3. 关键设计技巧

### 3.1 GQA 支持

通过 `G = HQ // H` 实现 Grouped Query Attention：

- 每个 KV head 对应 G 个 Q heads
- `b_q` shape 为 `[G, BK]`，一次处理一组 Q heads
- 输出 `b_o` 和 `b_lse` 也按组存储

### 3.2 Variable-Length 支持

使用 `token_indices` 和 `chunk_offsets` 支持变长序列：

```python
# token_indices: [total_tokens, 2]
# 每行为 (sequence_idx, position_in_sequence)
# 例如 offsets=[0, 2, 6] 对应 token_indices:
# [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
```

### 3.3 Block Pointer

使用 `tl.make_block_ptr` 简化多维张量访问：

```python
p_k = tl.make_block_ptr(
    base=k + offset,        # 基址偏移
    shape=(K, TC),          # 逻辑 shape
    strides=(1, H*K),       # 各维度步长
    offsets=(0, i_c),       # block 起始偏移
    block_shape=(BK, BC),   # 每次加载的 block 大小
    order=(0, 1)            # 内存布局顺序
)
```

## 4. 反向传播

**源码位置**: [parallel.py#L140-L326](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L140-L326)

反向传播分为两个内核：

1. `parallel_nsa_compression_bwd_kernel_dq`: 计算 dQ
2. `parallel_nsa_compression_bwd_kernel_dkv`: 计算 dK, dV

核心公式：

$$
\begin{aligned}
\mathbf{ds} &= \mathbf{p} \odot (\mathbf{dp} - \delta) \\
\mathbf{dq} &= \text{scale} \cdot \mathbf{ds} \cdot \mathbf{K}^\top
\end{aligned}
$$

```python
# dQ 计算核心逻辑
for i_c in range(0, NC, BC):
    # ... 加载 K, V ...

    b_s = tl.dot(b_q, b_k)
    b_p = tl.exp(b_s - b_lse[:, None])                     # [G, BC]
    b_p = tl.where((o_c < NC)[None, :], b_p, 0)

    b_dp = tl.dot(b_do, b_v)                               # [G, BV] @ [BV, BC] -> [G, BC]
    b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])  # [G, BC]
    b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))      # [G, BC] @ [BC, BK] -> [G, BK]

b_dq *= scale
```
