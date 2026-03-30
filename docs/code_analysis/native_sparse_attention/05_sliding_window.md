---
tags:
  - Flash Attention
  - Sliding Window
---
# 滑动窗口注意力

本节分析 NSA 的滑动窗口注意力（Sliding Window Attention），通过集成 FlashAttention 实现高效的局部注意力。

## 1. 算法概述

滑动窗口注意力只关注当前 token 附近的 $W$ 个 token：

$$
\mathbf{o}_{\text{swa},t} = \sum_{i=\max(1, t-W+1)}^{t} \text{softmax}\left(\frac{\mathbf{q}_t \cdot \mathbf{K}_{[t-W+1:t]}^\top}{\sqrt{d}}\right)_i \mathbf{v}_i
$$

复杂度从 $O(T^2)$ 降至 $O(T \cdot W)$。

## 2. 实现方式

NSA 直接调用 FlashAttention 的滑动窗口变体，而非自行实现 Triton 内核。

**源码位置**: [parallel.py#L1413-L1431](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L1413-L1431)

```python
if window_size > 0:
    if cu_seqlens is not None:
        '''
        变长序列：使用 flash_attn_varlen_func
        '''
        max_seqlen = q.shape[1]
        o_swa = flash_attn_varlen_func(
            q.squeeze(0),           # [T, HQ, D]
            k.squeeze(0),           # [T, H, D]
            v.squeeze(0),           # [T, H, D]
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
            window_size=(window_size-1, 0)  # (left, right)
        ).unsqueeze(0)
    else:
        '''
        定长序列：使用 flash_attn_func
        '''
        o_swa = flash_attn_func(
            q, k, v,
            causal=True,
            window_size=(window_size-1, 0)
        )

    # 门控融合
    o = torch.addcmul(o, o_swa, g_swa.unsqueeze(-1))
```

## 3. FlashAttention 窗口参数

`window_size=(left, right)` 定义注意力窗口：

- `left`: 向左看的 token 数
- `right`: 向右看的 token 数（因果模式下通常为 0）

例如 `window_size=(63, 0)` 表示窗口大小为 64（包含当前 token）。

!!! note "窗口大小设置"
    代码中使用 `window_size-1` 作为 left 参数：
    ```python
    window_size=(window_size-1, 0)
    ```
    因为 FlashAttention 的 left 参数不包含当前 token，所以实际窗口大小为 `left + 1`。

## 4. 朴素实现参考

**源码位置**: [naive.py#L158-L164](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/naive.py#L158-L164)

```python
if window_size > 0:
    # 截取窗口范围内的 K, V
    k_i_swa, v_i_swa = map(
        lambda x: x[max(0, i_q - window_size + 1):i_q + 1],
        (k_b, v_b)
    )
    # 计算窗口内的注意力
    attn_swa = torch.einsum('h d, n h d -> n h', q_i, k_i_swa).softmax(0)

    if not varlen:
        o_swa[i, i_q] = torch.einsum('n h, n h v -> h v', attn_swa, v_i_swa) * g_swa_i.unsqueeze(-1)
    else:
        o_swa[0][cu_seqlens[i]+i_q] = torch.einsum('n h, n h v -> h v', attn_swa, v_i_swa) * g_swa_i.unsqueeze(-1)
```

## 5. 三路融合

**源码位置**: [parallel.py#L1409-L1431](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L1409-L1431)

最终输出通过门控机制融合三路注意力：

```python
# 选择注意力（必选）
o_slc = ParallelNSAFunction.apply(q, k, v, block_indices, block_counts,
                                   block_size, scale, cu_seqlens)
o = o_slc * g_slc.unsqueeze(-1)

# 压缩注意力（可选）
if o_cmp is not None:
    # torch.addcmul: out = input + tensor1 * tensor2
    o = torch.addcmul(o, o_cmp, g_cmp.unsqueeze(-1))

# 滑动窗口注意力（可选）
if window_size > 0:
    o_swa = flash_attn_func(...)
    o = torch.addcmul(o, o_swa, g_swa.unsqueeze(-1))
```

数学形式：

$$
\mathbf{o} = g_{\text{slc}} \odot \mathbf{o}_{\text{slc}} + g_{\text{cmp}} \odot \mathbf{o}_{\text{cmp}} + g_{\text{swa}} \odot \mathbf{o}_{\text{swa}}
$$

## 6. 设计考量

### 6.1 为什么使用 FlashAttention

1. **成熟优化**: FlashAttention 已高度优化，IO-aware 设计
2. **窗口支持**: 原生支持滑动窗口，无需额外开发
3. **梯度正确**: 自动处理反向传播
4. **变长支持**: `varlen` 变体支持 packed sequences

### 6.2 与选择注意力的分工

| 分支 | 覆盖范围 | 实现方式 |
|------|----------|----------|
| 压缩注意力 | 全局粗粒度 | 自研 Triton 内核 |
| 选择注意力 | 稀疏重要 block | 自研 Triton 内核 |
| 滑动窗口 | 局部精确 | FlashAttention |

滑动窗口确保近邻信息不丢失，选择注意力捕获远程重要信息，压缩注意力提供全局概览。

### 6.3 门控学习

三个门控参数 $g_{\text{cmp}}, g_{\text{slc}}, g_{\text{swa}}$ 是可学习的：

```python
# 在 modeling_nsa.py 中定义
g_cmp = torch.rand((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)
g_slc = torch.rand((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)
g_swa = torch.rand((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)
```

模型学习如何动态权衡三路信息。
