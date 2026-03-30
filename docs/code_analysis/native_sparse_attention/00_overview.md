---
tags:
  - Triton
  - Sparse Attention
---
# Native Sparse Attention 总览

!!! abstract "TL;DR"
    NSA 是一种硬件友好的稀疏注意力机制，通过三路注意力融合（压缩 + 选择 + 滑动窗口）实现高效的长序列建模，核心创新在于 online top-k 选择避免物化完整注意力矩阵。

**源码仓库**: [fla-org/native-sparse-attention](https://github.com/fla-org/native-sparse-attention)

**论文**: [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089)

## 核心思想

标准 Attention 的复杂度为 $O(T^2)$，对于长序列训练和推理代价高昂。NSA 通过稀疏化策略将复杂度降至近线性：

$$
\mathbf{o} = g_{\text{cmp}} \cdot \mathbf{o}_{\text{cmp}} + g_{\text{slc}} \cdot \mathbf{o}_{\text{slc}} + g_{\text{swa}} \cdot \mathbf{o}_{\text{swa}}
$$

三个分支分别负责：

| 分支 | 作用 | 复杂度 |
|------|------|--------|
| **Compression** ($\mathbf{o}_{\text{cmp}}$) | 全局粗粒度信息，mean pooling 压缩 KV | $O(T \cdot T/B)$ |
| **Selection** ($\mathbf{o}_{\text{slc}}$) | 局部细粒度信息，top-k block 选择 | $O(T \cdot S \cdot B)$ |
| **Sliding Window** ($\mathbf{o}_{\text{swa}}$) | 近邻精确信息，固定窗口注意力 | $O(T \cdot W)$ |

其中 $B$ 为 block size，$S$ 为选择的 block 数量，$W$ 为窗口大小。

## 代码结构

```
native_sparse_attention/
├── ops/
│   ├── parallel.py      # 核心 Triton 内核
│   ├── naive.py         # PyTorch 参考实现
│   └── utils.py         # 工具函数（bitonic sort）
├── modeling_nsa.py      # 模型层封装
└── configuration_nsa.py # 配置类
```

**核心内核一览**：

| 内核 | 功能 | 源码位置 |
|------|------|----------|
| `parallel_nsa_compression_fwd_kernel` | 压缩注意力前向 | [parallel.py#L38](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L38) |
| `parallel_nsa_kernel_topk` | Online Top-k 选择 | [parallel.py#L339](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L339) |
| `parallel_nsa_fwd_kernel` | 选择注意力前向 | [parallel.py#L472](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L472) |
| `_bitonic_merge` | Bitonic Sort 合并 | [utils.py#L49](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/utils.py#L49) |

## 文档导航

1. [数学公式推导](01_math_formulation.md) - 注意力计算的数学基础
2. [压缩注意力实现](02_compression_attention.md) - Mean pooling + Triton 内核
3. [Top-k 选择机制](03_topk_selection.md) - Online selection + Bitonic sort
4. [选择注意力实现](04_selected_attention.md) - Block-wise 稀疏注意力
5. [滑动窗口注意力](05_sliding_window.md) - FlashAttention 集成
