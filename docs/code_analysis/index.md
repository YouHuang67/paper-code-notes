# 代码分析

这里收录各类代码拆解与分析笔记。

## GPU Kernel

- [Native Sparse Attention](native_sparse_attention/00_overview.md) - 硬件友好的稀疏注意力 Triton 实现
- [xformers Memory Efficient Attention](xformers_memory_efficient_attention/00_overview.md) - Split-K Triton 前向推理内核（mask 融合 + 量化）
- [CUTLASS Memory Efficient Attention](cutlass_mem_eff_attention/00_overview.md) - PyTorch ATen CUTLASS 前向 kernel（Online Softmax + 模板分发）
- [CUB Block 级原语](cub_block_primitives/01_block_radix_sort_and_scan.md) - BlockRadixSort 与 BlockScan 内部实现详解
- [CuTe 编程模型详解](cub_block_primitives/02_cute_programming_model.md) - Tensor/Layout、MMA Atom、Copy Atom、Swizzle 全栈抽象
