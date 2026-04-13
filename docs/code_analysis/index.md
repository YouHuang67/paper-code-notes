# 代码分析

这里收录各类代码拆解与分析笔记。

## CUDA 基础

- [CUDA Kernel 基础：导读](cuda_foundations/00_overview.md) - CUDA / CUTLASS / CuTe 公共引用页与阅读路径
- [CUDA 执行模型与内存访问](cuda_foundations/01_cuda_execution_model_and_memory.md) - 线程层级、shared memory、同步、warp 原语、浮点原子技巧
- [CUDA CUTLASS/CuTe 编程模型](cuda_foundations/02_cuda_cutlass_cute_programming_model.md) - GemmShape/OpClass、Tensor/Layout、MMA/Copy Atom、cp.async、Swizzle

## GPU Kernel

- [Native Sparse Attention](native_sparse_attention/00_overview.md) - 硬件友好的稀疏注意力 Triton 实现
- [xformers Memory Efficient Attention](xformers_memory_efficient_attention/00_overview.md) - Split-K Triton 前向推理内核（mask 融合 + 量化）
- [CUTLASS Memory Efficient Attention](cutlass_mem_eff_attention/00_overview.md) - PyTorch ATen CUTLASS 前向 kernel（Online Softmax + 模板分发）
- [CUB Block 级原语](cub_block_primitives/01_block_radix_sort_and_scan.md) - BlockRadixSort 与 BlockScan 内部实现详解
