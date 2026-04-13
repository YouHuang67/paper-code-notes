---
tags:
  - CUDA
  - CUTLASS
---
# CUDA 基础：导读

这一组文档收拢各类 CUDA kernel 分析里会反复出现的公共知识点，目标是减少项目目录中的重复解释。

目前分成两层：

1. [CUDA 基础：执行模型与内存访问](01_cuda_execution_model_and_memory.md)
   - 线程层级、`__launch_bounds__`、shared memory、`__syncthreads()`、`__shfl_sync`、浮点原子技巧、`exp2f`
2. [CUDA 基础：CUTLASS/CuTe 编程模型](02_cuda_cutlass_cute_programming_model.md)
   - CUTLASS 模板分发、`GemmShape`、`OpClass` / `ArchTag`、MMA pipeline、Epilogue、CuTe Tensor/Layout、MMA/Copy Atom、`cp.async`、Swizzle

使用原则：

- 项目文档只保留“这个实现为什么这样选”的内容
- 通用概念只在这一组文档里主讲一次，其它地方尽量引用
- 新知识点只有在至少两个项目会复用时，才上收到这里

当前主要引用方：

- [CUTLASS Memory Efficient Attention - 总览](../cutlass_mem_eff_attention/00_overview.md)
- [CUB Block 级原语：BlockRadixSort 与 BlockScan](../cub_block_primitives/01_block_radix_sort_and_scan.md)
