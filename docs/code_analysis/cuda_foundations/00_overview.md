---
tags:
  - CUDA
  - CUTLASS
---
# CUDA 基础：导读

这一组文档收拢各类 CUDA kernel 分析里会反复出现的公共知识点，目标是减少项目目录中的重复解释。

目前分成六篇：

1. [CUDA 基础：执行模型与内存访问](01_cuda_execution_model_and_memory.md)
   - 线程层级、`__launch_bounds__`、shared memory、`__syncthreads()`、`__shfl_sync`、浮点原子技巧、`exp2f`
2. [CUDA 基础：CUTLASS/CuTe 编程模型](02_cuda_cutlass_cute_programming_model.md)
   - CUTLASS 模板分发、`GemmShape`、`OpClass` / `ArchTag`、MMA pipeline、Epilogue、CuTe Tensor/Layout、MMA/Copy Atom、`cp.async`、Swizzle
3. [CUDA 基础：性能模型与 Occupancy](03_cuda_performance_model_and_occupancy.md)
   - block / warp / SM 调度关系、latency hiding、occupancy 与 `kBlockM / kBlockN / kNWarps / smem / registers` 的联动
4. [CUDA 基础：分块、数据搬运与局部性](04_cuda_tiling_data_movement_and_locality.md)
   - arithmetic intensity、tiling、shared memory 复用、coalescing、predication 与 tiled kernel 的数据流
5. [CUDA 基础：归约、Scan 与在线归一化](05_cuda_reduction_scan_and_online_normalization.md)
   - reduction tree、分层归约、scan 的工作效率视角、online softmax 与逐行统计量
6. [CUDA 基础：计算、带宽与算存权衡](06_cuda_compute_memory_tradeoffs.md)
   - Roofline 直觉、memory-bound / compute-bound、recomputation 与 block shape 的算存平衡

使用原则：

- 项目文档只保留“这个实现为什么这样选”的内容
- 通用概念只在这一组文档里主讲一次，其它地方尽量引用
- 新知识点只有在至少两个项目会复用时，才上收到这里

当前主要引用方：

- [Flash Attention V2 - 总览](../flash_attention_v2/00_overview.md)
- [Flash Attention V2 - 参数与硬件配置](../flash_attention_v2/01_params_and_traits.md)
- [Flash Attention V2 - 前向核心](../flash_attention_v2/02_forward_kernel.md)
- [CUTLASS Memory Efficient Attention - 总览](../cutlass_mem_eff_attention/00_overview.md)
- [CUB Block 级原语：BlockRadixSort 与 BlockScan](../cub_block_primitives/01_block_radix_sort_and_scan.md)
