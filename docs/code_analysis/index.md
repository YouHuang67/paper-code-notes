# 代码分析

这里收录各类代码拆解与分析笔记。

## 公共基础（CUDA）

- [CUDA 基础：导读](cuda_foundations/00_overview.md) - CUDA / CUTLASS / CuTe 公共引用页与阅读路径
- [CUDA 基础：执行模型与内存访问](cuda_foundations/01_cuda_execution_model_and_memory.md) - 线程层级、shared memory、同步、warp 原语、浮点原子技巧
- [CUDA 基础：CUTLASS/CuTe 编程模型](cuda_foundations/02_cuda_cutlass_cute_programming_model.md) - GemmShape/OpClass、Tensor/Layout、MMA/Copy Atom、cp.async、Swizzle
- [CUDA 基础：性能模型与 Occupancy](cuda_foundations/03_cuda_performance_model_and_occupancy.md) - CTA/warp/SM 调度、latency hiding、occupancy 与资源约束如何塑造 kernel 形态
- [CUDA 基础：分块、数据搬运与局部性](cuda_foundations/04_cuda_tiling_data_movement_and_locality.md) - arithmetic intensity、tiling、coalescing、shared memory 复用与边界 predication
- [CUDA 基础：归约、Scan 与在线归一化](cuda_foundations/05_cuda_reduction_scan_and_online_normalization.md) - reduction/scan 并行模式、online softmax 与逐行统计量的统一视角
- [CUDA 基础：计算、带宽与算存权衡](cuda_foundations/06_cuda_compute_memory_tradeoffs.md) - Roofline 直觉、recomputation、memory-bound / compute-bound 与 block shape 取舍
- [CUDA 基础：TileLang 编程模型](cuda_foundations/07_tilelang_programming_model.md) - DeepSeek V4 所需 TileLang 原语：Kernel/Shared/Fragment/Pipelined/GEMM/Reduce
- [CUDA 基础：块量化与低精度 GEMM](cuda_foundations/08_blockwise_quantization_and_low_precision_gemm.md) - FP8/FP4 block-wise 量化、scale 张量组织、低精度 GEMM 的 scale correction

## CuTe 教程（翻译）

翻译自 [NVIDIA/cutlass CuTe 官方教程](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute/)，BSD-3-Clause 许可证。

- [00 快速入门](cute/00_quickstart.md) - CuTe 概述、库组织与调试打印
- [01 Layout](cute/01_layout.md) - Layout 核心抽象：Shape/Stride/坐标映射/索引映射
- [02 Layout 代数](cute/02_layout_algebra.md) - Coalesce/Composition/Complement/Division/Product
- [03 Tensor](cute/03_tensor.md) - Tensor 容器：Engine/创建/访问/Tiling/Slicing/Partitioning
- [04 算法](cute/04_algorithms.md) - copy/gemm/axpby/fill/clear 等算法接口
- [05 MMA Atom](cute/05_mma_atom.md) - MMA 指令元信息与接口（Volta/Hopper）
- [06 GEMM 教程](cute/06_gemm_tutorial.md) - 使用 CuTe 从零构建 GEMM 的完整实现
- [07 谓词](cute/07_predication.md) - 分块不整除时的谓词化处理
- [08 TMA Tensor](cute/08_tma_tensors.md) - ArithTuple 与基向量步长用于 TMA 坐标生成

## GEMM 优化之路（翻译）

翻译自 Kapil Sharma 的 [Learn CUTLASS the Hard Way](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/) 系列博客，CC BY 4.0 许可证。

- [00 总览](cutlass_gemm_blog/00_overview.md) - 系列导读与性能演进总览
- [01 GEMM 基础与朴素优化](cutlass_gemm_blog/01_gemm_basics_and_naive.md) - GEMM 定义、Naive → Memory Coalescing → Shared Memory → Occupancy 分析
- [02 SIMT 分块优化](cutlass_gemm_blog/02_simt_tiling.md) - 1D/2D Block Tiling → Vectorized → Warp Tiling
- [03 Tensor Core 与 CUTLASS](cutlass_gemm_blog/03_tensorcore_and_cutlass.md) - WMMA/TC → Double Buffering → CUTLASS 2.x → Swizzling → Autotuning
- [04 Hopper 架构与 CUTLASS 3.x](cutlass_gemm_blog/04_hopper_cutlass3x.md) - TBC/TMA/WGMMA → CUTLASS 3.x Warp Specialized → Persistent → Stream-K → Autotuning

## 项目分析

- [CuTe sgemm_sm80 实战拆解](cute/09_sgemm_sm80.md) - 三版本对比、Swizzle SMEM、TiledCopy/TiledMMA、ldmatrix retiling、双层流水线
- [Flash Attention V2](flash_attention_v2/00_overview.md) - FA2 SM80 CUDA 实现：CuTe + CUTLASS 2.x 的 tiling attention 完整拆解（前向/反向/调度）
- [Native Sparse Attention](native_sparse_attention/00_overview.md) - 硬件友好的稀疏注意力 Triton 实现
- [xformers Memory Efficient Attention](xformers_memory_efficient_attention/00_overview.md) - Split-K Triton 前向推理内核（mask 融合 + 量化）
- [CUTLASS Memory Efficient Attention](cutlass_mem_eff_attention/00_overview.md) - PyTorch ATen CUTLASS 前向 kernel（Online Softmax + 模板分发）
- [CUB Block 级原语](cub_block_primitives/01_block_radix_sort_and_scan.md) - BlockRadixSort 与 BlockScan 内部实现详解
- [FlashInfer Variable Block Sparse Attention](flashinfer_variable_block_sparse/00_overview.md) - 本地剥离版 `variable_block_attn`：metadata 翻译层 → FA2 paged prefill runtime → JIT → C++ binding
- [DeepSeek V4](deepseek_v4/00_overview.md) - mHC + Hybrid Attention + MoE + TileLang 低精度推理实现
