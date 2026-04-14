---
tags:
  - CUTLASS
  - CUDA
---

# GEMM 优化之路：从 Naive 到 CUTLASS（翻译）

> **原文出处**: Kapil Sharma,
> [Learn CUTLASS the Hard Way!](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/)
> 与 [Learn CUTLASS the Hard Way - Part 2!](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way-2/)
> **许可证**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)（Creative Commons Attribution 4.0 International）
> **代码仓库**: [gpusgobrr/explore-gemm](https://github.com/gpusgobrr/explore-gemm)
>
> 本系列为上述博客的中文翻译与整理。根据 CC BY 4.0 许可证，翻译内容保留原作者署名，并标注原文链接。
> 翻译过程中对交互式 JS 可视化部分做了省略，保留了原文中的静态图片和全部代码。

本系列通过动手实现一系列 GEMM kernel，从最朴素的 FP32 实现一路优化到 CUTLASS BF16 kernel，涵盖 Ada（RTX 4090）和 Hopper（H100）两代架构。

## 文章目录

| 文档 | 内容 |
|------|------|
| [01 GEMM 基础与朴素优化](01_gemm_basics_and_naive.md) | GEMM 定义、硬件规格、Naive → Memory Coalescing → Shared Memory → GPU Occupancy 分析 |
| [02 SIMT 分块优化](02_simt_tiling.md) | 1D Block Tiling → 2D Block Tiling → Vectorized Memory Access → Warp Tiling |
| [03 Tensor Core 与 CUTLASS](03_tensorcore_and_cutlass.md) | 16-bit GEMM → WMMA/Tensor Core → Double Buffering → CUTLASS 2.x → Swizzling → Persistent Kernel → Autotuning |
| [04 Hopper 架构与 CUTLASS 3.x](04_hopper_cutlass3x.md) | Hopper 新特性（TBC/TMA/WGMMA）→ CUTLASS 3.x Warp Specialized → Persistent Cooperative → Ping-Pong → Stream-K → CTA Rasterization → Autotuning |

## 性能演进总览

### Part 1: RTX 4090（Ada Lovelace）

从 Naive kernel 到 CUTLASS autotuned，4096×4096 FP32 矩阵乘法的性能变化：

| Kernel | TFLOPS | 相对 PyTorch |
|--------|--------|-------------|
| Naive | 0.64 | 0.8% |
| Memory Coalesced | 4.73 | 5.8% |
| Shared Memory | 6.10 | 7.8% |
| 1D Block Tiling | 18.49 | 21.8% |
| 2D Block Tiling | 30.90 | 36.5% |
| Vectorized | 39.00 | 46.1% |
| Warp Tiling | 45.80 | 54.1% |
| CUTLASS（autotuned）| ~54% of PyTorch | ~54% |

### Part 2: H100（Hopper）

使用 CUTLASS 3.x 在 H100 上逐步逼近 PyTorch/cuBLAS：

- 基线（Ada kernel on H100）：~400 TFLOPS
- TMA Warp Specialized + Auto Stage：~2× 基线
- Persistent Cooperative：60-70% of PyTorch
- Stream-K + Autotuning：**90% of PyTorch**（大矩阵），小矩阵甚至超越
