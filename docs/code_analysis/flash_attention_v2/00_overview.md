---
tags:
  - CUDA
  - CUTLASS
  - Flash Attention
---
# Flash Attention V2 代码分析：总览

**源码仓库**: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

**论文**: [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.09288)

**分析范围**: `csrc/flash_attn/src/` 目录下的 SM80（Ampere）CUDA 实现

## 核心思想

FA2 通过三个关键策略将标准 Attention 的 $O(N^2)$ 内存开销降至 $O(N)$：

- **Tiling**：将 Q/K/V 分块加载到 shared memory，在 SRAM 层完成 Q·K^T 和 P·V 两次矩阵乘法
- **Online Softmax**：在 K 块迭代过程中增量维护 softmax 的 max 和 sum，避免物化完整的 $N \times N$ attention matrix
- **Recomputation**：反向传播时重新计算 attention matrix 而非存储中间结果，用计算换内存

与 FA1 的核心区别：FA2 改进了工作分配策略——外层循环遍历 Q 的行块（每个 thread block 处理一行 Q 块），内层循环遍历 K/V 块，提升了 GPU 并行度。

## 代码架构

```
csrc/flash_attn/src/
├── flash.h                         # 参数结构体 (Flash_fwd_params, Flash_bwd_params)
├── kernel_traits.h                 # 硬件配置：MMA Atom、Block 大小、Smem Layout
├── flash_fwd_kernel.h              # 前向核心：compute_attn_1rowblock (1294 行)
├── flash_fwd_launch_template.h     # 前向 host 端调度
├── flash_bwd_kernel.h              # 反向核心：compute_dq_dk_dv (841 行)
├── flash_bwd_launch_template.h     # 反向 host 端调度
├── flash_bwd_preprocess_kernel.h   # 反向预处理：dO 归一化
├── softmax.h                       # Online softmax 实现
├── utils.h                         # Warp reduce、类型转换、GEMM 工具
├── mask.h                          # Causal mask、local/sliding window mask
├── dropout.h                       # Dropout (Philox RNG)
├── rotary.h                        # RoPE 旋转位置编码
├── block_info.h                    # 变长序列 (varlen) 偏移计算
├── static_switch.h                 # 编译期模板分发宏
└── [74 个 .cu 文件]                 # 自动生成的模板实例化
```

### 文件依赖关系

```
flash_fwd_kernel.h ──┬── kernel_traits.h ── cute/tensor.hpp, cutlass/*
                     ├── utils.h
                     ├── softmax.h
                     ├── mask.h
                     ├── dropout.h
                     ├── rotary.h
                     └── block_info.h

flash_fwd_launch_template.h ── flash_fwd_kernel.h, static_switch.h

flash_bwd_kernel.h ──── (类似依赖)

[*.cu 实例化文件] ── flash_fwd_launch_template.h / flash_bwd_launch_template.h
```

### 框架使用

FA2 混合使用 **CuTe** 和 **CUTLASS 2.x**：

- **CuTe**（核心）：所有 tensor 抽象（`make_tensor`、`local_tile`、`partition`）、MMA Atom（`SM80_16x8x16_F32F16F16F32_TN`）、Copy Atom（`SM80_CP_ASYNC_CACHEGLOBAL`、`SM75_U32x4_LDSM_N`）、Swizzle Layout
- **CUTLASS 2.x**（辅助）：仅用于数据类型（`cutlass::half_t`、`cutlass::bfloat16_t`、`cutlass::Array`）和数值转换，未使用其 GEMM kernel 框架

### 模板实例化模式

通过 `generate_kernels.py` 生成 74 个 `.cu` 文件，每个文件约 10-14 行，组合覆盖：

- **Head dim**: 32, 64, 96, 128, 192, 256
- **数据类型**: fp16, bf16
- **Causal 模式**: causal, non-causal
- **方向**: forward (含 split-KV), backward

## 文档导航

| 文档 | 内容 |
|------|------|
| [01 参数与硬件配置](01_params_and_traits.md) | `flash.h` 参数结构体、`kernel_traits.h` MMA/Copy/Layout 选择 |
| [02 前向核心](02_forward_kernel.md) | `compute_attn_1rowblock` 逐段拆解 |
| [03 计算原语](03_softmax_mask_utils.md) | online softmax、mask、warp reduce、dropout |
| [04 反向传播](04_backward_kernel.md) | recomputation 策略与 dQ/dK/dV 计算 |
| [05 调度与实例化](05_dispatch_and_instantiation.md) | `static_switch.h` 宏、launch template、grid 配置 |

## 源码浏览

| 文件 | 行数 | 源码页 |
|------|------|--------|
| flash.h | 194 | [浏览](src/flash_h.md) |
| kernel_traits.h | 344 | [浏览](src/kernel_traits_h.md) |
| flash_fwd_kernel.h | 1294 | [浏览](src/flash_fwd_kernel_h.md) |
| flash_fwd_launch_template.h | 304 | [浏览](src/flash_fwd_launch_template_h.md) |
| flash_bwd_kernel.h | 841 | [浏览](src/flash_bwd_kernel_h.md) |
| flash_bwd_launch_template.h | 308 | [浏览](src/flash_bwd_launch_template_h.md) |
| flash_bwd_preprocess_kernel.h | 383 | [浏览](src/flash_bwd_preprocess_kernel_h.md) |
| softmax.h | 189 | [浏览](src/softmax_h.md) |
| utils.h | 413 | [浏览](src/utils_h.md) |
| mask.h | 214 | [浏览](src/mask_h.md) |
| dropout.h | 95 | [浏览](src/dropout_h.md) |
| rotary.h | 153 | [浏览](src/rotary_h.md) |
| block_info.h | 49 | [浏览](src/block_info_h.md) |
| static_switch.h | 111 | [浏览](src/static_switch_h.md) |

## 前置知识与关联笔记

阅读本系列需要以下 CuTe/CUDA 基础，对应本站已有笔记：

- **CuTe Layout 与 Tensor**：[Layout](../cute/01_layout.md)、[Tensor](../cute/03_tensor.md) — 理解 `make_tensor`、`local_tile`、`partition` 等核心操作
- **MMA Atom**：[MMA Atom](../cute/05_mma_atom.md) — 理解 `SM80_16x8x16` 指令与 TiledMMA 构建
- **CuTe 算法**：[算法](../cute/04_algorithms.md) — `copy`、`gemm` 的 CuTe 抽象
- **sgemm_sm80 实战**：[sgemm_sm80](../cute/09_sgemm_sm80.md) — 与 FA2 编码风格高度一致的 GEMM 示例，含 cp.async pipeline
- **CUDA 执行模型**：[执行模型与内存](../cuda_foundations/01_cuda_execution_model_and_memory.md) — shared memory、warp、同步
- **CUTLASS 编程模型**：[CUTLASS/CuTe 编程模型](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)
- **GEMM 优化**：[Tensor Core 与 CUTLASS](../cutlass_gemm_blog/03_tensorcore_and_cutlass.md) — Swizzle、bank conflict 优化
