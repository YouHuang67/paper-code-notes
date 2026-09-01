---
tags:
  - CUDA
  - CUTLASS
  - LLM Inference
---
# DeepGEMM 代码分析：总览

**源码仓库**: [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)（对照提交 `88965b0`）

**团队**: DeepSeek

**分析范围**: `deep_gemm/include/deep_gemm/` 的分层内核、`csrc/apis/gemm.hpp` 的 grouped 入口，以及 SGLang `MoeRunnerBackend.DEEP_GEMM` 把标准 MoE 接到 grouped GEMM 的核外契约。证据以本地 `dev-kit/refs/DeepGEMM` 与 `dev-kit/refs/sglang` 为准。

相关背景：

- [CUDA 基础：CUTLASS/CuTe 编程模型](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)
- [CUDA 基础：分块、数据搬运与局部性](../cuda_foundations/04_cuda_tiling_data_movement_and_locality.md)
- [CUDA 基础：块量化与低精度 GEMM](../cuda_foundations/08_blockwise_quantization_and_low_precision_gemm.md)
- [DeepEP](../deepep/00_overview.md)：同团队的 EP all-to-all，masked grouped GEMM 的典型上游

## 核心思想

DeepGEMM 是面向 Hopper（SM90）与 Blackwell（SM100）的运行时 JIT tensor-core 库。它把 LLM 热路径上的矩阵乘收成同一套执行语言：layout、scheduler、mma、epilogue 由 impl 拼成 persistent 流水。数学对象统一为 NT GEMM

\[
D = C + A B^{\mathsf{T}}
\]

累加器为 FP32。标准 MoE 路径上，转置、cast、permute、SwiGLU 与 router 加权由调用方在核外完成。

一次 grouped launch 覆盖 \(G\) 个同形状 expert：只沿 \(M\) 分组，\(N\) 与 \(K\) 固定。两次 GEMM、精度顺序与 dispatcher 接入见 [03](03_grouped_gemm_moe_contract.md)。Mega MoE / paged MQA 见 [04](04_mega_moe_and_paged_mqa.md)。

## 记号

以下符号在后续文档中保持不变。

- \(G\)：expert 数（组数）
- \(K\)：隐藏维（GEMM 的内积维）
- \(N_{\mathrm{ffn}}\)：单路 FFN 中间宽；gate 与 up 沿输出维拼接后 \(N = 2N_{\mathrm{ffn}}\)
- \(m_g\)：组 \(g\) 的有效 token 数
- \(M\)：contiguous 下 \(\sum_g \mathrm{align}(m_g, BLOCK_M)\)；masked 下为每组上限 \(M_{\max}\)
- \(k_{\mathrm{top}}\)：top-\(k\) 路由数
- CTA：CUDA thread block；grid 大小等于 \(N_{\mathrm{SM}}\)
- TMA：Tensor Memory Accelerator
- WGMMA / UMMA：Hopper warpgroup MMA / Blackwell tensor-core MMA
- SF：scaling factor。SM90 FP8 为 FP32；SM100 为 packed UE8M0（4 个 scale 打进一个 `int`）
- `Kernel1D2D`：SM90 grouped FP8 前向（`BLOCK_K = 128`，带 SF）
- `Kernel1D1D`：SM100 FP8/FP4 与部分 K-grouped 路径
- `KernelNoSF`：BF16 路径，无 SF 张量，`BLOCK_K = 64`

## 各文职责

四篇正文对应 `dev-kit` 四份 DeepGEMM 报告，按「执行语言 → 两代流水 → MoE 契约 → 专用布局」读。每篇只建立下一篇要用的对象。

- [01 分层内核架构](01_layered_architecture.md)：layout / scheduler / mma / epilogue / impl 各做什么几何变换。输出：persistent tile 流与角色分工。
- [02 SM90 到 SM100 流水](02_sm90_sm100_pipeline.md)：同一分层在两代硬件上换哪四件（描述符、TMA、累积位置、epilogue）。输出：grouped 前向为何选 1D2D / 1D1D / NoSF。
- [03 grouped GEMM 与标准 MoE 契约](03_grouped_gemm_moe_contract.md)：M-grouped 调用面、FP8/BF16 三条分叉、SGLang permute–GEMM–SwiGLU–finalize。这是标准 MoE 适配的主文。
- [04 Mega MoE 与 paged MQA](04_mega_moe_and_paged_mqa.md)：换任务几何（token 池、`(q_atom, kv_split)`）后仍用 01/02 的执行语言。标准 MoE 读完 03 即可接入。

## 代码架构

```
deep_gemm/
├── __init__.py                         # Python 入口、legacy 别名
├── include/deep_gemm/
│   ├── common/                         # 类型、TMA copy、SM90/SM100 工具
│   ├── scheduler/gemm.cuh              # persistent tile 与 grouped 索引
│   ├── mma/sm90.cuh, mma/sm100.cuh     # GMMA / UMMA 描述符
│   ├── epilogue/                       # TMEM / accum 写回
│   ├── layout/mega_moe.cuh             # Mega MoE token 池
│   └── impls/                          # 架构 × 精度 × GEMM 类型
└── csrc/
    ├── apis/gemm.hpp                   # Host 断言与 grouped 分发
    └── jit_kernels/heuristics/         # BLOCK_* 候选与占用率排序
```

`GemmType` 与 `KernelType` 定义见 [`types.hpp`](src/types_hpp.md#__codelineno-0-18)。

## 源码浏览

- [types.hpp](src/types_hpp.md)：`GemmType` / `KernelType`
- [gemm.cuh](src/gemm_cuh.md)：persistent scheduler
- [tma_copy.cuh](src/tma_copy_cuh.md)：TMA 2D/3D、multicast / 2SM
- [sm90.cuh](src/mma_sm90_cuh.md) / [sm100.cuh](src/mma_sm100_cuh.md)：描述符构造
- [sm90_fp8_gemm_1d2d.cuh](src/sm90_fp8_gemm_1d2d_cuh.md)：SM90 grouped FP8 前向
- [sm90_bf16_gemm.cuh](src/sm90_bf16_gemm_cuh.md)：SM90 BF16 NoSF
- [sm100_fp8_fp4_gemm_1d1d.cuh](src/sm100_fp8_fp4_gemm_1d1d_cuh.md)：SM100 1D1D
- [gemm.hpp](src/gemm_hpp.md)：Host grouped API
- [sm90.hpp](src/heuristics_sm90_hpp.md)：`block_n` 格子
- [generators.py](src/generators_py.md)：contiguous / masked 生成器
- [mega_moe.cuh](src/mega_moe_cuh.md)：Mega MoE 池容量
- [paged_mqa_logits.cuh](src/paged_mqa_logits_cuh.md)：paged MQA 任务流
