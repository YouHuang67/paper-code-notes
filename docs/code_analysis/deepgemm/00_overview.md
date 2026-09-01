---
tags:
  - CUDA
  - CUTLASS
  - LLM Inference
---
# DeepGEMM 代码分析：总览

**源码仓库**: [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/tree/88965b0)（提交 [`88965b0`](https://github.com/deepseek-ai/DeepGEMM/commit/88965b0)）

**团队**: DeepSeek

**分析范围**: DeepGEMM 的分层内核（[`deep_gemm/include/deep_gemm/`](https://github.com/deepseek-ai/DeepGEMM/tree/88965b0/deep_gemm/include/deep_gemm)）与 grouped Host 入口（[`csrc/apis/gemm.hpp`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/csrc/apis/gemm.hpp)）。[03](03_grouped_gemm_moe_contract.md) 另对照 SGLang 提交 [`62c505a`](https://github.com/sgl-project/sglang/tree/62c505a) 的 `MoeRunnerBackend.DEEP_GEMM`，说明标准 MoE 如何接到 grouped GEMM。正文交叉引用同时给出 GitHub 行锚点与站内 [源码浏览](src/index.md)（行号与官方文件一致）。

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

四篇正文按「执行语言 → 两代流水 → MoE 契约 → 专用布局」读。每篇只建立下一篇要用的对象。

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

`GemmType` 与 `KernelType` 定义见 [`types.hpp`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/common/types.hpp#L18-L39) · [types.hpp:L18-L39](src/types_hpp.md#__codelineno-0-18)。

## 源码浏览

站内副本与官方文件对照见 [src/index.md](src/index.md)。每条同时链到 GitHub 具体文件：

- [types.hpp](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/common/types.hpp) · [站内](src/types_hpp.md)：`GemmType` / `KernelType`
- [scheduler/gemm.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/scheduler/gemm.cuh) · [站内](src/gemm_cuh.md)：persistent scheduler
- [common/tma_copy.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/common/tma_copy.cuh) · [站内](src/tma_copy_cuh.md)：TMA 2D/3D、multicast / 2SM
- [common/sm90_utils.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/common/sm90_utils.cuh) · [站内](src/sm90_utils_cuh.md)：WGMMA 包装
- [ptx/wgmma.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/ptx/wgmma.cuh) · [站内](src/ptx_wgmma_cuh.md)：1D2D 实际调用的 `ptx::warpgroup_*`
- [mma/sm90.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/mma/sm90.cuh) · [站内](src/mma_sm90_cuh.md) / [mma/sm100.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/mma/sm100.cuh) · [站内](src/mma_sm100_cuh.md)：描述符构造
- [sm90_fp8_gemm_1d2d.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh) · [站内](src/sm90_fp8_gemm_1d2d_cuh.md)：SM90 grouped FP8 前向
- [sm90_bf16_gemm.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh) · [站内](src/sm90_bf16_gemm_cuh.md)：SM90 BF16 NoSF 内核
- [sm90_bf16_gemm.hpp](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/csrc/jit_kernels/impls/sm90_bf16_gemm.hpp) · [站内](src/sm90_bf16_gemm_hpp.md)：SM90 BF16 Host JIT
- [sm100_fp8_fp4_gemm_1d1d.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh) · [站内](src/sm100_fp8_fp4_gemm_1d1d_cuh.md)：SM100 1D1D
- [gemm.hpp](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/csrc/apis/gemm.hpp) · [站内](src/gemm_hpp.md)：Host grouped API
- [heuristics/sm90.hpp](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/csrc/jit_kernels/heuristics/sm90.hpp) · [站内](src/heuristics_sm90_hpp.md) / [sm100.hpp](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/csrc/jit_kernels/heuristics/sm100.hpp) · [站内](src/heuristics_sm100_hpp.md) / [runtime.hpp](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/csrc/jit_kernels/heuristics/runtime.hpp) · [站内](src/heuristics_runtime_hpp.md)
- [tests/generators.py](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/tests/generators.py) · [站内](src/generators_py.md)：contiguous / masked 生成器
- [layout/mega_moe.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/layout/mega_moe.cuh) · [站内](src/mega_moe_cuh.md) / [sm100_fp8_fp4_mega_moe.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh) · [站内](src/sm100_fp8_fp4_mega_moe_cuh.md)
- [paged_mqa_logits.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/scheduler/paged_mqa_logits.cuh) · [站内](src/paged_mqa_logits_cuh.md) / [sm100_fp4_paged_mqa_logits.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm100_fp4_paged_mqa_logits.cuh) · [站内](src/sm100_fp4_paged_mqa_logits_cuh.md)
