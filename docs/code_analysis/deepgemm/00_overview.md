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

累加器为 FP32。库本身不负责转置、cast、permute、SwiGLU 与 router 加权；这些步骤由调用方在核外完成。

对标准 MoE，一次 launch 覆盖 \(G\) 个同形状 expert：只沿 \(M\) 分组，\(N\) 与 \(K\) 固定。contiguous 把各 expert 的 token 沿 \(M\) 拼成一条对齐带；masked 保留前缀维 \(G\)，用 `masked_m[G]` 标有效行。SGLang 在核外做 permute / scatter，两次 grouped GEMM（\(W_{13}\) 与 \(W_2\)）共用同一份行索引，SwiGLU 与 finalize 发生在 BF16 交接之后。

Mega MoE 与 paged MQA 走专用路径：把 routed token 池或 paged KV 先收成规则任务流，再交给同一类 TMA / UMMA 原语。标准 MoE 适配走 grouped GEMM 契约。

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

## 吞吐从哪里来

一次 grouped 前向把下列机制叠在同一 CTA 上：

- persistent 抢 tile：CTA \(x\) 第 \(\mathrm{iter}\) 轮领取 \(\mathrm{next}=(\mathrm{iter}+1)\cdot N_{\mathrm{SM}}+\mathrm{blockIdx.x}\)
- TMA 按 stage 装 A/B（FP8 另装 SF），math warp 做 WGMMA/UMMA，整段 \(K\) 累进 FP32
- 写回前 `__float22bfloat162_rn`（grouped 前向 \(D\) 为 BF16）
- contiguous 把 \(BLOCK_M\) 锁成进程内 `get_mk_alignment_for_contiguous_layout()`（SM90 默认 128）
- 形状进入 JIT 键；CUDA graph 捕获前必须热编译对应 `(M,N,K,G,layout,dtype)`

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

## 标准 MoE 在库边界上的形状

expert 前向是两段线性：\(h W_{13}^{\mathsf{T}}\) 得 gate/up，SwiGLU 后 \(h' W_2^{\mathsf{T}}\) 回到隐藏维。DeepGEMM 只看见

- \(A\)：按 expert 排好的激活（contiguous `[M,K]` 或 masked `[G,M_{\max},K]`）
- \(B\)：堆叠权重 `[G,N,K]`（NT 下与 \(A\) 做 \(A B^{\mathsf{T}}\)）
- `grouped_layout` 或 `masked_m`：行到 expert 的索引

同一份索引喂两次 kernel，中间激活与量化在核外。细节在 [03 grouped GEMM 与标准 MoE 契约](03_grouped_gemm_moe_contract.md)。

## 文档导航

- [01 分层内核架构](01_layered_architecture.md)：layout / scheduler / mma / epilogue / impl，以及 persistent 角色分工
- [02 SM90 到 SM100 流水](02_sm90_sm100_pipeline.md)：WGMMA 协议、UMMA/TMEM、TMA 单 SM 与 2SM
- [03 grouped GEMM 与标准 MoE 契约](03_grouped_gemm_moe_contract.md)：contiguous / masked / psum、FP8 与 BF16 分叉、SGLang permute–GEMM–SwiGLU–finalize
- [04 Mega MoE 与 paged MQA](04_mega_moe_and_paged_mqa.md)：共享 token 池与元数据驱动的 indexer 流水（标准 MoE 之外的专用路径）

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
