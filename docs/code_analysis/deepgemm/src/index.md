---
tags:
  - CUDA
  - CUTLASS
  - LLM Inference
---
# DeepGEMM 源码

**仓库**: [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/tree/88965b0) · **提交**: [`88965b0`](https://github.com/deepseek-ai/DeepGEMM/commit/88965b0) · **解读**: [代码分析](../00_overview.md)

站内每页是官方文件的完整副本（`linenums="1"` 与仓库行号一致）。标题下的「仓库」链到该文件在 `88965b0` 上的 blob。

**类型与调度**

- [types.hpp](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/common/types.hpp) · [站内](types_hpp.md)：`GemmType` / `KernelType`
- [scheduler/gemm.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/scheduler/gemm.cuh) · [站内](gemm_cuh.md)：persistent tile 与 grouped 索引
- [common/tma_copy.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/common/tma_copy.cuh) · [站内](tma_copy_cuh.md)：TMA 2D/3D、multicast / 2SM
- [common/sm90_utils.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/common/sm90_utils.cuh) · [站内](sm90_utils_cuh.md)：WGMMA 包装
- [ptx/wgmma.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/ptx/wgmma.cuh) · [站内](ptx_wgmma_cuh.md)：`ptx::warpgroup_*`

**MMA 与实现**

- [mma/sm90.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/mma/sm90.cuh) · [站内](mma_sm90_cuh.md)：GMMA 描述符
- [mma/sm100.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/mma/sm100.cuh) · [站内](mma_sm100_cuh.md)：UMMA 描述符
- [impls/sm90_fp8_gemm_1d2d.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh) · [站内](sm90_fp8_gemm_1d2d_cuh.md)：SM90 grouped FP8
- [impls/sm90_bf16_gemm.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh) · [站内](sm90_bf16_gemm_cuh.md)：SM90 BF16 NoSF 内核
- [impls/sm100_fp8_fp4_gemm_1d1d.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh) · [站内](sm100_fp8_fp4_gemm_1d1d_cuh.md)：SM100 1D1D

**Host 与测试**

- [csrc/apis/gemm.hpp](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/csrc/apis/gemm.hpp) · [站内](gemm_hpp.md)：grouped API 与断言
- [csrc/jit_kernels/impls/sm90_bf16_gemm.hpp](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/csrc/jit_kernels/impls/sm90_bf16_gemm.hpp) · [站内](sm90_bf16_gemm_hpp.md)：BF16 Host JIT
- [heuristics/sm90.hpp](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/csrc/jit_kernels/heuristics/sm90.hpp) · [站内](heuristics_sm90_hpp.md)：`block_n` 候选
- [heuristics/sm100.hpp](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/csrc/jit_kernels/heuristics/sm100.hpp) · [站内](heuristics_sm100_hpp.md)：SM100 layout 候选
- [heuristics/runtime.hpp](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/csrc/jit_kernels/heuristics/runtime.hpp) · [站内](heuristics_runtime_hpp.md)：contiguous alignment
- [tests/generators.py](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/tests/generators.py) · [站内](generators_py.md)：contiguous / masked 生成器

**专用布局**

- [layout/mega_moe.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/layout/mega_moe.cuh) · [站内](mega_moe_cuh.md)：token 池与 Workspace
- [impls/sm100_fp8_fp4_mega_moe.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh) · [站内](sm100_fp8_fp4_mega_moe_cuh.md)：Mega MoE 融合核
- [scheduler/paged_mqa_logits.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/scheduler/paged_mqa_logits.cuh) · [站内](paged_mqa_logits_cuh.md)：paged MQA metadata
- [impls/sm100_fp4_paged_mqa_logits.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm100_fp4_paged_mqa_logits.cuh) · [站内](sm100_fp4_paged_mqa_logits_cuh.md)：paged MQA 主核
