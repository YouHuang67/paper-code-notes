---
tags:
  - CUDA
  - CUTLASS
  - LLM Inference
---
# DeepGEMM 源码

**仓库**: [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) · **提交**: `88965b0` · **解读**: [代码分析](../00_overview.md)

**类型与调度**

- [types.hpp](types_hpp.md)：`GemmType` / `KernelType`
- [scheduler/gemm.cuh](gemm_cuh.md)：persistent tile 与 grouped 索引
- [common/tma_copy.cuh](tma_copy_cuh.md)：TMA 2D/3D、multicast / 2SM

**MMA 与实现**

- [mma/sm90.cuh](mma_sm90_cuh.md)：GMMA 描述符
- [mma/sm100.cuh](mma_sm100_cuh.md)：UMMA 描述符
- [impls/sm90_fp8_gemm_1d2d.cuh](sm90_fp8_gemm_1d2d_cuh.md)：SM90 grouped FP8
- [impls/sm90_bf16_gemm.cuh](sm90_bf16_gemm_cuh.md)：SM90 BF16 NoSF
- [impls/sm100_fp8_fp4_gemm_1d1d.cuh](sm100_fp8_fp4_gemm_1d1d_cuh.md)：SM100 1D1D

**Host 与测试**

- [csrc/apis/gemm.hpp](gemm_hpp.md)：grouped API 与断言
- [csrc/jit_kernels/heuristics/sm90.hpp](heuristics_sm90_hpp.md)：`block_n` 候选
- [tests/generators.py](generators_py.md)：contiguous / masked 生成器

**专用布局**

- [layout/mega_moe.cuh](mega_moe_cuh.md)：token 池与 Workspace
- [scheduler/paged_mqa_logits.cuh](paged_mqa_logits_cuh.md)：paged MQA metadata
