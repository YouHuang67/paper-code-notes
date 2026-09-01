---
tags:
  - CUDA
  - CUTLASS
  - LLM Inference
---
# DeepGEMM 代码分析：SM90 到 SM100 流水

上一节把内核收成 layout / scheduler / mma / epilogue / impl。本节说明两代架构在 **mma 描述符、TMA 装载原语、中间累积位置、epilogue 独立性** 上如何换件。grouped GEMM 的调用面在下一节使用这里的 tile 与 SF 语义。

## 1. SM90：warpgroup 是计算原子

Hopper 路径的同步主语是 warpgroup。SM90 1D2D 调用 `ptx::warpgroup_arrive` / `commit_batch` / `wait`（[`ptx/wgmma.cuh`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/ptx/wgmma.cuh)）；[`common/sm90_utils.cuh`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/common/sm90_utils.cuh) 提供同一组包装。指令序列见[附录 A](#app-wgmma-ptx)。

**源码位置**: [`ptx::warpgroup_arrive`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/ptx/wgmma.cuh#L7-L23) · [wgmma.cuh:L7-L23](src/ptx_wgmma_cuh.md#__codelineno-0-7)；[`sm90_utils.cuh:L229-L245`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/common/sm90_utils.cuh#L229-L245) · [站内 L229](src/sm90_utils_cuh.md#__codelineno-0-229)

shared memory 经 `mma::sm90::make_smem_desc` 变成 `cute::GmmaDescriptor`。1D2D FP8 的 math 循环是：等 TMA full barrier → `warpgroup_arrive` → 沿 `BLOCK_K / WGMMA::K` 发 `F32E4M3E4M3` WGMMA → `commit` / `wait<0>` → `scale_a * scale_b` 乘进 FP32 `final_accum`。

**源码位置**: [sm90_fp8_gemm_1d2d.cuh:L283-L338](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh#L283-L338) · [站内 L283](src/sm90_fp8_gemm_1d2d_cuh.md#__codelineno-0-283)

心智模型：warpgroup 计算、smem 做 GMMA staging、barrier 把装载与计算接起来。累加结果主要活在寄存器里，epilogue 从寄存器 round 到 BF16 再 TMA store。

## 2. SM100：描述符体系切到 UMMA

Blackwell 工具层围绕 `cute::UMMA::SmemDescriptor`：`version_=1`、layout type 由 swizzle（0/32/64/128）与可选 `SWIZZLE_128B_BASE32B` 决定。

**源码位置**: [mma/sm100.cuh:L14-L80](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/mma/sm100.cuh#L14-L80) · [站内 L14](src/mma_sm100_cuh.md#__codelineno-0-14)

impl 侧 UMMA 形状是 `UMMA_M = 128 * kNumMulticast`、`UMMA_K = 32`、`BLOCK_K = 128`。SF 粒度 `kGranKA/kGranKB` 为 32 或 128，对应 FP4 细块与 FP8 粗块。

**源码位置**: [sm100_fp8_fp4_gemm_1d1d.cuh:L48-L68](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh#L48-L68) · [站内 L48](src/sm100_fp8_fp4_gemm_1d1d_cuh.md#__codelineno-0-48)

## 3. TMA：单 SM load、SM90 multicast、SM100 2SM

[`tma_copy.cuh`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/common/tma_copy.cuh) 的 2D 路径按 `num_tma_multicast` 分叉：

- `num_tma_multicast == 1`：一律 `cute::SM90_TMA_LOAD_2D`
- `num_tma_multicast > 1` 且 `__CUDA_ARCH__ >= 1000`：`cute::SM100_TMA_2SM_LOAD_2D`，信号只送到 leader CTA
- `num_tma_multicast > 1` 且 SM90：leader CTA 发 `cute::SM90_TMA_LOAD_MULTICAST_2D`

**源码位置**: [tma_copy.cuh:L27-L56](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/common/tma_copy.cuh#L27-L56) · [站内 L27](src/tma_copy_cuh.md#__codelineno-0-27)

SM90 grouped 前向用 scheduler 的 `is_tma_multicast_valid` 在运行时把 multicast 降成 1。SM100 grouped 启发式把 `cluster_n` 固定为 1 或 2（\(N\) 方向 tile 数与 \(N_{\mathrm{SM}}\) 都为偶数时取 2）。奇数组成组修正见 [01](01_layered_architecture.md) 的 SM90 swizzle。

## 4. TMEM 成为独立生产-消费阶段

SM100 1D1D 为累积、SFA、SFB 规划 TMEM 列数，并给 epilogue 单独的 full/empty barrier 与 `tmem_ptr_in_smem`。

**源码位置**: [sm100_fp8_fp4_gemm_1d1d.cuh:L94-L157](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh#L94-L157) · [站内 L94](src/sm100_fp8_fp4_gemm_1d1d_cuh.md#__codelineno-0-94)

epilogue 等 `tmem_full_barriers[accum_stage_idx]`，从该 stage 对应的 TMEM 列（`accum_stage_idx * UMMA_N`，其中 `UMMA_N` 为 UMMA 在 \(N\) 向的边长）读出，写入 smem CD 后再 TMA store，最后 arrive `tmem_empty_barriers`。矩阵乘结果先沉到 TMEM，由独立线程集分段搬运。

## 5. 换件之后，grouped 前向怎么选核

对 M-grouped 前向，上述四件决定 Host 分发：SM90 FP8 走 `Kernel1D2D`（`BLOCK_K=128`，FP32 SF）；SM100 FP8/FP4 走 `Kernel1D1D`（packed UE8M0）；BF16 走 `KernelNoSF`（`BLOCK_K=64`）。tile 格子、alignment 与两次 GEMM 的精度顺序见 [03](03_grouped_gemm_moe_contract.md)。paged MQA 把同一角色拆分扩成四类 warp，见 [04](04_mega_moe_and_paged_mqa.md)。

## 附录 A {#app-wgmma-ptx}

```cpp
CUTLASS_DEVICE void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
CUTLASS_DEVICE void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N>
CUTLASS_DEVICE void warpgroup_wait() {
    DG_STATIC_ASSERT(N >= 0 and N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}
```
