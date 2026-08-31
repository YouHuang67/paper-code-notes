---
tags:
  - CUDA
  - CUTLASS
  - LLM Inference
---
# DeepGEMM 代码分析：SM90 到 SM100 流水

上一节把内核收成 layout / scheduler / mma / epilogue / impl。本节说明两代架构在 **mma 描述符、TMA 装载原语、中间累积位置、epilogue 是否独立** 上如何换件。grouped GEMM 的调用面在下一节使用这里的 tile 与 SF 语义。

## 1. SM90：warpgroup 是计算原子

Hopper 路径的同步主语是 warpgroup。`sm90_utils.cuh` 把 WGMMA 协议收成 arrive / commit / wait：

```cpp
__forceinline__ __device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__forceinline__ __device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N>
__forceinline__ __device__ void warpgroup_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}
```

**源码位置**: [`warpgroup_arrive`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/common/sm90_utils.cuh#L229-L245)

shared memory 经 `make_smem_desc` 变成 `cute::GmmaDescriptor`。1D2D FP8 的 math 循环是：等 TMA full barrier → `warpgroup_arrive` → 沿 `BLOCK_K / WGMMA::K` 发 `F32E4M3E4M3` WGMMA → `commit` / `wait<0>` → `scale_a * scale_b` 乘进 FP32 `final_accum`。[`sm90_fp8_gemm_1d2d.cuh:L283-L338`](src/sm90_fp8_gemm_1d2d_cuh.md#__codelineno-0-283)

心智模型：warpgroup 计算、smem 做 GMMA staging、barrier 把装载与计算接起来。累加结果主要活在寄存器里，epilogue 从寄存器 round 到 BF16 再 TMA store。

## 2. SM100：描述符体系切到 UMMA

Blackwell 工具层围绕 `cute::UMMA::SmemDescriptor`：`version_=1`、layout type 由 swizzle（0/32/64/128）与可选 `SWIZZLE_128B_BASE32B` 决定。[`mma/sm100.cuh:L14-L80`](src/mma_sm100_cuh.md#__codelineno-0-14)

impl 侧 UMMA 形状是 `UMMA_M = 128 * kNumMulticast`、`UMMA_K = 32`、`BLOCK_K = 128`。SF 粒度 `kGranKA/kGranKB` 为 32 或 128，对应 FP4 细块与 FP8 粗块。[`sm100_fp8_fp4_gemm_1d1d.cuh:L48-L68`](src/sm100_fp8_fp4_gemm_1d1d_cuh.md#__codelineno-0-48)

关注点是把 smem 与 TMEM 收成 UMMA 可消费几何。

## 3. TMA：单 SM load、SM90 multicast、SM100 2SM

[`tma_copy.cuh`](src/tma_copy_cuh.md) 的 2D 路径按 `num_tma_multicast` 分叉（`88965b0`）：

- `num_tma_multicast == 1`：一律 `cute::SM90_TMA_LOAD_2D`
- `num_tma_multicast > 1` 且 `__CUDA_ARCH__ >= 1000`：`cute::SM100_TMA_2SM_LOAD_2D`，信号只送到 leader CTA
- `num_tma_multicast > 1` 且 SM90：leader CTA 发 `cute::SM90_TMA_LOAD_MULTICAST_2D`

[`tma_copy.cuh:L27-L56`](src/tma_copy_cuh.md#__codelineno-0-27)

SM90 grouped 前向用 scheduler 的 `is_tma_multicast_valid` 在运行时把 multicast 降成 1。SM100 grouped 启发式把 `cluster_n` 固定为 1 或 2（\(N\) 方向 tile 数与 \(N_{\mathrm{SM}}\) 都为偶数时取 2）；2-CTA 保持固定 cluster，scheduler 省略奇数组成组修正。

## 4. TMEM 成为独立生产-消费阶段

SM100 1D1D 为累积、SFA、SFB 规划 TMEM 列数，并给 epilogue 单独的 full/empty barrier 与 `tmem_ptr_in_smem`。[`sm100_fp8_fp4_gemm_1d1d.cuh:L94-L157`](src/sm100_fp8_fp4_gemm_1d1d_cuh.md#__codelineno-0-94)

epilogue 等 `tmem_full_barriers[accum_stage_idx]`，从 `accum_stage_idx * UMMA_N` 读 TMEM，写入 smem CD 后再 TMA store，最后 arrive `tmem_empty_barriers`。矩阵乘结果先沉到 TMEM，由独立线程集分段搬运。

## 5. 对 grouped MoE 前向的直接后果

SM90 grouped FP8：`Kernel1D2D`，`BLOCK_K=128`；SF 为 FP32 且 LHS 须 MN-major；contiguous 的 \(BLOCK_M\) 取 `get_mk_alignment_for_contiguous_layout()`（默认 128）；\(BLOCK_N\) 走启发式格子，1D2D 上界 192；累加在寄存器，再 BF16 TMA store。

SM100 grouped FP8/FP4：`Kernel1D1D`，`BLOCK_K=128`；SF 为 packed UE8M0 `int`；\(BLOCK_M\) 同源，理论值可随 `expected_m` 从 240 以 16 步进下调；\(BLOCK_N\) 固定 128 且 `swap_ab=true`；累加经 TMEM staging 由 epilogue 写回 BF16。

Host 在 [`gemm.hpp`](src/gemm_hpp.md) 按 `arch_major` 与 SF dtype 把 `m_grouped_fp8_fp4_gemm_nt_contiguous` 分到 `sm90_*_1d2d` 或 `sm100_*_1d1d`。[`gemm.hpp:L193-L205`](src/gemm_hpp.md#__codelineno-0-193)

BF16 走 `KernelNoSF`，无 SF 流水；SM90 上 `BLOCK_K = 128 / sizeof(bf16) = 64`。分叉细节与 MoE 两次 GEMM 的精度顺序见 [03](03_grouped_gemm_moe_contract.md)。
