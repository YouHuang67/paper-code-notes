---
tags:
  - CUDA
  - CUTLASS
  - LLM Inference
---
# DeepGEMM 代码分析：分层内核架构

上一节给出统一数学对象 \(D=C+AB^{\mathsf{T}}\) 与记号。本节说明内核如何按几何角色拆层；下一节用同一分层解释 SM90 与 SM100 流水差在哪一层。

DeepGEMM 的内核按五层拆开，各做一种几何变换，在 impl 汇合：

- **layout**：数据池、工作区、块对齐、存储视图
- **scheduler**：全局问题切成 CTA / warpgroup 可领取的 tile 流
- **mma**：shared memory 视图转成 GMMA / UMMA 描述符
- **epilogue**：TMEM / accum 写回最终张量
- **impl**：用具体架构的 barrier 与角色把上述层接成流水

模板参数是这些层的笛卡尔积：A/B major、K 向 SF 粒度、CTA tile、smem swizzle、warp 角色数、multicast、`GemmType`、epilogue 类型。impl 文件编码的是执行架构。

**源码位置**: [`sm100_fp8_fp4_gemm_1d1d_impl`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh#L19-L40) · [sm100_fp8_fp4_gemm_1d1d.cuh:L19-L40](src/sm100_fp8_fp4_gemm_1d1d_cuh.md#__codelineno-0-19)

## 1. impl 先暴露角色，再暴露数学

[`sm100_fp8_fp4_gemm_1d1d.cuh`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh) 把线程分成非 epilogue 与 epilogue 两套 launch bound。shared memory 按 stage 切成 CD store、A tile、B tile、SFA/SFB，再用 `PatternVisitor` 把裸偏移收成「第 \(i\) 个 stage 的视图」。barrier 同样按职责切：TMA full/empty、带 SF 的 `with_sf_full`、TMEM 交给 epilogue 的 full/empty。

**源码位置**: [`PatternVisitor` smem / barrier](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh#L131-L157) · [sm100_fp8_fp4_gemm_1d1d.cuh:L131-L157](src/sm100_fp8_fp4_gemm_1d1d_cuh.md#__codelineno-0-131)

```cpp
auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
});
auto smem_a  = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<a_dtype_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
});
auto full_barriers          = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
auto empty_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
auto with_sf_full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
auto tmem_full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + i); });
```

SM90 FP8 1D2D 用同一模式，只是 SF 为 FP32、无独立 TMEM barrier：smem 顺序为 D、A stages、B stages、SFA stages、整段 SFB，然后 full/empty barrier。

**源码位置**: [sm90_fp8_gemm_1d2d.cuh:L110-L126](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh#L110-L126) · [站内 L110](src/sm90_fp8_gemm_1d2d_cuh.md#__codelineno-0-110)

## 2. scheduler 输出硬件可执行的块顺序

`sched::Scheduler` 输入 `shape_m, shape_n, shape_k` 与可选 `grouped_layout`，输出 `(m_block_idx, n_block_idx)`。grid 常驻 \(N_{\mathrm{SM}}\) 个 CTA，每个 CTA 用

\[
\mathrm{next} = (++\mathrm{current\_iter})\cdot N_{\mathrm{SM}} + \mathrm{blockIdx.x}
\]

领取下一 tile。

**源码位置**: [`get_next_block`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/scheduler/gemm.cuh#L171-L172) · [gemm.cuh:L171-L172](src/gemm_cuh.md#__codelineno-0-171)

swizzle 把 tile 编成对 L2 与 TMA multicast 友好的组：主轴按 8 或 16 个 1D block 成组，组内走次轴。奇数组成组修正只编译进 SM90（`#if __CUDA_ARCH__ < 1000`）；SM100 2-CTA 保持固定 cluster。

**源码位置**: [gemm.cuh:L95-L131](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/scheduler/gemm.cuh#L95-L131) · [站内 L95](src/gemm_cuh.md#__codelineno-0-95)

`get_global_idx` 把块坐标译成 TMA 用的全局偏移。contiguous 用 `grouped_layout[m_{\mathrm{blk}}\cdot BLOCK_M]` 当 expert id；padding 行为 \(-1\)，`max(0,\cdot)` 把负 id 收成 0 偏移，配合 `is_computation_valid` 丢掉 padding 行上的 MMA。

**源码位置**: [`get_global_idx`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/scheduler/gemm.cuh#L133-L160) · [gemm.cuh:L133-L160](src/gemm_cuh.md#__codelineno-0-133)；[`is_computation_valid`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/scheduler/gemm.cuh#L282-L295) · [gemm.cuh:L282-L295](src/gemm_cuh.md#__codelineno-0-282)

multicast 在 contiguous 且 multicast 落在 A 的对侧时，要求 peer tile 的 expert id 相同，否则退回单 CTA 装载。

**源码位置**: [`is_tma_multicast_valid`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/scheduler/gemm.cuh#L261-L277) · [gemm.cuh:L261-L277](src/gemm_cuh.md#__codelineno-0-261)

## 3. 内核本体按角色领取 scheduler 任务

SM90 FP8 1D2D 中，math warpgroup 占前 `kNumMathThreads` 线程；其后的 warp 做 TMA。TMA 侧选出一个 elected thread，在 `get_next_block` 循环里按 stage 发 A/B/SFA 的 TMA，并用 `arrive_and_expect_tx` 把本 stage 字节数（含 SFA）登记到 full barrier。

**源码位置**: [sm90_fp8_gemm_1d2d.cuh:L167-L206](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh#L167-L206) · [站内 L167](src/sm90_fp8_gemm_1d2d_cuh.md#__codelineno-0-167)

math 侧同样 `while (scheduler.get_next_block(...))`：等 full barrier、WGMMA、用 `scale_a * scale_b` 提升到 FP32 累加器、empty barrier 放行生产者。TMA 与 MMA 解耦后，同一套 scheduler 可以驱动多阶段流水。

SM100 1D1D 把 TMA warp、UMMA warp、epilogue warp 分得更开：TMA 只生产 smem，UMMA 只消费描述符并写入 TMEM，epilogue 等 `tmem_full_barriers` 再 store。角色流见 [02](02_sm90_sm100_pipeline.md)。

## 4. mma 层只构造描述符几何

[`mma/sm90.cuh`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/mma/sm90.cuh) 的 `make_smem_desc` 把 smem 指针编成 `cute::GmmaDescriptor`（start address、layout type、leading/stride byte offset）。[`mma/sm100.cuh`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/mma/sm100.cuh) 的主语换成 `cute::UMMA::SmemDescriptor`：version、swizzle layout type、base32、MN/K major。

**源码位置**: [`mma::sm90::make_smem_desc`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/mma/sm90.cuh#L196-L210) · [sm90.cuh:L196-L210](src/mma_sm90_cuh.md#__codelineno-0-196)；[sm100.cuh:L14-L80](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/deep_gemm/include/deep_gemm/mma/sm100.cuh#L14-L80) · [站内 L14](src/mma_sm100_cuh.md#__codelineno-0-14)

K-major 时 swizzle 等于 `BLOCK_K * sizeof(dtype)`，且每个 block 在 K 轴上只有一个 swizzle atom。这些约束留在 mma 层。

## 5. 分层的收益落在 impl 的交界

layout、scheduler、mma、epilogue 边界固定之后，impl 只协调：

- 哪个 warp 发 TMA
- 哪个 stage 等哪组 barrier
- SF 与主矩阵如何交错
- TMEM / 寄存器结果何时交给 epilogue

SM90 1D2D 的交界是 full/empty barrier + warpgroup arrive/commit/wait。SM100 1D1D 多出 TMEM full/empty 与独立 epilogue 线程集。下一节按这条交界对比两代流水。
