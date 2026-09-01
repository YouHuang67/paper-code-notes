---
tags:
  - CUDA
  - CUTLASS
  - LLM Inference
---
# DeepGEMM 代码分析：grouped GEMM 与标准 MoE 契约

01/02 给出 persistent tile 流与 1D2D / 1D1D / NoSF 的选型。本节把同一套 kernel 读成 MoE 调用面：先固定 M-grouped 几何与 FP8/BF16 三条分叉，再把标准 MoE（permute → GEMM1 → SwiGLU → GEMM2 → finalize）接到这条边界上。Mega MoE 见 [04](04_mega_moe_and_paged_mqa.md)。

记号沿用 [总览](00_overview.md)。证据：DeepGEMM `88965b0`，SGLang `62c505a`。

## 1. 机制：一次 launch、整段 \(K\)、三条分叉

吞吐对象是一组同形状 expert 的 NT GEMM \(D=C+AB^{\mathsf{T}}\)（定义见总览）。M-grouped 前向：\(N,K\) 固定，\(M\) 按组变。contiguous 下 \(A\in\mathbb{R}^{M\times K}\)，\(B\in\mathbb{R}^{G\times N\times K}\)，\(D\in\mathbb{R}^{M\times N}\)，\(A\) 沿 \(M\) 按 expert 分段。masked 下 \(A,D\) 带前缀维 \(G\)，组 \(g\) 的有效行 \(m_g\le M_{\max}\)。可选 \(C\) 与 \(D\) 同形。同一 CTA 走完整 \(K\)，FP32 累加后 identity store 为 BF16。K-grouped（权重反传）\(M,N\) 固定、\(K\) 按组变；FP8 每段 \(K\) 整除 128，1D1D 写 FP32 `TMA_REDUCE_ADD`。

grid 等于 \(N_{\mathrm{SM}}\)，领取公式见 [01](01_layered_architecture.md)。contiguous 的组号来自 `grouped_layout[m_{\mathrm{blk}}\cdot BLOCK_M]`，因此 \(BLOCK_M\) 必须等于 `get_mk_alignment_for_contiguous_layout()`。

同一 scheduler 上有两条 grouped 前向：FP8 为 SM90 `Kernel1D2D`（SM100 `Kernel1D1D`）并携带 SF；BF16 为 `KernelNoSF`。相对 FP8 grouped 前向，BF16 只改三处。

### 1.1 无 SF，因而无 ensure_zero_padding

NoSF 的 smem 只有 D/A/B 与 barrier，`smem_sfa_per_stage = 0`。Host 入口 `m_grouped_bf16_gemm_nt_contiguous` 不调用 `transform_sf_pair_into_required_layout`。[`sm90_bf16_gemm.hpp`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/csrc/jit_kernels/impls/sm90_bf16_gemm.hpp) 填 `KernelType::KernelNoSF`，并断言 \(K\) 整除 64。

FP8 的 SF 要把 MN 扩到 TMA 16 字节对齐（ensure_zero_padding）：packed UE8M0 的 torch 路径先 `zeros` 再拷入有效区；CUDA pack 对越界 K 写 0，且只 store `global_mn_idx < mn`。K-grouped Host 注释为 `Transform SF with padding`。contiguous 激活对齐尾仍由生成器写成 \(A=0\)、`grouped_layout=-1`，与 SF 对齐是两套契约。

`use_psum_layout` 在 BF16 与 `m_grouped_fp8_fp4_gemm_nt_contiguous` 上默认 `false`；SGLang wrapper 使用逐行 `m_indices`。

### 1.2 官方 `block_n` 格子与格子外的 16

SM90 `get_layout_candidates`：`step = lcm(16, block_n_multiple_of)`，`start = step`，`end` 为 NoSF 256 / 1D2D 192 / 1D1D 160，再 `for (i = start; i <= end; i += step)`。contiguous / psum 的 \(BLOCK_M\) 只有 alignment。候选还要求 \(BLOCK_M\) 与 \(BLOCK_N\) 至少一个 \(\le 128\)（alignment 为 128 时 NoSF 可取 `block_n=256`）。排序量是 \(\max(\text{L1 cycles},\text{L2 cycles})/\text{wave\_efficiency}\)：分母是最后一波 SM 占用，分子是字节流折成的片上/片外周期。

[`sm90.hpp:L38-L60`](src/heuristics_sm90_hpp.md#__codelineno-0-38)

`block_n=16` 作为独立项只在 1D1D 且 \(D\) 为 FP32 时出现：先把 `start` 改成 24（躲开 FP32 写回 bank conflict），再 `push_back(16)`。SM100 非 grouped 同样在 32-step 格子外可加入 16。SM100 grouped 固定 `block_n=128`、`swap_ab=true`。[`sm100.hpp:L31-L42`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/csrc/jit_kernels/heuristics/sm100.hpp#L31-L42)

### 1.3 入口是两套符号

SGLang 按 `w13_weight.dtype` 分流。FP8/FP4 走 `grouped_gemm_nt_f8f8bf16_contig` → `deep_gemm.m_grouped_fp8_gemm_nt_contiguous`（C++ 短名等于 `m_grouped_fp8_fp4_*`）。BF16 走 `grouped_gemm_nt_bf16_contig` → `m_grouped_bf16_gemm_nt_contiguous`。`fp8_m_grouped_gemm_nt_masked` / `bf16_m_grouped_gemm_nt_masked` 标了 `TODO: remove these later`。`deep_gemm.legacy.*_tl` 模块头写明可能弃用。Serving 现行名是 `grouped_gemm_nt_f8f8bf16_contig` 与 `grouped_gemm_nt_bf16_contig`。

## 2. 约束：调用面几何与类型

Host 在 [`gemm.hpp`](src/gemm_hpp.md) 按 `arch_major` 与 SF dtype 把 `m_grouped_fp8_fp4_gemm_nt_contiguous` 分到 `sm90_*_1d2d` 或 `sm100_*_1d1d`。[`gemm.hpp:L193-L205`](src/gemm_hpp.md#__codelineno-0-193)

SM90 FP8 要求 A、B 都是 K-major。Grouped 前向 \(D\) 为 BF16，`with_accumulation = false`。FP8 的 LHS SF 必须 MN-major，并经 ensure_zero_padding。BF16 入口无 SF 参数。\(G\) 个 expert 共享同一 \((N,K)\)。

布局语义：

- contiguous 逐行：`grouped_layout[M]`，\(M=\sum_g \mathrm{align}(m_g)\)；行 \(i\) 的 expert id，padding \(-1\)
- contiguous psum：`grouped_layout[G]`；组 \(g\) 有效结束行
- masked：`masked_m[G]`，\(A\) 为 `[G,M_{\max},K]`；组 \(g\) 的 \(m_g\)

对齐值必须等于进程内 `get_mk_alignment_for_contiguous_layout()`，与 \(BLOCK_M\) 同一来源。SM90 默认 128；SM100 可用 `expected_m` 把理论 alignment 从 240 以 MMA step 16 下调，且不小于 32。[`runtime.hpp:L47-L57`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/csrc/jit_kernels/heuristics/runtime.hpp#L47-L57)

`expected_m` 还进入启发式。\(m_g=1\) 时该组仍占一个 \(BLOCK_M\) tile，math 用 `is_computation_valid` 丢掉 padding。生成器把对齐尾写成 \(A=0\)、逐行 layout 为 \(-1\)。[`generators.py:L294-L321`](src/generators_py.md#__codelineno-0-294)

**源码位置**: [`m_grouped_fp8_fp4_gemm_nt_contiguous`](https://github.com/deepseek-ai/DeepGEMM/blob/88965b0/csrc/apis/gemm.hpp#L143-L170)

## 3. 标准 MoE 接入契约

对照 SGLang `MoeRunnerBackend.DEEP_GEMM`（[`moe_runner/deep_gemm.py`](https://github.com/sgl-project/sglang/blob/62c505a/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py)）。框架在核外构造 grouped 输入；两次 GEMM 保持同一行序；激活与 finalize 发生在 BF16 交接之后。

标准 MoE 一层的数据流：

1. Router 给出 `topk_ids`、`topk_weights`，形状 `[T, k_{\mathrm{top}}]`。
2. **pre-permute**：把 token 排成 kernel 布局，并（FP8 路径）量化 hidden。
3. **GEMM1**：\(W_{\mathrm{gate}},W_{\mathrm{up}}\) 在 \(N\) 维拼成 \(W_{13}\)，一次 grouped GEMM 得到 `gateup_output`。
4. **SwiGLU**：gate/up 在 FP32 完成 SiLU 与乘法，再 round 到 BF16；FP8 路径随后做 per-token group 量化。
5. **GEMM2**：同一份 `m_indices` / `masked_m` 乘 \(W_2\)。
6. **post-permute / finalize**：按 token 把 \(k_{\mathrm{top}}\) 路 expert 输出用 `topk_weights` 在 FP32 加权写回。

### 3.1 三条 dispatcher 到 grouped 布局

**standard → masked**。`pre_permute_standard_to_deep_gemm` 调用 `moe_ep_deepgemm_preprocess`：按 expert 做 stable 分段，得到 `masked_m`、`expected_m`、逆置换 `src2dst`，以及 `[G, M_{\max}, K]` 的 hidden（FP8 时附 scale）。`use_masked_gemm=True`。CUDA graph 下 CPU 不必知道每组真实 \(m_g\)，kernel 只算 `masked_m` 标出的有效行。

**DeepEP low-latency → masked**。DeepEP LL 的输出已经是 masked 布局，`pre_permute_deepep_ll_to_deep_gemm` 直接转发 `masked_m` / `expected_m`。

**DeepEP normal → contiguous**。`pre_permute_deepep_normal_to_deep_gemm` 用 `ep_scatter` 把 recv token 写入对齐后的 `input_tensor`，并写逐行 `m_indices`（即 `grouped_layout`）。对齐块 `BLOCK_E=128`，与 SM90 默认 \(BLOCK_M\) 一致。`all_tokens = sum(num_recv_tokens_per_expert)` 已含 padding。

### 3.2 两次 GEMM 与精度顺序

contiguous FP8 路径（`_run_contiguous_gemm`）：

1. 量化 hidden 与权重（权重以 `(w, scale)` pair 进入）。
2. GEMM1：核内 FP32 累加，`__float22bfloat162_rn` 写 BF16 `gateup_output`，形状 `[all_tokens, N]`。
3. SwiGLU：`silu_and_mul_contig_post_quant`（或 fallback：BF16 `silu_and_mul` 再 `sglang_per_token_group_quant_fp8`）。SiLU 与乘法在 FP32 完成，乘完 `.to(bf16)` 后再按 group size 128 量化。
4. GEMM2：同样 FP32→BF16，输出 `[all_tokens, K]`。
5. DeepEP normal 的 combine 用 `ep_gather`；standard 的 finalize 见下。

contiguous BF16 路径无量化与 post-quant，两次 `grouped_gemm_nt_bf16_contig` 夹一次 `_legacy_silu_and_mul`。

masked 路径把张量形状换成 `[G, M_{\max}, ·]`，两次 `grouped_gemm_nt_f8f8bf16_masked` / `_bf16_masked`，SwiGLU 走 `_varlen_deep_gemm_silu_mul_quant` 或 `silu_and_mul_masked_fwd`。FP4 expert 把 `recipe_a=(1,128)`、`recipe_b=(1,32)` 传进 DeepGEMM。

两次 kernel 共用同一份 `m_indices` / `masked_m`，GEMM2 才能对准对应 expert 的 \(W_2\)。

### 3.3 finalize：按 token 的 FP32 顺序加权

standard 路径的 `post_permute_deep_gemm_to_standard` 启动 `post_reorder_triton_kernel`，grid 为 token 数 \(T\)。每个 program 顺序扫 \(k_{\mathrm{top}}\)：

- 用 `src2dst` 把 (token, expert-slot) 映到 masked 输出行
- 将该行 hidden 升到 FP32，乘 `topk_weights`
- 在寄存器 `sum_vec` 里累加后一次 `tl.store`

DeepGEMM 写出的 masked 行互不重叠，hidden 维用顺序累加。若配置了 `routed_scaling_factor`，在 kernel 外再乘一次。

**源码位置**: [`post_reorder_triton_kernel`](https://github.com/sgl-project/sglang/blob/62c505a/python/sglang/srt/layers/moe/ep_moe/kernels.py#L681-L719)

### 3.4 JIT 与 CUDA graph

JIT 键含 `BLOCK_*`、`kNumStages`、`compiled_dims`、`KernelType`。捕获 CUDA graph 前必须对将出现的 `(M,N,K,G,layout,dtype)` 热编译，否则 graph 内会撞上首次 NVCC。workspace（TMA tensor map 等）也要在捕获前分配到稳定地址。

## 4. 结论

每次 grouped 前向把同形状、变 \(M\) 的 expert 收成一次 GPU 启动。吞吐来自 TMA 与 WGMMA/UMMA 多 stage 重叠、角色拆分、\(N_{\mathrm{SM}}\) 常驻抢 tile、contiguous 把 \(BLOCK_M\) 锁成组对齐、FP32 整段 \(K\) 累加、按形状 JIT。BF16 NoSF 用 \(BLOCK_K=64\) 去掉 SF 流水。

框架契约：

- `m_indices` / `masked_m` 与 alignment 一致
- \(W_{13}\) 拼 \(N\) 后同一索引喂两次 GEMM
- SwiGLU 在 FP32 乘完再 round；FP8 再量化
- finalize 按 token 做 FP32 顺序加权
- CUDA graph 前完成对应 dtype 的 workspace 与 JIT

Serving 入口是 `grouped_gemm_nt_f8f8bf16_contig` 与 `grouped_gemm_nt_bf16_contig`（及对应 masked 符号）。permute / SwiGLU / finalize 在核外。Mega MoE 见 [04](04_mega_moe_and_paged_mqa.md)。

## 附录：源码摘录

### A. 类型与 persistent 领取

[`types.hpp`](src/types_hpp.md)：

```cpp
enum class GemmType {
    Normal                              = 0,
    MGroupedContiguous                  = 1,
    MGroupedMasked                      = 2,
    KGroupedContiguous                  = 3,
    Batched                             = 4,
    MGroupedContiguousWithPsumLayout    = 5,
};
enum class KernelType {
    Kernel1D1D = 0,
    Kernel1D2D = 1,
    KernelNoSF = 2
};
```

scheduler 领取下一 tile，contiguous multicast 要求 peer expert id 相同：

```cpp
const auto next_block_idx = (++ current_iter) * kNumSMs + blockIdx.x;
// ...
const auto group_idx = grouped_layout[m_block_idx * BLOCK_M];
const auto peer_group_idx = grouped_layout[(m_block_idx ^ 1) * BLOCK_M];
return group_idx == peer_group_idx;
```

[`gemm.cuh:L171-L172`](src/gemm_cuh.md#__codelineno-0-171)、[`gemm.cuh:L273-L275`](src/gemm_cuh.md#__codelineno-0-273)

### B. FP8 1D2D 生产者与 scale 提升

TMA 按组坐标装 A/B，`arrive_and_expect_tx` 含 SFA：

```cpp
tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, __nv_fp8_e4m3, kIsBatchedMM>(
    &tensor_map_a, &full_barrier, smem_a[stage_idx], k_idx,
    scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx),
    num_tma_multicast_a, batch_idx);
full_barrier.arrive_and_expect_tx(
    SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE);
```

[`sm90_fp8_gemm_1d2d.cuh:L194-L205`](src/sm90_fp8_gemm_1d2d_cuh.md#__codelineno-0-194)

math 侧在 `is_computation_valid` 为真时发 WGMMA，再用 `scale_a * scale_b` 提升。FP8 WGMMA 为 \(64\times N\times 32\) `F32E4M3E4M3`；BF16 为 \(64\times N\times 16\) `F32BF16BF16`。NoSF 无 scale 乘。

### C. SGLang 两次 contiguous GEMM

```python
deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
    (hidden_states, hidden_states_scale), w13_weight_fp8, gateup_output, m_indices, ...)
silu_and_mul_contig_post_quant(...)
deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
    (down_input_fp8, down_input_scale), w2_weight_fp8, down_output, m_indices, ...)
```

[deep_gemm.py L206-L293](https://github.com/sgl-project/sglang/blob/62c505a/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py#L206-L293)

finalize 按 token 扫 \(k_{\mathrm{top}}\)，FP32 累加后一次 store，见 [kernels.py L681-L719](https://github.com/sgl-project/sglang/blob/62c505a/python/sglang/srt/layers/moe/ep_moe/kernels.py#L681-L719)。
