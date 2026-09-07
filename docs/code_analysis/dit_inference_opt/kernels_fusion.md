---
tags:
  - Diffusion Model
  - CUDA
  - Triton
  - LLM Inference
---
# DiT 推理优化：Kernels & Fusion

**源码 pin**: `refs/codes/sglang` @ `db75dfe10ff7ef1f735d79178d2f2090683e3eb0`  
**运行时入口**: `python/sglang/kernels/ops/diffusion/`、`multimodal_gen/runtime/layers/layernorm.py`  
**官方现网库存**（比本 pin 新）: [Fused Kernels](https://docs.sglang.io/docs/sglang-diffusion/fused_kernels)  
本地快照: `refs/dit-inference-opt/crawl/results/pages/docs.sglang.io__docs__sglang-diffusion__fused_kernels__7a764b1f32ca925e.html`（SHA-256 `40158abec3de…`）

返回：[专题总览](overview.md)

本篇回答三件事：DiT 热路径上非 GEMM 时间花在哪；SGLang 用什么公式把哪些 launch 折成一个 kernel；什么时候这条路径是默认打开的，什么时候必须 `quality="high"`。

## 抓住重点

- 一个 DiT block 的主干是 GEMM + Attention；真正拖慢 e2e 的，经常是 AdaLN / QK-Norm / RoPE / 残差门控这类 **短 elementwise 链**：每次一次 launch、一次 HBM 来回。层数 \(L\)、步数 \(N_{\mathrm{step}}\) 把这笔开销乘上去。
- 融合有两类：**折 launch**（把 4～7 个 pointwise 合成 1 个）；**删中间 Tensor**（不物化 `[tokens, 4D]` 激活、不物化 temb `chunk` 出来的 GB 级 strided 切片）。后者往往比「再抠 5% GEMM」更明显。
- 接线分三层：公开 `register_kernel` 只覆盖少数 op；模型真正走的是 `layernorm.py` / `elementwise.py` 的共享 helper；**非 bit-exact** 的融合走 denoise 阶段的 site 协议，只在 `quality="high"` 挂上。
- 默认 `quality="lossless"` 保持参考链 bit-for-bit。评价融合必须关 [Feature Cache](feature_cache.md)：cache 跳过的 block 上，融合收益同时消失。

## 1. 这条轴在优化什么

典型 AdaLN-DiT block（符号：隐状态 \(x\in\mathbb{R}^{B\times S\times D}\)，调制向量 \(s,b,g\) 由 timestep embedding 投影得到）：

\[
\hat x = \mathrm{Norm}(x)\odot(1+s)+b,\qquad
y = x + g\odot F(\hat x)
\]

\(F\) 是 Attention 或 MLP。eager PyTorch 把 \(\mathrm{Norm}\)、乘加、门控拆成多次 kernel。SGLang 内部技能文档 `existing-fast-paths.md` 把可复用融合按家族列出来，并规定：**先证明现有路径因 shape/dtype/连续性没挂上，再提新 kernel**。

稀疏 Attention 算法不在本篇展开 → [overview 稀疏边界](overview.md#4-与稀疏-attention-的边界)。Attention **backend** 选择见 `attention_backends.mdx`。

## 2. 三条接入层

| 层 | 作用 | pin 证据 |
|----|------|----------|
| Registry | `KernelSpec` 注册 `diffusion.*`，`get_kernel` 按后端取实现 | `kernels/ops/diffusion/__init__.py` |
| 共享 runtime | 模型调 `apply_qk_norm_rope`、`LayerNormScaleShift`、`fuse_scale_shift_kernel` 等，内部再选 Triton / JIT / sgl_kernel | `runtime/layers/layernorm.py`、`elementwise.py`、`fused_scale_shift_gate.py` |
| Quality site | `mark_*_site` 默认关；denoise 在 batch 边界 `mount` / `unmount` | `fused_linear_gelu.py`、`fused_ln_modulate.py`、`fused_gate_rmsnorm.py`；`pipelines_core/stages/denoising.py` `_maybe_toggle_quality_fusions` |

公开 `__all__` 只有 `apply_group_norm_silu`、`residual_gate_add`、`fused_inplace_qknorm_rope`。同目录还有 `triton/`、`cutedsl/`、`flydsl/`、`ltx2_qknorm_split_rope.py`、`usp_relayout.py` 等，由模型或 helper **直接 import**，所以「registry 条目少」不等于「融合少」。

现网文档称 registry 上约 45 个 op、51 套实现（含 KDA / JIT / Triton / CuTe-DSL / FlyDSL / AOT）。**本 pin 的 `__init__.py` 只注册了 5 个 `diffusion.*` op**；其余以模块函数存在。读本仓笔记以 pin 为准，现网库存当前瞻索引。

## 3. 数值契约：lossless 与 high

多步去噪会把单步 rounding 差放大。pin 里 denoise 注释写明：cublasLt Linear+GELU 与 fused gate-RMSNorm **只在半精度 rounding 量级等价，不是 bit-exact**，因此：

- 默认 `quality="lossless"`：参考链，bit-for-bit。
- `quality="high"`：在 batch 边界对 transformer 做 all-or-nothing mount（任一 marked site 静态检查失败，整模该家族保持参考路径）。
- `quality` 进入 dynamic-batch signature，混质量流量不会同批。

pin 实际挂上的三族（`denoising.py`）：

1. Linear + tanh-GELU（cublasLt epilogue）  
2. Gate RMSNorm（复用 Z-Image Triton：`RMSNorm(x)*scale` 与 `residual + tanh(gate)*RMSNorm(x)`）  
3. LayerNorm + modulate 折成一次 affine `layer_norm(x, weight=(1+s), bias=b)`  

现网文档在 pin 之上又拆了 `extra-high`（只加 request-gated kernel，不含 cache/稀疏），并把 `high` 定义为 extra-high **再加** 模型自有的 high-only 策略（例如审计过的 Cache-DiT）。现网还写：ERNIE-Image 上一次「看起来无害」的 fp32 单 pass norm 融合把 50 步轨迹打到 18.83 dB PSNR，因此后来把部分 norm 路径改成 **对照 eager 链的 bit-exact 复刻**。该数字是官方文档自报，本仓未复现。

H3 对 `quality="high"` 另有 admission 约束，见 [Denoise Loop](../minimax_h3/08_denoise_loop_state_machine.md)。

## 4. AdaLN、调制、残差门控

### 4.1 通用 scale / shift

`triton/scale_shift.py` 的 `fuse_scale_shift_kernel` 实现 \(x\odot(1+s)+b\)。要求 `x` CUDA 且 contiguous；`s/b` 支持多种 broadcast。NPU 走 `npu_fallback`。

`LayerNormScaleShift` / `RMSNormScaleShift`（`layernorm.py`）把 **Norm + 上式** 收成一个 CustomOp。CuTe-DSL 路径 `fused_norm_scale_shift` / `fused_scale_residual_norm_scale_shift` 还要求 \(D\bmod 256=0\) 且 \(D\le 8192\)；不满足则 warning 后回退 PyTorch。

`ScaleResidual*` 再往前折一步：先做 \(x+g\odot u\)，再 Norm，再 scale/shift。这是「残差门控 + AdaLN」在同一段 fused dataflow 里。

### 4.2 Qwen-Image：select-0/1 门控

`fused_scale_shift_gate.py`：`FusedLayerNormScaleShiftGateSelect01` 用 Triton 一次完成 LN + 两套 \((s,b,g)\) 按 boolean index 选择。native 回退是 `torch.where` 选参数再 `F.layer_norm`。对应 Qwen 双流 / 双调制表。

### 4.3 LTX-2：`residual + gate * update`

`residual_gate_add.py` JIT CUDA，bit-exact 目标。约束：同 dtype（fp16/bf16/fp32）、同设备、contiguous、`update.shape == residual.shape`；`gate` 可全 shape 或行广播。`ltx_2.py` 的 `_ltx2_residual_gate_add` 在 runtime 异常时 **进程级关掉** fast path，避免反复失败。

### 4.4 MiniMax-H3：按行 index 的调制

H3 packed 序列每行有 `combined_indices`，不能先 `index_select` 再 pointwise。`triton/indexed_modulation.py`：

- `indexed_scale_shift_bf16_`：按行取 \((s,b)\)，并在 Triton 里 **显式复刻 eager BF16 rounding**（`_round_bf16_to_fp32`），再算 \((x\cdot(1+s))+b\)。  
- `indexed_gate_bf16_`：按行取 gate，做门控残差。

这不是「数学上等价即可」：换一种收缩会破坏 H3 的 BF16 边界。业务含义见 [H3 效率主线 §5.2](../minimax_h3/05_efficiency_in_sglang.md)。

### 4.5 Quality-gated：折 LN 与 Ideogram 门控 RMSNorm

`fused_ln_modulate.py`：用 `F.layer_norm(..., weight=1+s, bias=b)` 代替「无 affine 的 LN + 单独 modulate」。`1+s` 仍按 eager 对调制行 rounding，但 scale/shift 作用在未 round 的归一化值上，故 **非 bit-exact**。仅 `x.shape[0]==1` 且调制为 `[1,D]` 时允许。

`fused_gate_rmsnorm.py`：Ideogram-4 每 block 四条 RMSNorm 调制/门控链，复用 Z-Image `zimage_native_norm`。Z-Image 自身参考就是 native bf16，可无条件走 Triton；Ideogram 参考是 `F.rms_norm`（fp32 统计），融合后统计经 bf16，故只在 `quality="high"` 挂上。静态限制：bf16 权重、\(D\le 8192\)、Triton 可用。

## 5. QK-Norm 与 RoPE

Attention 前处理的典型 eager 链：Q/K RMSNorm（可能 per-head）→ 写回 → RoPE 再读。融合目标是 **原地、一次过**。

`apply_qk_norm`（`layernorm.py`）：CUDA、inplace、`q_eps==k_eps`、fp16/bf16、contiguous、`head_dim ∈ {64,128,256,512,1024}` 时走 `fused_inplace_qknorm`（`kernels/ops/layernorm/norm.py`）；否则各做一次 RMSNorm。调用面包括 flux / flux_2 / qwen_image / zimage / wanvideo / ltx_2 / hunyuanvideo。

`apply_qk_norm_rope`：再叠 RoPE。默认 `SGLANG_ENABLE_FUSED_QKNORM_ROPE=1`。额外约束：4D 同 shape 的 q/k、`head_dim ∈ {64,128,256}`、`rope_dim` 整除 per-thread 宽度、**不在 `torch.compile` 区域**。失败则 `apply_qk_norm` + FlashInfer inplace RoPE。JIT 实现见 `qknorm_rope.py`（`diffusion/qknorm_rope.cuh`）。

H3 直接调 `fused_inplace_qknorm_rope`：BF16、`head_dim=128`、`rope_dim=96`、NeoX、`round_norm_before_rope=True`（这是 H3 eager 数值合同的一部分）。compile 下故意拆回分开的 eager op。见 [H3 §5.1](../minimax_h3/05_efficiency_in_sglang.md)。

LTX-2 另有 `apply_ltx2_split_rotary_emb`（`triton/ltx2_rotary.py`）：`[B,S,H*D]` 上拆开的 cos/sin。profile 若仍是大段 split-RoPE PyTorch 链，应先查 shape/dtype 是否把这条路径打掉。

单独 GPT-J 风格 RoPE：`triton/rotary.py`；Q/K 优先 FlashInfer。

## 6. GEMM epilogue 与 packed 投影

### 6.1 Linear + tanh-GELU

MLP up-proj 常见 \( \mathrm{gelu}(xW^\top + b,\ \mathrm{approx=tanh}) \)。eager 会物化 `[tokens, 4D]` 再跑带宽受限的 GELU。`fused_linear_gelu_tanh` 走 `torch._addmm_activation(..., use_gelu=True)`，把 bias+GELU 吃进 cublasLt epilogue。cublasLt GELU 相对 `F.gelu(approximate="tanh")` 在 fp32 上 max abs ~5e-6，半精度下只剩 rounding-order 差。

静态拒绝：量化 linear、`skip_bias_add`、多 rank 的 `gather_output`、无 bias、非半精度。CPU offload 时权重可在 host，设备检查放在 **每次 forward**。site 标记在 flux / qwen_image / glm_image 等 FFN；`quality="high"` 才 mount。

量化 FLUX 另有 Nunchaku `_fused_gelu_mlp`：把 `fc1 GEMM + GELU + shift + re-quant + fc2.lora_down` 收进第二段 GEMM 之前，避免独立 GELU Tensor。这是 checkpoint 路径，不受上面 cublasLt site 协议管理。见 [Quantization](quantization.md)。

### 6.2 Packed QKV：少一次全局写

NVFP4 / Nunchaku FLUX：磁盘上 QKV packed，runtime 用 `MergedColumnParallelLinear`（`to_qkv` / `to_added_qkv`），而不是三次独立投影再 `cat`。SANA self-attn 同样 `to_qkv`；cross-attn 用 `to_kv`（K/V 共享步不变的 encoder 状态，Q 仍单独）。profile 若出现拆开的 `to_q/to_k/to_v`，先当 **已有 packed 路径没挂上**，不要先写新 GEMM fusion。

## 7. 删中间 Tensor 与纯数据搬移

这类 kernel **不改算术**，只少 copy / 少 `contiguous`。

| 路径 | 替换什么 | pin 文件 |
|------|----------|----------|
| `usp_merge_heads` | Ulysses 输出 `permute(2,1,0,3,4).contiguous()`，bit-exact | `usp_relayout.py`；`torch.compile` 内禁用 |
| `pack_qkv_destination_major` | 三次独立准备 Ulysses Q/K/V 交换 → 一次 destination-major pack + 一次 collective | `triton/ulysses_qkv.py`；H3 |
| `fused_pack_qkv` / `fused_scatter_to_padded` | masked varlen attention 的 gather/scatter | `triton/varlen_pack_pad.py` |
| `apply_group_norm_silu` | VAE / LTX upsampler 的 GroupNorm+SiLU | HunyuanVAE、`latent_upsampler.py`；无 env 开关，guard 失败回退 |

现网文档另举 Wan `temb_table_slices`：eager `(scale_shift_table + temb.float()).chunk(6, dim=2)` 在 704p/121f 会物化约 **8 GB fp32**，六个 strided slice 再各自 `.contiguous()`。融合一次写出六个 contiguous slice。**本 pin 无 `temb_table_slices` 符号**，当作现网已落地、pin 未跟上的例子，用来理解「删中间 Tensor」量级。

因果 Conv3d 的 cat+pad 见 `causal_conv3d_cat_pad.py`。

## 8. pin 上能对上的模型调用点

| 模型 | pin 内可核验的融合 |
|------|-------------------|
| FLUX / FLUX.2 | `apply_qk_norm(_rope)`；`quality=high` 的 linear+GELU、LN modulate；量化 packed QKV / Nunchaku GELU MLP |
| Qwen-Image | select-0/1 LN+gate；linear+GELU site |
| Z-Image | `zimage_rmsnorm_scale` / `tanh_residual`（native bf16，可默认） |
| Ideogram-4 | gate RMSNorm sites（`quality=high`） |
| LTX-2 | residual-gate CUDA、split RoPE、ada values Triton |
| HunyuanVideo | `apply_qk_norm`；VAE `apply_group_norm_silu` |
| SANA | packed `to_qkv` / `to_kv` |
| MiniMax-H3 | indexed AdaLN、fused qknorm+rope、`silu_and_mul`、packed Ulysses QKV、`usp_merge_heads` |

现网「Coverage by model」表更长（FLUX.2 QKV epilogue、LingBot MoE top-k、SANA-Video 线性注意力等）。那些 op **不必假装已经在本 pin 的 `__init__.py` 里**；要对齐现网需重新 pin。

## 9. 与 Graph / Cache / Quant / Parallel

- **BCG**：静段（Norm / RoPE / Residual / MLP）适合进 graph；动态 Attention 与通信保持 Eager → [Graph Runtime](graph_runtime.md)。pin 的 BCG CLI 与 torch.compile、Cache-DiT 互斥。现网文档额外规定：**不要把 request-gated DiT 融合和 BCG 一起开**——warmup 捕获的是 lossless 分支，replay 会绕过后来 mount 的 kernel。本 pin 的 `server_args` **尚未**搜到这条硬拒绝；落地以运行时日志为准，评测不要混用。  
- **Cache**：跳过的 block 上融合不执行 → 用未开 cache 的基线比 kernel。  
- **Quant**：Producer fusion（LN+modulate 直接写出 FP8/NVFP4）决定量化是否吃到 e2e；只报 GEMM TOP 不够 → [Quantization](quantization.md)。  
- **Parallel**：Ulysses pack / merge、varlen USP 与通信重叠是同一条热路径上的另一面 → [Parallelism](parallelism.md)、[H3 DiT Runtime](../minimax_h3/07_dit_runtime_and_collectives.md)。

## 10. 源码锚点

| 主题 | 路径 |
|------|------|
| 公开 registry | `kernels/ops/diffusion/__init__.py` |
| 家族清单与新模型 checklist | `multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/existing-fast-paths.md` |
| 共享 Norm / QK / AdaLN | `runtime/layers/layernorm.py` |
| Quality mount | `runtime/pipelines_core/stages/denoising.py` `_maybe_toggle_quality_fusions` |
| JIT QKNorm+RoPE | `kernels/ops/diffusion/qknorm_rope.py` |
| H3 indexed AdaLN | `kernels/ops/diffusion/triton/indexed_modulation.py` |
| 单测 / microbench | `test/registered/kernels/ops/diffusion/`、`test/registered/kernels/benchmark/diffusion/` |

## 相关阅读

- [专题总览](overview.md) · [Graph Runtime](graph_runtime.md) · [Quantization](quantization.md) · [Correctness](correctness.md) · [Feature Cache](feature_cache.md)  
- H3 把融合嵌进 packed row contract：[效率主线](../minimax_h3/05_efficiency_in_sglang.md)

## 附录：pin 与现网文档差在哪

- 现网 `fused_kernels.mdx` **不在** 本 pin 的 `docs/docs/sglang-diffusion/` 目录里；以 crawl 快照为准。  
- 现网质量档是 lossless / extra-high / high；pin 的 denoise mount 只分 lossless / high。  
- 现网库存含 KDA 实现、Wan temb slices、FLUX.2 token-cat FP8/NVFP4 producers、LingBot group-limited top-k 等，pin 树中部分文件不存在。  
- 读本专题时：机制与公式用 pin；「还有哪些 op」用现网库存对照，发现缺失就标 pin lag，不要把未 pin 的 kernel 写成已经可在本仓复现。
