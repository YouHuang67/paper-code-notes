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
- 融合路径分为参考等价路径和 quality-gated 路径；后者只在请求的数值合同允许时启用。实现名称与文件位置放在附录。
- 默认 `quality="lossless"` 保持参考链 bit-for-bit。评价融合必须关 [Feature Cache](feature_cache.md)：cache 跳过的 block 上，融合收益同时消失。

## 1. 这条轴在优化什么

典型 AdaLN-DiT block（符号：隐状态 \(x\in\mathbb{R}^{B\times S\times D}\)，调制向量 \(s,b,g\) 由 timestep embedding 投影得到）：

\[
\hat x = \mathrm{Norm}(x)\odot(1+s)+b,\qquad
y = x + g\odot F(\hat x)
\]

\(F\) 是 Attention 或 MLP。eager PyTorch 把 \(\mathrm{Norm}\)、乘加、门控拆成多次 kernel。SGLang 内部技能文档 `existing-fast-paths.md` 把可复用融合按家族列出来，并规定：**先证明现有路径因 shape/dtype/连续性没挂上，再提新 kernel**。

稀疏 Attention 算法不在本篇展开 → [overview 稀疏边界](overview.md#4-与稀疏-attention-的边界)。Attention **backend** 选择见 `attention_backends.mdx`。

## 2. 融合的适用域

令 \(\mathcal D\) 表示输入的 shape、dtype、layout 与设备组成的域。一条融合仅在 \(z\in\mathcal D\) 时替换参考计算。若其满足逐元素相同的舍入序列，可作为默认路径；若只满足误差界 \(\lVert\tilde f(z)-f(z)\rVert\le\epsilon\)，则只能在明确质量等级下启用。模型、共享算子与运行时选择层共同完成这个域判定；文件级接线见附录。

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

对 \(x\in\mathbb R^{B\times S\times D}\)、调制向量 \(s,b\in\mathbb R^{B\times 1\times D}\)，融合目标是一次遍历完成
\[
z_{bsd}=x_{bsd}(1+s_{bd})+b_{bd}.
\]
适用域要求布局可线性访问且 broadcast 规则明确；否则回退参考实现。

若 \(\mu,\sigma\) 是按 token 计算的均值和标准差，则 Norm+调制可写为
\[
z=\frac{x-\mu(x)}{\sqrt{\sigma^2(x)+\epsilon}}(1+s)+b.
\]
融合把归一化后的中间张量留在寄存器/共享内存中，减少一次 HBM 写回；维度对齐或设备条件不满足时保持分步实现。

残差门控进一步计算 \(z=\operatorname{Norm}(x+g\odot u)(1+s)+b\)，把残差、归一化和调制串成单一数据流。

### 4.2 Qwen-Image：select-0/1 门控

双流模型可令选择变量 \(q\in\{0,1\}\)，从两套参数中选取 \((s_q,b_q,g_q)\)，再执行同一 AdaLN 公式；融合避免先生成两套候选张量再用 \(q\) 选择。选择维度或 layout 不满足时回退。

### 4.3 LTX-2：残差门控更新

LTX 类更新的数学形式是 \(y=x+g\odot u\)。要保持 bit-exact，\(x,u,g\) 必须具有兼容 shape、dtype、设备和广播规则；运行时失败应回退并记录，不能静默改变公式。

### 4.4 MiniMax-H3：按行 index 的调制

H3 packed 序列每行都有调制索引；先取出调制表再做 pointwise 会额外物化中间量。

设 packed 序列第 \(i\) 行的调制索引为 \(j_i\)，则 \(z_i=x_i(1+s_{j_i})+b_{j_i}\)。若参考路径在中间操作执行 BF16 舍入 \(R_{bf16}\)，融合必须逐项复现同一舍入顺序；这属于数值合同而非可选优化。

这不是「数学上等价即可」：换一种收缩会破坏 H3 的 BF16 边界。业务含义见 [H3 效率主线 §5.2](../minimax_h3/05_efficiency_in_sglang.md)。

### 4.5 Quality-gated：折 LN 与 Ideogram 门控 RMSNorm

把无 affine 的 LN 与调制合并为 affine LN 等价于令 \(\gamma=1+s,\beta=b\)。由于 \(\gamma\) 作用在未独立舍入的归一化值上，该变换通常只有 \(\lVert\tilde f-f\rVert\le\epsilon\) 意义，必须归入 `quality=high`，并限制到已审计的 shape。

RMSNorm 门控链可写为 \(y=x+\tanh(g)\odot \operatorname{RMSNorm}(x)\odot s\)。若参考统计在 FP32、融合统计在 BF16，误差来源已超出 bit-exact 合同，只能在质量门控路径启用。

## 5. QK-Norm 与 RoPE

Attention 前处理的典型 eager 链：Q/K RMSNorm（可能 per-head）→ 写回 → RoPE 再读。融合目标是 **原地、一次过**。

对每个 head 的 \(q,k\in\mathbb R^d\)，QK-Norm+RoPE 计算
\[
q'=\operatorname{RoPE}\left(q/\sqrt{\operatorname{mean}(q^2)+\epsilon}\right),\quad
k'=\operatorname{RoPE}\left(k/\sqrt{\operatorname{mean}(k^2)+\epsilon}\right).
\]
原地融合要求 q/k layout、dtype、head_dim 和两侧 epsilon 兼容；否则拆成参考 RMSNorm 与 RoPE。

融合 kernel 还要求 q/k 的 4D shape 一致、RoPE 子维度可整除线程布局，且不位于 `torch.compile` 捕获区；任一条件不满足即回退，保证输出路径明确。

H3 的已审计域为 BF16、head dimension 128、RoPE dimension 96 和 NeoX 布局，并要求先按参考路径完成 norm 的舍入再应用 RoPE；这些是数值合同的一部分。compile 下故意拆回分开的 eager op。见 [H3 §5.1](../minimax_h3/05_efficiency_in_sglang.md)。

LTX-2 的另一条 RoPE 路径处理 \([B,S,H D]\) 布局上拆开的 cos/sin；profile 若仍是大段 split-RoPE PyTorch 链，应先查 shape/dtype 是否落在已审计域。单独 GPT-J 风格 RoPE 也有对应实现；Q/K 优先 FlashInfer。源码位置见附录。

## 6. GEMM epilogue 与 packed 投影

### 6.1 Linear + tanh-GELU

MLP up-proj 计算 \(y=\operatorname{GELU}_{tanh}(xW^\top+b)\)。epilogue fusion 直接在 GEMM 写回阶段计算 GELU，避免物化 \([tokens,4D]\) 中间量。它要求 bias、半精度和非量化线性层等条件满足；量化路径见 [Quantization](quantization.md)。

当线性层缺少 bias、采用量化表示、需要跨 rank 聚合或不满足半精度条件时，当前路径回退。其舍入顺序与参考 GEMM 不完全相同，故只在 `quality="high"` 的审计域启用。

量化 MLP 还可把 GEMM、GELU、重定标和重新量化串联，避免在两次 GEMM 间生成高精度中间量；其适用域由 checkpoint 家族决定，见 [Quantization](quantization.md)。

### 6.2 Packed QKV：少一次全局写

若 QKV 权重已按列打包，投影可写为一次 \([Q,K,V]=XW_{qkv}^\top\)，随后按视图切分；相比三次 GEMM 加 concat，它减少全局写和布局转换。Cross-Attention 中 encoder 的 K/V 可复用，Q 仍单独投影。是否能采用取决于权重布局与并行切分，见 [Parallelism](parallelism.md)。

## 7. 删中间 Tensor 与纯数据搬移

设中间张量大小为 \(m\) bytes，eager 链若产生 \(k\) 次完整 materialization，至少会增加 \(km\) 写和随后读取；布局融合直接将最终布局写出，使额外读写降为常数次。它不改变算术，却可能在长序列/高分辨率下成为主要收益来源。Ulysses 的 pack/merge 与变长 gather/scatter 同时决定通信布局，见 [Parallelism](parallelism.md)。

## 8. 与 Graph / Cache / Quant / Parallel

- **BCG**：静段（Norm / RoPE / Residual / MLP）适合进 graph；动态 Attention 与通信保持 Eager → [Graph Runtime](graph_runtime.md)。pin 的 BCG CLI 与 torch.compile、Cache-DiT 互斥。现网文档额外规定：**不要把 request-gated DiT 融合和 BCG 一起开**——warmup 捕获的是 lossless 分支，replay 会绕过后来 mount 的 kernel。本 pin 的 `server_args` **尚未**搜到这条硬拒绝；落地以运行时日志为准，评测不要混用。  
- **Cache**：跳过的 block 上融合不执行 → 用未开 cache 的基线比 kernel。  
- **Quant**：Producer fusion（LN+modulate 直接写出 FP8/NVFP4）决定量化是否吃到 e2e；只报 GEMM TOP 不够 → [Quantization](quantization.md)。  
- **Parallel**：Ulysses pack / merge、varlen USP 与通信重叠是同一条热路径上的另一面 → [Parallelism](parallelism.md)、[H3 DiT Runtime](../minimax_h3/07_dit_runtime_and_collectives.md)。

## 附录：源码锚点与覆盖

| 主题 | 路径 |
|------|------|
| 公开 registry | `kernels/ops/diffusion/__init__.py` |
| 家族清单与新模型 checklist | `multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/existing-fast-paths.md` |
| 共享 Norm / QK / AdaLN | `runtime/layers/layernorm.py` |
| Quality mount | `runtime/pipelines_core/stages/denoising.py` `_maybe_toggle_quality_fusions` |
| JIT QKNorm+RoPE | `kernels/ops/diffusion/qknorm_rope.py` |
| H3 indexed AdaLN | `kernels/ops/diffusion/triton/indexed_modulation.py` |
| 单测 / microbench | `test/registered/kernels/ops/diffusion/`、`test/registered/kernels/benchmark/diffusion/` |

| 模型族 | pin 内可核验的机制 |
|------|------|
| FLUX / FLUX.2 | QK-Norm/RoPE、quality-gated MLP/AdaLN、量化 QKV/MLP |
| Qwen-Image | 双调制选择、quality-gated MLP |
| Z-Image / Ideogram-4 | RMSNorm 门控；后者受质量门控 |
| LTX-2 / HunyuanVideo | 残差门控或 QK 前处理；VAE 融合 |
| SANA / MiniMax-H3 | packed QKV，或 indexed AdaLN 与 Ulysses 布局融合 |

数据搬移实现覆盖 Ulysses head merge、destination-major QKV pack、变长 gather/scatter 与 VAE GroupNorm+SiLU；现网还有本 pin 不含的 Wan temb slice 融合。覆盖表是实现索引，公式及适用域以正文为准。

## 相关阅读

- [专题总览](overview.md) · [Graph Runtime](graph_runtime.md) · [Quantization](quantization.md) · [Correctness](correctness.md) · [Feature Cache](feature_cache.md)  
- H3 把融合嵌进 packed row contract：[效率主线](../minimax_h3/05_efficiency_in_sglang.md)

## 附录：pin 与现网文档差在哪

- 现网 `fused_kernels.mdx` **不在** 本 pin 的 `docs/docs/sglang-diffusion/` 目录里；以 crawl 快照为准。  
- 现网质量档是 lossless / extra-high / high；pin 的 denoise mount 只分 lossless / high。  
- 现网库存含 KDA 实现、Wan temb slices、FLUX.2 token-cat FP8/NVFP4 producers、LingBot group-limited top-k 等，pin 树中部分文件不存在。  
- 读本专题时：机制与公式用 pin；「还有哪些 op」用现网库存对照，发现缺失就标 pin lag，不要把未 pin 的 kernel 写成已经可在本仓复现。
