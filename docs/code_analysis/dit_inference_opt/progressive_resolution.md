---
tags:
  - Diffusion Model
  - Video Generation
  - LLM Inference
---
# DiT 推理优化：Progressive Resolution

**文档**: `docs/docs/sglang-diffusion/progressive_resolution.mdx`  
**实现**: `runtime/pipelines_core/stages/progressive_resolution/`  
**论文线索**: Spectral Progressive Diffusion（文档引用 arXiv 2605.18736）

早期 denoise 步在更粗 latent 分辨率上跑，再谱上采样到全分辨率继续，从而砍掉早期步的二次 Attention 成本。属 **quality-tradeoff**。

## 1. 模式

| `progressive_mode` | 含义 |
|--------------------|------|
| `fullres` | 关闭，等价标准生成 |
| `dct_rewind` | 谱上采样 + scheduler rewind（推荐） |
| `dct` | 谱上采样，不 rewind |

参数：`--progressive-levels`（分辨率减半次数）、`--progressive-delta`（噪声主导容差 δ，越大粗分辨率步越多）。

## 2. 收益与条件

文档给出的 denoise 加速（各自固定测试配置，**不是**万能 e2e 数字）：FLUX.1 / FLUX.2 / Z-Image / Wan2.1 / Qwen-Image / Ideogram 约 1.5×–2.8× 量级。

半分辨率 token 数约为全分辨率的 1/4，早期步 Attention 成本近似降到约 6% 量级（文档表述）。

**重要**：建议 `--dit-cpu-offload false`。整模 CPU offload 每步固定 PCIe 成本，会冲淡分辨率带来的加速。

## 3. 与少步蒸馏的边界

Progressive 改的是 **轨迹上的空间分辨率日程**；few-step distillation 改的是 \(N_{\mathrm{step}}\)。二者正交，但验收都要看画质。少步模型上再叠加 progressive 的收益曲线需单独测。

## 4. 本轮缺口

- `spectral_ops.py` / rewind 与 scheduler 状态同步  
- δ 与 Bayes-optimal frequency-activation 准则的公式级对照  
- 各 pipeline adapter（flux / wan / qwen_image / …）差异
