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

返回：[专题总览](overview.md)

## 抓住重点

- 早期 denoise 步在更粗 latent 分辨率上跑，再 **谱上采样** 到全分辨率继续 → 砍早期步的二次 Attention 成本。
- 属 **quality-tradeoff**；推荐模式 `dct_rewind`（谱上采样 + scheduler rewind）。
- **必须**尽量让 DiT GPU-resident（`--dit-cpu-offload false`），否则每步固定 PCIe 成本冲淡加速。
- **不可**与 Ulysses/Ring SP、`torch.compile` 同开；与 Cache-DiT 组合标为 experimental。

## 1. 模式与参数

| `progressive_mode` | 含义 |
|--------------------|------|
| `fullres` | 关闭，等价标准生成 |
| `dct_rewind` | 谱上采样 + scheduler rewind（推荐） |
| `dct` | 谱上采样，不 rewind |

| 参数 | 默认 | 含义 |
|------|------|------|
| `--progressive-levels` | 1 | 分辨率减半次数；1 = 一次粗阶段（如 64²→128² latent） |
| `--progressive-delta` | 0.01 | 噪声主导容差 δ；越大粗分辨率步越多、越快 |

半分辨率 token 数约为全分辨率的 1/4；文档表述早期步 Attention 成本可降到约 6% 量级（相对全分辨率步）。

## 2. 收益与条件

文档给出的 denoise 加速（各自固定测试配置，**不是**万能 e2e 数字）量级约 1.5×–2.8×，覆盖 FLUX.1 / FLUX.2 / Z-Image / Wan2.1 / Qwen-Image / Ideogram 等；示例（A6000、`--dit-cpu-offload false`）：

- FLUX 类：`dct_rewind` L1 δ=0.05 可到约 1.6× denoise  
- FLUX.2-klein：δ=0.10 约 1.9×  
- Z-Image：文档示例可达约 2.3×  

（精确表以 pin 内 mdx 为准。）

**Tip（官方）**：加 `--dit-cpu-offload false`。整模 CPU offload 每步固定 PCIe 成本，与分辨率无关，会稀释加速 → 也见 [Offload](memory_offload.md)。

Z-Image 特殊：5-D latent `[B,C,1,H,W]` 需 squeeze/unsqueeze；阶段切换时重算 caption+image RoPE。

## 3. 组合约束

来自官方 Limitations 节：

| 组合 | 行为 |
|------|------|
| Progressive + `--ulysses-degree` / `--ring-degree` | RuntimeError（SP 不兼容） |
| Progressive + `--enable-torch-compile` | 不兼容（固定序列长 vs 分辨率切换） |
| Progressive + Cache-DiT | experimental：换分辨率会 refresh cache context，需自测 |
| Progressive + 整模 DiT CPU offload | 强烈建议关闭 offload |

与少步蒸馏正交：Progressive 改 **空间分辨率日程**；蒸馏改 \(N_{\mathrm{step}}\)。少步模型上再叠 progressive 的收益曲线需单独测。验收 → [Correctness](correctness.md#4-建议验收清单)。

## 3.1 阶段切换的状态

实现位于 `stages/progressive_resolution/`：denoise stage 根据当前噪声水平和 `progressive_delta` 选择粗分辨率，使用 DCT 频域上采样恢复 latent，再在 `dct_rewind` 模式同步 scheduler 状态。切换点会刷新与分辨率相关的 RoPE 或 cache context；Z-Image 还需处理 `[B,C,1,H,W]` latent 的 squeeze/unsqueeze。这里减少的是早期 token 数和 Attention 二次复杂度，Encoder/VAE 与最终输出尺寸保持原协议。

## 4. 源码 / 文档锚点

| 主题 | 路径 |
|------|------|
| 用户文档与测例表 | `docs/.../progressive_resolution.mdx` |
| Stage 实现 | `runtime/pipelines_core/stages/progressive_resolution/` |
| Sampling 字段 | `configs/sample/sampling_params.py`（`progressive_*`） |

## 相关阅读

- [专题总览](overview.md) · [Memory Offload](memory_offload.md) · [Feature Cache](feature_cache.md) · [Parallelism](parallelism.md) · [Correctness](correctness.md)

## 附录：仍可加深

- `spectral_ops.py` / rewind 与 scheduler 状态同步  
- δ 与 Bayes-optimal frequency-activation 准则的公式级对照  
- 各 pipeline adapter（flux / wan / qwen_image / …）差异
