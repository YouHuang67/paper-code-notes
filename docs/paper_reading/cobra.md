---
tags:
  - Sparse Attention
  - Diffusion Model
---

# Cobra: Efficient Line Art Colorization with Broader References

[arXiv 2504.12240](https://arxiv.org/abs/2504.12240) | [项目页](https://zhuang2002.github.io/Cobra/) | Tsinghua University / CUHK / Tencent ARC Lab

## 概述

漫画线稿上色需要大量参考图来保持角色颜色一致性，但现有方法（如 ColorFlow）限于 12 张参考图，且推理慢、内存大。Cobra 提出了一套工业级长上下文线稿上色框架，支持 200+ 参考图，核心是 **Causal Sparse DiT** 架构，通过三项改进解决参考图数量与效率的矛盾：

1. **Causal Sparse Attention + KV-Cache**：消除参考图之间的 pairwise 注意力，并将参考图处理改为因果单向（仅在 timestep 0 去噪一次），将复杂度从 $O(N^2)$ 降至 $O(N)$（相对于参考图数 N）
2. **Localized Reusable Position Encoding**：复用局部位置编码，支持任意数量参考图而不改变预训练 DiT 的 2D 位置编码
3. **Line Art Guider**：仅保留 Self-Attention 的 ControlNet 式控制网络，接收线稿和颜色提示

主要结果（Cobra-Bench，30 个漫画章节）：
- 12 参考图下：ColorFlow 1.03s/36.4GB → Cobra 0.31s/9.3GB，质量全面提升
- 64 参考图下：Causal Sparse Attention 比 Full Attention 快约 15×
- 用户研究：色彩 ID 一致性以 79.1% vs 20.9% 超过 ColorFlow

## 方法

### 问题：全量注意力的复杂度

Cobra 将线稿 latent（序列长 $S_l$）与 $N$ 张参考图 latent（每张序列长 $S_r$）拼接后做自注意力。全量注意力复杂度：

$$O\!\left(T \times (S_l^2 + 2N \cdot S_l \cdot S_r + N^2 \cdot S_r^2)\right)$$

其中 $T$ 为去噪步数。$N^2 \cdot S_r^2$ 项随参考图数量二次增长，是主要瓶颈。

### Causal Sparse DiT

**第一步：去除参考图间注意力（Sparse Attention）**

参考图的作用是向线稿提供颜色 ID 信息，参考图之间不需要相互交互。去除参考图 pairwise 计算后：

$$O\!\left(T \times (S_l^2 + 2N \cdot S_l \cdot S_r + N \cdot S_r^2)\right)$$

**第二步：因果注意力 + KV-Cache（Causal Sparse Attention）**

参考图是干净图像（clean latents），不需要随噪声 latent 做完整的 $T$ 步去噪。将双向注意力改为单向因果注意力：噪声 latent 可 attend 到参考图 latent，但参考图 latent 不 attend 到噪声 latent。因此参考图只需在 timestep 0 做一次去噪，其 KV 值缓存后在全程 $T$ 步推理中重复使用：

$$O\!\left(T \times (S_l^2 + N \cdot S_l \cdot S_r) + N \cdot S_r^2\right)$$

$N \cdot S_r^2$ 项从 $O(T)$ 降为 $O(1)$（只算一次），使总复杂度关于 $N$ 为线性。

消融对比（24 参考图，FP16）：

| 注意力类型 | 时间/步↓ | FLOPs | FID↓ |
|----------|---------|-------|------|
| Full Attention | 1.99s | 38.2T | – |
| Sparse Attention | 0.81s | 14.7T | 21.07 |
| Causal Sparse Attention | **0.35s** | **9.0T** | **20.98** |

### Localized Reusable Position Encoding

预训练 PixArt-Alpha 的 2D 位置编码支持宽高比 0.25–4.0，超过 8 张参考图水平/垂直拼接后会产生极端宽高比，导致生成质量下降。

Cobra 将线稿图分为四个空间 patch（左上、左下、右上、右下），每个 patch 单独检索 top-k 最相似的参考图（四组各自独立）。位置编码复用：参考图的局部位置编码与对应的 patch 区域相同，而非延伸全图。这样每张参考图在位置空间中仍处于"接近中心区域"，无论添加多少参考图都不会影响位置编码范围。

训练时从每组参考集中随机采样，总数固定为 3/6/12，增强模型对不同参考数量的适应能力。

### Line Art Guider

ControlNet 风格控制网络，接受线稿 latent $Z_L$ 和可选颜色提示 latent $Z_C$ 及提示掩码 $M$：

- 去除 Cross-Attention 层，仅保留 Self-Attention（减少参数，保持控制效果）
- **线稿风格增强**：混合两种不同风格线稿提取器的输出（随机比例混合），提升对多样线稿风格的鲁棒性
- **颜色提示采样策略**：约束颜色提示点内 RGB 方差 ≤ 0.01，避免提示点落在边缘交叉处造成训练歧义

训练目标：

$$\mathcal{L} = \mathbb{E}_{t,\epsilon}\left[\|\epsilon - D_{cs}(G(Z_L, Z_C, M, t), Z_R^{0:N-1}, Z_t, t)\|_2^2\right]$$

基于 PixArt-Alpha 预训练权重，对 Line Art Guider 和 Causal Sparse DiT 的 LoRA 权重进行微调（78K 步，lr=1e-5，batch=16，640×1024 分辨率）。

## 实验

### 与 baseline 对比（Cobra-Bench，24 参考图）

| 方法 | CLIP-IS↑ | FID↓ | PSNR↑ | SSIM↑ | AS↑ |
|------|---------|------|-------|-------|-----|
| MC-v2（无参考） | – | – | – | – | – |
| IP-Adapter | 0.828 | 76.01 | 8.11 | 0.556 | 4.511 |
| ColorFlow | 0.903 | 26.29 | 15.20 | 0.805 | 4.630 |
| **Cobra** | **0.918** | **20.98** | **16.08** | **0.814** | **4.641** |

### 效率对比（12 参考图）

| 方法 | FID↓ | 时间(s) | 内存(GB) |
|------|------|--------|---------|
| ColorFlow | 26.29 | 1.03 | 36.4 |
| Cobra | **21.86** | **0.31** | **9.3** |

Cobra 推理时间缩短 3.3×，内存减少 4×，质量更好。

### 参考图数量消融

| 参考图数 | CLIP-IS↑ | FID↓ | PSNR↑ |
|--------|---------|------|-------|
| 4 | 0.908 | 23.18 | 15.61 |
| 12 | 0.913 | 21.86 | 15.94 |
| 24 | 0.918 | 20.98 | 16.08 |
| 36 | 0.918 | 20.64 | 16.13 |

参考图增多始终有提升，验证了更广泛参考上下文的必要性。64 参考图时 Causal Sparse Attention 比 Full Attention 快约 15×。

## 关键启示

- **参考图间 pairwise 注意力是冗余的**：参考图作用是向目标提供色彩 ID，互相之间的交互没有必要，去除后 FLOPs 从 38.2T 降到 14.7T，质量不下降
- **因果注意力 + KV-Cache 将参考图开销从 $O(T)$ 降至 $O(1)$**：将参考图视为静态条件（类似 LLM 的 prefix），只需一次前向并缓存 KV，后续 $T$ 步推理直接复用，是效率跃升的核心
- **局部位置编码复用解决长上下文位置编码问题**：不需要扩展预训练模型的位置编码范围，通过分组检索 + 复用局部编码，200+ 参考图都能保持与目标区域的位置邻近性
- **颜色提示采样策略是训练细节**：避免提示点落在边缘交叉处，这种边界条件若不处理会引入训练歧义，方差约束是简单有效的过滤方式
