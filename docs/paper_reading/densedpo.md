---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
  - DPO
---

# DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models

- 论文：https://arxiv.org/abs/2506.03517
- 代码：https://snap-research.github.io/DenseDPO/
- 团队：Snap Research, University of Toronto, Vector Institute
- 发表：NeurIPS 2025

## 概述

DenseDPO 针对视频 DPO 中的「运动偏见」问题提出解决方案。传统 VanillaDPO 从独立噪声生成视频对，标注者倾向于选择低运动但无伪影的视频，导致模型生成的视频越来越静态。DenseDPO 做了三个改进：(1) 通过对真实视频加噪再去噪（guided sampling）生成结构对齐的视频对，消除运动偏见；(2) 在时间片段级别标注偏好（1秒粒度），提供更密集更精确的学习信号；(3) 利用现成 VLM（GPT-o3）在短片段上自动标注偏好，效果接近人工标注。用仅 1/3 的标注数据量，DenseDPO 在保持视觉质量和文本对齐的同时，显著提升了 dynamic degree（从 VanillaDPO 的 80.25% 恢复到 85.38%，接近预训练模型的 84.16%）。

## 背景与动机

### 视频 DPO 的运动偏见

传统 VanillaDPO 流程：对同一 prompt 用不同随机种子生成两个视频 → 人工标注哪个更好 → DPO 训练。

问题：从独立噪声生成的两个视频运动模式差异很大。视频模型擅长生成高质量的慢动作视频，但动态场景常有伪影（如肢体变形、闪烁）。标注者自然倾向于选择无伪影的静态视频，导致偏好数据系统性地偏向低运动内容。DPO 训练进一步强化这一偏见。

实验验证：VanillaDPO 训练后，dynamic degree 从预训练的 84.16% 降至 80.25%（VideoJAM-bench），同样的问题在此前的 VideoDPO（使用 VideoCrafter-v2）和 VisionReward（使用 CogVideoX）的实验中也被观察到。

## 方法

### StructuralDPO：结构对齐的视频对生成

核心思想：受 SDEdit 启发，对真实视频部分加噪再去噪来生成视频对，而非从纯噪声生成。

具体做法：给定真实视频 $x$ 和引导强度 $\eta \in [0, 1]$，在第 $n = \text{round}(\eta \cdot N)$ 步注入噪声：

$$x^0_n = (1 - \eta)x + \eta\epsilon^0, \quad x^1_n = (1 - \eta)x + \eta\epsilon^1$$

然后从第 $n$ 步去噪到第 1 步生成视频对。

效果：
- 因为扩散模型的早期步骤控制全局运动，结果视频对共享高层语义和运动轨迹，只在局部细节上不同
- 消除运动偏见：标注者不再需要在「质量 vs 动态性」之间取舍
- 降低数据生成成本：只需更少的去噪步数
- 实验中 $\eta$ 在 [0.65, 0.8] 范围内随机采样

但有代价：视频对多样性降低，导致视觉质量和文本对齐不如 VanillaDPO。

### 理论分析：StructuralDPO 的学习信号退化

论文用数学推导解释了 StructuralDPO 性能不佳的原因。核心论点：

因为 guided sampling 生成的 winning 和 losing 视频共享大量相同像素（对应真实数据分布的区域），DPO 损失的梯度由 losing 样本在这些共享区域的损失主导，导致模型在正确区域「反学习」（unlearn）真实数据分布。

DPO 梯度的关键项是 $\nabla_\theta(\Delta^w_\theta - \Delta^l_\theta)$，其中 $\Delta^w_\theta$ 和 $\Delta^l_\theta$ 分别是 winning 和 losing 样本的去噪误差。由于 SFT 后模型对 winning 样本的重建误差更小，$\Delta^w_\theta[I_{\text{same}}] < \Delta^l_\theta[I_{\text{same}}]$ 在共享像素区域成立，导致 losing 样本的梯度贡献主导，模型在好的区域反向优化。

使用 Flux-dev 模型的实验验证了这一现象。

### DenseDPO：片段级密集偏好标注

核心创新：将视频分割为短片段（默认 1 秒），对每个片段独立标注偏好。

数学形式化：给定两个视频 $(x^0, x^1)$ 和片段长度 $s$，分割为 $F = \lceil T/s \rceil$ 个片段，得到密集偏好标签 $l \in \{-1, 0, +1\}^F$（0 表示平局）。DenseDPO 目标：

$$\mathcal{L}(\theta) = -\mathbb{E}\left[\log \sigma\left(-\beta \sum_{f=1}^{F} l(x^0_f, x^1_f) \cdot (s(x^0, c, t, \theta)_f - s(x^1, c, t, \theta)_f)\right)\right]$$

关键优势：
- 超过 60% 的视频对在不同片段有不同偏好方向，传统全局标注会处理为 tie 或选择伪影少的一方
- DenseDPO 只在有明确差异的片段上优化，避免在模糊区域引入错误信号
- tie 标签（$l=0$）的帧直接跳过，不参与损失计算
- 10k 视频对中 80% 以上有至少 1 个非 tie 片段，大幅提高数据利用率

### VLM 自动标注

发现：现有 VLM 在评估长视频（5s）时准确率较低，但在评估短片段（1s）时表现良好。

方案：GPT-o3 Segment —— 将视频分割为 1 秒片段，分别送入 GPT-o3 判断哪个更好，通过多数投票聚合全局偏好。

准确率（短片段 / 长视频）：
- VisionReward（fine-tuned）：72.45% / 62.11%
- GPT-o3（zero-shot）：70.03% / 53.45%
- GPT-o3 Segment（聚合）：70.03% / 70.59%

GPT-o3 Segment 在长视频偏好预测上超过所有 fine-tuned 模型。

## 实验

### 训练配置

- 基础模型：DiT-based latent flow model，32 DiT blocks，MAGVIT-v2 autoencoder
- DPO 训练：$\beta=500$，LoRA rank 128，AdamW，全局 batch size 256，1000 步
- 学习率 $1 \times 10^{-5}$，前 250 步线性 warmup，gradient clipping 1.0
- 64 张 A100 GPU，约 16 小时

### 主要结果（VideoJAM-bench）

| 方法 | Aesthetic | Imaging Quality | Dynamic Degree | Text Alignment |
|------|-----------|----------------|----------------|----------------|
| Pre-trained | 54.65 | 55.85 | 84.16 | 0.770 |
| VanillaDPO | 57.25 | 60.38 | **80.25** | 0.867 |
| StructuralDPO | 56.38 | 59.78 | 84.69 | 0.843 |
| DenseDPO | 56.99 | 60.92 | **85.38** | 0.863 |

DenseDPO 在保持 VanillaDPO 级别的视觉质量和文本对齐的同时，dynamic degree 甚至超过预训练模型。

### 消融实验

- **片段粒度**：$s=1$ 和 $s=0.5$ 效果接近，$s=2$ 明显下降。0.5 秒太短不足以评估时间质量
- **标签数量**：2x 标签效果最好，0.5x 标签仍优于翻转 40% 标签，说明标签质量比数量更重要
- **VLM 标签数量**：从 10k 到 55k GPT 标签，各指标持续提升，方法可扩展
- **引导视频选择**：不同的 ground-truth 视频集得到类似结果，方法对数据选择鲁棒
- **多数投票 vs 全局标注**：聚合片段标签得到的全局偏好与直接全局标注效果接近，证明 DenseDPO 的提升来自片段级监督而非标注者偏差

### VLM 标注偏见分析

论文发现 VisionReward 存在运动偏见：给定原始动态视频和复制单帧构成的静态视频，VisionReward 在约 70% 的情况下偏好静态视频。而 VideoReward 因为简单平均各维度分数，约 80% 偏好原始动态视频。原因：VisionReward 通过逻辑回归学习人类偏好的各维度权重，继承了人类标注中的运动偏见。

## 关键启示

- **视频 DPO 中运动偏见是系统性问题**：从独立噪声生成视频对 → 标注者偏好静态视频 → DPO 训练强化静态偏见。这是从图像 DPO 迁移到视频 DPO 时忽略的关键差异
- **结构对齐 + 密集标注是组合解法**：guided sampling 消除运动偏见但降低多样性，密集片段标注从每对视频中提取更丰富的学习信号来补偿
- **时间维度的偏好不是一致的**：60%+ 的视频对在不同时间片段有不同偏好方向，全局二元标签丢失了这些信息
- **VLM 评估短视频可靠、长视频不可靠**：通过分割-评估-聚合策略可以用现成 VLM 替代人工标注，且效果随数据量增加持续提升
- **标签质量比数量更重要**：翻转 20% 标签造成的性能损失大于减少 50% 数据量
