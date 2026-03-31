---
tags:
  - Diffusion Model
  - Reinforcement Learning
  - DPO
---

# DeDPO: Debiased Direct Preference Optimization for Diffusion Models

- 论文：https://arxiv.org/abs/2602.06195
- 团队：Cornell University, Rutgers University, NYU

## 概述

DeDPO 解决 DPO 训练中人类偏好标签成本高昂的问题。提出半监督框架：用少量人类标注数据（25%）加上大量廉价的 AI 合成偏好标签（75%）进行训练，通过引入因果推断中的去偏估计技术（Doubly Robust Estimator），显式校正合成标注器的系统性偏差。DeDPO 在 SD1.5 和 SDXL 上用 25% 人类标注数据达到甚至超过 100% 人类标注 DPO 的性能（PickScore 21.91 vs 21.88），且对合成标签的噪声和数量变化保持鲁棒。

## 动机

DPO 需要大量人类偏好标签，但标注成本高。直接使用 AI 合成标签（如 VLM 预测）会引入系统性偏差——合成标注器的错误模式与随机噪声不同，而是有规律的。例如 Qwen-VLM 偏好语义连贯性和艺术风格约束，而人类偏好真实感纹理。

现有噪声鲁棒 DPO 方法（label smoothing、DRO）假设随机翻转噪声模型，不适用于这种系统性偏差。

## 方法

### DPO 重新解释为二分类

将 DPO 损失重写为二元交叉熵形式：

$$\mathcal{L}_{\text{DPO}}(\theta) = \mathbb{E}_{t,c,x^0_t,x^1_t}\left[\mathcal{L}(G_\theta(c, x^0_t, x^1_t), z)\right]$$

其中 $G_\theta$ 是基于去噪误差差的隐式 reward margin（通过 sigmoid 映射到 [0,1]），$z$ 是偏好标签，$\mathcal{L}$ 是二元交叉熵。

### DeDPO 去偏损失

设有标记数据 $y_l$（人类标签 $z_l$）和未标记数据 $y_u$（合成标签 $\hat{G}(y_u)$），DeDPO 损失为：

$$\mathcal{L}_{\text{DeDPO}}(\theta) = \mathbb{E}_{n_l+n_u}\mathcal{L}(G_\theta(y), \hat{G}(y)) + \mathbb{E}_{n_l}[\mathcal{L}(G_\theta(y_l), z_l) - \mathcal{L}(G_\theta(y_l), \hat{G}(y_l))]$$

三项含义：
- 第一项：在全部数据上用合成标签训练
- 第二项：在标记数据上用人类标签的监督信号
- 第三项：在标记数据上减去合成标签的信号（校正项）

等价理解（Proposition 2）：对未标记数据直接用合成标签；对标记数据，用合成标签加一个放大的校正项 $\frac{n_l+n_u}{n_l}(z - \hat{G}(y))$，将合成标签推向真实标签。

### 理论性质

**无偏性（Proposition 1）**：$\mathbb{E}[\mathcal{L}_{\text{DeDPO}}(\theta)] = \mathbb{E}[\mathcal{L}_{\text{DPO}}(\theta)]$，无论合成标签是否正确。当合成标签完美时，等效于在全部数据上训练；当完全错误时，退化为仅在标记数据上训练。

**收敛速率（Theorem 1）**：

$$\|\hat{\theta} - \theta^*\|^2_2 \leq O\left(\frac{1}{n_l + n_u}\right) + O\left(\|\hat{G} - G^*\|^4_4\right)$$

关键是第二项的 4 次方：合成标注器只需以 $(n_l+n_u)^{-1/4}$ 的慢速率收敛，模型参数就能以 $(n_l+n_u)^{-1}$ 的最优速率收敛。模型对合成标签的误差非常鲁棒。

重要前提：合成标注器必须在独立数据上训练（sample splitting），否则校正项会过拟合为 0。这意味着预训练 VLM 是更优选择，自训练（self-training）违反此假设。

### 合成标签来源

- **Self-training**：用当前模型的隐式 reward margin $G_{\hat{\theta}}$ 作为合成标签，用不同 timestep 作为数据增强避免确认偏差
- **VLM**：使用 Qwen2.5-VL-7B 评估图像对偏好（准确率约 80%）
- **CLIP**：用 CLIP 相似度评估（准确率约 50%，接近随机猜测）

## 实验

### 训练配置

- 数据集：FiFA-5K（从 Pick-a-Pic v2 过滤的高质量子集），25% 标记 + 75% 未标记
- SD1.5：1000 步，AdamW，学习率 $1 \times 10^{-7}$，batch size 128
- SDXL：100 步，AdaFactor，学习率 $2 \times 10^{-8}$
- 评估：PartiPrompt（1632 prompts），HPSv2 benchmark（3200 prompts）

### 主要结果（FiFA-5K）

| 方法 | PickScore | HPSv2 | Aesthetic |
|------|-----------|-------|----------|
| DPO + 25% | 21.76 | 27.76 | 5.38 |
| DPO + 100% | 21.88 | 27.79 | 5.38 |
| DPO + synthetic | 21.71 | 27.39 | 5.33 |
| DeDPO + synthetic | **21.91** | **27.80** | **5.43** |

DeDPO 用 25% 人类标签 + 75% Qwen 合成标签超过了 100% 人类标签的 DPO。

### 消融

- **合成标签来源**：Qwen > Self-training > CLIP，但 DeDPO 对所有来源均有改善。即使 CLIP 准确率仅 50%，DeDPO 仍能有效利用
- **标记/未标记比例**：固定 1.2K 标记数据，未标记从 3K 增至 98K，DPO 在 8K 后性能下降（噪声积累），DeDPO 保持稳定
- **训练集规模**：总量从 5K 增至 100K（保持 25:75 比例），DPO 在 20K 后饱和甚至下降，DeDPO 保持稳定
- **与其他鲁棒方法对比**：DeDPO 优于 Label Smoothing、IPO、DRO 等，因为这些方法假设随机噪声而非系统性偏差

### 合成标签偏差分析

Qwen-VLM 与人类标注的系统性差异：Qwen 优先考虑语义连贯性和艺术风格约束，对 "8K"、"hyper-detailed" 等真实感关键词不敏感；而人类强烈偏好照片级真实感，即使图像存在幻觉或遗漏细节。这种偏差是有规律的，不是随机噪声。

## 关键启示

- **合成偏好的误差是系统性偏差而非随机噪声**：VLM 和人类在审美判断上的分歧有规律可循，传统噪声鲁棒方法（label smoothing、DRO）无法有效纠正
- **因果推断的去偏技术可直接移植到 DPO**：少量高质量标注数据可以校正大量合成标签的偏差，且收敛速率对合成标签质量不敏感（4 次方项）
- **25% 人类标注 + 75% AI 标注可达到甚至超过 100% 人类标注**：大幅降低标注成本
- **合成数据越多不一定越好（对 DPO）**：vanilla DPO 随合成数据增加性能下降，而 DeDPO 保持稳定
