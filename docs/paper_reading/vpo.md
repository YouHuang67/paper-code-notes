---
tags:
  - Video Generation
  - Reinforcement Learning
---

# VPO: Aligning Text-to-Video Generation Models with Prompt Optimization

- 论文：https://arxiv.org/abs/2503.20491
- 代码：https://github.com/thu-coai/VPO
- 团队：Tsinghua University (CoAI Group + KEG), Zhipu AI

## 概述

VPO 从 prompt 优化角度对齐视频生成模型，而非直接优化生成模型本身。核心观察：视频生成模型训练时使用详细精心编写的描述，但推理时用户输入往往简短、模糊、结构差，这一差距严重影响生成质量。现有 prompt 优化方法（基于 LLM in-context learning）存在三个问题：可能引入有害内容、扭曲用户意图、不考虑对视频质量的实际影响。VPO 提出三原则——无害（Harmless）、准确（Accurate）、有用（Helpful），通过两阶段训练构建 prompt 优化器：(1) 基于原则的 SFT，使用 LLM 生成初始数据后经原则驱动的 critique-refine 流程提升质量；(2) 多反馈偏好优化，结合文本级反馈（LLM-as-judge 评估安全性和准确性）和视频级反馈（VisionReward 评估生成质量）进行 DPO。在 CogVideoX-5B 上，VPO 在人类评估中胜率超过原始查询 37.5%、超过官方 prompt 优化方法 14%，且性能优于 Diffusion DPO 并可与之叠加获得额外增益。

## 动机

训练-推理差距：视频生成模型训练数据中的文本描述详细、结构化，但用户输入如 "Dog lying grass tongue out" 这样简短模糊。这一差距是制约生成质量的关键因素。

现有 prompt 优化方法的局限：
1. **安全隐患**：LLM few-shot 重写不显式处理安全性，可能生成包含暴力等有害内容的 prompt
2. **意图偏移**：重写可能遗漏关键细节或引入偏差，偏离用户原始意图
3. **忽视视频质量**：仅追求语义丰富度，不考虑 prompt 对实际生成视频质量的影响
4. **过度拒绝**：LLM 可能拒绝处理含敏感关键词的查询，即使查询本身无害

## 方法

### Stage 1: 基于原则的 SFT

**数据构建**：
- 从 VidProM 数据集筛选 ~18k 通用查询 + 2k 安全相关查询（10k 用于 SFT，10k 用于 DPO）
- 用 GPT-4o in-context learning 生成初始优化 prompt
- 原则驱动的 critique-refine：LLM-as-judge 根据三原则评估每个 (query, prompt) 对，识别有害内容、遗漏细节、描述模糊等问题，生成 critique $c$，再据此生成修正版 $p_{\text{refined}}$
- 最终 SFT 数据：无问题的保留原始 prompt，有问题的使用修正版

**训练**：标准交叉熵损失 SFT。

### Stage 2: 多反馈偏好优化

从 SFT 模型为每个查询 $x$ 采样 $K$ 个 prompt，通过两种反馈构造偏好对：

**文本级反馈**：LLM-as-judge 检查每个 prompt 是否违反三原则。如有问题，生成 critique 并修正为 $p_j^{\text{refined}}$，构造偏好对 $(x, p_j \prec p_j^{\text{refined}})$，组成 $\mathcal{D}_{\text{text}}$。

**视频级反馈**：对通过文本级检查的 prompt，用目标视频生成模型生成视频，VisionReward 评分。按分数排序构造偏好对 $(x, p_m \prec p_{m+1})$（$r_m < r_{m+1}$），组成 $\mathcal{D}_{\text{video}}$。

**训练**：合并两种偏好数据 $\mathcal{D}_{\text{dpo}} = \mathcal{D}_{\text{text}} \cup \mathcal{D}_{\text{video}}$，标准 DPO 损失。

### 与 Diffusion DPO 的关系

VPO 和 Diffusion DPO 是正交的对齐方法：
- VPO 优化 prompt 优化器（LLM），不修改视频生成模型
- Diffusion DPO 直接优化视频生成模型的权重
- 两者可叠加使用获得额外增益

## 实验

### 训练配置

- Prompt 优化器骨干：GLM-4（SFT + DPO）
- 目标模型：CogVideoX-2B/5B、Open-Sora 1.2
- 视频级 reward：VisionReward
- 数据：10k SFT 对 + 10k DPO 对

### 主要结果（CogVideoX-5B）

| 方法 | MonetBench Overall | VBench Human Action | VBench Multiple Objects |
|------|-------------------|--------------------|-----------------------|
| Original Query | 3.77 | 88.00 | 45.67 |
| GLM-4 Few-Shot | 3.98 | 98.40 | 72.38 |
| GPT-4o Few-Shot | 4.03 | 99.20 | 72.21 |
| VPO-SFT | 4.01 | 97.20 | 73.70 |
| VPO w/o TL FDBK | 4.12 | 97.60 | 72.99 |
| **VPO** | **4.15** | **99.60** | **75.73** |

VPO 在所有指标上超过基线。移除文本级反馈（w/o TL FDBK）导致性能下降，说明安全和准确性优化对整体质量也有贡献。

### 文本级对齐

| 方法 | Aligned ↑ | Unsafe ↓ | Imprecise ↓ | Refusal ↓ |
|------|-----------|----------|-------------|-----------|
| GLM-4 Few-Shot | 83.4 | 5.4 | 10.0 | 1.2 |
| GPT-4o Few-Shot | 86.4 | 2.4 | 8.6 | 2.6 |
| VPO | **94.8** | **0.4** | **4.8** | **0.0** |

VPO 的对齐率从 83.4% 提升至 94.8%，不安全率从 5.4% 降至 0.4%，且完全消除了拒绝问题。

### 与 Diffusion DPO 对比

使用 VisionReward pairwise 评估：VPO 胜率 > Diffusion DPO。两者结合（Diffusion DPO + VPO）获得最高性能，证明正交性。

### 跨模型泛化

在 CogVideoX-2B 上训练的 VPO 直接应用于 Open-Sora 1.2，仍能获得显著提升（VBench Human Action: 88.80 → 97.00，Multiple Objects: 55.99 → 67.88），说明 prompt 优化器可跨模型迁移。

## 关键启示

- **Prompt 优化是视频生成对齐的高效正交维度**：不修改生成模型本身，通过优化输入 prompt 即可显著提升质量，且可与 Diffusion DPO 等方法叠加
- **文本级和视频级反馈缺一不可**：仅用视频级 reward 会损害安全性（video-only DPO 安全率低于 SFT），文本级反馈确保安全和准确
- **Critique-Refine 是高效的 SFT 数据提升策略**：LLM-as-judge 根据预定义原则自动识别问题并修正，比纯 LLM 生成质量更高
- **Prompt 优化器具有跨模型迁移能力**：在一个模型上训练的优化器可直接用于其他模型，降低适配成本
- **训练-推理文本差距是被低估的性能瓶颈**：简单的 prompt 重写就能带来巨大提升（原始查询 vs GLM-4 Few-Shot），VPO 在此基础上进一步改进
