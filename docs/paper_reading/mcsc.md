---
tags:
  - Video Generation
  - Reinforcement Learning
  - DPO
  - GRPO
  - Reward Model
---

# McSc: Motion-Corrective Preference Alignment for Video Generation

- 论文：https://arxiv.org/abs/2511.22974
- 团队：Tongyi Lab, Alibaba Group

## 概述

McSc 提出三阶段框架解决视频生成偏好对齐中的运动偏差问题。核心观察：视频质量与运动动态性之间存在负相关（-0.1 到 0.3），直接 DPO 对齐会导致模型偏向低运动内容（reward hacking）。方法包括：(1) Self-Critic Dimensional Reasoning (ScDR)，用 GRPO 训练 Reward Model 对单维度做推理式评分；(2) Hierarchical Comparative Reasoning (HCR)，在维度推理基础上做整体配对比较；(3) Motion-Corrective DPO (McDPO)，通过运动感知的动态权重调整 DPO 损失。在 VideoCrafter2 上 VBench 从 80.44 提升到 83.97，同时 Dynamic Degree 从 42.50 提升到 43.26（标准 DPO 下降到 32.64）。

## 动机

- 视频偏好的多维冲突：运动动态性与视觉质量呈负相关
- 现有方法（CLIP score、美学评分、VLM 线性投影）无法捕捉人类判断逻辑
- 直接偏好对齐导致 reward hacking：模型倾向生成静态但画质高的视频
- 缺乏推理过程建模——现有方法只预测标量分数或配对排名

## 方法

### Stage 1: Self-Critic Dimensional Reasoning (ScDR)

冷启动阶段，训练 RM（Qwen2-VL-7B-Instruct）对单维度做推理评分。

**偏好数据分解**：将多标注数据集拆分为单维度实例 $D = \{(v_i, y_{i,d})\}$，共 679.5k 维度标签来自 121.1k 偏好视频（VisionReward、MJ-Video、Lift-HRA、VideoDPO、Charades）。

**结构化推理**：对每个 (video $v$, dimension query $q_d$)，RM 输出 `<think>reasoning</think><answer>prediction</answer>`。

**三重 Reward**：

$$r = r_{\text{format}} + r_{\text{acc}} + r_{\text{sc}}$$

- $r_{\text{format}}$：输出格式是否正确（0/1）
- $r_{\text{acc}}$：预测是否匹配 ground truth（0/1）
- $r_{\text{sc}}$：Self-Critic 分数——用当前 RM 作为 critic 验证推理是否逻辑支持预测（0/1）

训练配置：GRPO，rollout $G=8$，$\beta_1=0.07$，batch size 16，LR $1 \times 10^{-6}$，2 epochs

### Stage 2: Hierarchical Comparative Reasoning (HCR)

在 ScDR 基础上实现整体视频配对比较。

**层次化评估**：给定视频对 $(v_a, v_b)$，RM 生成结构化输出——对每个维度 $d$ 分别推理评分，最后汇总得出整体偏好判断。

**三重层次 Reward**：

$$r_{\text{total}} = r_{\text{hier}} + r_{\text{dim}} + r_{\text{com}}$$

- $r_{\text{hier}}$：是否包含多样的维度标签格式（0/1）
- $r_{\text{dim}}$：各维度格式正确性的平均值 $\frac{1}{D}\sum_d r_{dim,d}$
- $r_{\text{com}}$：最终配对预测是否正确（0/1）

### Stage 3: Motion-Corrective DPO (McDPO)

**偏好对构建**：对文本 prompt $x$ 采样 $N$ 个候选视频，用训练好的 RM 评分，选最高/最低分作为正/负样本。

**运动校正权重**：

$$w_w = 0.5 + \sigma[(s_{om}^w - s_{om}^l) + (s_{cm}^w - s_{cm}^l)]$$
$$w_l = 2.0 - w_w$$

其中 $s_{om}$、$s_{cm}$ 分别是 RM 给出的物体运动和镜头运动分数。

**修正后 DPO 损失**：

$$\mathcal{L}_{\text{McDPO}} = -\mathbb{E}\left[\log \sigma\left(-\beta_2(w_w \cdot r(x, v_w) - w_l \cdot r(x, v_l))\right)\right]$$

$\beta_2 = 2500$。当正样本运动低时降低其权重，当负样本运动高时降低其惩罚——动态纠正运动偏差。

训练配置：10k 视频对，AdamW，LR $6 \times 10^{-6}$，8×A100-80G，20 epochs

## 实验

### RM 偏好预测（MonetBench + GenAI-Bench）

| 方法 | MonetBench tau | MonetBench diff | GenAI-Bench tau | GenAI-Bench diff |
|------|:---:|:---:|:---:|:---:|
| UnifiedReward | - | - | 60.7% | 77.2% |
| ScHR (full) | **72.0%** | **81.5%** | **62.9%** | **82.7%** |

消融：去掉 ScDR -1.3% tau，去掉 HCR -2.7% tau，去掉推理 -3.9% tau

### 视频生成对齐（VBench）

| 方法 | VBench Total Quality | Visual Quality |
|------|:---:|:---:|
| VideoCrafter2 基线 | 80.44 | 82.20 |
| VideoDPO | 81.93 | - |
| Flow-DPO | 81.35 | - |
| **McSc** | **83.97** | **85.03** |

Wan2.1-T2V-1.3B：83.96→85.71 (+1.75)

### 运动指标保持（关键对比）

| 方法 | Dynamic Degree | Object Motion | Camera Motion |
|------|:---:|:---:|:---:|
| 基线 | 42.50 | 97.73 | 96.85 |
| VideoDPO | **32.64** | 92.18 | 95.69 |
| **McSc** | **43.26** | **97.88** | **97.24** |

VideoDPO 导致运动崩溃（DD -9.86），McSc 反而提升运动质量（DD +0.76）

### 人类评估（100 组视频）

- McSc vs SFT：运动质量胜 68%，视觉质量胜 72%
- McSc vs VideoDPO：运动质量胜 62%，总体胜 59%

## 关键启示

- **运动与质量的负相关是偏好对齐的核心陷阱**：简单 DPO 放大此偏差，模型学会"不动就是好"——McDPO 通过动态权重纠正
- **层次化推理式 RM 优于直接评分**：两阶段训练（单维度→整体比较）捕捉维度间冲突，Self-Critic 确保推理一致性
- **运动校正权重的设计直觉**：正样本运动低说明模型走捷径、应降权，负样本运动高说明可能只是"动得多"而非质量差、应减惩罚
