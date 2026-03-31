---
tags:
  - Video Generation
  - Reinforcement Learning
---

# REACT: Thinking with Frames — Generative Video Distortion Evaluation via Frame Reward Model

- 论文：https://arxiv.org/abs/2601.04033
- 团队：USTC, Kling Team (Kuaishou), Institute of Software CAS

## 概述

REACT 是一个帧级别的 reward 模型，专门评估生成视频中的结构性失真（structural distortion），如肢体变形、多余肢体、肢体不完整、躯干变形、面部变形、非动物物体坍塌、运动模糊和网格穿透。现有视频 reward 模型（VideoReward、VideoScore2）侧重视觉质量、运动质量和文本对齐，但对结构性失真评估不足——即使视频有严重的肢体变形也可能给高分。REACT 基于 Qwen2.5-VL-7B，通过 CoT 推理对每帧输出点数评分和归因标签。训练分两阶段：(1) 带 masked loss 的 SFT 注入领域知识；(2) GRPO 强化推理，使用创新的 pairwise reward（基于 BTT loss）对齐人类偏好。推理时采用动态帧采样机制，自适应聚焦最可能出现失真的帧。在 REACT-Bench 上，偏好对齐准确率 0.813（w/o tie），远超 UnifiedReward 的 0.701；失真识别 F1 0.845，远超 MagicAccessor 的 0.554。

## 动机

### 帧级别 vs 视频级别

结构性失真适合帧级评估：(1) 失真在空间上局部化，可在单帧中检测；(2) 现有视频 reward 模型采样率低（2 fps），难以捕捉帧级伪影；(3) 帧级标注效率高，同一视频可产生多个训练样本。

### 帧级别 vs 图像评估器

图像生成中的伪影清晰锐利，但视频生成中的失真表现为由时间不一致和运动动态导致的模糊碎片区域，存在领域差距。图像评估器（MagicAccessor、Q-Insight）直接用于视频帧效果差。

## 方法

### 结构性失真分类体系

两大类八个子类：

**异常物体外观**：
- 肢体变形、多余肢体、肢体不完整（仅针对四肢）
- 躯干变形、面部变形
- 非动物物体坍塌与变形
- 运动模糊

**异常物体交互**：
- 网格穿透（物体边界不合理地交叉或融合）

### 数据构建

**视频采集与帧对构建**：从社交媒体收集复杂运动真实视频 → 用 Kling、HaiLuo、Seedream、Pika、Sora、Luma 生成视频 → 同一 prompt 不同模型生成的视频取相同时间戳帧配对 → 15k+ 帧对（约 30k 帧）。

**高效 CoT 合成**：标注者仅需对失真区域画 bounding box（远比写文本描述简单），然后用 Gemini 2.5 Pro 根据标注帧和失真区域模拟推理过程，生成 CoT 数据，按标签和区域准确率过滤，得到 6k 高质量 CoT 实例。

**伪点数评分**：无真实点数评分，根据失真标签数量随机分配：0 个标签 → [4.0, 5.0]，1 个 → [3.0, 4.0]，2 个 → [2.0, 3.0]，3+ → [1.0, 2.0]。保持排序一致性的同时增加评分多样性。

### 两阶段训练

**Stage 1: Masked SFT**

第一个 epoch：完整 CoT 数据训练（推理过程 + 标签 + 分数均参与 loss），教模型推理失真模式。第二个 epoch：masked SFT，仅对最终标签和分数计算 loss，推理轨迹不参与 loss。

目的：平衡领域知识注入和推理多样性。完整 SFT 过度训练会让模型死记 CoT 模式，降低 GRPO 阶段的推理轨迹多样性。

**Stage 2: GRPO**

给定帧对 $\{f^A, f^B\}$，对每帧分别生成 $G$ 个 rollout。三个 reward 组件：

- **Format Reward**：输出结构正确（`<think>` + `<answer>` 标签）
- **Attribution Accuracy Reward**：$R_{\text{attr}} = 0.6 \cdot a_{\text{right}} - 0.2 \cdot (a_{\text{wrong}} + a_{\text{missing}})$
- **Preference Reward（基于 BTT）**：用 BTT loss 计算 rollout 对之间的偏好概率，与 ground truth 偏好标签对齐

$$R_{\text{pref}}(o_i^A, o_i^B) = \mathbb{1}(f^A \succ f^B)\log P(o_i^A \succ o_i^B) + \mathbb{1}(f^A \prec f^B)\log P(o_i^A \prec o_i^B) + \mathbb{1}(f^A = f^B)\log P(o_i^A = o_i^B)$$

创新点：虽然没有真实点数评分，但通过 pairwise reward 让模型的点数评分输出对齐人类偏好排序。

### 动态帧采样

推理时两阶段采样：
1. 以 1/2 fps 均匀采样，用 REACT 评分
2. 根据分数分布决定第二阶段策略：
   - 所有帧高分 → 在第一阶段帧之间稀疏采样
   - 存在低分帧 → 在低分帧邻域 1/4 fps 密集采样
   - 混合情况 → 优先在低分帧邻域采样

## 实验

### 训练配置

- 基础模型：Qwen2.5-VL-7B
- SFT：lr $5 \times 10^{-4}$，LoRA rank 32，batch size 64，2 epochs
- GRPO：lr $1 \times 10^{-6}$，rollout $G=8$，300 步，rollout batch 256

### 主要结果

**人类偏好对齐（REACT-Video, 500 对）**

| 模型 | Acc w/ Tie | Acc w/o Tie |
|------|-----------|-----------|
| VideoScore2 | 0.342 | 0.521 |
| VideoReward | 0.415 | 0.551 |
| UnifiedReward | 0.416 | 0.701 |
| VisualQuality-R1 | 0.376 | 0.610 |
| Gemini-2.5-Pro | 0.370 | 0.534 |
| **REACT** | **0.610** | **0.813** |

**失真识别（REACT-Frame, 2.1k 帧）**

| 模型 | Distorted F1 | Normal F1 |
|------|-------------|-----------|
| GPT-o3 | 0.641 | 0.379 |
| MagicAccessor | 0.554 | 0.285 |
| Q-Insight | 0.334 | 0.300 |
| Qwen2.5-VL-7B | 0.162 | 0.292 |
| **REACT** | **0.845** | **0.671** |

REACT 在失真帧 F1 上大幅领先，且是唯一在正常帧 F1 上也远超基线的模型。

## 关键启示

- **结构性失真是现有视频 reward 模型的盲区**：VideoReward 等侧重美学和运动平滑度，对肢体变形等结构问题不敏感。REACT 作为互补组件填补这一空白
- **帧级评估比视频级更适合结构性失真**：失真空间局部化、现有模型采样率低易遗漏、帧级标注效率高
- **Masked SFT 平衡知识注入和推理多样性**：第一 epoch 完整训练教推理模式，第二 epoch mask 推理仅训练输出，避免 GRPO 前过拟合
- **无点数评分时可通过 pairwise reward 间接对齐**：BTT loss 将帧对偏好转化为每个 rollout 的 reward 信号，使点数评分对齐人类排序
- **动态帧采样比均匀采样更有效**：利用生成视频时间一致性特征，在疑似失真区域密集采样
