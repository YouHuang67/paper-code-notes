---
tags:
  - Video Generation
  - Reward Model
  - Reinforcement Learning
  - VLM
---

# RewardDance: Reward Scaling in Visual Generation

- 论文：https://arxiv.org/abs/2509.08826
- 团队：ByteDance Seed

## 概述

RewardDance 提出可扩展的视觉 Reward Model 框架，核心创新是生成式 reward 范式——将 reward 分数重新表述为 VLM 预测 "yes" token 的概率（表示一张图/视频优于参考），天然对齐 VLM 的 next-token prediction 机制。这解锁两个维度的扩展：(1) 模型扩展：系统化地将 RM 从 1B 扩展到 26B 参数；(2) 上下文扩展：整合任务指令、参考样例和 CoT 推理。关键发现：大规模生成式 RM 在 RL 训练后期维持高 reward 方差（抵抗 reward hacking），而回归式 RM 方差迅速收缩（mode collapse）。在 Seedream-3.0 (T2I) 上 26B RM 带来 alignment score +10.7；在 Seedance-1.0 (T2V) 上 GSB +49%。

## 动机

- CLIP-based RM 受架构和输入模态限制，难以扩展
- VLM-based 回归式 RM（Bradley-Terry loss + 回归头）与 VLM 的 next-token prediction 机制根本不匹配，阻碍有效扩展
- 回归式 RM 极易被 reward hacking：训练后期 reward 方差趋近零，模型生成均匀高分但无多样性的输出

## 方法

### 生成式 Reward 建模

将 reward 评估重构为配对比较生成任务。输入两张图/视频 $x_1, x_2$、prompt $y$ 和 CoT 任务指令 $i$，reward 分数为模型预测 "yes" 的概率：

$$r_\theta(x_1, x_2, y, i) = P_\theta(\text{"yes"} | x_1, x_2, y, i)$$

天然对齐 VLM 自回归机制，无需额外回归头。

### 模型扩展（1B→26B）

系统化训练 1B、2B、4B、8B、26B 五个规模的 RM。关键发现：
- **OOD 准确率是关键指标**：ID 准确率与参数规模无严格正相关，但 OOD 准确率（泛化能力）持续提升
- 1B: 69.10% OOD → 26B: 80.90% OOD
- OOD 泛化能力是 RM 指导 RL 效果的更好预测器

### 上下文扩展

超越简单 image-text pair，整合三类上下文：
- **任务感知指令**：定义具体评估标准
- **参考样例**：Best-of-N 采样的参考图作为比较基线
- **CoT 推理**：训练模型先生成评判理由再给分，增强推理能力

### 训练流程

- **Stage 1**：大规模偏好数据 SFT（格式对齐 + 任务理解）
- **Stage 2**：针对性偏好数据微调（包含 CoT 推理数据）
- 下游应用：RL fine-tuning（GRPO 驱动）或 test-time scaling（推理时 Best-of-N）

### 抗 Reward Hacking 机制

生成式范式的内在优势：
- **回归式 2B RM**：RL 后期 reward 方差 $\sigma = 4.2 \times 10^{-3}$（几乎为零，mode collapse）
- **生成式 2B RM**：$\sigma = 6.1 \times 10^{-3}$（更高方差）
- **生成式 26B RM**：$\sigma = 5.4 \times 10^{-2}$（方差比回归式高一个数量级，维持多样性）

## 实验

### 模型扩展效果（T2I）

| RM 规模 | OOD Acc | FLUX.1-dev Align | Seedream-3.0 Align (RL) | Seedream-3.0 Align (TTS) |
|:---:|:---:|:---:|:---:|:---:|
| No RM | - | 67.0 | 74.1 | 74.1 |
| 1B | 69.10% | 70.7 (+3.7) | 74.9 (+0.8) | 75.1 (+1.0) |
| 2B | 69.59% | 72.4 (+5.4) | 75.3 (+1.2) | 76.3 (+2.2) |
| 4B | 71.92% | 72.2 (+5.2) | 79.5 (+5.4) | 78.4 (+4.3) |
| 8B | 71.94% | 73.0 (+6.0) | 81.6 (+7.5) | 79.3 (+5.2) |
| 26B | 80.90% | 73.6 (+6.6) | **84.8** (+10.7) | 80.5 (+6.4) |

### 视频生成扩展（T2V/I2V，GSB 提升）

| 任务 | 1B RM | 2B RM | 4B RM | 8B RM | 26B RM |
|------|:---:|:---:|:---:|:---:|:---:|
| T2V RL | +28% | +32% | +41% | +45% | **+49%** |
| I2V RL | +29% | +34% | +37% | +41% | **+47%** |

### GenEval

Seedream-3.0 w/ RewardDance Overall: 0.79（w/o: 0.69，+0.10）

## 关键启示

- **生成式 > 回归式 reward 范式**：将 reward 建模为 token 生成任务，天然对齐 VLM 架构，同等参数下更抗 reward hacking
- **Reward Model 扩展规律**：RM 从 1B 扩展到 26B 带来持续且显著的生成质量提升，这是视觉生成领域首次系统验证的 RM scaling law
- **OOD 泛化 > ID 准确率**：RM 在未见分布上的判断准确性（泛化能力）是预测 RL 最终效果的更好指标
- **高 reward 方差是健康信号**：RL 训练后期 reward 方差维持高位说明模型仍在有效探索多样模式，方差趋零则是 mode collapse 的征兆
