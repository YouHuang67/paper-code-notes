---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
  - GRPO
  - Flow Matching
  - VLM
---

# What Happens Next: Next Scene Prediction with a Unified Video Model

- 论文：https://arxiv.org/abs/2512.13015
- 代码：https://nextsceneprediction.github.io/
- 团队：Penn State University, Amazon

## 概述

提出 Next Scene Prediction（NSP）任务：给定前一场景的文字描述，生成后续场景的视频。与标准 T2V 不同，NSP 要求模型进行时间推理和因果推断。架构上采用 Query/Connector 范式，冻结 Qwen-VL（7B）做多模态理解，LTX 做视频生成，两者通过 256 个可学习 query embedding 和 connector 模块（Linear-SiLU-Linear-RMSNorm + 可学习 scale）桥接。三阶段训练：T2V 预训练（课程学习：图像→混合→视频）→ NSP 数据集上 SFT → GRPO + 因果一致性 reward。最终因果一致性从 baseline 的 0.23 提升到 0.73。

## 动机

- 统一多模态模型（理解+生成）在传统任务（T2V、编辑）上取得进展，但时间推理潜力未被充分挖掘
- NSP 任务要求更深层次的能力：时间推理、因果推断、叙事一致性、视觉合理性
- 现有 T2V 模型只描述当下，不推理未来——NSP 推动模型从"描述"到"预测"

## 方法

### 架构设计

- **理解模块**：冻结 Qwen-VL 2.5（7B），接收文本 + 可学习 query embedding（256 个，最大序列长度 1024）
- **生成模块**：LTX 视频 DiT + 冻结 VAE（latent space 编解码）
- **Connector**：两层 Linear + SiLU 激活 + RMSNorm（权重初始化 $\sqrt{5.5}$，可学习 scale 初始化 0.01）
  - Qwen-VL 输出 embedding 的方差比 LTX 原始 text encoder 大几个数量级，RMSNorm + scale 是关键稳定化设计

### 三阶段训练

**Stage 1-3：T2V 预训练（课程学习）**

| 阶段 | 数据 | 训练内容 | 配置 |
|------|------|---------|------|
| Stage 1 | BLIP-3o 20M 图像 | 仅 connector | 3 epochs, batch 32 |
| Stage 2 | BLIP-3o 7M + VidGen 1M + OpenVid 0.31M + OpenS2V 1.33M | Connector + DiT | 5 epochs, batch 8 |
| Stage 3 | OpenHumanVid 10.8M | Connector + DiT | 2 epochs, batch 8 |

- 优化器：Prodigy，初始 LR 1.0，32×80G A100
- 去掉预训练直接 SFT：因果一致性仅 0.06（灾难性失败）

**SFT 阶段**

- NSP 数据集 0.97M 样本（从 OpenS2V + OpenVid 构建）
- 构建方式：LLM 生成前置场景描述 + LLM 过滤验证（因果一致性 + 非冗余性，3 轮迭代，不合格丢弃）
- 输入 prompt："Please generate a video showing what happens after this scene: \<preceding scene description\>"
- 32×80G A100, batch 8

**GRPO + 因果一致性 Reward**

- 数据集：8K 样本
- Reward 设计（二值化）：
  1. Judge 模型（Claude 3.7 Sonnet）对生成视频做 caption
  2. Judge 模型评估 caption 与前置场景描述的因果一致性
  3. 二值 reward：$r(s, q) = 1$ (Pass) 或 $0$ (Fail)
- GRPO 目标：标准组内归一化优势 + clipped surrogate
- Flow Matching 的 SDE 转换（同 Flow-GRPO / DanceGRPO）：

$$dx_t = \left[ v_\theta(x_t, c, t) + \frac{\sigma_t^2}{2t} (x_t + (1-t)v_\theta(x_t, c, t)) \right] dt + \sigma_t dw$$

- 训练配置：16×80G A100，梯度累积 8，60 优化步，batch 1，CFG 3.0，分辨率 480×832×65，20 采样步
- 每个输入生成 24 个候选视频，Best-of-N（N=8）选 top-4 + bottom-4 做 reward 优化

## 实验

### T2V 质量（VBench）

| 模型 | Quality | Semantic | Total |
|------|:---:|:---:|:---:|
| LTX 原始 | 0.7799 | 0.6738 | 0.7586 |
| 本文（预训练后）| **0.8051** | **0.6945** | **0.7830** |

### NSP 因果一致性

| 方法 | Causal Consistency |
|------|:---:|
| LTX | 0.23 |
| Wan 2.1 1.3B | 0.25 |
| Omni-Video | 0.46 |
| **本文** | **0.73** |

### 消融实验

| 阶段 | Causal Consistency |
|------|:---:|
| 仅预训练 | 0.54 |
| + SFT | 0.60 |
| + RL（GRPO）| **0.73** |
| 无预训练 | 0.06 |

### 效率

| 方法 | GPU 内存 | 采样时间 |
|------|:---:|:---:|
| LTX | 14.9G | 12s |
| Wan | 16.9G | 2m22s |
| Omni-Video | 37.6G | 2m16s |
| 本文 | 26.4G | **13s** |

比 Wan/Omni-Video 快约 10 倍。

## 关键启示

- **二值 reward + GRPO 可行**：无需复杂标量 reward 设计，LLM judge 的 Pass/Fail 判断即可驱动有效的 RL 优化
- **预训练不可跳过**：T2V 预训练是 NSP 成功的必要条件（去掉后因果一致性从 0.54 崩溃到 0.06）
- **Query/Connector 范式的效率优势**：冻结理解模块 + 轻量 connector + 高效生成器，13s 采样 vs 其他方法 2m+，兼顾质量和效率
- **NSP 作为时间推理的代理任务**：要求模型理解因果关系而非简单描述，为统一多模态模型提供新的能力评测维度
