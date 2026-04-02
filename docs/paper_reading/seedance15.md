---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
---

# Seedance 1.5 pro: A Native Audio-Visual Joint Generation Foundation Model

- 论文：https://arxiv.org/abs/2512.13507
- 团队：ByteDance Seed

## 概述

Seedance 1.5 pro 提出一个原生音视频联合生成基础模型，基于双分支 Diffusion Transformer（MMDiT）架构，视频和音频各一路 branch，通过 Cross-Modal Joint Module 实现时序同步和语义绑定。后训练采用 SFT + RLHF（多维奖励模型覆盖运动质量、视觉美学、音频保真度），并对 RLHF pipeline 做基础设施优化实现约 3x 训练加速。在内部 SeedVideoBench-1.5 上，T2V/I2V 的指令遵循、视觉美学、运动质量均具竞争力，中文语音生成和唇形同步优于 Veo 3.1 和 Kling 2.6。

## 动机

- 现有视频生成模型将音频作为后处理步骤，缺乏原生语义和时序绑定，唇形同步差、音视频情感割裂
- 专业内容创作需要精确多语言/方言口型同步和叙事连贯性
- slow-motion trick 刷稳定性指标但牺牲运动活力（vividness），实际体验差

## 方法

### 架构：双分支 Diffusion Transformer

- 基于 MMDiT，视频和音频各一个 diffusion branch
- Cross-Modal Joint Module 负责两路特征的时序对齐和语义绑定
- 支持 T2VA（文本到视频+音频）、I2VA、T2V、I2V 多任务

### 数据

- 多阶段数据筛选（music coherence、motion expressiveness）
- Curriculum-based data scheduling
- 专业级 captioning system 覆盖视频和音频两个模态

### 后训练（SFT + RLHF）

论文为 technical report 风格，后训练部分披露极简略：

- **SFT**：在高质量音视频数据集上监督微调
- **RLHF**：多维奖励模型分别覆盖运动质量、视觉美学、音频保真度
  - 引用 Flow-GRPO、DanceGRPO、RewardDance、UniFl 等 RL 对齐工作
  - 对 RLHF pipeline 做基础设施优化，实现约 3x 训练加速
- 具体奖励函数形式、RL 算法选择、训练轮数均未披露

### 推理加速

- 多阶段蒸馏框架（参考 Mean Flows / HyperSD / RayFlow）
- NFE 大幅压缩 + 量化 + 并行推理，端到端加速超 10x

## 实验

评测基于内部 SeedVideoBench-1.5（专业影视从业者 5-point Likert + GSB 对比），无具体数值表格：

- T2V：指令遵循排名第一，视觉美学和运动质量与 Kling 2.6 / Veo 3.1 具竞争力
- Audio GSB：中文语音生成和唇形同步优于 Veo 3.1 和 Kling 2.6
- 支持四川话、台湾普通话、粤语、上海话等方言

## 关键启示

- **多维奖励模型分解优于单一奖励**：将 RLHF 奖励拆分为运动质量/视觉美学/音频保真度，可针对性改善各能力短板
- **RLHF 基础设施优化是工程关键**：RL 训练瓶颈往往不在算法而在工程，pipeline 优化可带来 3x 加速
- **"Vividness" 应作为独立评测维度**：slow-motion bias 是视频模型的系统性缺陷，将运动活力单独量化可遏制该 shortcut 行为
- **本文的 RL alignment 技术细节极少**：核心价值在架构和数据 pipeline，RL 相关技术应参考其引用的 DanceGRPO、RewardDance、Flow-GRPO
