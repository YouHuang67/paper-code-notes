---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
  - DPO
---

# DreaMontage: Montage Video Generation via Multi-Shot Diffusion Model Alignment

- 论文：https://arxiv.org/abs/2512.21252
- 代码：https://dreamontage.github.io/
- 团队：ByteDance

## 概述

DreaMontage 解决蒙太奇视频生成中的两大核心缺陷：镜头间的跳切（jump cuts）和镜头内的运动异常。提出三阶段后训练流程：Adaptive Tuning → SFT → Tailored DPO，针对性地优化多镜头视频的连贯性和运动质量。核心创新在于 DPO 阶段设计了两条独立的数据构建管线：Pipeline A 利用 VLM 判别器自动筛选跳切偏好对，Pipeline B 通过人工标注筛选运动质量偏好对，各生成约 1k 偏好对。此外在超分辨率阶段引入 Shared-RoPE 增强帧间位置一致性，推理时采用 SAR（Segmented Auto-Regressive）策略生成长视频。

## 动机

- 蒙太奇视频（多镜头拼接）是实际视频创作的核心形式，但现有视频生成模型主要针对单镜头优化
- 多镜头生成的两大痛点：
  - **跳切（Jump Cuts）**：相邻镜头之间缺乏视觉连贯性，产生突兀的视觉跳跃
  - **运动异常（Motion Artifacts）**：镜头内物体运动不自然、变形、抖动
- 现有方法缺乏针对多镜头场景的系统性后训练方案

## 方法

### 三阶段训练流程

**阶段 1：Adaptive Tuning**

- 在高质量多镜头视频数据上微调基础模型，适应多镜头生成范式
- 学习镜头转换的基本模式（切换、淡入淡出等）

**阶段 2：SFT（Supervised Fine-Tuning）**

- 在精选的高质量蒙太奇视频上进行监督微调
- 提升整体生成质量和镜头间连贯性

**阶段 3：Tailored DPO**

核心创新阶段，设计两条独立的偏好数据构建管线，分别针对两类缺陷：

**Pipeline A — 跳切修复**：
- 使用 VLM（视觉语言模型）作为判别器
- 对同一 prompt 生成多个候选视频
- VLM 自动评估相邻镜头间的视觉连贯性，筛选出跳切严重的样本作为 lose、连贯的作为 win
- 生成约 1k 偏好对

**Pipeline B — 运动质量**：
- 通过人工标注进行质量筛选
- 评估标准：运动自然性、物体一致性、时间平滑性
- 同样生成约 1k 偏好对

- 两条管线的偏好对合并后进行 DPO 训练，约 10k steps

### Shared-RoPE（超分辨率阶段）

- 在超分辨率（SR）模型中引入 Shared-RoPE 机制
- 不同镜头的帧共享相同的旋转位置编码基准
- 增强跨镜头的帧间位置一致性，减少 SR 阶段引入的不一致

### SAR 推理策略（Segmented Auto-Regressive）

- 将长蒙太奇视频分段自回归生成
- 每段以前一段的末尾帧作为条件，保持跨段连贯性
- 支持任意长度的多镜头视频生成

## 实验

- 基础模型：ByteDance 内部视频生成模型
- DPO 数据规模：Pipeline A + Pipeline B 各约 1k 偏好对，共约 2k 对
- DPO 训练：约 10k steps
- 评估维度：跳切频率、运动质量、整体视觉连贯性
- DPO 阶段显著降低跳切率，运动质量同步提升
- Shared-RoPE 在 SR 阶段有效减少帧间不一致

## 关键启示

- **分治策略**：将多镜头视频的两类核心缺陷（跳切、运动异常）解耦为独立的 DPO 数据管线，比混合训练更有针对性
- **VLM 作为自动判别器**：利用 VLM 自动构建跳切偏好对，降低人工标注成本，为自动化偏好数据生成提供新思路
- **位置编码的跨镜头共享**：Shared-RoPE 是一种轻量但有效的方法，解决多镜头场景中 SR 阶段的帧间不一致问题
- **小规模 DPO 数据即可生效**：每条管线仅 1k 偏好对即可显著改善对应缺陷，说明高质量偏好数据比数据量更重要
