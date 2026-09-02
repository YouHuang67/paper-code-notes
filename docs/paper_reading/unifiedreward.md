---
tags:
  - Video Generation
  - Reward Model
  - VLM
  - DPO
  - Post Training
---

# UnifiedReward: Unified Reward Model for Multimodal Understanding and Generation

- 论文：https://arxiv.org/abs/2503.05236
- 代码：https://github.com/CodeGoat24/UnifiedReward
- 团队：Fudan University, Shanghai Innovation Institute, Shanghai AI Lab

## 概述

UnifiedReward 是 2025 年被后续工作当作默认开源视觉 RM baseline 的统一模型：一张 VLM 同时做图像/视频的理解与生成评估，同一套指令同时支持 pairwise 排序和 pointwise 打分。训练目标是标准 next-token 交叉熵（只在答案 token 上），因此它已经是生成式 RM（GRM）而不是 BT 线性头。数据约 236K，拼 EvalMuse / HPD / OIP、LLaVA-Critic、VideoDPO / LiFT-HRA / VideoFeedback、ShareGPTVideo 等。训练后用「成对排序划 chosen/rejected 列表 + 点数筛选极值」构造偏好对，再对 SDXL-Turbo、T2V-Turbo、LLaVA-OneVision、LLaVA-Video 做 DPO。联合多任务比单任务 RM 更强：VideoGen-RewardBench 上 Overall tau/diff 60.7 / 77.2，超过 VideoReward 的 50.2 / 73.3。

## 动机

- 当时 RM 任务割裂：PickScore/HPS/ImageReward 只管图生成，VideoScore/LiFT/VideoReward 只管视频生成，LLaVA-Critic 只管理解。
- 假设跨任务协同：图像理解帮图像生成评估，图像评估帮视频帧评估。

## 数据（约 236K）

- 图像生成：EvalMuse 均分 + 元素一致性做 pointwise；HPD 票数构造 pairwise；OIP 7.4K 对直接用。
- 图像理解：LLaVA-Critic-113K 中各取 25K pair / point。
- 视频生成：VideoDPO 10K pairwise；LiFT-HRA、VideoFeedback 做 pointwise。
- 视频理解：ShareGPTVideo 等。
- pairwise 统一成「谁更好 + 可选理由」；pointwise 不强制统一分数格式，按源数据改 instruction。

## 方法

骨干：LLaVA-OneVision-7B（另训 Qwen2.5-VL 验证）。输入视觉 token + instruction + caption/问题，输出排序或分数（可带短理由）。损失：交叉熵，只在预测答案上。8×H100，batch 2，grad accum 16，LR $2.5\times 10^{-6}$，warmup 0.3。直接回答约 1s，带短理由约 3s。

偏好构造：模型采样 $N=10$ 候选 → 两两排序分成 chosen 列表 $C$ 与 rejected 列表 $R$ → 点数 $S(\cdot)$ 取 $O_c^*=\arg\max_{O\in C}S(O)$，$O_r^*=\arg\min_{O\in R}S(O)$。

DPO：理解侧 $\beta_u=0.1$；生成侧 $\beta_g=5000$（扩散 DPO 常用大 $\beta$）。视频生成 10K 对，其余任务 14K，3 epoch。

## 实验

VLRewardBench Overall / Macro：UnifiedReward 66.1 / 66.5，高于 GPT-4o 65.8 / 62.4 和 LLaVA-Critic 46.9 / 46.6。只训图像理解远弱于全任务。

视频生成（VideoGen-RewardBench tau / diff）：VideoScore 42.1 / 49.9，VisionReward 46.8 / 66.4，VideoReward 50.2 / 73.3，UnifiedReward **60.7 / 77.2**。视频训练数据相对少，多任务仍抬高视频评估。

T2V-Turbo DPO 相对 VideoDPO 数据构造，定性上运动与细节更稳。

## 关键启示

- 开源视觉 GRM 的第一代形态：不改架构，把排序/打分写成 VLM 生成任务，靠指令和多任务数据。
- 它不是 BT 头，但输出仍可当 pointwise reward 喂 DPO/GRPO；后续 Think/Flex/RewardDance 都在这条生成式轴上加推理深度。
- pair ranking + point sifting 比单一策略更适合从同一 RM 蒸馏对齐数据。
