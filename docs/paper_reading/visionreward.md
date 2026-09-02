---
tags:
  - Video Generation
  - Reward Model
  - VLM
  - DPO
  - Post Training
---

# VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning

- 论文：https://arxiv.org/abs/2412.21059
- 代码：https://github.com/THUDM/VisionReward（现 zai-org/VisionReward）
- 团队：Tsinghua University, Z.AI
- 发表：AAAI 2026

## 概述

VisionReward 是 ImageReward（THUDM，NeurIPS 2023）在图+视频上的后继：把黑盒标量 RM 拆成层级诊断题（9 个主维度、64 道是/否题），用微调后的 CogVLM2 / CogVLM2-Video 做二值 VQA，再用 logistic 回归学线性权重把答案合成可解释总分。权重学习的 pairwise logistic 即 Bradley-Terry。用于 Diffusion-DPO 时提出 MPO：只有在所有维度上都支配对方的样本对才进入优化，避免总分 hack 某一维。视频偏好预测准确率相对 VideoScore 提升 17.2%；CogVideoX 上用 VisionReward 做 DPO 的人类 pairwise 胜率比 VideoScore 高 31.6%。

## 动机

- ImageReward / PickScore / HPSv2 不透明，多因素权衡无法审计。
- 通用 VLM judge（GPT-4o、Gemini）可解释但细粒度偏好弱于专用 RM。
- 图像 RM 评帧忽略时序；VideoScore 已做视频打分，但偏好预测与优化仍不够。

## 数据

- 图像：从 ImageRewardDB、HPDv2、Pick-a-Pic 筛 48k 图，约 3M 二值 QA。
- 视频：VidProM 经 Rouge-L / 语义过滤 / ChatGPT 清洗得 10k prompt；CogVideoX、VideoCrafter2、OpenSora 生成 30k + Panda-70M 真实 3k，共 33k 视频、约 2M QA。
- 标注员一致性：图像 89.29%，视频 89.33%。合计约 81k 样本、5M 二值标注。

## 方法

### 细粒度评估

先平衡每题正负例，再 instruction tuning 骨干 VLM。每题答案 $A_i\in\{\text{yes},\text{no}\}$，特征 $x_i=\mathbf{1}[A_i=\text{yes}]$。

### 可解释 BT 合成

$$R=\sum_{i=1}^{N} w_i \mathbf{1}[A_i=\text{yes}]$$

成对差 $\Delta X=X_i-X_j$，对人类偏好 $y$ 做 logistic 回归（BT）：

$$\mathcal{L}(W)=-\mathbb{E}\big[y\log\sigma(\Delta X W^\top)+(1-y)\log(1-\sigma(\Delta X W^\top))\big]$$

维度分 $R(\mathrm{dim}_k)$ 只对属于该维度的题加权。图像权重用 HPDv2 24k + ImageRewardDB 20k 对；视频权重用 1795 个人工偏好对。

VQA 训练：CogVLM2 / CogVLM2-Video，batch 64，LR $1\times 10^{-6}$ / $4\times 10^{-6}$，1500 step。

### MPO

$R_i$ 支配 $R_j$ 当且仅当每个维度 $R_i(\mathrm{dim}_k)\ge R_j(\mathrm{dim}_k)$。普通 DPO 按总分选对；MPO-DPO 只保留支配对，再跑标准 Diffusion-DPO。Pick-a-Pic 上标准 DPO 会在安全、对称、肢体等维出现负增益，MPO 用于压这种偏置。

## 实验

自建 MonetBench（图/视频各 1000 prompt）。视频时长到 6s 时，仅 VisionReward 能稳定超过随机；相对 VideoScore，MonetBench 视频 tau/diff 为 64.0 / 72.1 vs 49.1 / 54.9。问题数量从 4 增到 64，GenAI-Bench 准确率单调上升。

人类评 DPO：SDXL 与 CogVideoX 上 VisionReward 均优于 Pick-a-Pic 原始对、HPSv2、VideoScore。

## 关键启示

- BT 不必绑死在「一个线性头出标量」：checklist 二值特征 + 成对 logistic 仍是 BT，但可解释、可按维做 MPO。
- 图像级 BT RM 直接搬到长视频会失效；需要视频骨干和时序相关问题。
- 与 VideoReward（BTT 标量头）对照：VisionReward 解释性强、优化约束更严；VideoReward 更适合连续多维分数和 tie。
