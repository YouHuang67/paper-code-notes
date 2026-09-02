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

VisionReward 把通用图/视频生成的人类偏好拆成层级诊断题，用微调后的 CogVLM2 / CogVLM2-Video 做二值 VQA，再用成对 logistic（Bradley-Terry）学线性权重合成可解释总分。视频侧 9 个主维度、20 个子维、64 道是/否题。用于 Diffusion-DPO 时提出 MPO：只有在所有维度上都支配对方的样本对才进入优化。相对 VideoScore，偏好预测准确率 +17.2%；CogVideoX-2B 上 MPO 的人类 pairwise 胜率比 VideoScore DPO 高 31.6%。这是 ImageReward（THUDM）在视频生成上的通用后继，不是垂类 RM。

## 动机

黑盒标量 RM（ImageReward、PickScore、HPSv2）无法审计多因素权衡。通用 VLM judge 可解释但细粒度偏好弱。图像 RM 评帧忽略时序；VideoScore 已做视频打分，但偏好预测与优化仍不够。

## 数据

相对 VideoScore（38K 样本、5 维、0.2M 标注），VisionReward 同时覆盖图与视频：81K 样本、18–20 维、61–64 题、5.0M 二值标注。

- 图像：ImageRewardDB、HPDv2、Pick-a-Pic 筛 48k 图，约 3M QA。层级为 Alignment / Composition / Quality / Fidelity / Safety，共 18 子维、61 题（含主体、配色、光、细节、肢体、手、脸、安全、情绪等）。
- 视频：VidProM 经 Rouge-L、语义过滤、ChatGPT 清洗得 10k prompt；CogVideoX、VideoCrafter2、OpenSora 生成 30k，Panda-70M 抽 3k 真实视频，共 33k、约 2M QA。9 个主维度：Alignment、Composition、Quality、Fidelity、Safety、Dynamic、Physics、Stability、Preservation。子维覆盖运动平滑、相机稳定、形变、物理、文字、动态物体等。
- 专业公司培训标注员，每题 ≥10 个示例。二值结果一致性：图像 89.29%，视频 89.33%。

## 方法

### 细粒度 VQA

先对每题平衡正负例，再 instruction tuning。图像骨干 CogVLM2，视频骨干 CogVLM2-Video。batch 64，LR $1\times 10^{-6}$（图）/ $4\times 10^{-6}$（视频），1500 step。

每题答案 $A_i\in\{\text{yes},\text{no}\}$，特征 $x_i=\mathbf{1}[A_i=\text{yes}]$。

### 可解释 BT 合成

输入特征差 $\Delta X=X_i-X_j$ 与人类偏好标签 $y\in\{0,1\}$，学权重 $W$，使总分

$$R=\sum_{i=1}^{N} w_i \mathbf{1}[A_i=\text{yes}]$$

最大化成对正确率。损失为 logistic / BT：

$$\mathcal{L}(W)=-\mathbb{E}\big[y\log\sigma(\Delta X W^\top)+(1-y)\log(1-\sigma(\Delta X W^\top))\big]$$

维度分只在该维的题上加权：$R(\mathrm{dim}_k)=\sum_{i\in\mathrm{dim}_k} w_i \mathbf{1}[A_i=\text{yes}]$。图像权重用 HPDv2 24k + ImageRewardDB 20k 对；视频权重用 1795 个人工偏好对。权重可 mask 掉接近零的题，GenAI-Bench 准确率基本保持。

### MPO

定义 $R_i$ 支配 $R_j$：对每个维度 $R_i(\mathrm{dim}_k)\ge R_j(\mathrm{dim}_k)$。普通 DPO 按总分 $R$ 选对；MPO 只保留支配对，再跑 Diffusion-DPO。Pick-a-Pic 上按总分 DPO 会在安全、对称、肢体等维出现负增益；MPO 用更严的 Pareto 约束压这种偏置。

## 实验

自建 MonetBench（图/视频各 1000 prompt）。另用 HPDv2、GenAI-Bench。tau 计 tie，diff 丢掉标签 tie。

偏好准确率（视频 MonetBench tau / diff）：VideoScore 49.1 / 54.9，VisionReward **64.0 / 72.1**。GenAI-Bench 视频（约 2s）上图像 RM 仍有竞争力；时长到 6s（MonetBench）时只有 VisionReward 稳定高于随机（相对随机 +22.1 点，次优约 +12.5）。题数从 4 增到 64，GenAI-Bench 准确率单调上升。

视频 MPO（CogVideoX-2B）：每 prompt 生成 4 条，MPO 筛约 9400 对（约 22000 条优化后的 VidProM prompt）。batch 32，LR $5\times 10^{-6}$，warmup 100，$\beta=500$，约 500 step / 2 epoch，每 40 step 按验证集 reward 选 ckpt。

VBench（CogVideoX-2B）：

| | Human Action | Scene | Multiple Objects | Appearance Style |
|--|:---:|:---:|:---:|:---:|
| Original | 98.20 | 55.60 | 68.43 | 24.20 |
| VideoScore DPO | 97.60 | 56.25 | 68.66 | 23.96 |
| VisionReward MPO | **98.40** | **57.57** | **71.54** | 24.02 |

VideoScore 优化后若干维下降；VisionReward 在 Scene / Multiple Objects 上抬得更明显。人类 pairwise：相对 VideoScore DPO 胜率 +31.6%。

## 关键启示

- 通用视频 BT 可以是「checklist 二值特征 + 成对 logistic」，不必只用一个线性头出标量；MPO 把多维一致性写进选对规则。
- 短视频上图像 RM 会虚高；6s 级时序必须用视频骨干和 Dynamic / Physics / Stability 题。
- 与 VideoReward（BTT 连续多维分数）对照：VisionReward 解释性强、选对更严；VideoReward 对 tie 和连续分更自然。
