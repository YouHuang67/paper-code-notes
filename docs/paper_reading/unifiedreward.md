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

UnifiedReward 用一张 VLM 同时评估图像/视频的生成与理解，指令里切换 pairwise 排序和 pointwise 打分。训练是标准 next-token 交叉熵（只在答案 token 上），属于生成式 RM（GRM），不是 BT 线性头。视频生成 pairwise 数据相对少（VideoDPO 10K），靠多任务把 VideoGen-RewardBench Overall 做到 tau/diff **60.7 / 77.2**（VideoReward 50.2 / 73.3）。再用「成对排序划列表 + 点数取极值」构造偏好对，对 T2V-Turbo 做扩散 DPO。后续 Think / Flex 都以此为基座。评估面包含理解任务，主贡献仍是通用视觉生成 RM，不是垂类。

## 动机

当时 RM 按任务切开：图生成（PickScore / HPS / ImageReward）、视频生成（VideoScore / LiFT / VideoReward）、理解（LLaVA-Critic）。假设图像理解改善生成评估，图像评估改善视频帧评估。

## 数据（约 236K）

| 任务 | Pair | Point |
|------|------|-------|
| 图像生成 | EvalMuse* 3K，HPD* 25.6K，OIP 7.4K | EvalMuse* 32.7K |
| 图像理解 | LLaVA-Critic 25K | LLaVA-Critic 25K |
| 视频生成 | VideoDPO 10K | LiFT-HRA 20K，VideoFeedback 36.6K |
| 视频理解 | ShareGPTVideo 17K | ShareGPTVideo* 34K |

EvalMuse：至少三人打 1–5 分 + 元素是否出现；pointwise 用均分，pairwise 取同 prompt 最高/最低均分。HPD：按票数排 pairwise。

pairwise 答案统一为 “image/video/response X is better than … Y”；有理由的源数据保留理由。pointwise 不统一分数区间，靠 instruction 对齐各套评分标准。

## 方法

### GRM 训练

骨干 LLaVA-OneVision-7B（另训 Qwen2.5-VL）。生成评估输入：视觉 token + instruction + caption；理解评估把 caption 换成 question。模型按 instruction 输出排序或分数，可带短理由。损失：交叉熵，只在预测答案上。8×H100，batch 2，grad accum 16，LR $2.5\times 10^{-6}$，warmup 0.3。直接回答约 1s，短理由约 3s。

### 偏好构造

给定 prompt，生成器（或 VLM）采样 $N=10$ 个输出 $\{O_1,\ldots,O_N\}$。

1. Pair rank：配成 $N/2$ 对，RM 排序后得到 chosen 列表 $C$ 与 rejected 列表 $R$。
2. Point sift：对 $C,R$ 全部打点分 $S(\cdot)$，取

$$O_c^*=\arg\max_{O\in C} S(O),\quad O_r^*=\arg\min_{O\in R} S(O)$$

相对只用排序或只用打分，同时用相对比较和绝对质量两端。

### 扩散 DPO（视频/图像生成）

偏好对 $\mathcal{D}_{\mathrm{Gen}}=\{(x_0^w,x_0^l)\}$。在噪声步 $t$ 比较微调模型与参考模型的噪声预测误差：

$$\mathcal{L}(\theta)=-\mathbb{E}\log\sigma\Big(-\beta_g T_\omega(\lambda_t)\big(\|\epsilon^w-\epsilon_\theta(x_t^w,t)\|^2-\|\epsilon^w-\epsilon_{\mathrm{ref}}\|^2-\|\epsilon^l-\epsilon_\theta(x_t^l,t)\|^2+\|\epsilon^l-\epsilon_{\mathrm{ref}}\|^2\big)\Big)$$

文中 $T_\omega(\lambda_t)$ 取成常数，实际用 $\beta_g=5000$。T2V-Turbo：10K 对，batch 16，3 epoch。SDXL-Turbo：14K 对，batch 32。理解侧标准 DPO，$\beta_u=0.1$。

## 实验

视频生成评估（VideoGen-RewardBench tau / diff）：VideoScore 42.1 / 49.9，VisionReward 46.8 / 66.4，VideoReward 50.2 / 73.3，只训视频生成 48.2 / 69.4，图+视频生成 52.0 / 73.6，**UnifiedReward 60.7 / 77.2**。GenAI-Bench 图像 tau/diff：UnifiedReward 54.8 / 70.9。

视频理解 Acc：OV-7B 48.2 → 只训视频理解 74.2 → 视频理解+图像理解 76.6 → 视频理解+生成 78.6 → 全任务 **84.0**。视频 pairwise 虽少，多任务仍抬高视频生成 RM。

T2V-Turbo 上相对 VideoDPO 构造的偏好数据，DPO 后运动与语义更稳（文中定性对比）。

## 关键启示

- 开源通用视频 GRM 的第一代：不改架构，把排序/打分写成 VLM 生成；视频对不足时靠图+理解任务补。
- pair ranking + point sifting 适合从同一 RM 蒸馏对齐数据。
- 输出可当 pointwise reward；Think 加长 CoT，Flex 改动态标准，RewardDance 改 yes-token 扩展。
