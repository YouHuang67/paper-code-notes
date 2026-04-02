---
tags:
  - Video Generation
  - Reinforcement Learning
  - GRPO
  - DPO
  - Diffusion Model
---

# MapReduce LoRA: Multi-Preference Alignment via Iterative Expert Merging

- 论文：https://arxiv.org/abs/2511.20629
- 代码：https://github.com/SHI-Labs/MapReduce-LoRA
- 团队：Georgia Tech, Adobe

## 概述

MapReduce LoRA 解决多维度偏好对齐中的"对齐税"问题——优化一个 reward 维度往往导致其他维度退化。核心方法：Map 阶段并行训练多个 reward-specific LoRA 专家（GRPO 驱动），Reduce 阶段均匀平均并折叠到基础模型，迭代多轮逼近 Pareto 前沿。理论上基于近端共识优化证明逐步合并优于一次性合并。额外提出 Reward-aware Token Embedding (RaTE) 实现推理时偏好控制。在 SD 3.5 M 上 GenEval +36.1%、OCR +55.7%；在 HunyuanVideo 上 VQ +48.09%、MQ +89.96%。

## 动机

- 多维度人类偏好对齐面临"对齐税"：优化多个 reward 信号时，改进一个维度退化其他维度
- 先验法（加权混合 reward）易被易优化目标主导
- 后验法（Rewarded Soup 一次性合并权重）性能受限
- 目标：建立高效可控的多偏好对齐框架，跨文生图/文生视频/语言任务

## 方法

### MapReduce LoRA（迭代式 LoRA 专家合并）

**Map 阶段**：冻结基础模型 $\theta^{(k)}$，各 reward $R_i$ 独立训练 LoRA adapter $\phi_i^{(k)}$，使用 GRPO：

$$J_{\text{GRPO}} = \mathbb{E}_p\left[\frac{1}{G}\sum_{g=1}^G \frac{1}{T}\sum_{t=1}^T \min(r_t^g \hat{A}_g, \text{clip}(r_t^g, 1-\epsilon, 1+\epsilon)\hat{A}_g)\right] - \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

组归一化优势：$\hat{A}_g = (R(y_g, p) - \text{mean}(R)) / \text{std}(R)$

**Reduce 阶段**：均匀平均专家 LoRA 权重 $\bar{\phi}^{(k)} = \frac{1}{n}\sum_{i=1}^n \phi_i^{(k)}$，折叠到基础模型 $\theta^{(k+1)} = \text{MergeLoRA}(\theta^{(k)}, \bar{\phi}^{(k)})$，重置 adapter，进入下一迭代。

**理论保证**：假设各目标 $f_i(\theta)$ 局部 L-光滑，聚合目标满足 Polyak-Łojasiewicz 条件，迭代 $m$ 轮后误差界：

$$\|F(\theta^{(m)}) - F^*\| \leq (1 - c\eta\mu)^m \|F(\theta^{(0)}) - F^*\|$$

### Reward-aware Token Embedding (RaTE)

推理时偏好控制机制，基于 Textual Inversion 思想：

- 教师：专家 LoRA adapter + 冻结基础模型
- 学生：冻结基础模型 + 特殊 token 嵌入 $\theta_{\text{token}}^i$（唯一可训练参数）
- Flow Matching 蒸馏：

$$\mathcal{L} = \mathbb{E}_{p,z,\epsilon,t}\left[\|M(z_t, t, c(p, \theta_{\text{token}}^i)) - \epsilon + z_{\text{teacher},0,i}\|_2^2\right]$$

训练开销仅为 MapReduce LoRA 的 0.1579 倍。推理时用户自由拼接多个 reward token 控制偏好。

## 实验

### 文生图（SD 3.5 M / FLUX.1-dev）

3 个 reward：GenEval（文本-图像对齐）、PickScore（人类偏好）、OCR（文本渲染）

| 模型 | 方法 | GenEval | PickScore | OCR |
|------|------|:---:|:---:|:---:|
| SD 3.5 M | 基线 | 0.68 | 21.78 | 0.601 |
| | MapReduce LoRA | **0.98** (+36.1%) | **22.54** (+4.6%) | **0.916** (+55.7%) |
| FLUX.1-dev | 基线 | 0.67 | 22.01 | 0.573 |
| | MapReduce LoRA | **0.96** (+32.7%) | **22.95** (+4.3%) | **0.968** (+67.1%) |

泛化到非目标 reward：VQAScore +1.85%，MPS +6.49%，VILA +19.96%

### 文生视频（HunyuanVideo）

2 个 reward：Visual Quality (VQ) + Motion Quality (MQ)，3 轮合并

| 方法 | VQ | MQ |
|------|:---:|:---:|
| 基线 | 3.25 | 0.95 |
| 单独专家 | 4.37 (+34.55%) | 1.66 (+74.10%) |
| Rewarded Soup | 4.13 (+27.37%) | 1.43 (+50.42%) |
| **MapReduce LoRA** | **4.81** (+48.09%) | **1.81** (+89.96%) |

### 语言任务（Llama-2 7B）

- Helpful: +43.4%
- Harmless: **+136.7%**

### 迭代次数消融

- $k=1$：初步获益但收敛不充分
- $k=4$（默认）：平衡点，OCR 达 0.916
- $k=10$：边际收益递减（OCR 仅 +0.03%）

训练配置：32×A100，全局 batch size 576，SD 3.5 M 用 LoRA ($\alpha=64$, $r=32$)，4 轮迭代

## 关键启示

- **迭代合并优于一次性合并**：逐步折叠 LoRA 通过近端共识机制持续推进 Pareto 前沿，理论和实验双重验证
- **专家并行训练避免梯度竞争**：各 reward 独立优化消除多目标冲突，比加权混合训练稳定且高效
- **RaTE 实现零额外推理成本的偏好控制**：轻量级 token 嵌入蒸馏专家知识，适用于有显式 cross-attention 的模型（SD 系列），对联合序列建模（FLUX）效果有限
- **跨模态通用性**：统一框架适配文生图（两种架构）、文生视频、语言任务
