---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
  - GRPO
  - Flow Matching
  - Reward Model
---

# Euphonium: Process Reward Guided Flow Matching for Video Generation

- 论文：https://arxiv.org/abs/2602.04928
- 团队：SJTU, Tencent Hunyuan

## 概述

Euphonium 提出将过程奖励模型（Process Reward Model, PRM）的梯度注入 Flow Matching 的 SDE 漂移项，实现去噪过程中的逐步引导。建立 NETS 统一框架，证明现有方法（ReFL、DRaFT、DDPO 等）均为 NETS 在不同噪声水平和估计策略下的特例。核心创新包括：(1) KL 正则化引导动力学公式，将 PRM 梯度与 KL 散度约束结合；(2) Dual-Reward GRPO，同时使用 PRM 和 ORM（Outcome Reward Model）训练策略；(3) 蒸馏机制，将推理时 PRM 引导的知识蒸馏到模型参数中，消除推理时对 PRM 的依赖。在 HunyuanVideo 14B 上实现 VBench2 最优总分，收敛速度提升 1.66 倍。

## 动机

- 现有视频生成 RL 方法（如 DDPO、DanceGRPO）仅使用结果奖励（ORM），即对最终生成的完整视频打分
- ORM 的问题：反馈信号稀疏，无法指导中间去噪步骤的优化方向，导致收敛慢、信用分配困难
- 过程奖励模型（PRM）可以在每个去噪步骤提供即时反馈，但如何将 PRM 与 Flow Matching 框架结合缺乏理论基础
- 推理时使用 PRM 引导增加计算开销，需要蒸馏消除依赖

## 方法

### NETS 统一框架（Noise-level Estimation and Timestep Selection）

建立统一视角将现有方法归纳为 NETS 框架的特例：

- **ReFL**：固定单一噪声水平，单步去噪估计
- **DRaFT**：固定起始噪声，$K$ 步去噪估计，反向传播通过 $K$ 步
- **DDPO/DanceGRPO**：全程去噪，策略梯度估计（不通过去噪链反向传播）

NETS 的关键维度：
1. **噪声水平选择**：在哪些 timestep 施加 reward 引导
2. **估计策略**：单步 vs 多步 vs 全程去噪
3. **梯度类型**：反向传播梯度 vs 策略梯度

### KL 正则化引导动力学

将 PRM 梯度注入 Flow Matching SDE 的漂移项，同时加入 KL 正则化防止偏离基础模型分布：

$$dx_t = \left[ f(x_t, t) + g(t)^2 \left( \nabla_{x_t} \log p_t(x_t) + \lambda \nabla_{x_t} R_{\text{PRM}}(x_t, t) \right) \right] dt + g(t) dW_t$$

其中 $f(x_t, t)$ 为 Flow Matching 的漂移项，$R_{\text{PRM}}(x_t, t)$ 为过程奖励，$\lambda$ 控制引导强度。

KL 约束确保引导后的轨迹不偏离原始流太远：

$$\min_\pi \mathbb{E}_\pi[R] - \alpha \text{KL}(\pi \| \pi_{\text{ref}})$$

### Dual-Reward GRPO

同时使用 PRM 和 ORM 训练，融合过程级和结果级反馈：

- **PRM 信号**：在每个去噪步骤提供局部梯度引导
- **ORM 信号**：对完整生成视频提供全局质量评分
- GRPO 框架下，组内归一化奖励结合两种信号：

$$R_{\text{dual}} = \alpha \cdot R_{\text{PRM}} + (1 - \alpha) \cdot R_{\text{ORM}}$$

- 双奖励互补：PRM 加速收敛（局部精确引导），ORM 保证最终质量（全局目标一致性）

### 蒸馏：消除推理时 PRM 依赖

- 推理时使用 PRM 引导需要额外前向传播，增加 30-50% 计算开销
- 蒸馏方案：用 PRM 引导后的去噪轨迹作为教师，训练学生模型直接拟合引导后的输出
- 蒸馏后模型不需要 PRM 即可生成高质量视频，推理速度恢复到基线水平

## 实验

- **基础模型**：HunyuanVideo 14B
- **PRM 构建**：基于视频质量评估模型，在中间去噪状态上微调
- **评估基准**：VBench2
- **主要结果**：
  - VBench2 总分达到最优，超越纯 ORM 方法
  - 相比纯 ORM GRPO，收敛速度提升 1.66 倍（达到相同质量所需训练步数减少 40%）
  - Dual-Reward GRPO 优于单独使用 PRM 或 ORM
  - 蒸馏后模型质量接近 PRM 引导推理，但推理速度恢复到无引导水平
- **消融实验**：
  - PRM 在高噪声 timestep 的引导效果最显著（早期去噪决定全局结构）
  - $\alpha$ 的最优值约 0.5，平衡局部和全局信号

## 关键启示

- **过程奖励 vs 结果奖励的互补性**：PRM 提供稠密的逐步引导加速收敛，ORM 保证最终输出质量，两者结合优于任一单独使用
- **NETS 统一视角**：将看似不同的方法（ReFL、DRaFT、DDPO）统一到同一框架下，有助于理解各方法的本质差异和适用场景
- **蒸馏消除推理开销**：推理时引导→蒸馏到参数→无开销推理的三步范式，是将 test-time compute 转化为模型能力的通用思路
- **高噪声 timestep 是关键**：PRM 在高噪声阶段效果最大，与 TAGRPO 等工作的发现一致，暗示早期去噪步骤决定生成质量的上限
