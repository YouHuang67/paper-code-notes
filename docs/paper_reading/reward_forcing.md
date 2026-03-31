---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
---

# Reward-Forcing: Autoregressive Video Generation with Reward Feedback

- 论文：https://arxiv.org/abs/2601.16933
- 团队：UCLA

## 概述

Reward-Forcing 提出一种将双向视频扩散模型转换为自回归模型的轻量方案，避免传统方法对强双向 teacher 模型的重度依赖。核心观察：扩散模型先学习全局运动结构，后学习纹理细节。基于此，Reward-Forcing 分两阶段训练：(1) 用少量 ODE 轨迹（1.4K）从 teacher 学习运动先验；(2) 用纯 reward 信号（ImageReward）引导纹理质量提升，无需进一步蒸馏。基于 Wan2.1-1.3B，在 VBench 上总分 84.92，超过 Self Forcing 的 84.31 和 CausVid 的 81.20，且训练过程显著更轻量——不需要加载 teacher/critic 模型，也不需要训练 critic 来近似生成器分布。

## 动机

现有自回归视频扩散方法（CausVid、Self Forcing）依赖从双向 teacher 进行异构蒸馏，存在两个问题：
1. **性能受限于 teacher**：自回归 student 的性能天花板由双向 teacher 决定
2. **训练开销大**：需要加载 teacher 和 critic 模型，训练 critic 近似生成器分布

关键观察：扩散模型生成过程中，全局粗糙结构（运动）先于纹理细节出现。在将双向模型转为自回归模型时同样成立——ODE 训练后模型已学会一致的运动模式，但缺乏纹理信息。

## 方法

### 阶段 1：ODE 轨迹初始化

先用 DMD 将模型蒸馏为 4 步模型。然后从双向 teacher 采样 1.4K 条 ODE 轨迹 $\{x_t^i\}$，训练 student：

$$\mathcal{L}_{\text{ode}} = \mathbb{E}_{x, t_i} \|G_\phi(\{x_{t_i}^i\}, \{t_i\}) - \{x_0^i\}\|^2$$

此阶段后，模型学会生成一致的运动，但纹理信息匮乏。

### 阶段 2：Reward 引导优化

用 ImageReward 作为可微分 reward 模型，直接反向传播梯度引导视频扩散模型：

$$\mathcal{L}_{\text{reward}}(\theta) = -\mathbb{E}_{z \sim \mathcal{Z}}[R(\hat{x}_T)]$$

其中 $\hat{x}_T$ 是生成视频的最后一帧。

**为什么只监督最后一帧**：监督随机帧会导致运动质量下降（dynamic degree 降低 10+ 个百分点），因为 ImageReward 是图像级 reward，缺乏时间连续性概念，监督多帧会鼓励模型生成静态内容以获取高分。只监督最后一帧保留了运动，可能因为它是自回归过程的末端，受运动先验保护最少。

### 关键设计

- **不需要第二阶段蒸馏**：省略 CausVid/Self Forcing 中的 DMD 第二阶段，无需加载 teacher/critic，训练更轻量
- **data-free**：只需 ODE 轨迹和 reward 模型，不需要真实训练数据
- **Self-Rollout**：训练时使用自生成帧而非 ground-truth 帧，与 Self Forcing 一致
- **EMA**：训练过程启用 EMA（decay 0.99）

## 实验

### 训练配置

- 基础模型：Wan2.1-T2V-1.3B，先蒸馏为 4 步模型
- ODE 初始化：1.4K 轨迹
- Reward：ImageReward（仅监督最后一帧）
- 优化器：AdamW（$\beta_1=0$, $\beta_2=0.999$, lr $2 \times 10^{-6}$, weight decay 0.01）
- batch size 8，8 × H100 GPU

### 主要结果（VBench）

| 模型 | 类型 | Total | Quality | Semantic | 吞吐量 (FPS) |
|------|------|-------|---------|----------|-------------|
| Wan2.1 | 双向扩散 | 84.26 | 85.30 | 80.09 | 0.78 |
| CausVid | chunk 自回归 | 81.20 | 84.05 | 69.80 | 17.0 |
| Self Forcing | chunk 自回归 | 84.31 | 85.07 | 81.28 | 17.0 |
| ODE Only | chunk 自回归 | 68.77 | 73.27 | 50.81 | 17.0 |
| **Ours** | chunk 自回归 | **84.92** | **85.91** | 80.97 | 17.0 |

Reward-Forcing 在 Quality Score 上超过所有方法（包括双向 teacher），Total Score 超过 Self Forcing 0.61 分。ODE Only 仅 68.77 分，说明 reward 阶段至关重要。

### 消融实验

- **Reward + Distillation**：同时使用蒸馏损失和 reward 损失联合训练，性能下降至 82.55。原因：蒸馏要求 student 模仿 teacher 输出分布，而 reward 鼓励生成偏离 teacher 的高分输出，两个目标冲突。需解耦 teacher 监督和 reward 优化
- **随机帧 vs 最后帧**：监督随机帧导致 dynamic degree 下降 10+ 个百分点，因为图像级 reward 鼓励静态内容
- **不同 teacher 模型**：Wan2.1-14B vs Wan2.1-1.3B 作为 teacher，性能无显著差异。可能因 1.3B student 的容量限制以及双向/自回归架构差异

## 关键启示

- **运动和纹理可以解耦训练**：少量 ODE 轨迹提供运动先验，reward 模型负责纹理质量，两阶段互不干扰
- **纯 reward 信号可替代重度蒸馏**：不需要持续的 teacher-student 分布匹配，reward 引导就能产生高质量纹理
- **蒸馏和 reward 目标冲突**：联合训练两者会相互破坏，应顺序执行而非并行
- **图像级 reward 应用于视频需要谨慎**：监督多帧会抑制运动，仅监督最后一帧是有效的折中
- **自回归模型的性能天花板不一定受限于 teacher**：通过 reward 引导，student 可在某些维度（如美学、动态度）超越 teacher
