---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
  - DPO
  - Flow Matching
---

# RealDPO: Aligning Video Diffusion Models with Real-World Preference

- 论文：https://arxiv.org/abs/2510.14955
- 团队：Shanghai AI Lab, NTU

## 概述

RealDPO 提出直接使用真实视频作为 DPO 的 win 样本，模型生成视频作为 lose 样本，从而将真实世界视频的质量分布作为对齐目标。核心动机是真实视频天然具有最优的运动合理性、物理一致性和视觉质量，无需复杂的 reward 模型或人工标注。方法上将 DPO loss 适配到 DiT/Flow Matching 的 latent space，引入 EMA 参考模型更新策略（$\omega = 0.996$）稳定训练。构建 RealAction-5K 数据集（5k 真实-生成视频对），在 CogVideoX-5B 上验证，显著提升视频质量。

## 动机

- 现有视频 DPO 方法依赖 reward 模型或人工标注构建偏好对，成本高且引入 reward 偏差
- 模型生成的 win 样本仍受限于模型自身能力上限，无法突破模型分布的质量天花板
- 真实视频天然满足物理规律、运动合理性、时间一致性，是最理想的 win 样本来源
- 核心问题：真实视频与模型生成视频的分布差异较大，直接做 DPO 训练不稳定

## 方法

### Latent-Space DPO for DiT/Flow Matching

将 DPO loss 从像素空间适配到 latent space，兼容 DiT 架构和 Flow Matching 训练范式：

- 标准 DPO 的偏好概率通过 latent denoising loss 差值计算
- 对于 Flow Matching 模型，loss 基于 velocity prediction 的 MSE：

$$\mathcal{L}_{\text{DPO}} = -\log \sigma \left( \beta \left[ \sum_t \left( \|\epsilon_\theta(x_t^l, t) - v^l\|^2 - \|\epsilon_{\text{ref}}(x_t^l, t) - v^l\|^2 \right) - \sum_t \left( \|\epsilon_\theta(x_t^w, t) - v^w\|^2 - \|\epsilon_{\text{ref}}(x_t^w, t) - v^w\|^2 \right) \right] \right)$$

其中 $x^w$ 为真实视频编码的 latent，$x^l$ 为模型生成视频的 latent，$v$ 为目标 velocity。

### EMA 参考模型

- 训练过程中参考模型 $\theta_{\text{ref}}$ 通过 EMA 更新而非固定：

$$\theta_{\text{ref}} \leftarrow \omega \cdot \theta_{\text{ref}} + (1 - \omega) \cdot \theta$$

- $\omega = 0.996$，缓慢跟踪策略模型变化
- 动机：真实视频与模型生成视频的分布差异导致固定参考模型训练不稳定，EMA 平滑过渡

### RealAction-5K 数据集

- 5,000 个真实-生成视频对
- 真实视频来源：高质量动作视频数据集
- 生成视频：使用 CogVideoX-5B 基于真实视频对应的 prompt 生成
- 每个真实视频配对一个生成视频，真实为 win、生成为 lose

## 实验

- **基础模型**：CogVideoX-5B
- **训练数据**：RealAction-5K（5k 对）
- **评估指标**：VBench 各维度、人类偏好评估
- RealDPO 在运动质量、时间一致性、视觉保真度上均优于基础模型
- 与传统 DPO（模型生成 win/lose 对）相比，RealDPO 在物理合理性维度提升更显著
- EMA 参考模型相比固定参考模型，训练更稳定、最终质量更高
- 消融实验验证 $\omega = 0.996$ 是最优选择，过大（0.999）更新太慢，过小（0.99）训练不稳定

## 关键启示

- **真实视频作为天然上界**：绕过 reward 模型和人工标注，直接以真实视频质量为对齐目标，思路简洁有效
- **EMA 参考模型解决分布差异**：真实-生成视频的 latent 分布差异是核心挑战，EMA 缓慢更新是关键稳定化手段
- **小数据 DPO 的有效性**：仅 5k 对即可显著提升，再次验证高质量偏好数据比规模更重要
- **局限性**：真实视频的 prompt 匹配度有限（真实视频的内容不完全受 prompt 控制），可能限制 prompt 跟随能力的提升
