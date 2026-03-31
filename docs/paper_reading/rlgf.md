---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
---

# RLGF: Reinforcement Learning with Geometric Feedback for Autonomous Driving Video Generation

- 论文：https://arxiv.org/abs/2509.16500
- 团队：University of Macau, Li Auto Inc.
- 发表：NeurIPS 2025

## 概述

RLGF 针对自动驾驶视频生成中的几何失真问题，提出用感知模型驱动的几何反馈来强化学习优化视频扩散模型。现有视频生成模型虽然视觉逼真（2D 检测 mAP 43.8 vs 真实数据 44.7），但存在严重的 3D 几何失真（3D 检测 mAP 25.7 vs 真实数据 35.5）——消失点偏移、车道拓扑不一致、深度错误。RLGF 包含两个核心组件：(1) Latent-Space Windowing Optimization，在扩散去噪过程的随机滑动窗口内提供中间步反馈，避免全链路反向传播；(2) Hierarchical Geometric Reward (HGR)，基于两个 latent 空间感知模型（几何感知 Pgeo + 占据预测 Pocc），提供点-线-面-体素-特征五级几何奖励。应用于 DiVE 后，消失点误差降低 21%，深度误差降低 57%，3D 检测 mAP 提升 +5.67（25.75 → 31.42），显著缩小与真实数据的差距。

## 动机

当前自动驾驶视频生成模型的核心矛盾：**视觉逼真但几何失真**。

- 2D 检测几乎无差距：YOLOv5 在合成数据上 mAP 43.8，真实数据 44.7
- 3D 检测差距巨大：BEVFusion 在合成数据上 mAP 25.7，真实数据 35.5

根源：像素级 MSE 损失将每个像素独立处理，无法建模高阶几何关系（如透视一致性）；条件注入仅强制局部像素对齐，不保证全局 3D 场景结构的合理性。

为量化几何失真，论文提出 GeoScores 指标套件：消失点误差（VP）、车道拓扑分数（Lane F1）、深度误差（Depth RMSE），通过对比合成视频和真实视频在感知模型上的输出差异来衡量。

## 方法

### Latent-Space Windowing Optimization

扩散模型去噪过程中几何结构逐步形成：早期步骤建立粗粒度全局几何，晚期步骤细化局部细节。直接对全 $T$ 步去噪链做反向传播计算开销不可承受。

策略：在 $T$ 步采样过程 $z_T \to z_0$ 中，随机采样起始步 $t'$，在长度为 $w$ 的滑动窗口 $[t'-w, t']$ 内应用 reward。感知模型以 $z_k$（窗口末端的噪声 latent）和步数 $k$ 为输入，直接在 latent 空间评估几何质量。

$$J_{\text{practical}}(\theta_{\text{LoRA}}) = \mathbb{E}_{c,v,z_k}[R(z_k, z_v)]$$

梯度仅通过窗口内的步骤反传：

$$\nabla_{\theta_{\text{LoRA}}} R_k(z_k, z_v) = \frac{\partial R_k}{\partial z_k} \cdot \frac{\partial z_k}{\partial \theta_{\text{LoRA}}}$$

窗口大小 $w=5$，起始步 $t'$ 从 $[8, 30]$ 随机采样（总步数 $T=30$），覆盖早期全局结构和晚期细节细化。

### Micro-Decode Module

为避免在每个中间步做完整 VAE 解码，用 VAE 解码器的前几层浅层构建轻量 Micro-Decode 模块 $F_{\text{micro}}$，将噪声 latent $z_k$ 和 timestep $k$（Fourier embedding）转换为增强特征 $f_k^f$，供下游感知模型使用。

### Latent 空间感知模型

**Pgeo（几何感知模型）**：DINOv2-ViT-S/14 骨干（从 Depth Anything V2 初始化），多任务头：
- 消失点检测：热力图回归，MSE 损失
- 车道解析：拓扑感知分割
- 深度估计：SiLog 损失

在 latent 空间即可达到接近像素空间模型的性能（VP NormDist 0.024 vs 像素空间 VPD 的 0.032，Lane F1 0.865 vs PriorLane 的 0.879）。

**Pocc（占据预测模型）**：从帧序列特征推断 3D 占据网格 $O_i \in \mathbb{R}^{X \times Y \times Z}$，latent 空间 mIoU 29.96（vs 像素空间 FlashOcc 的 32.08）。

### Hierarchical Geometric Reward (HGR)

总 reward $R = R_{\text{geo}} + R_{\text{occ}}$，五级反馈：

**几何 reward**（点-线-面）：

$$R_{\text{geo}} = \lambda_{\text{vp}} r_{\text{vp}} + \lambda_{\text{lane}} r_{\text{lane}} + \lambda_{\text{depth}} r_{\text{depth}}$$

- 点级（消失点）：$r_{\text{vp}} = -\|p_{\text{vp}} - v_{\text{ref}}\|_2^2$
- 线级（车道）：$r_{\text{lane}} = \text{F1-Score}(L_{\text{pred}}, L_{\text{ref}})$
- 面级（深度）：$r_{\text{depth}} = -(\sum(D_{\text{pred}} \odot M_{\text{road}} - D_{\text{ref}} \odot M_{\text{road}})^2 + \sum(D_{\text{pred}} \odot M_{\text{vehicle}} - D_{\text{ref}} \odot M_{\text{vehicle}})^2)$

**占据 reward**（体素+特征）：

$$R_{\text{occ}} = r_{\text{align}} + r_{\text{iou}}$$

- 特征级：$r_{\text{align}} = -D_{\text{KL}}(p(\text{feat}_{\text{occ}}^{\text{real}}) \| p(\text{feat}_{\text{occ}}^{\text{gen}}))$，对齐中间占据特征分布
- 体素级：$r_{\text{iou}} = \text{IoU}(O^{\text{gen}}, O^{\text{real}})$，最大化 3D 占据网格重叠

## 实验

### 训练配置

- 基线模型：DiVE、MagicDrive-V2
- LoRA rank 16，应用于 DiT 的 attention 层（Q/K/V）
- 窗口大小 $w=5$，起始步范围 $[8, 30]$
- Reward 权重：$\lambda_{\text{vp}}=0.1, \lambda_{\text{lane}}=0.1, \lambda_{\text{depth}}=0.5$
- AdamW，lr $1 \times 10^{-4}$，batch size 1（8 帧/clip）

### 主要结果

| 方法 | FVD ↓ | 3D mAP ↑ | 3D NDS ↑ | VP ↓ | Lane F1 ↑ | Depth ↓ |
|------|-------|----------|----------|------|-----------|---------|
| Real Data | - | 35.53 | 41.20 | - | - | - |
| DiVE | 68.4 | 25.75 | 33.61 | 0.086 | 0.792 | 1.822 |
| DiVE + RLGF | 67.6 | 31.42 | 36.07 | 0.068 | 0.879 | 0.772 |
| MagicDrive-V2 | 101.2 | 18.95 | 21.10 | 0.092 | 0.787 | 1.732 |
| MagicDrive-V2 + RLGF | 99.8 | 23.21 | 27.80 | 0.079 | 0.854 | 0.983 |

RLGF 作为即插即用模块在两个基线上均显著提升几何保真度，同时不损害视觉质量（FVD 甚至略有改善）。DiVE + RLGF 的 3D mAP 从 25.75 提升至 31.42，接近真实数据的 35.53。

### 消融实验

| HGR 组件 | 3D mAP ↑ | 3D NDS ↑ |
|---------|----------|----------|
| DiVE 基线 | 25.75 | 33.61 |
| + $r_{\text{vp}}$ | 26.31 | 33.66 |
| + $r_{\text{vp}} + r_{\text{lane}}$ | 26.93 | 33.98 |
| + $r_{\text{vp}} + r_{\text{lane}} + r_{\text{depth}}$ | 27.12 | 34.82 |
| + $r_{\text{align}} + r_{\text{iou}}$（仅占据） | 28.06 | 35.11 |
| 全部五级 reward | **31.42** | **36.07** |

五级 reward 协同效应显著——全部组合（31.42）远超各子集之和。占据 reward 单独使用（28.06）已超过几何 reward 的三级组合（27.12），说明 3D 场景级反馈比 2.5D 几何反馈更强。

### 超参数消融

| 超参数 | 值 | 3D mAP |
|--------|-----|--------|
| 窗口大小 | 3 / **5** / 8 | 30.89 / **31.42** / 31.25 |
| 窗口位置 | 早期 [20,30] / **中全 [8,30]** / 晚期 [1,15] | 30.55 / **31.42** / 29.91 |

覆盖全范围（早期+晚期）的窗口位置策略最优，仅监督晚期（细节阶段）效果最差。

## 关键启示

- **视频生成的几何保真度是独立于视觉质量的关键维度**：2D 外观几乎无损但 3D 几何严重失真，像素级损失无法捕捉高阶几何关系
- **Latent 空间感知模型可替代像素空间模型提供 reward**：Micro-Decode + DINOv2 在 latent 空间达到接近像素空间的感知精度，避免 VAE 解码开销
- **分层多粒度 reward 的协同效应远超各组件之和**：点-线-面-特征-体素五级反馈从不同尺度约束几何一致性，全组合效果 >> 部分组合
- **滑动窗口策略平衡效率和覆盖范围**：随机采样窗口位置覆盖早期（全局结构）和晚期（局部细节），$w=5$ 是效率和效果的最优平衡点
- **3D 占据反馈比 2.5D 几何反馈更强**：场景级体积信息直接约束物体布局和动态，比消失点/车道/深度等 2.5D 线索更有效
