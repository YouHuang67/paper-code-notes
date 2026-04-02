---
tags:
  - Video Generation
  - Reinforcement Learning
  - GRPO
  - Flow Matching
---

# AR-Drag: Real-Time Motion-Controllable Autoregressive Video Diffusion

- 论文：https://arxiv.org/abs/2510.08131
- 团队：NTU, Xmax.AI, ZJU, SMU

## 概述

AR-Drag 是首个 RL 增强的 few-step 自回归视频扩散模型，实现实时 I2V 运动控制。核心解决两个问题：(1) AR 训练中 teacher forcing 导致的 train-test mismatch（破坏 MDP 性质），通过 Self-Rollout 逐步在自生成历史上训练；(2) 长决策链探索成本，通过 selective stochasticity（仅随机一步 SDE，其余 ODE）降低方差。设计轨迹 reward 模型评估运动对齐度。仅 1.3B 参数，延迟 0.44s（Tora 176.51s 的 <1%），FID 28.98（最优），FVD 187.49。

## 动机

- 双向 VDM（Tora、MagicMotion）延迟极高（68-1426s），无法实时交互
- 现有 AR VDM 仅支持 T2V 或简单控制信号，few-step 模型存在质量退化和运动伪影
- 将 GRPO 应用于 AR 视频生成面临三个障碍：
  - Markov 性质：典型 AR 训练用 teacher forcing 条件化 ground-truth 帧，而非自生成帧
  - 长决策过程：$M$ 帧 × $N$ 步去噪，全链 SDE 方差过高
  - 缺乏可控视频生成的 reward 模型

## 方法

### Step 1: 构建实时 AR 基础模型

**数据整理**：收集含多样运动的真实和合成视频，用关键点检测器自动生成轨迹控制信号，人工验证过滤。

**双向 fine-tuning**：在 Wan2.1-1.3B-I2V 上用三种控制信号训练——轨迹嵌入 $c_m^{\text{traj}}$（VAE 编码坐标热力图）、文本 $c_{\text{text}}$、首帧参考图 $c_{\text{ref}}$（仅 $m=0$）。

**蒸馏为 few-step AR**：用 DMD + 对抗损失将多步双向教师蒸馏为 3 步因果学生（双向注意力→因果注意力）。

**Self-Rollout（Markovize AR 训练）**：维护 KV cache 存储先前去噪帧。训练时从纯噪声开始逐帧去噪：对第 $m$ 帧随机采样步 $n$，逐步去噪从 $x_{m,0}$ 到 $x_{m,n}$ 计算损失，继续去噪到 $x_{m,N}$ 更新 KV cache。后续帧条件化于自生成 cache 而非 ground-truth。

与 Self-Forcing 的区别：Self-Forcing 将 $x_{m,n}$ 到 $x_{m,N}$ 折叠为单步，Self-Rollout 逐步完成，更忠实匹配推理动态。

### Step 2: AR VDM 上的 GRPO

**MDP 建模**：
- 状态 $s_{m,n} = (c_m, t_n, X_{m,n})$
- 动作 $a_{m,n} = x_{m,n+1} \sim p_\theta(\cdot | c_m, t_n, X_{m,n})$
- Reward 仅在帧完成去噪时给出：$R(s_{m,n}, a_{m,n}) = \mathbb{1}[n=N] \cdot (R_{\text{quality}} + R_{\text{motion}})$

**GRPO 目标**：采样 $G$ 个视频，帧级组归一化优势：

$$\hat{A}_{m,n}^{(i)} = \frac{R(x_{m,N}^{(i)}, c_m) - \text{mean}(\{R(x_{m,N}^{(j)}, c_m)\}_{j=1}^G)}{\text{std}(\{R(x_{m,N}^{(j)}, c_m)\}_{j=1}^G)}$$

**Selective Stochasticity**：每帧仅随机选一步 $\tilde{n}$ 用 SDE，其余用确定性 ODE。在保持探索的同时将有效 horizon 缩短 5-20×。

**Reward 设计**：
- 质量 reward：$R_{\text{quality}} = f_{\text{AQ}}(x_{m,N})$（LAION Aesthetic Quality Predictor，1-5 分）
- 运动 reward：$R_{\text{motion}} = \lambda \max(0, \alpha - \|\hat{c}_m^{\text{traj}} - c_m^{\text{traj}}\|_2^2)$（Co-Tracker 估计实际轨迹与目标轨迹的 L2 距离）

训练配置：Wan2.1-1.3B-I2V，3 步去噪，AdamW LR $1 \times 10^{-5}$，8×H20 GPU，KV cache 7 帧

## 实验

### 与运动可控 VDM 对比

| 方法 | Latency (s) | FID | FVD | Aesthetic | Motion Smooth | Motion Consist |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| DragNUWA | 94.26 | 36.31 | 376.39 | 3.30 | 0.9759 | 3.71 |
| DragAnything | 68.76 | 38.13 | 367.74 | 3.22 | 0.9811 | 3.63 |
| Tora | 176.51 | 32.84 | 283.43 | 3.86 | 0.9855 | 3.97 |
| MagicMotion | 1426.37 | 30.04 | 230.53 | 4.01 | 0.9871 | 3.95 |
| Self-Forcing | 0.95 | 34.47 | 315.87 | 3.70 | 0.9920 | 4.06 |
| **AR-Drag** | **0.44** | **28.98** | **187.49** | **4.07** | **0.9948** | **4.37** |

- 延迟 0.44s（Tora 的 <1%，Self-Forcing 的 46%）
- 所有质量和运动指标均最优，包括超越 5B 参数的 MagicMotion

### 消融

| 方法 | FID | FVD | Aesthetic | Motion Consist |
|------|:---:|:---:|:---:|:---:|
| AR-Drag | 28.98 | 187.49 | 4.07 | 4.37 |
| w/o RL | 31.65 | 210.35 | 3.92 | 4.12 |
| w/o Self-Rollout | 38.13 | 353.75 | 3.38 | 4.02 |
| Initial model | 35.94 | 303.16 | 3.84 | 3.22 |
| Teacher model | 29.38 | 151.46 | 4.15 | 4.36 |

- RL 训练贡献：FID -2.67，Motion Consistency +0.25
- Self-Rollout 极关键：去掉后 FVD 从 187.49 崩溃到 353.75（train-test mismatch 的严重后果）
- AR 学生 vs 双向教师：延迟从 45.64s 降到 0.44s（100×），FID 仅差 0.4

## 关键启示

- **Self-Rollout 是 AR 视频 RL 的前提**：teacher forcing 破坏 MDP，Self-Rollout 通过逐步在自生成历史上训练消除 train-test 分布偏移
- **Selective stochasticity 平衡探索与效率**：每帧仅一步 SDE 足够提供 GRPO 所需随机性，避免全 SDE 的高方差问题
- **Few-step AR 模型可通过 RL 弥补蒸馏损失**：RL 后训练使 3 步学生在 FID 上接近甚至超越多步教师，同时保持 100× 速度优势
- **帧级 reward 设计**：与全序列 reward 不同，每帧独立给 reward 实现精确归因，适合 AR 的逐帧生成范式
