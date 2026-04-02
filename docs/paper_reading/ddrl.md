---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
  - Flow Matching
  - GRPO
---

# DDRL: Data-regularized Reinforcement Learning for Diffusion Models at Scale

- 论文：https://arxiv.org/abs/2512.04332
- 团队：Stanford University, NVIDIA（Cosmos2.5 团队，含 James Zou、Stefano Ermon 等）

## 概述

DDRL（Data-regularized Diffusion RL）解决扩散模型 RL 对齐中的 reward hacking 问题。现有方法（DanceGRPO、FlowGRPO）使用在线策略的 KL 正则化，当策略偏离数据流形时正则信号失效，导致 reward 数值高但人类实际不偏好。DDRL 核心洞见：用**前向 KL 散度**替代反向 KL，将正则化锚定在离线数据分布上，数学上等价于在 RL 目标中直接加入标准 diffusion loss。在 Cosmos2.5-2B/14B 上消耗超 100 万 H100 GPU 小时验证，DDRL 是唯一同时提升 reward 和人类偏好投票的方法。

## 动机

标准 diffusion RL 目标为：

$$J_{\text{RL}}(p_\theta) = \mathbb{E}_{p_\theta(x_0|c)}\left[\frac{r(x_0,c)}{\beta}\right] - D_{\text{KL}}(p_\theta \| p_{\text{ref}})$$

KL 正则化在策略采样点上计算：

$$D_{\text{KL}}(p_\theta \| p_{\text{ref}}) = \mathbb{E}_{x_{0:T} \sim p_\theta} \sum_t D_{\text{KL}}(p_\theta(\cdot|x_t,c) \| p_{\text{ref}}(\cdot|x_t,c))$$

**根本问题**：$x_t$ 从 $p_\theta$ 采样，当 $p_\theta$ 被 reward 驱动偏移时，$x_t$ 进入 $p_{\text{ref}}$ 未训练过的 OOD 区域，参考模型的正则化信号变得不可靠。多步 Markov 链累积这种漂移，即使 KL 数值保持稳定仍出现噪声纹理和过度风格化。

控制 KL ≠ 控制生成质量，因为 KL 的计算点本身可以被策略操控到 OOD 区域。

## 方法

### DDRL 目标函数

将正则化从在线策略改为离线数据分布：

$$J_{\text{DDRL}}(p_\theta) = \mathbb{E}_{p_\theta(x_0|c)}\left[\lambda\left(\frac{r(x_0,c) - Z}{\beta}\right)\right] - D_{\text{KL}}(\tilde{p}_{\text{data}} \| p_\theta)$$

其中 $\lambda(x) = -\exp(-x)$ 是单调变换，$Z = \beta \log \mathbb{E}_{p_{\text{ref}}}[\exp(r/\beta)]$ 近似为组内平均奖励（类似 GRPO advantage）。

**关键等价**：前向 KL $D_{\text{KL}}(\tilde{p}_{\text{data}} \| p_\theta)$ 直接等价于标准 diffusion loss：

$$\tilde{J}_{\text{DDRL}} = \mathbb{E}_{p_\theta}\left[\lambda\left(\frac{r - Z}{\beta}\right)\right] - \mathcal{L}(\theta;\, \tilde{p}_{\text{data}}(x_0|c))$$

### 理论保证

**Theorem 3.1**：最优解为 $p^*_\theta(x_0|c) \propto \tilde{p}_{\text{data}}(x_0|c) \exp(r(x_0,c)/\beta)$，与传统 RL 目标的最优解一致，但正则化不依赖在线采样。

### 算法实现

每步优化：
1. 从数据集采样 $\tilde{x}_0$，从当前策略 rollout $N$ 条轨迹
2. 计算 advantage：$A_n = (r_n - \text{mean}) / (\beta \cdot \text{std})$
3. 对每个 timestep $t$，联合优化：

$$\mathcal{L}_t = \|\epsilon_\theta(\tilde{x}_t, t, c) - \epsilon\|^2 - A_n \log p_\theta(x_t^n | x_{t+1}^n, c)$$

前项 = diffusion loss（离线数据），后项 = RL reward（在线 rollout）。

**与标准方法的核心差异**：
- 正则项完全使用离线数据的 diffusion loss，不需要在线策略的 log-ratio
- 不需要训练时保留参考模型副本（节省显存）
- 可将 diffusion loss 每条数据只算一次（随机采样 $t$），NFE 与 DanceGRPO 持平

### SFT 与 RL 的统一

DDRL 目标本身就是 diffusion loss + reward maximization 的加权和。实验显示直接从预训练模型施加 DDRL（跳过 SFT）可达到与 SFT+DDRL 几乎相同的 diffusion loss（0.119 vs 0.121），数据效率大幅提升。

## 实验

### 训练配置

- 模型：Cosmos2.5-2B（256 H100）、Cosmos2.5-14B（1024 H100）
- 视频：720p/1080p，93 帧（16 FPS），Wan-2.1 VAE
- Rollout size N=8，batch size=16，AdamW（lr=1e-5 for 2B，3e-6 for 14B）
- 奖励模型：VideoAlign（基于 Qwen2-VL，3 子分数均值）+ VBench（5 子分数均值）

### 主要结果（T2V，Cosmos2.5-2B）

| 方法 | VideoAlign Reward | ΔVote vs DDRL |
|------|------------------|---------------|
| w/o RL | 0.408 | -22.9% |
| DanceGRPO | **0.715**（hacked） | -10.5% |
| FlowGRPO | 0.408（失败） | -6.7% |
| **DDRL** | 0.604 | **0** (baseline) |

- DanceGRPO 的 reward 最高（0.715）但人类偏好落后 DDRL 10.5%——典型 reward hacking
- FlowGRPO 在该设置下 reward 未提升（优化失败）
- DDRL 是唯一 reward 提升且人类偏好也领先的方法

### T2I 实验（SD3.5-Medium，OCR reward）

- DDRL OCR 0.823 vs DanceGRPO 0.846，但人类偏好 DDRL 领先 53%（ΔVote=-53%）
- DDRL 的 OOD reward 全面保持（ClipScore/PickScore/ImageReward 均不下降）
- 合成数据（从基础模型采样）也足够提供有效正则化

### 消融

- 训练 256 步 vs 128 步，reward 继续提升无 hacking
- Diffusion loss 每条数据只算一次效果持平，计算开销大幅降低
- Post-trained 模型的 diffusion loss 上升超过 10% 是 reward hacking 的强信号

## 关键启示

- **Reward hacking 的根因是在线正则化**：控制 KL 不等于控制生成质量，因为 KL 的计算点本身可被策略操控到 OOD 区域
- **前向 KL = diffusion loss 是核心数学洞见**：允许直接用真实/合成数据替代在线参考模型，省去训练时保留参考模型的显存开销
- **合成数据也足够正则化**：当无法获取真实训练数据时，基础模型采样的合成数据提供的背景/风格信息足以防止 OOD 退化
- **SFT 与 RL 可理论统一**：DDRL 目标天然包含 diffusion loss + reward 项，可跳过独立 SFT 阶段
- **Diffusion loss 上升是 reward hacking 的代理检测指标**：post-trained 模型的 diffusion loss 上升超 10% 时应立即预警
