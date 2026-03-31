---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
  - Flow Matching
  - GRPO
---

# SAGE-GRPO: Manifold-Aware Exploration for Reinforcement Learning in Video Generation

- 论文：https://arxiv.org/abs/2603.21872
- 团队：HKUST, Tencent Hunyuan

## 概述

SAGE-GRPO（Stable Alignment via Exploration）解决视频 GRPO 训练中的稳定性问题。现有视频 GRPO 方法（FlowGRPO、DanceGRPO）在 ODE-to-SDE 转换时使用一阶近似，在高噪声区域注入过多噪声能量，导致 rollout 质量下降、reward 估计不可靠。SAGE-GRPO 将问题形式化为流形约束的探索：在微观层面，推导精确的流形感知 SDE（带对数曲率修正），并引入梯度范数均衡器平衡各 timestep 的优化压力；在宏观层面，提出双重信赖域（周期性移动锚点 + 逐步约束）防止策略长期漂移。在 HunyuanVideo 1.5 上使用 VideoAlign 作为 reward，SAGE-GRPO 在 VQ、MQ、TA 和视觉指标上全面超越 DanceGRPO、FlowGRPO 和 CPS，用户偏好研究中胜率高达 80%+。

## 动机

视频 GRPO 远不如语言模型和图像领域可靠。根本原因：

1. **ODE-to-SDE 噪声注入过量**：FlowGRPO 和 DanceGRPO 使用 Euler 式离散化的一阶近似计算 SDE 噪声标准差，在高噪声步骤引入截断误差，注入多余噪声能量，将采样推离数据流形，产生时间抖动和伪影
2. **梯度跨 timestep 不平衡**：扩散过程中梯度范数在高噪声（$t \to 1$）时消失、低噪声（$t \to 0$）时爆炸，变化超过一个数量级，导致学习偏向特定阶段
3. **稳定性-可塑性困境**：固定 KL（锚定初始模型）限制最优策略可达性；逐步 KL（约束相邻步更新幅度）不约束累积位移，允许策略缓慢但持续地漂离流形

## 方法

### 核心视角：流形约束的探索

将预训练模型定义的合法视频空间视为数据流形 $\mathcal{M} \subset \mathbb{R}^D$，GRPO 的核心问题是：在提高 reward 的同时，约束探索始终在流形邻域内。

### 微观层面：精确 SDE + 梯度均衡

#### 精确流形感知 SDE

**第一阶段：从 ODE 到保边缘 SDE**

Rectified Flow 预训练的速度场 $v_\theta$ 定义了确定性 ODE 轨迹：

$$dx_t = v_\theta(x_t, t) dt$$

为 RL 探索注入布朗运动 $dw_t$，强度由扩散系数 $\varepsilon_t$ 控制。为抵消噪声引起的分布偏移，根据 Fokker-Planck 方程在漂移项中加入基于分数函数 $s_\theta(x_t)$ 的 Itô 校正，形成保边缘概率的 SDE：

$$dx_t = \left( v_\theta(x_t, t) - \frac{1}{2}\varepsilon_t^2 s_\theta(x_t) \right) dt + \varepsilon_t dw_t$$

流形感知的扩散系数：让注入噪声与 Flow 轨迹的信号/噪声几何收缩率匹配：

$$\varepsilon_t = \eta \sqrt{\frac{\sigma_t}{1-\sigma_t}}$$

**第二阶段：SDE 离散化与 $\Sigma_t$ 的本质**

对连续 SDE 在时间区间 $[\sigma_{t+1}, \sigma_t]$（步长 $\Delta t = \sigma_t - \sigma_{t+1}$）上求定积分：

$$x_{t+\Delta t} - x_t = \underbrace{\int_{\sigma_{t+1}}^{\sigma_t} v_\theta(x_s, s) ds}_{\text{速度场积分}} - \underbrace{\int_{\sigma_{t+1}}^{\sigma_t} \frac{1}{2}\varepsilon_s^2 s_\theta(x_s) ds}_{\text{Itô 校正项积分}} + \underbrace{\int_{\sigma_{t+1}}^{\sigma_t} \varepsilon_s dw_s}_{\text{随机噪声积分}}$$

确定性积分的欧拉近似：$\Delta t$ 足够短，$v_\theta$ 和 $s_\theta$ 视为常数。速度场积分 $\approx v_\theta(x_t, t) \Delta t$，Itô 校正项积分 $\approx -\frac{1}{2} s_\theta(x_t) \int_{\sigma_{t+1}}^{\sigma_t} \varepsilon_s^2 ds$。

随机噪声积分的精确处理：布朗运动微分 $dw_s \sim \mathcal{N}(0, ds)$，无数独立微小高斯增量的累加仍为高斯分布，均值为 0。根据 **Itô 等距同构定理**，独立高斯变量相加的总方差等于各部分方差的积分，即累积方差严格定义为：

$$\Sigma_t = \int_{\sigma_{t+1}}^{\sigma_t} \varepsilon_s^2 ds$$

因此随机噪声积分服从 $\mathcal{N}(0, \Sigma_t I)$，可用 $\Sigma_t^{1/2}\epsilon$（$\epsilon \sim \mathcal{N}(0, I)$）等效替换。注意 Itô 校正项的积分同样是 $\Sigma_t$，合并后得到离散更新公式：

$$x_{t+\Delta t} = x_t + v_\theta(x_t, t)\Delta t + \frac{\Sigma_t}{2}s_\theta(x_t) + \Sigma_t^{1/2}\epsilon$$

**第三阶段：精确求解 $\Sigma_t$**

代入 $\varepsilon_t = \eta\sqrt{\sigma_t/(1-\sigma_t)}$：

$$\Sigma_t = \eta^2 \int_{\sigma_{t+1}}^{\sigma_t} \frac{\sigma}{1-\sigma} d\sigma$$

加一减一分离被积函数：$\frac{\sigma}{1-\sigma} = -1 + \frac{1}{1-\sigma}$

求原函数（第二项令 $u = 1-\sigma$，$d\sigma = -du$）：

$$\int \left( -1 + \frac{1}{1-\sigma} \right) d\sigma = -\sigma - \ln(1-\sigma)$$

代入上下限（Newton-Leibniz）：

$$\Sigma_t = \eta^2 \left[ -\sigma - \ln(1-\sigma) \right]_{\sigma_{t+1}}^{\sigma_t} = \eta^2 \left[ -(\sigma_t - \sigma_{t+1}) + \ln\frac{1-\sigma_{t+1}}{1-\sigma_t} \right]$$

对数修正项 $\ln\frac{1-\sigma_{t+1}}{1-\sigma_t}$ 捕捉了信号系数 $(1-\sigma_t)$ 的几何收缩，线性近似无法表示。取平方根得到噪声标准差：

$$\Sigma_t^{1/2} = \eta\sqrt{-(\sigma_t - \sigma_{t+1}) + \ln\frac{1-\sigma_{t+1}}{1-\sigma_t}}$$

**对比现有方法的噪声标准差**：

- DanceGRPO：$\eta\sqrt{\sigma_t - \sigma_{t+1}}$
- FlowGRPO：$\eta\sqrt{\frac{\sigma_t}{1-\sigma_t}(\sigma_t - \sigma_{t+1})}$
- SAGE（精确）：$\eta\sqrt{-(\sigma_t - \sigma_{t+1}) + \ln\frac{1-\sigma_{t+1}}{1-\sigma_t}}$

精确 SDE 产生更小方差，探索区域紧贴流轨迹，而非传统一阶近似的大范围偏离流形。$\ln$ 曲率修正项解决了传统线性近似在高噪声区引发的方差爆炸问题。

#### 梯度范数均衡器

高斯转移分布 $\pi(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta, \Sigma_t I)$ 下：

$$\|\nabla_\mu \log \pi\| \propto \frac{1}{\Sigma_t^{1/2}}$$

导致低噪声 timestep 的梯度远大于高噪声 timestep。估计每个 timestep 的梯度尺度 $N_t$，用鲁棒归一化：

$$S_t = \frac{\text{Median}(\{N_\tau\}_{\tau=1}^T)}{N_t + \epsilon}$$

对每个 timestep 的梯度乘以 $S_t$，使结构性更新和纹理更新的贡献均等化。

### 宏观层面：双重信赖域

将 KL 散度解释为策略空间的动态锚定机制，设计 position-velocity 控制器：

**周期性移动锚点（Position Control）**：每 $N$ 步更新参考策略 $\pi_{\text{ref}} \leftarrow \pi_\theta$，约束策略不偏离最近的流形一致检查点：

$$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}_N}) = \frac{(\mu_\theta - \mu_{\text{ref}_N})^2}{2\Sigma_t^2}$$

**逐步约束（Velocity Control）**：约束相邻步之间的更新幅度：

$$D_{\text{KL}}(\pi_\theta \| \pi_{k-1})$$

**双重 KL 目标**：

$$\mathcal{L}_{\text{KL}} = \beta_{\text{pos}} \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}_N}) + \beta_{\text{vel}} \cdot D_{\text{KL}}(\pi_\theta \| \pi_{k-1})$$

Position 项防止长期漂移（类似多阶段松弛版 TRPO），Velocity 项平滑即时更新。相比之下：
- 固定 KL 太严格，随训练推进最优策略可能远离初始模型
- 仅逐步 KL 不约束累积位移，允许缓慢但持续的流形偏离

## 实验

### 训练配置

- 基础模型：HunyuanVideo 1.5
- 视频：81 帧，每 20 去噪步做一次 GRPO 更新
- 有效 batch size：8（per-GPU 2 × 4 梯度累积）
- Reward：VideoAlign（VQ + MQ + TA），冻结不微调
- KL 权重调度：$\lambda_{\text{KL}} \in [10^{-7}, 10^{-5}]$，两阶段递增

### 主要结果

两种 reward 配置：Setting A（均等权重 $w_{vq}=w_{mq}=w_{ta}=1.0$）和 Setting B（对齐侧重 $w_{vq}=0.5, w_{mq}=0.5, w_{ta}=1.0$）。

Setting B（对齐侧重，最佳效果）：

| 方法 | VQ | MQ | TA | CLIPScore | PickScore |
|------|-----|------|------|-----------|-----------|
| HunyuanVideo 1.5 | 0.0654 | -0.7539 | -0.5870 | 1.4063 | 0.5409 |
| DanceGRPO + Fixed KL | 0.1290 | -0.7739 | -0.5083 | 1.4112 | 0.5452 |
| FlowGRPO + Fixed KL | 0.2103 | -0.6654 | -0.5506 | 1.4263 | 0.5427 |
| CPS + Fixed KL | 0.3705 | -0.6121 | -0.4787 | 1.4613 | 0.5458 |
| SAGE-GRPO + Dual Mov KL | **0.8066** | **-0.4765** | **-0.2384** | **1.5216** | **0.5484** |

SAGE-GRPO 在所有指标上显著领先，VQ 达到 0.8066 vs 次优 CPS 的 0.3705。

### 用户偏好研究

29 位评估者，32 个 prompt，SAGE-GRPO 对比各基线的胜率：

| 对比基线 | 视觉质量 | 运动质量 | 语义对齐 |
|----------|---------|---------|---------|
| vs DanceGRPO | 85.9% | 75.8% | 79.2% |
| vs FlowGRPO | 83.8% | 79.2% | 71.9% |
| vs CPS | 80.2% | 70.8% | 67.9% |

### 消融实验

- **梯度范数均衡器**：对所有 SDE 公式（DanceGRPO、FlowGRPO、CPS、SAGE）均有效。无均衡时 reward 曲线不稳定或停滞；有均衡时曲线平滑且持续上升，梯度尺度变化从超过一个数量级缩小到小常数倍
- **KL 策略**：Dual Moving KL 在收敛速度和最终 reward 上均最优。Moving KL 早期探索充分但后期衰减；Dual Moving KL 全程保持更高且稳定的探索水平
- **KL 权重调度**：两阶段 $10^{-7} \to 10^{-5}$ 效果最佳，支持渐进收紧信赖域的策略
- **Reward 配置**：侧重语义对齐（Setting B）比均等权重（Setting A）更能抑制 reward hacking，产生更稳定的增益

## 关键启示

- **视频 GRPO 的核心瓶颈是 SDE 探索噪声过量**：一阶近似在高噪声区域注入多余能量，推离数据流形。精确积分 + 对数曲率修正是直接改进
- **梯度范数均衡器是通用组件**：对所有 SDE 公式和 KL 策略均有效，因为它解决的是扩散过程固有的信噪比不平衡问题
- **周期性移动锚点是 Fixed KL 和 Step-wise KL 的有效折中**：position control（防漂移）+ velocity control（防突变）的 dual 设计，兼顾稳定性和可塑性
- **侧重语义对齐的 reward 配置可减少 reward hacking**：均等权重下加 KL 正则通常改善视觉但恶化 reward，而侧重对齐则同时改善两者
