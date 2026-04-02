---
tags:
  - Reinforcement Learning
  - GRPO
  - Flow Matching
  - Diffusion Model
---

# CPS: Coefficients-Preserving Sampling for Flow Matching RL

- 论文：https://arxiv.org/abs/2509.05952
- 团队：CreateAI

## 概述

CPS 解决 Flow-GRPO/DanceGRPO 中 SDE 采样引入严重噪声伪影的问题。理论分析揭示根因：Flow-SDE 推导中的 Taylor 展开在 $t \to 0$ 时引入 $1/t$ 数值不稳定项，导致注入的噪声超出调度器预设水平，积累形成非零终端噪声。受 DDIM 启发，CPS 重新设计采样过程确保每个时间步的信号-噪声系数严格保持调度器约束，消除噪声伪影。在 FLUX.1-dev 上 PickScore 从 23.90（Flow-GRPO）提升到 24.25；HPSv2 从 0.364（DanceGRPO）提升到 0.377。

## 动机

- Flow-GRPO 和 DanceGRPO 将 Flow Matching 的确定性 ODE 转为 SDE 引入随机性用于 RL 探索
- **关键发现**：SDE 采样在训练时产生显著噪声伪影（Fig. 1），噪声强度随 $\sigma$ 增加而加剧
- 噪声样本送入 reward 模型导致评分不准确（审美/人类偏好模型误判噪声伪影为"纹理细节"），误导训练
- 问题在 few-step 模型（如 FLUX.1-schnell 4 步）上尤为严重

## 分析

### Flow-SDE 的噪声问题

Flow-GRPO 的 SDE 更新规则：

$$x_{t+\Delta t} = x_t + \left[v_\theta(x_t, t) + \frac{\sigma_t^2}{2t}(x_t + (1-t)v_\theta(x_t, t))\right]\Delta t + \sigma_t\sqrt{\Delta t}\,\epsilon$$

**根因**：推导过程使用 Taylor 展开近似 score function，引入 $1/t$ 项。当 $\Delta t$ 不趋近于零（实际采样步数 4-20）时：
- 近似误差显著
- $t \to 0$ 时数值不稳定
- 注入噪声量超出 ODE 调度器预设，累积导致终端噪声非零

### Coefficients-Preserving 原则

在采样过程中，任意时间步 $t$ 的潜变量应满足：

$$x_t = \alpha_t \hat{x}_0 + \sigma_t \epsilon$$

其中 $\alpha_t$ 和 $\sigma_t$ 由调度器唯一确定。Flow-SDE 违反此约束（噪声系数偏离调度器）。

## 方法

受 DDIM 启发，CPS 重新推导采样公式：
- 从当前 $x_t$ 和模型预测 $\hat{v}_\theta$ 估计干净样本 $\hat{x}_0$
- 使用调度器的精确系数 $(\alpha_{t+\Delta t}, \sigma_{t+\Delta t})$ 重建下一步 $x_{t+\Delta t}$
- 随机性通过重采样 $\epsilon$ 引入（控制信号-噪声分解），而非叠加额外噪声

保证每步的系数严格保持，消除噪声积累。当 $\sigma_t = 0$ 退化为 DDIM（确定性）。

## 实验

### GenEval（SD3.5-M）

| 方法 | Overall |
|------|:---:|
| SD3.5-M 基线 | 0.63 |
| + Flow-GRPO w/ KL | 0.97 |
| + Flow-CPS w/ KL | **0.97** |

GenEval 上两者持平（验证 CPS 不损失对齐能力）。

### PickScore（FLUX.1）

| 方法 | PickScore |
|------|:---:|
| FLUX.1-schnell 基线 | 21.86 |
| + Flow-GRPO | 23.39 |
| + Flow-CPS | **23.78** |
| FLUX.1-dev 基线 | 22.06 |
| + Flow-GRPO | 23.90 |
| + Flow-CPS | **24.25** |

### HPSv2（FLUX.1-schnell）

| 方法 | HPSv2 |
|------|:---:|
| 基线 | 0.304 |
| + Dance-GRPO | 0.364 |
| + Flow-CPS | **0.377** |

### OCR（SD3.5-M）

| 方法 | OCR |
|------|:---:|
| 基线 | 0.579 |
| + Flow-GRPO | 0.966 |
| + Flow-CPS | **0.975** |

- 审美/人类偏好 reward（PickScore、HPSv2）上 CPS 收敛更快、终值更高
- 检测类 reward（GenEval、OCR）上 CPS 收敛更快，终值相当或略优

## 关键启示

- **SDE 噪声伪影是 reward hacking 的隐性来源**：审美/偏好 reward 模型将噪声误判为有益特征，导致模型学会"加噪声提分"
- **Coefficients-Preserving 是 SDE 采样的必要约束**：确保每步信号-噪声系数与调度器一致，消除 ODE→SDE 转换的副作用
- **Few-step 模型更脆弱**：$\Delta t$ 越大，Taylor 近似误差越严重，CPS 在低步数场景收益更大
- **方法正交且易集成**：CPS 可直接替换 Flow-GRPO/DanceGRPO 的 SDE 采样模块，无需修改其余训练流程
