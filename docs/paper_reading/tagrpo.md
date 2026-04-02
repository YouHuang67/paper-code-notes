---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
  - GRPO
  - Flow Matching
---

# TAGRPO: Trajectory-Aligned GRPO for Diffusion Model Post-Training

- 论文：https://arxiv.org/abs/2601.05729
- 团队：HKU, Tencent Hunyuan

## 概述

TAGRPO 针对 DanceGRPO 在 Image-to-Video（I2V）任务上失效的问题，提出轨迹对齐的 GRPO 变体。核心发现：DanceGRPO 在 HunyuanVideo-1.5 的 I2V 任务上，HPSv3 指标反而下降，原因是标准 GRPO 的重要性比率仅在单一 timestep 内计算，忽略了去噪轨迹的整体一致性。TAGRPO 引入跨轨迹重要性比率（cross-trajectory importance ratios），通过 memory bank 存储历史 rollout 提升采样效率，并仅在高噪声 timestep 进行优化（低噪声 timestep 的梯度信号弱且噪声大）。

## 动机

- DanceGRPO 是视频生成 GRPO 的代表性工作，但在 I2V 任务上失效：
  - 在 HunyuanVideo-1.5 上，DanceGRPO 训练后 HPSv3 反而下降
  - T2V 任务上有效但 I2V 无效，暗示方法本身存在局限性
- 根因分析：
  - DanceGRPO 在每个 timestep $t$ 独立计算重要性比率 $\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$
  - 这种逐 timestep 独立计算忽略了去噪轨迹的整体一致性
  - I2V 任务中条件图像约束更强，轨迹偏差的影响被放大
- 另一个效率问题：GRPO 需要多次完整去噪 rollout，计算开销巨大

## 方法

### 轨迹对齐损失（Trajectory Alignment Loss）

标准 GRPO 的重要性比率仅考虑单一 timestep 的自身转移：

$$r_t^i = \frac{\pi_\theta(x_{t-1}^i | x_t^i, c)}{\pi_{\theta_{\text{old}}}(x_{t-1}^i | x_t^i, c)}$$

TAGRPO 的核心创新：将组内最高奖励轨迹 $x^+$ 和最低奖励轨迹 $x^-$ 作为锚点，每个样本同时学习向正锚靠拢、远离负锚。引入跨轨迹重要性比率：

$$r_t^{i,+} = \frac{\pi_\theta(x_{t-1}^+ | x_t^i, c)}{\pi_{\theta_{\text{old}}}(x_{t-1}^+ | x_t^i, c)}, \quad r_t^{i,-} = \frac{\pi_\theta(x_{t-1}^- | x_t^i, c)}{\pi_{\theta_{\text{old}}}(x_{t-1}^- | x_t^i, c)}$$

轨迹对齐 loss：

$$J_{\text{align}} = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{T} \sum_{t=0}^{T-1} \left[ \min(r_t^{i,+} \hat{A}^+, \text{clip}(r_t^{i,+}, 1-\epsilon, 1+\epsilon) \hat{A}^+) + \min(r_t^{i,-} \hat{A}^-, \text{clip}(r_t^{i,-}, 1-\epsilon, 1+\epsilon) \hat{A}^-) \right]$$

最终目标函数：

$$J_{\text{TAGRPO}} = J_{\text{GRPO}} + \gamma \cdot J_{\text{align}}$$

$\gamma = 1$（实验确定）。关键区别：标准 GRPO 仅让每个样本学习自己的重要性比率，TAGRPO 额外引入跨样本的轨迹对齐信号，利用 I2V 中同一条件图像下样本的结构相似性。

### Memory Bank 机制

借鉴对比学习，维护 FIFO 缓冲区存储历史 rollout：

- 存储：latent 表示 $b_i$ 和对应奖励 $R_i$
- 新迭代复用历史数据作为额外的正/负锚点，丰富对齐信号
- FIFO 刷新保持数据新鲜度
- 消融实验显示 memory bank 贡献约 60% 的增益

### 高噪声 Timestep 优化

- Wan 2.2：仅优化高噪声区域（冻结低噪声模型）
- HunyuanVideo-1.5：仅优化 $t > 900$ 的 timestep
- 理由：高噪声 timestep 决定全局结构和语义，是优化的关键阶段；低噪声阶段梯度信号弱且方差大

## 实验

- **基础模型**：Wan 2.2, HunyuanVideo-1.5（I2V）
- **Reward 模型**：Q-Save（VQ + DQ + IA 综合分）、HPSv3（采样 2 帧/秒）
- **训练数据**：约 10k image-text 对，TAGRPO-Bench 200 个挑战性 I2V 样本
- **超参**：组大小 $G=8$，训练分辨率 320p / 53 帧，推理步数 16，CFG 3.5，$\epsilon=0.2$

### I2V 主要结果

| 模型 | 方法 | HPSv3 (320p) | HPSv3 (720p) | Q-Save (320p) | Q-Save (720p) |
|------|------|:---:|:---:|:---:|:---:|
| HunyuanVideo-1.5 | Baseline | 2.00 | 4.42 | 8.01 | 10.02 |
| HunyuanVideo-1.5 | DanceGRPO | 1.84 | 4.33 | 8.01 | 10.02 |
| HunyuanVideo-1.5 | TAGRPO | **2.41** | **4.58** | **8.05** | **10.05** |
| Wan 2.2 | Baseline | 3.63 | 4.34 | 8.73 | 10.13 |
| Wan 2.2 | DanceGRPO | 3.70 | 4.40 | 8.75 | 10.17 |
| Wan 2.2 | TAGRPO | **4.29** | **5.03** | **8.81** | **10.26** |

- DanceGRPO 在 HunyuanVideo-1.5 上 HPSv3 从 2.00 下降到 1.84（确认失效）
- TAGRPO 在 HunyuanVideo-1.5 上 HPSv3 提升至 2.41（+31%）
- 320p 训练直接迁移到 720p，无需额外训练

### T2V 泛化结果

- Wan 2.2 T2V：HPSv3 达 6.27 vs DanceGRPO 5.42（+15.7%）
- HunyuanVideo-1.5 T2V：HPSv3 达 1.99 vs DanceGRPO 1.29（+54.3%）
- TAGRPO 不仅修复 I2V 失效，在 T2V 上也全面超越 DanceGRPO

### 消融实验

- 去掉 $J_{\text{align}}$：Q-Save 从 8.66 降至 8.48（-0.18）
- 去掉 memory bank：Q-Save 从 8.66 降至 8.55（-0.11）
- Memory bank 贡献约 60%，$J_{\text{align}}$ 贡献约 40%

## 关键启示

- **轨迹一致性是关键**：逐 timestep 独立计算重要性比率在 I2V 等强条件任务上不够，需要考虑完整去噪轨迹的一致性
- **DanceGRPO 的失效模式**：I2V 任务是检验 GRPO 方法鲁棒性的重要试金石，条件图像的强约束放大了逐步独立优化的问题
- **高噪声 timestep 优化**：与 Euphonium 的发现一致，早期去噪步骤（高噪声）是决定生成质量的关键阶段，集中优化这些步骤是性价比最高的策略
- **Memory bank 的实用价值**：历史 rollout 复用显著降低计算成本，是大规模视频 GRPO 训练的实用优化
