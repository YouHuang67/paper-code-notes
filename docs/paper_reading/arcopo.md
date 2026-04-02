---
tags:
  - Video Generation
  - Reinforcement Learning
  - GRPO
  - Flow Matching
---

# AR-CoPO: Align Autoregressive Video Generation with Contrastive Policy Optimization

- 论文：https://arxiv.org/abs/2603.17461
- 团队：CUHK MMLab, Vivix Group Limited, HKUST

## 概述

AR-CoPO 解决流式自回归（streaming AR）视频生成模型的 RLHF 对齐问题。现有 SDE-based GRPO（如 DanceGRPO）在 few-step consistency model 上完全失效——因为这类模型近似确定性映射，中间噪声注入几乎不影响输出，SDE 探索无效。AR-CoPO 基于 Neighbor GRPO 的对比学习重新诠释，提出 chunk-level forking 机制：在随机选择的 pivot chunk 处分叉构建邻域候选，赋予序列级 reward，仅在该 chunk 做 GRPO 更新。进一步提出 semi-on-policy 训练策略，结合 on-policy 探索和 off-policy 利用，各训练独立 LoRA adapter 后合并。在 Self-Forcing 上 VideoAlign 从 7.76 提升到 8.22，同时保持 VBench 质量。

## 动机

- 流式 AR 视频生成器（Self-Forcing、Causal-Forcing）+ few-step 蒸馏实现低延迟高质量合成，但难以用 RL 对齐
- SDE-based GRPO 在这类模型上失败的根因（实验验证 Fig. 6）：
  - **近确定性动力学**：替换初始 chunk 噪声导致输出巨大变化，替换 CM solver 中间噪声几乎无变化
  - **SDE 探索无效**：DanceGRPO 冻结初始噪声、在中间 SDE 噪声注入上定义动作空间，但中间噪声熵接近零 → 策略梯度信号接近零
  - **架构不匹配**：few-step consistency model 偏离标准 flow matching ODE

## 方法

### Neighbor GRPO 基础

通过扰动共享初始噪声构建邻域候选：

$$\epsilon^{(i)} = \sqrt{1-\sigma^2}\,\epsilon^* + \sigma\,\delta^{(i)}, \quad \delta^{(i)} \sim \mathcal{N}(0, I)$$

$\sigma \in (0,1)$ 控制探索半径。基于距离定义代理策略分布：

$$\pi_\theta^{(i)} = \frac{\exp(-d^{(i)}/\tau)}{\sum_{k=1}^G \exp(-d^{(k)}/\tau)}, \quad d^{(i)} = \|x_t^{(i)} - x_t^{(\theta)}\|_2^2$$

### Chunk-Level Forking

三阶段流程：

1. **共享上下文生成**：随机采样 pivot chunk $p \in \{1, \ldots, L\}$，模型顺序生成前 $p-1$ 个 chunk，建立共享 KV cache $h_{p-1}$
2. **动作空间分叉**：在第 $p$ 个 chunk，从共享基础噪声 $\epsilon_p^*$ 构建 $G$ 个扰动噪声 $\{\epsilon_p^{(i)}\}$。每个分支完成 T 步去噪，轨迹存入 replay buffer
3. **序列级 Reward**：每个分支确定性生成剩余 $L-p$ 个 chunk（无额外扰动），对完整视频计算 reward $r^{(i)}$

**受控噪声共享**（关键设计）：分支间唯一差异是 pivot chunk 的初始噪声 $\epsilon_p^{(i)}$。所有非 pivot chunk 的初始噪声和所有 CM solver 噪声完全共享。因此 reward 差异 $r^{(i)} - r^{(j)}$ 可完全归因于 pivot chunk 的生成质量——clean credit assignment。

### CM 模型的 CoPO 适配

对 consistency model，在 $\hat{x}_0$ 预测空间而非中间 $x_t$ 空间计算距离：

$$d_{0,t}^{(i)} = \|\hat{x}_{0,t}^{(i)} - \hat{x}_{0,t}^{(\theta)}\|_2^2, \quad \pi_\theta^{(i|s_t)} = \frac{\exp(-d_{0,t}^{(i)}/\tau_0)}{\sum_{k=1}^G \exp(-d_{0,t}^{(k)}/\tau_0)}$$

### Semi-On-Policy 训练

**问题**：纯 on-policy 训练对全局语义 reward（如 Text Alignment）无效——局部噪声扰动产生语义相似的输出，梯度信号弱且噪声大。实验证实：on-policy 优化 TA 导致 MQ 从 1.68 崩溃到 0.25（reward hacking）。

**方案**：训练两个独立 LoRA adapter（rank 64, $\alpha=128$）：

- **On-policy adapter**（探索）：从演化的 $\pi_\theta$ 生成新候选，驱动 reward 提升
- **Semi-on-policy adapter**（利用）：固定所有 rollout 为参考策略 $\pi_{\text{ref}}$，从预收集的 replay buffer（100 组 rollout）中上调高 reward / 下调低 reward 候选

两个 adapter 在推理时通过 LoRA 权重缩放合并。Ratio clipping 保持 trust region，防止分布漂移。

## 实验

- **基础模型**：Self-Forcing（few-step AR consistency model）
- **训练数据**：MovieGen Video Bench
- **Reward**：VideoAlign（VQ + MQ + TA）
- **超参**：$G=12$, LR $1 \times 10^{-5}$, $\sigma=0.5$, 100 iterations, 24 GPUs

### 主要结果

| 方法 | VBench Total | VideoAlign Overall | VQ | MQ | TA |
|------|:---:|:---:|:---:|:---:|:---:|
| Self-Forcing | 82.15 | 7.76 | 3.80 | 1.68 | 2.28 |
| + AR-CoPO (semi) | 82.45 | 7.61 | 3.70 | 1.60 | 2.30 |
| + AR-CoPO (on-policy) | 81.99 | 8.51 | 4.15 | 2.06 | 2.30 |
| + AR-CoPO (merged, scale 0.8) | **82.17** | **8.22** | 4.00 | 1.86 | 2.36 |

- Merged 模型：VideoAlign +0.46 且 VBench 保持（genuine alignment, not reward hacking）
- SDE-based GRPO 在 Self-Forcing 上完全无法提升 reward（训练曲线平坦）

### 训练策略消融（仅优化 TA reward）

| 策略 | VBench Total | MQ | TA |
|------|:---:|:---:|:---:|
| Self-Forcing | 82.15 | 1.68 | 2.28 |
| On-policy only | 79.26 | **0.25** | 2.63 |
| Off-policy (no clip) | 67.99 | -0.15 | 2.16 |
| Semi-on-policy | **82.45** | 1.60 | 2.30 |

On-policy 优化 TA 导致 MQ 崩溃（reward hacking），semi-on-policy 通过 trust region 避免。

### LoRA Merging Scale 权衡

| Scale | VBench Total | VideoAlign Overall |
|-------|:---:|:---:|
| 0 (semi only) | 82.45 | 7.61 |
| 0.8 ✓ | 82.17 | 8.22 |
| 1.0 | 81.99 | 8.33 |

**双基准准则**：选择 scale=0.8（VBench 和 VideoAlign 同时提升），scale=1.0 虽 VideoAlign 更高但 VBench 下降（过度优化信号）。

### 泛化到 Causal-Forcing

- VBench Total: 82.28→82.54, VideoAlign: 7.79→8.01（scale=0.5）
- 确认方法的跨模型泛化能力

## 关键启示

- **Few-step AR 模型的确定性本质**：初始噪声主导输出，中间噪声可忽略——这从根本上使 SDE-based GRPO 失效，需要新的探索机制
- **Chunk-level credit assignment**：通过受控噪声共享，将序列级 reward 精确归因到单个 chunk，避免全序列反向传播的计算开销
- **探索与利用分离**：on-policy（探索新模式）和 semi-on-policy（利用已有高质量分布）各训 LoRA 再合并，优于两者任一单独使用
- **双基准验证的必要性**：单一 in-domain metric 不足以区分 genuine alignment 和 reward hacking，需要同时关注 out-of-domain 质量指标
