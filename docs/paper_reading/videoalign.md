---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
  - Flow Matching
  - DPO
---

# VideoAlign: Improving Video Generation with Human Feedback

- 论文：https://arxiv.org/abs/2501.13918
- 代码：https://gongyeliu.github.io/videoalign/
- 团队：MMLab CUHK, Tsinghua University, Kling Team (Kuaishou), Shanghai AI Lab
- 发表：NeurIPS 2025

## 概述

VideoAlign 是一套完整的视频生成人类偏好对齐流水线。(1) 构建 182k 规模的多维度人类偏好数据集，覆盖 12 个 T2V 模型（含 Pre-Sora 和现代模型），标注 Visual Quality（VQ）、Motion Quality（MQ）和 Text Alignment（TA）三个维度；(2) 提出 VideoReward，基于 Qwen2-VL-2B 的多维度 reward 模型，系统研究了 BT vs 回归、tie 建模、token 位置等设计选择；(3) 从统一 RL 视角推导三种 flow-based 对齐算法：Flow-DPO（训练时）、Flow-RWR（训练时）和 Flow-NRG（推理时 reward guidance）。关键发现：将 Diffusion-DPO 直接适配到 rectified flow 时，时间相关的 $\beta_t = \beta(1-t)^2$ 会导致 reward hacking，改用常数 $\beta$ 后 Flow-DPO 在所有维度上稳定优于其他方法。

## VideoReward：多维度 Reward 模型

### 偏好数据集

- 16k 高质量 prompt（8 个元类别，GPT-4o 扩展后筛选）
- 12 个 T2V 模型生成 108k 视频，组成 182k 标注三元组（prompt + 两个视频）
- 三维度配对标注：VQ / MQ / TA，每维度 A wins / Ties / B wins
- 同时收集 1-5 Likert 点数评分，便于对比 pointwise 和 pairwise 监督
- 验证集：13k 三元组（prompt 不与训练集重叠）

### Reward 模型设计

骨干：Qwen2-VL-2B，以 2 fps 采样，约 448×448 分辨率。系统研究三个关键设计：

**Bradley-Terry vs 回归**：随数据规模增长，两者均提升，但 BT 模型始终优于回归。原因：pairwise 标注更能捕捉微妙的相对差异，即使两个视频获得相同点数评分，标注者仍可在配对比较中区分质量差异。

**Ties 建模（BTT）**：采用 Bradley-Terry with Ties 模型，扩展为三元偏好分布：

$$P_\theta(c|y, x_0^A, x_0^B) = \begin{cases} \frac{(\theta^2-1)\exp(r_A)\exp(r_B)}{(\exp(r_A)+\theta\exp(r_B))(\theta\exp(r_A)+\exp(r_B))} & \text{Tie} \\ \frac{\exp(r_A)}{\exp(r_B)+\theta\exp(r_A)} & A \succ B \\ \frac{\exp(r_B)}{\theta\exp(r_A)+\exp(r_B)} & B \succ A \end{cases}$$

$\theta > 1$ 控制 tie 倾向，实验中设 $\theta = 5.0$。BT 模型对许多 tie 对赋予较大分数差，将 tie 与明确偏好混淆；BTT 学到灵活的决策边界，tie 对聚集在零附近，明确胜负保持大间距。

**Token 位置策略**：常规做法用最后一个 token 预测多维度分数，但会造成上下文泄漏（同一视频搭配不同 prompt 获得不同 VQ 分数）。解决方案：在视频 token 之后、prompt 之前插入 [VQ] 和 [MQ] 两个上下文无关 token（只能关注视觉内容），在 prompt 之后放置 [TA] 上下文相关 token（可关注视频和文本），通过共享线性层映射为各维度分数。消除上下文泄漏，稳定视觉和运动评估。

### 评估结果

在 VideoGen-RewardBench（现代 T2V 模型）上的准确率：

| 方法 | Overall (w/ Ties) | VQ (w/o Ties) | MQ (w/o Ties) | TA (w/o Ties) |
|------|-------------------|---------------|---------------|---------------|
| VideoScore | 41.80 | 47.72 | 51.09 | 50.34 |
| LiFT | 39.08 | 55.97 | 54.91 | 55.43 |
| VisionReward | 56.77 | 59.03 | 60.98 | 61.15 |
| VideoReward | **61.26** | **75.66** | **74.70** | **72.20** |

VideoReward 在所有维度大幅领先，尤其在需要考虑 ties 的设置下。

## 对齐算法

### 统一 RL 目标

$$\max_{p_\theta} \mathbb{E}_{y \sim \mathcal{D}_c, x_0 \sim p_\theta(x_0|y)}[r(x_0, y)] - \beta D_{\text{KL}}[p_\theta(x_0|y) \| p_{\text{ref}}(x_0|y)]$$

### Flow-DPO

将 Diffusion-DPO 适配到 rectified flow：噪声预测误差 $\|\epsilon^* - \epsilon_{\text{pred}}\|^2 = (1-t)^2 \|v^* - v_{\text{pred}}\|^2$，代入 DPO 损失得到：

$$\mathcal{L}_{\text{FD}}(\theta) = -\mathbb{E}\left[\log\sigma\left(-\frac{\beta_t}{2}\left(\|v^w - v_\theta(x_t^w,t)\|^2 - \|v^w - v_{\text{ref}}(x_t^w,t)\|^2 - \|v^l - v_\theta(x_t^l,t)\|^2 + \|v^l - v_{\text{ref}}(x_t^l,t)\|^2\right)\right)\right]$$

其中 $\beta_t = \beta(1-t)^2$。

**关键发现：时间相关 $\beta_t$ 导致 reward hacking**。$\beta_t$ 在 $t \to 1$ 时趋于 0，使模型在高噪声层几乎不受 KL 惩罚，优先在高噪声层过度对齐。与 DDPM 中去掉去噪 score matching 的加权系数能提升采样质量类似，改用常数 $\beta$ 后训练更稳定，所有维度对齐效果均改善。这是因为 T2V 模型在不同噪声级别共享权重，不均匀的 $\beta$ 导致不均匀的训练。

### Flow-RWR

基于 EM 的 reward-weighted velocity 回归：

$$\mathcal{L}_{\text{RWR}}(\theta) = \mathbb{E}\left[\exp(r(x_0, y)) \|v - v_\theta(x_t, t, y)\|^2\right]$$

同样去掉 $(1-t)^2$ 因子。

### Flow-NRG（推理时 Reward Guidance）

利用闭式最优解 $p_\theta(x_0|y) \propto p_{\text{ref}}(x_0|y)[\exp(r(x_0,y))]^w$，通过修改速度场实现 reward guidance：

$$\tilde{v}_t(x_t|y) = v_t(x_t|y) - w \cdot \frac{t}{1-t} \nabla r(x_t, y)$$

为避免在像素空间反向传播 VAE 解码器梯度，在 latent 空间训练轻量级时间相关 reward 模型 $r_\theta(\cdot, t)$（复用预训练骨干的前几层）。允许用户在推理时为多个目标自定义权重，无需重训练。

## 实验

### 对齐结果

多维度对齐（VQ:MQ:TA = 1:1:1），VideoGen-Eval win rate：

| 方法 | VQ | MQ | TA | VBench Total |
|------|-----|------|------|-------------|
| Pretrained | 50.0 | 50.0 | 50.0 | 83.19 |
| SFT | 51.28 | 65.21 | 52.84 | 82.31 |
| Flow-RWR | 51.55 | 63.90 | 53.43 | 82.27 |
| Flow-DPO ($\beta_t$) | 87.78 | 82.36 | 51.02 | 80.90 |
| Flow-DPO (const $\beta$) | **93.42** | 69.08 | **75.43** | **83.41** |

时间相关 $\beta_t$ 版本 VQ/MQ 高但 TA 低于基线（reward hacking），常数 $\beta$ 版本三维度均大幅提升。

### Reward Guidance

用户可通过调整权重 $w_{\text{vq}}:w_{\text{mq}}:w_{\text{ta}}$ 在推理时控制对齐方向：
- 0:0:1 → TA 70.42%（专注文本对齐）
- 0.5:0.5:0 → VQ 86.43%, MQ 93.23%（专注视觉/运动质量，TA 下降）

### 人类评估

Flow-DPO vs 预训练模型（200 样本，3 标注者）：
- 总体：DPO wins 44.0%, Ties 29.0%, Pretrained wins 27.0%
- TA：DPO wins 48.2%

## 关键启示

- **Rectified flow 下时间相关 KL 系数 $\beta_t = \beta(1-t)^2$ 导致 reward hacking**：因共享权重下不均匀训练，改用常数 $\beta$ 是简单但关键的修正
- **Pairwise 标注（BT 模型）优于 Pointwise 回归**：即使数据充足，pairwise 仍更能捕捉微妙偏好差异
- **Tie 建模不可忽视**：BTT 在 tie 区域形成更清晰的决策边界，避免将 tie 误判为明确偏好
- **Token 位置策略消除上下文泄漏**：VQ/MQ token 放在 prompt 之前只看视频，TA token 放在 prompt 之后看两者，是 VLM reward 模型的简洁设计
- **推理时 reward guidance 允许免训练的多目标控制**：在 latent 空间训练时间相关 reward 模型避免 VAE 解码开销，用户自定义权重即可调整生成偏好
