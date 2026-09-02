---
tags:
  - Video Generation
  - Reward Model
  - VLM
  - Reinforcement Learning
  - GRPO
  - DPO
  - Post Training
---

# UnifiedReward-Flex: Unified Personalized Reward Model for Vision Generation

- 论文：https://arxiv.org/abs/2602.02380
- 项目：https://codegoat24.github.io/UnifiedReward/flex
- 代码：https://github.com/CodeGoat24/UnifiedReward
- 团队：Fudan University, Shanghai Innovation Institute, Shanghai Jiao Tong University, Shanghai AI Lab

## 概述

UnifiedReward-Flex（2026-02）做通用图/视频**生成** RM：先读 prompt 意图和视觉证据，在预定义锚点维下实例化子标准，必要时新增高层维（叙事互动、动作物理），再汇总 overall winner。Think 的 rubric 是固定清单；标量 BT（VideoReward 等）学一条全局 $r(x)$。训练：闭源 VLM 蒸馏 SFT → 按「结论对错 + 推理质量」做 DPO。8B 默认接入 Pref-GRPO。RM 评估：GenAI-Bench 视频 82.5、MJ-Bench-Video 72.0（Think 为 80.3 / 70.9）。Wan2.1-14B 上 Dynamic Degree **58.6 → 70.8**（VideoReward GRPO 降到 41.6）。

## 动机

三类通用 RM 都不随样本改标准：CLIP/PickScore 式固定打分、VideoAlign 式 BT 标量、Think 式固定 checklist。人类评估则是内容自适应的：先理解意图，再选相关维。文中视频例：狐狸登山跳跃需要「动作物理 / 运动清晰」，治愈麒麟故事需要额外「Narrative & Interaction」。

## 方法

### 层次评估

输入 prompt $p$ 与一对生成 $v^{(0)},v^{(1)}$。图像生成锚点维：语义对齐、视觉质量、美学。每维实例化 prompt 相关子标准；上下文需要时添加新高层维及其子标准。每维给出比较理由与 dimension winner，再合成 overall winner。

### Stage I：蒸馏 SFT

闭源 VLM 教师轨迹 $y^{\mathcal{T}}$，标准条件 LM：

$$\mathcal{L}_{\mathrm{SFT}}(\theta)=-\sum_i\sum_t\log p_\theta(y^{\mathcal{T}}_{i,t}\mid x_i,y^{\mathcal{T}}_{i,<t})$$

### Stage II：推理感知 DPO

人类标签 $w^\star\in\{0,1\}$ 标明哪条视觉更好。从 SFT 模型采两条评估 $y^{(a)},y^{(b)}$，正确性 $c(y)=\mathbf{1}[\hat{w}(y)=w^\star]$。

- 仅一条正确：该条为 $y^+$。
- 都对：闭源 judge（文中 GPT-5.2）比较推理质量，再人工核验。
- 都错：丢弃。

$$\mathcal{L}_{\mathrm{DPO}}(\theta)=-\mathbb{E}\log\sigma\Big(\beta_{\mathrm{dpo}}\big(\log\pi_\theta(y^+|x)-\log\pi_\theta(y^-|x)-\log\pi_{\mathrm{ref}}(y^+|x)+\log\pi_{\mathrm{ref}}(y^-|x)\big)\Big)$$

$\beta_{\mathrm{dpo}}=0.1$。骨干 UnifiedReward-Think-Qwen3-VL，2B–32B；下游 GRPO 默认 8B。32×H200，batch 2，grad accum 2，LR $2.5\times 10^{-6}$，warmup 0.1。DPO 采样温度 0.7。

### Pref-GRPO 中的个性化奖励

Flow 生成器对 prompt $c$ 采组 $\{x_0^i\}_{i=1}^{G}$。Pref-GRPO 不用绝对标量，用组内胜率

$$R(x_0^i,c)=\frac{1}{G-1}\sum_{j\neq i}\mathbf{1}[x_0^i\succ x_0^j]$$

再组内标准化成 $\hat{A}_i$。Flex 把比较拆成锚点维胜率均值 $\bar{R}_{\mathrm{dim}}$ 与 overall 胜率 $R_{\mathrm{overall}}$，分别标准化后

$$\hat{A}^i=\alpha\hat{A}^i_{\mathrm{dim}}+(1-\alpha)\hat{A}^i_{\mathrm{overall}}$$

动态新增维不一定每对都出现，所以 overall 单独一项。代入标准 clip GRPO（重要性比在相邻去噪步上）。

T2I：FLUX.1-dev，UniGenBench++ prompt，15 步、9 rollout、同噪声，LR $3\times 10^{-6}$，$\beta_{\mathrm{KL}}=0$。T2V：Wan2.1-T2V-14B LoRA $r=64,\alpha=128$，20 步、6 rollout，训练 240×416×33，推理 480×832、30 步、CFG 5，$\beta_{\mathrm{KL}}=0.004$。

## 实验

RM 准确率（论文 Table 1）：

| | 图像 GenAI-Bench | MMRB2 | 视频 GenAI-Bench | MJ-Bench-Video |
|--|:---:|:---:|:---:|:---:|
| VideoReward | — | — | 73.1 | 63.4 |
| UnifiedReward | 71.5 | 60.0 | 76.8 | 68.8 |
| Think | 72.3 | 66.0 | 80.3 | 70.9 |
| Flex | **73.4** | **69.2** | **82.5** | **72.0** |

相对 Think：MMRB2 +3.2，视频 GenAI-Bench +2.2。去掉 DPO 或「双对时不比推理质量」都低于完整 Flex。

FLUX.1-dev UniGenBench 语义一致性：基座 59.39，VideoReward 类标量 RM 无增益或下降，UnifiedReward 60.87，Think 68.89，**Flex 73.95**。

VBench（Wan2.1-14B）摘录：

| | Aesthetic | Imaging | Dynamic Degree | Human Action | Color | Scene |
|--|:---:|:---:|:---:|:---:|:---:|:---:|
| 基座 | 62.4 | 64.9 | 58.6 | 79.4 | 87.7 | 28.8 |
| + VideoReward | 62.9 | 66.5 | 41.6 | 78.2 | 87.8 | 28.2 |
| + Think | 63.9 | 65.2 | 58.3 | 78.4 | 86.1 | 27.2 |
| + Flex | **65.1** | **66.9** | **70.8** | **79.9** | **89.6** | **30.5** |

VideoReward 抬成像、压动态，是标量视频 RM 刷静帧的典型模式；Flex 的动态维把运动保住并抬高。

## 关键启示

- 2026 通用视频 GRM 的增量是「标准随 prompt/内容变」，并在 Pref-GRPO 里把维胜率与 overall 胜率合成优势。
- 视频 RL 必须同时看 Imaging 和 Dynamic Degree。
- pairwise judge 成本高于标量 BT；延迟敏感用 VideoReward，质量与运动同时要时用 Flex。
