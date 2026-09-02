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

UnifiedReward-Flex（2026-02）针对 GRM 的下一瓶颈：Think 仍用固定评分清单，BT 标量 RM（HPSv3、VideoReward）则学一条全局偏好函数。Flex 让模型先读 prompt 意图和视觉证据，再在少量预定义粗维度下实例化细粒度子标准，必要时新增高层维度（如叙事互动、动作物理）。训练：闭源 VLM 蒸馏 SFT → 对「结论对错 + 推理质量」构造的偏好对做 DPO。8B 变体作为默认 RM，接入 FLUX.1-dev / Wan2.1-14B 的 GRPO。相对 Think：MMRB2 +3.2、GenAI-Bench-Video +2.2。Wan2.1 上 Dynamic Degree 从 58.6 升到 70.8（VideoReward GRPO 反而降到 41.6，典型运动 hacking）。这是 UnifiedReward 系列在生成侧的个性化 / 动态 rubric 版本，与 2026-08 的 RubricRM 同属动态标准 GRM，但 Flex 已直接接到图/视频 GRPO。

## 动机

论文把当时 RM 分成三类并指出共同缺陷：

- 固定判别器（CLIP、PickScore）与 BT 模型（VideoAlign、HPSv3）：一条全局 $r(x)$，假设偏好分布单一。
- VLM-as-judge（UnifiedReward-Think）：生成式推理，但 rubric 静态，对 prompt 特异线索不敏感。

人类评估是内容自适应的：先理解意图，再选相关维度，必要时加新维度。

## 方法

输入 $(p, v^{(0)}, v^{(1)})$。层次评估：语义对齐 / 视觉质量 / 美学等锚点维度 → 实例化子标准 → 上下文需要时扩展新高层维度 → 各类 winner 再汇总 overall winner。

**Stage I SFT**：蒸馏闭源 VLM 的结构化评估轨迹，标准 LM 损失。

**Stage II DPO**：同一输入采样两条评估 $y^{(a)},y^{(b)}$。若仅一条最终 winner 与人类标签一致，该条为 win；若都对，再用闭源 judge（文中 GPT-5.2）比较推理质量并经人工核验。$\beta_{\mathrm{dpo}}=0.1$。

骨干：UnifiedReward-Think-Qwen3-VL，规模 2B–32B，下游 GRPO 默认 8B。训练 32×H200，batch 2，grad accum 2，LR $2.5\times 10^{-6}$。

## 下游 GRPO

- T2I：FLUX.1-dev，UniGenBench++ prompt，15 步、9 rollout、同噪声，LR $3\times 10^{-6}$，$\beta_{\mathrm{KL}}=0$。
- T2V：Wan2.1-T2V-14B LoRA r=64 $\alpha=128$，20 步、6 rollout，240×416×33 训练，推理 480×832，$\beta_{\mathrm{KL}}=0.004$。

## 实验

FLUX.1-dev UniGenBench 语义一致性：基座 59.39，HPSv3 57.98（略降），UnifiedReward 60.87，Think 68.89，**Flex 73.95**。OOD GenEval / T2I-CompBench 同样 Think 与 Flex 明显高于 BT 标量 RM。

VBench（Wan2.1-14B）摘录：

| | Aesthetic | Imaging | Dynamic Degree | Human Action |
|--|:---:|:---:|:---:|:---:|
| 基座 | 62.4 | 64.9 | 58.6 | 79.4 |
| + VideoReward | 62.9 | 66.5 | 41.6 | 78.2 |
| + Think | 63.9 | 65.2 | 58.3 | 78.4 |
| + Flex | **65.1** | **66.9** | **70.8** | **79.9** |

VideoReward 抬成像质量但压动态程度，符合 BT 标量 RM 在视频 RL 里刷静帧的已知模式；自适应 rubric 更能保住运动。

## 关键启示

- 2026 视觉 GRM 的增量从「会不会 think」转到「标准是否随样本变」。
- 下游应同时看质量维和 Dynamic Degree：只报 Aesthetic 会掩盖 BT RM 的运动 hacking。
- Flex 仍走 pairwise 文本 judge，在线 GRPO 成本高于 HPSv3 标量头；系列内部形成清晰分工：BT 低延迟、Think/Flex 高判别。
