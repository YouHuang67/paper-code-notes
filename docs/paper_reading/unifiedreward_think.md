---
tags:
  - Video Generation
  - Reward Model
  - VLM
  - Reinforcement Learning
  - GRPO
  - Post Training
---

# UnifiedReward-Think: Unified Multimodal Chain-of-Thought Reward Model

- 论文：https://arxiv.org/abs/2505.03318
- 代码：https://github.com/CodeGoat24/UnifiedReward
- 团队：Fudan University, Shanghai Innovation Institute, Shanghai AI Lab, Hunyuan (Tencent)
- 发表：NeurIPS 2025

## 概述

UnifiedReward-Think 把 UnifiedReward 从「短答案 / 浅理由 GRM」升级为显式长 CoT 视觉 RM：先按维度打分再汇总成最终 winner，用探索式强化学习激活 VLM 已有推理能力，而不是人工标注大规模 CoT。三阶段：(1) 用少量图生成偏好蒸馏 GPT-4o 的 CoT 做 cold start；(2) 在统一多模态偏好数据上 rejection sampling，只留最终答案对的轨迹；(3) 对答错样本做 GRPO（格式 + 准确率奖励）。主结果：掌握 CoT 后，即使推理时关掉显式 think，直接打分也超过原 UnifiedReward。视频生成 VideoGen-RewardBench 上 GRPO 后 diff 80.5（基座 79.3），GenAI-Bench 视频 diff 82.3（基座 77.2）。这是开源视觉 GRM 里「推理式 judge + RL」的代表性一条，与 RewardDance 的 yes-token 扩展、VideoScore2 的打分式 CoT+GRPO 并列。

## 动机

- 直接打分 / 浅理由在复杂场景不可靠，且推理与结论经常不一致。
- 大规模人工 CoT 奖励数据不可得；VLM 已有推理先验，需要被 elicit。
- 假设：显式长 CoT 提升可靠性；内化后隐式推理也能抬高无 CoT 准确率。

## 方法

基座：已训好的 UnifiedReward。输出格式 `<think>…</think><answer>…</answer>`，think 内对每个候选按任务相关维度打分并求和，强制推理与答案对齐。

**Cold start**：从图生成集随机抽，蒸馏 GPT-4o，只留最终答案与 GT 一致的轨迹 → ImageGen-CoT-Reward-5K。

**Rejection sampling**：大规模统一偏好（HPD 25.6K、OIP 7.4K、EvalMuse 3K、Rapidata 图 6.7K；VideoDPO 10K、T2V 人偏 5.7K；LLaVA-Critic 30K；ShareGPTVideo-DPO 17K）。模型自生成 CoT，答案对则留下微调。

**GRPO**：对仍错的样本采样 $N=8$ 条轨迹。$R=R_{\mathrm{fmt}}+R_{\mathrm{acc}}$（格式含 think/answer 标签；准确率看最终 winner）。组内标准化优势、clip、$\beta=0.04$ KL。64×H20，LR $1\times 10^{-6}$。前两阶段 8×H100，LR $2.5\times 10^{-6}$。

无 CoT 的 GRPO 对照：只强化最终答案，图像理解 Macro 几乎不动，说明有效信号来自推理过程而非答案模仿。

## 实验

VLRewardBench 消融 Overall / Macro：UnifiedReward 67.5 / 66.6 → +cold start 66.9 / 66.0（格式学习可能短暂掉点）→ +rejection 72.1 / 69.3 → +GRPO **73.8 / 72.3**。无 CoT 的 GRPO 仅到 69.0 / 67.4。

生成评估（diff，不含 tie）：

| | 图像 GenAI-Bench | 视频 GenAI-Bench | VideoGen-Reward |
|--|:---:|:---:|:---:|
| UnifiedReward | 70.9 | 77.2 | 79.3 |
| +GRPO w/o CoT | 71.3 | 78.4 | 79.5 |
| +GRPO Think | **72.5** | **82.3** | **80.5** |

## 关键启示

- 视觉 GRM 的第二阶段是「长 CoT + 可验证奖励的 GRPO」，与 LLM 侧 DeepSeek-R1 / GRM 同构。
- 多维打分再汇总，让只监督最终答案也能约束推理质量。
- 推理成本高；论文强调内化后可关 CoT 做低延迟 RM，这对在线 GRPO 很关键（对比 RewardDance 直接读 yes 概率）。
