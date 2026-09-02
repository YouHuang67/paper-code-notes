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

UnifiedReward-Think 在 UnifiedReward 上加显式长 CoT：对每个候选按任务维度打分再求和，得到最终 winner。用少量 GPT-4o 蒸馏轨迹教格式，再用拒绝采样把正确轨迹扩到图/视频生成与理解，最后对答错样本做 GRPO（格式奖励 + 答案准确率）。视频生成 GenAI-Bench diff **82.3**（基座 77.2），VideoGen-RewardBench diff **80.5**（基座 79.3）。掌握 CoT 后关掉显式 think，直接打分仍高于基座。与 RewardDance（yes 概率）和 VideoScore2（打分式 CoT）同属通用视频/视觉生成 GRM 的推理线。

## 动机

浅理由 GRM 在复杂场景不可靠，且推理与结论常不一致。大规模人工 CoT 奖励数据不可得。假设：显式长 CoT 提高可靠性；内化后隐式推理也能抬无 CoT 准确率。VLM 已有推理先验，用探索式 RL elicit，而不是从零标注。

## 数据

图像生成：HPD 25.6K、OIP 7.4K、EvalMuse 3K、Rapidata OpenAI-4o t2i 人偏 6.7K。视频生成：VideoDPO 10K、Rapidata Text2Video 人偏 5.7K。图像理解：LLaVA-Critic 抽 30K。视频理解：ShareGPTVideo-DPO 17K。Cold start 只用图生成子集蒸馏 GPT-4o，得到 ImageGen-CoT-Reward-5K，其余留给后两阶段。

## 方法

基座：已训好的 UnifiedReward。输出必须含 `<think>` 与 `<answer>`。think 内对每个候选按任务维打分（生成：语义一致性、美学、真实性等；理解：语义准确、事实正确、清晰度），加总后决定 winner，把推理和答案绑在一起。因此 cold start / 拒绝采样 / GRPO 都可以只过滤最终答案，仍约束推理质量。

### Cold start

图生成偏好对 + prompt（instruction + caption）+ GT → GPT-4o 写长推理。只留最终答案与 GT 一致的 $(x,y)$。目标为教师轨迹上的 LM：

$$\mathcal{L}_{\mathrm{cold}}(\theta)=-\sum_{i=1}^{T}\log p(y_i\mid x,y_{<i};\theta)$$

8×H100，batch 1，grad accum 16，LR $2.5\times 10^{-6}$，warmup 0.3。

### Rejection sampling

用 cold start 后的模型在统一偏好集上自生成 CoT（生成任务：caption + 图/视频对；理解任务：query + 图/视频）。答案对则留下，再用与 cold start 相同的 CE 微调。视频上会评时序一致性等，把图生成上学到的格式迁过去。

### GRPO

拒绝采样丢掉的难题用来探索。每输入采样 $N=8$ 条轨迹。

- $R_{\mathrm{fmt}}=1$ 当且仅当同时含合法 `<think>` / `<answer>`，否则 0。
- $R_{\mathrm{acc}}=1$ 当且仅当 `<answer>` 内 “Image/Video/Answer X is better” 与 GT 完全一致。

$R=R_{\mathrm{fmt}}+R_{\mathrm{acc}}$。组内标准化优势

$$\hat{A}^{(i)}=\frac{R^{(i)}-\mathrm{mean}(R)}{\mathrm{std}(R)}$$

重要性比 $r^{(i)}=\pi_{\theta_{\mathrm{new}}}(o^{(i)}|x)/\pi_{\theta_{\mathrm{old}}}(o^{(i)}|x)$，clip 到 $[1-\delta,1+\delta]$，加 $\beta D_{\mathrm{KL}}(\pi_{\mathrm{new}}\|\pi_{\mathrm{ref}})$，$\beta=0.04$。64×H20，batch 1，LR $1\times 10^{-6}$。

对照：无 CoT 的 GRPO 只强化最终答案，图像理解 Macro 几乎不动。

## 实验

VLRewardBench Overall / Macro：UnifiedReward 67.5 / 66.6 → +cold start 66.9 / 66.0（先学格式可能短暂掉点）→ +rejection 72.1 / 69.3 → +GRPO **73.8 / 72.3**。无 CoT GRPO：69.0 / 67.4。

生成评估（diff，去掉 tie）：

| | 图像 GenAI-Bench | 视频 GenAI-Bench | VideoGen-Reward |
|--|:---:|:---:|:---:|
| UnifiedReward | 70.9 | 77.2 | 79.3 |
| +GRPO w/o CoT | 71.3 | 78.4 | 79.5 |
| +GRPO Think | **72.5** | **82.3** | **80.5** |

视频 GenAI-Bench 的增益明显大于「只强化答案」的 GRPO。局限：显式 CoT 增加推理时延；论文用内化后的无 think 模式做低延迟 RM。

## 关键启示

- 通用视频 GRM 第二阶段：长 CoT + 可验证 $R_{\mathrm{fmt}}+R_{\mathrm{acc}}$ 的 GRPO，与 LLM 侧过程奖励同构。
- 多维打分再汇总，使只监督最终 winner 仍能压「理由胡编、答案碰巧对」。
- 在线视频 GRPO 若吃不消长推理，可关 think；要更高判别则保留 Think，或走 RewardDance 的 yes 概率。
