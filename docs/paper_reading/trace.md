---
tags:
  - LLM Post Training
  - On-Policy Distillation
  - Token Routing
  - SFT
---

# TRACE: Distilling Where It Matters via Token-Routed Self On-Policy Alignment

- 论文：[TRACE: Distilling Where It Matters via Token-Routed Self On-Policy Alignment](https://arxiv.org/abs/2605.10194)
- 代码：论文页面未给出官方代码链接
- 团队：南京大学、AMAP（Alibaba Group）、清华大学、University of Wisconsin–Madison
- 提交：2026-05-11，arXiv:2605.10194v1

## 概述

TRACE 研究 self-OPD 中“整条响应每个 token 都蒸馏”的粒度问题。作者观察到，全 token KL 会把梯度花在大量冗余位置，并放大 privileged information 泄漏，导致熵上升、推理变短和分布外性能下降。TRACE 让 privileged annotator 只标记每条 rollout 的关键 reasoning spans，再把蒸馏限制在这些位置；其他 token 继续由 GRPO 处理，KL 通道在短 warm-up 后退火。

论文在 Qwen3-8B 等模型上进行数学训练，在四个 held-out 数学基准和 GPQA-Diamond 上报告相对 GRPO 平均 2.76 个百分点的提升，并保持 Qwen3-8B 在 GPQA-Diamond 上的基础 OOD 分数。

## 1. 问题与方法

RLVR 给整条轨迹一个标量奖励，token 级 credit assignment 很稀疏。self-OPD 用 privileged context 生成逐 token 教师分布，把稀疏奖励变成密集信号，但全响应 KL 形成三种风险：冗余 token 获得不必要更新，特权信息暴露过多，错误或非关键位置的教师偏置积累。

TRACE 将一个 rollout 的 token 集合划分为关键正确 spans、局部错误 spans 和其余位置。对正确 rollout 的关键 spans 使用 forward KL；错误 spans 可以选择性使用 reverse KL；其余位置不使用 OPD KL，由 GRPO 更新。annotator 只提供 span 的粗粒度诊断类型，不把 span 文本直接交给教师，降低特权信息泄漏。

若 $M$ 表示被路由的 token 集合，$q$ 是教师分布，$p$ 是学生分布，关键 span 的 forward KL 可写为

$$\mathcal{L}_{\mathrm{FKL}}(M)=\sum_{t\in M}\mathrm{KL}(q_t\|p_t).$$

训练目标同时包含 GRPO 项、被路由的 KL 项和相对 SFT 参考策略的约束。KL 系数在 warm-up 后退火，限制累计 privileged-gradient exposure。论文理论说明，forward KL 能提升教师支持而学生低估的 token 概率；span mask 与 KL decay 则控制长期暴露。

## 2. 实验设置

主实验使用 Qwen3-8B，并在数学推理数据上比较 GRPO、all-token self-OPD、选择性 OPD 和 TRACE。评测包含四个 held-out 数学基准与 GPQA-Diamond；论文还测试 online self-annotation，即训练中的学生策略自己充当 annotator，不依赖外部 supervisor。训练细节、span-to-token 对齐、解码配置和 annotator prompt 在附录给出。

## 3. 主要结果

TRACE 相对 GRPO 在四个数学基准和 GPQA-Diamond 上平均提升 2.76 个百分点。它是比较的训练方法中唯一保持 Qwen3-8B base OOD 分数的方案；GRPO 与 all-token self-OPD 在 GPQA-Diamond 上出现退化。使用 online self-annotation 时，论文仍报告约 1.90 个百分点的平均增益，说明关键在于路由粒度和更新范围，而非必须拥有外部逐 token 教师。

消融实验围绕三项设计展开：只路由关键正确 spans、错误 spans 的局部 RKL，以及 KL 退火。论文还报告 all-token self-OPD 的三类失败症状、asymmetric thinking、NoThink-Eval robustness 和负结果，说明全 token 监督在长程数学训练中会出现熵与长度方向的异常变化。

## 4. 机制解释

TRACE 的核心不是增加教师信息量，而是控制教师信息落点。关键 span 具有较高的决策价值，能够在学生需要纠正的位置提供密集梯度；非关键 token 由学生自身的生成和 GRPO 保持，避免把表达表面、冗余连接词和特权上下文一起写入模型。局部 RKL 用于错误 span 的方向性排斥，但其使用范围仍由 annotator 路由。

## 5. 限制

实验主体是数学推理和 Qwen3 系列，关键 span 由 annotator 识别，路由质量会影响结果。论文没有图像 caption 或视觉 groundedness 评测，因此 span 的定义能否迁移到视觉描述需要重新设计。论文也不主张把所有 token 的 KL 都替换成 RKL；不同 span 类型使用不同信号。

## 6. 方法启示

1. 蒸馏粒度会影响信号效用；响应中的关键位置与冗余位置可以采用不同更新目标。
2. 使用特权标注时，路由信息与具体 span 文本的分离可限制信息泄漏。
3. 监督通道的强度和持续时间需要联合设计，退火用于控制训练期间累计的特权梯度。
4. 方法效果依赖基础模型状态；论文报告弱基础模型上的 FKL/RKL 最优方向与强模型不同。

## 来源

Wang et al., “TRACE: Distilling Where It Matters via Token-Routed Self On-Policy Alignment,” arXiv:2605.10194v1, 2026. [论文](https://arxiv.org/abs/2605.10194)
