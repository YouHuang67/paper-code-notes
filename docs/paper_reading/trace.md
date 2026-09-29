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

一次更新先由学生 $\pi_S$ 采样推理轨迹，verifier 给二值正确性。annotator $\pi_A$ 把响应切成编号 segment：正确轨迹标关键推理 span $K_y$，错误轨迹标局部错误 span $E_y$，余下为 $N_y$。segment 位置再映射到 token mask，总覆盖限制在响应长度的 25% 以内。annotator 给教师的额外信息只有粗类型标签（如遗漏分类讨论）；教师看不到完整轨迹、span 文本和具体位置。教师 $\pi_T$ 与学生共享并周期同步参数，在相同因果前缀上评分，但教师另看私有类型标签。这样类型信息参与教师判断，位置仅决定在哪些 token 计算 KL。

令 $q_t$ 为教师分布、$p_t$ 为学生分布。强基座的默认 action 在 $K_y$ 上使用 $\mathrm{KL}(q_t\|p_t)$（FKL），在 $E_y,N_y$ 上不用 KL；弱基座的有效 action 改为在 $E_y$ 上使用 $\mathrm{KL}(p_t\|q_t)$（RKL）。训练损失把非 span 的 GRPO、span 上逐步恢复的 GRPO 与按响应长度归一化的局部 KL 合并。实现对词表内 KL 项做 pointwise clip，阈值为 0.05；每个 span 分支先求 span 均值，再乘 $|S_y|/|y|$，使不同长度轨迹对应论文的序列归一化目标。

KL 权重 $\lambda_k$ 在前 10 步保持 $w_0=0.5$，随后 30 步线性降至 0；第 40 步后教师前向停止，训练退回纯 GRPO。span 上 GRPO 系数为 $1-\lambda_k/w_0$，因而随着 KL 关闭平滑恢复，而不会突然改变这些 token 的训练目标。教师在 KL 活跃阶段每 10 步同步一次；学生和教师均用 Think 模式。论文把学生 NoThink、教师 Think 的非对称设置作为负面消融。

FKL 与 RKL 的差别来自 softmax logit 梯度：FKL 对学生低估但教师支持的 token 给出与教师概率差成正比的提升；RKL 的压力还乘学生自身概率，因此更适合压低学生高度自信而教师反对的错误 token。理论中的“暴露有限”依赖 mask 覆盖上限和 $\sum_k\lambda_k^2$ 有界；它是在假设下控制特权梯度风险的结果，不是对任意 annotator 或任务的无条件质量保证。

## 2. 实验设置

主实验使用 Qwen3-8B，并在数学推理数据上比较 GRPO、all-token self-OPD、选择性 OPD 和 TRACE。评测包含四个 held-out 数学基准与 GPQA-Diamond；论文还测试 online self-annotation，即训练中的学生策略自己充当 annotator，不依赖外部 supervisor。训练细节、span-to-token 对齐、解码配置和 annotator prompt 在附录给出。

训练使用 OpenThoughts-114k 数学子集中的最多 30K problem–solution pairs，在 H100 上以 verl 训练，并采用 DAPO clip-higher（$\epsilon_{\rm low}=0.2,\epsilon_{\rm high}=0.28$）。Qwen3-8B 属于 strong-base regime，各 in-distribution 数学基准的 base avg@8 至少约 60%；论文另用 Qwen3-1.7B 检查弱基础模型。评测遵循 Qwen3 Thinking 模式，温度 0.6、top-p 0.95、top-k 20；avg@k 是每题 $k$ 次采样的平均准确率，五列均分为非加权平均。checkpoint 只按 OpenThoughts 验证集选择，不按最终基准挑选。

## 3. 主要结果

TRACE 相对 GRPO 在四个数学基准和 GPQA-Diamond 上平均提升 2.76 个百分点。它是比较的训练方法中唯一保持 Qwen3-8B base OOD 分数的方案；GRPO 与 all-token self-OPD 在 GPQA-Diamond 上出现退化。使用 online self-annotation 时，论文仍报告约 1.90 个百分点的平均增益，说明关键在于路由粒度和更新范围，而非必须拥有外部逐 token 教师。

表 2 中 Qwen3-8B 的 GRPO 五项均分为 78.75，TRACE-FKL 为 81.51；AIME25 从 68.96 到 73.54，GPQA-Diamond 从 53.85 到 58.33，后者接近 base 58.27。TRACE-RKL 均分为 80.14。Qwen3-1.7B 的排序反转：GRPO 57.73、TRACE-FKL 58.46、TRACE-RKL 60.16，而 base 为 58.77。两组结果支撑“按错误形态选择 KL 方向”，不能把 TRACE 简化为固定 FKL 配方。

消融实验围绕三项设计展开：只路由关键正确 spans、错误 spans 的局部 RKL，以及 KL 退火。论文还报告 all-token self-OPD 的三类失败症状、asymmetric thinking、NoThink-Eval robustness 和负结果，说明全 token 监督在长程数学训练中会出现熵与长度方向的异常变化。

定位消融在固定 1.7B 的 KL 调度后分别改成全 token、随机 25% 或反向选择非关键 25%；随机稀疏虽可缓解，但仍不及关键 span，说明“选哪些位置”比仅减少 token 数更重要。annotator 消融中，强 API annotator 带来 +2.76 点，训练中学生在线自标注仍有 +1.90 点；冻结 Qwen3-32B 和静态 base copy 分别为 +1.09、+0.92 点。作者还报告教师支持的关键 token 单步 log-prob lift：TRACE-FKL +0.145 nats、GRPO +0.054，作为局部提升机制的代理指标。

## 4. 机制解释

TRACE 的核心不是增加教师信息量，而是控制教师信息落点。关键 span 具有较高的决策价值，能够在学生需要纠正的位置提供密集梯度；非关键 token 由学生自身的生成和 GRPO 保持，避免把表达表面、冗余连接词和特权上下文一起写入模型。局部 RKL 用于错误 span 的方向性排斥，但其使用范围仍由 annotator 路由。

理论部分给出两个互补解释：FKL 会提升教师支持而学生低估的关键 token 概率；span mask 与 KL decay 则使 privileged-gradient exposure 在训练 horizon 内保持有限。online self-annotation 的增益说明 annotator 可以退化为“识别当前学生轨迹中的关键位置”，不必把完整外部推理轨迹直接写入学生。

## 5. 复现与边界

复现时需要实现学生 rollout 与 verifier、segment-to-token 对齐、类型标签与位置 mask 的隔离、按 span 类型切换 FKL/RKL、GRPO 主损失及精确 KL 退火；仅把全响应 KL 改成更小系数不能复现路由机制。all-token 对照在论文扫描配置内出现长度从 2027→1042 或 1873→759 token 的坍缩，并伴随熵升高；但作者重实现了部分未公开的基线代码，负例结论受其配置范围限制。AIME 子集只有 30 题，论文提供 bootstrap 置信区间，因此单列小幅差异不宜单独视为显著结论。实验主体是数学推理和 Qwen3 系列，span 路由质量依赖 annotator；视觉 caption 任务需重新定义关键与错误 span。

## 6. 方法启示

1. 蒸馏粒度会影响信号效用；响应中的关键位置与冗余位置可以采用不同更新目标。
2. 使用特权标注时，路由信息与具体 span 文本的分离可限制信息泄漏。
3. 监督通道的强度和持续时间需要联合设计，退火用于控制训练期间累计的特权梯度。
4. 方法效果依赖基础模型状态；论文报告弱基础模型上的 FKL/RKL 最优方向与强模型不同。

## 来源

Wang et al., “TRACE: Distilling Where It Matters via Token-Routed Self On-Policy Alignment,” arXiv:2605.10194v1, 2026. [论文](https://arxiv.org/abs/2605.10194)
