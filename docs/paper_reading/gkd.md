---
title: GKD：从学生轨迹学习自生成错误
tags:
  - LLM Post Training
  - On-Policy Distillation
  - Knowledge Distillation
  - Imitation Learning
category: LLM Post Training
---

# GKD：从学生轨迹学习自生成错误

> 论文：[On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649)，ICLR 2024。作者：Google DeepMind、Mila、University of Toronto。论文页面未提供可核验的官方代码仓库，本文仅解读论文。

## 概述

GKD 将自回归蒸馏写成在线模仿学习问题。学生先生成自己的序列，教师在这些学生前缀上提供完整 token 分布，学生再根据选定的 divergence 更新。核心控制量是学生生成样本所占比例 λ；λ=0 退化为固定数据蒸馏，λ=1 为纯 on-policy 蒸馏。论文同时比较 Forward KL、Reverse KL 与 JSD，并展示同一轨迹机制可与 RL 目标联合。

## 背景与问题

固定教师序列或标注序列上的 teacher forcing，使训练前缀与推理时学生自身前缀存在分布差异。早期错误会改变后续状态，学生因而缺少对自身错误状态的纠正信号。论文将该现象与 imitation learning 中的 exposure bias 联系起来，并提出在学生访问的状态上重新查询教师。

## 方法分析

给定输入 x、教师分布 p_T、学生分布 p_θ 和输出序列 y，论文定义逐 token divergence：

$$D(p_T\|p_\theta)(y|x)=\frac1{L_y}\sum_{n=1}^{L_y}D(p_T(\cdot|y_{<n},x)\|p_\theta(\cdot|y_{<n},x)).$$

GKD 的目标为

$$\mathcal L=(1-\lambda)\mathbb E_{(x,y)\sim\mathcal D}D(p_T\|p_\theta)(y|x)+\lambda\mathbb E_{x\sim\mathcal X,y\sim p_\theta}D(p_T\|p_\theta)(y|x).$$

训练时以概率 λ 从学生采样输出，以概率 1−λ 取固定数据；教师只在得到的前缀上计算分布，采样路径停止梯度。该 stop-gradient 使训练保持 token-level 蒸馏的稳定形式。Forward KL 覆盖教师支持，Reverse KL 集中于教师高概率模式，JSD 在二者之间调节覆盖与集中。

论文还把蒸馏项与结果奖励 r(y) 合并：

$$\mathbb E_{y\sim p_\theta}[(1-\alpha)r(y)-\alpha D(p_T\|p_\theta)(y|x)].$$

α 控制教师约束与 RL 优化的相对权重，说明 GKD 可作为 RL 微调中的密集辅助目标。

## 实验与证据

教师为约 3B 参数 T5-XL，学生为 77M、250M、800M 的 T5；任务包括 XSum 摘要、WMT14 英德翻译、GSM8K 数学推理和 FLAN 指令蒸馏。XSum 中 on-policy GKD 在不同学生规模上超过 Supervised KD、SeqKD、ImitKD 和 f-distill；GSM8K 中学生生成比例提高到至少 25% 后准确率持续改善，纯学生轨迹通常最好。WMT 中 JSD 变体在部分设置优于固定 KL，体现 divergence 与任务及采样温度的耦合。FLAN 的 MMLU 与 BBH 评测中，on-policy Reverse KL 优于固定数据基线。

RLAIF 实验在 XSum 上加入文本蕴含奖励。提高蒸馏权重会提升 ROUGE-2，同时奖励侧的事实一致性增益发生变化，说明联合目标需要调度。实验从 SFT 学生开始，论文没有证明随机初始化学生可以稳定进入相同训练区间。

## 讨论与边界

GKD 的主要贡献是把轨迹分布纳入蒸馏目标，并提供 λ 与 divergence 两个可解释旋钮。实验模型以 T5 为主，数据集与教师访问条件相对受控；对于超大 decoder-only 模型、黑盒教师和长多轮任务，论文未给出直接证据。教师在学生异常前缀上的校准质量也没有被单独测量。因而 GKD 适合作为 OPD 方法主线的基础定义，具体 divergence 的选择仍需结合任务和算力验证。

## 可迁移设计点

1. 将学生 rollout 与教师评分拆成独立阶段，λ 可作为暴露偏差与成本的连续控制量。
2. 以任务指标观察 divergence 的覆盖—集中取舍，避免把某一种 KL 设为普遍最优。
3. 在 reward 训练中保留教师 divergence，利用密集信号改善早期优化，再用结果奖励调整最终行为。

## 来源

- 论文正文与附录：[arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
