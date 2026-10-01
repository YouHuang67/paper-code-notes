---
title: ExOPD：带奖励外推的广义策略蒸馏
tags:
  - LLM Post Training
  - On-Policy Distillation
  - Reinforcement Learning
  - Knowledge Distillation
category: LLM Post Training
---

# ExOPD：带奖励外推的广义策略蒸馏

> 论文：[Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation](https://arxiv.org/abs/2602.12125)，2026。作者：中国人民大学高瓴人工智能学院、腾讯 LLM Department。代码：[RUCBM/G-OPD](https://github.com/RUCBM/G-OPD)。本文仅解读论文。

## 概述

ExOPD 将 OPD 放入带 KL 约束的 dense RL 形式，并引入 reward scaling factor λ 与 reference model。λ=1 对应标准 OPD，0<λ<1 产生奖励插值，λ>1 形成 reward extrapolation（ExOPD）。论文在数学、代码和多教师合并实验中观察到适度外推可以超过单个教师；强到弱蒸馏中，使用教师 RL 前的 base model 做 reference 可进一步校正隐式奖励。

## 背景与问题

标准 OPD 通过教师 logit 给学生提供密集方向，但教师分布也构成能力上限。纯 off-policy SFT 只学习教师轨迹，标准 OPD 仍以教师行为为中心。论文要回答的问题是：能否在保持教师 KL 约束的同时，让学生沿隐式奖励继续改进，并把多个领域教师的能力合并到一个学生。

## 方法分析

论文采用 KL 约束 RL 目标：

$$J(\theta)=\mathbb E_{x,y\sim\pi_\theta}[r(x,y)-\beta D_{KL}(\pi_\theta\|\pi_{ref})].$$

将教师相对 reference 的 log-probability 比值写成 token reward 后，G-OPD 引入 λ 调整 reward 与 KL 正则的相对权重。标准 OPD 是 λ=1；λ>1 时奖励项被外推，形成 ExOPD。reference 可以是学生初始模型，也可以是教师 RL 前的 base model。对应的最优 log-probability 形式为 $\log\pi_\theta=\lambda\log\pi^*+(1-\lambda)\log\pi_{ref}$，因此 λ 的作用可以直接解释为教师与 reference 之间的分布插值或外推。

多教师设置中，学生从同一 base model 出发，教师分别经过数学或代码 RL。学生执行 rollout，教师提供 token log-prob，reference 提供约束项，再以 G-OPD 更新。强到弱设置中，reward correction 用教师 pre-RL 模型作为 reference，以减少教师 RL 后 log-ratio 中的分布偏差。

## 实验与证据

同尺寸教师—学生实验使用 Qwen3-4B-Non-Thinking 及数学、代码 RL 教师。四个数学基准的教师平均准确率为 46.0，标准 OPD 为 46.5，ExOPD 为 48.0；三个代码基准的教师平均为 61.2，ExOPD 为 62.1。多教师合并时，ExOPD 的数学平均为 47.7，代码平均为 62.0，超过各领域教师；SFT、权重外推 ExPO 和标准 OPD 的跨基准表现较低。

强到弱实验使用 Qwen3-30B-A3B-Instruct-2507 教师和 Qwen3-1.7B/4B 学生。1.7B 学生在四个数学基准上的平均分从 SFT 的 13.5、OPD 的 23.1 提升到 ExOPD 的 25.4；4B 学生从 OPD 的 42.6 提升到 45.3。适度 λ=1.25 通常最好，λ=1.5 出现性能下降和长度膨胀。reward correction 在额外提供教师 pre-RL reference 时继续提升数学和代码平均准确率。

## 讨论与边界

ExOPD 的“超越教师”依赖隐式 reward 的可靠性和 λ 范围。论文观察到外推会增加输出长度与熵，过大 λ 可能放大 log-ratio 偏差并造成 reward hacking。多教师实验使用同一 base model 的领域 RL 变体，结果不能直接推广到完全不同架构或 tokenizer 的教师。reward correction 还需要教师 RL 前模型并增加前向开销。当前证据支持 ExOPD 作为 OPD 与 RL 融合的机制候选，长期泛化仍需独立复现。

## 可迁移设计点

1. 把 reference model、隐式 reward 和 KL 权重分开记录，使“模仿教师”和“探索教师外部区域”可独立调节。
2. 对 reward scaling 同时监控准确率、响应长度、熵和 token reward，及时识别外推导致的长度偏差。
3. 多教师合并应平衡各领域轨迹数量，并将统一学生与每个领域教师分别比较。

## 来源

- 论文正文与附录：[arXiv:2602.12125](https://arxiv.org/abs/2602.12125)
