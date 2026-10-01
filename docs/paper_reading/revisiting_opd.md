---
title: Revisiting OPD：失败模式与局部支持匹配
tags:
  - LLM Post Training
  - On-Policy Distillation
  - Training Stability
  - Knowledge Distillation
category: LLM Post Training
---

# Revisiting OPD：失败模式与局部支持匹配

> 论文：[Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes](https://arxiv.org/abs/2603.25562)，2026。本文仅解读论文。

## 概述

论文系统分析 sampled-token OPD 的三个脆弱点：单 token 奖励高度不均衡、教师在偏离前缀上的局部指导可能失真、教师与学生 tokenizer 不一致会扭曲比较。作者提出 Local Support Matching（LSM），在每个学生前缀上取教师 top-K 候选，重新归一化后计算截断 Reverse KL；同时配合 top-p rollout 和 special-token masking。

## 方法分析

在前缀 c_t，学生分布为 π_θ，教师分布为 q。完整 Reverse KL 为

$$L_{full}(c_t)=\sum_{v\in V}\pi_\theta(v|c_t)\log\frac{\pi_\theta(v|c_t)}{q(v|c_t)}.$$

sampled-token OPD 只用学生采样 token y_t 的单点估计。LSM 取教师支持集 S(c_t)=TopK_q(c_t)，并分别归一化：

$$\hat\pi(v)=\frac{\pi_\theta(v)}{\sum_{u\in S}\pi_\theta(u)},\quad \hat q(v)=\frac{q(v)}{\sum_{u\in S}q(u)}.$$

其目标为对 rollout 中每个位置的 $D_{KL}(\hat\pi\|\hat q)$ 求平均。top-p 采样限制极低概率延续，special-token masking 减少 marker 与 EOS 不一致造成的假负例。作者还比较 token-level、sequence-level 与折扣 return-to-go，指出未来回报耦合增大会提高梯度方差。

## 实验与证据

单任务数学实验使用 Qwen2.5-7B-Instruct 学生、OpenThinker3-7B 教师和 DAPO-Math-17K。五个数学基准的平均分为：学生 28.2，sampled-token OPD 36.4，加 special-token mask 后 40.7，LSM 不加 mask 41.7，LSM 加 mask 41.5，教师 56.0。多任务实验交替训练数学与 ALFWorld，LSM 不加 mask 将数学平均从 sampled-token OPD 的 34.8 提升到 41.7，并保持 ALFWorld 竞争力；mask 版本在 ALFWorld 达到 97.7，但数学平均为 38.6。

消融显示，教师 top-K、top-p rollout 和支持集内重新归一化需要组合使用；缺少归一化会导致训练快速崩溃，支持集过小或完全无约束 rollout 也会降低稳定性。作者观察到 sampled-token 在多数位置产生负奖励，少数正奖励 token 主导更新；长 rollout 后段的教师—学生 log-prob 差异更宽，提示局部指导可靠性随前缀深度下降。

## 讨论与边界

LSM 以教师 top-K 支持替代单点采样，降低单 token 波动，同时保留 token-level 更新的效率。实验主要集中于数学和一个 agent 环境，教师与学生规模接近；对跨 tokenizer、开放式生成和更长多轮任务的普适性仍待验证。论文的“各向异性优势导致梯度抵消”解释属于作者提出的假设，尚未由梯度方向分析直接证实。

## 可迁移设计点

1. 将单点 teacher signal 与支持集分布信号并列评估，区分采样方差和目标偏差。
2. 对截断分布执行独立归一化，并监控支持集大小、top-p 和特殊 token mask 的联合作用。
3. 长序列训练应记录按位置的 overlap、熵和梯度范数，以定位后缀先发生的失稳。

## 来源

- 论文正文与附录：[arXiv:2603.25562](https://arxiv.org/abs/2603.25562)
