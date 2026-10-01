---
title: DeepSeek-V4 报告中的 OPD：多教师 Full-Vocabulary 合并
tags:
  - LLM Post Training
  - On-Policy Distillation
  - Knowledge Distillation
  - Industrial Adoption
category: LLM Post Training
---

# DeepSeek-V4 报告中的 OPD：多教师 Full-Vocabulary 合并

> 论文：[DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)，2026。作者：DeepSeek-AI。本文只解读报告中的 OPD 后训练部分；架构、百万 token attention 和推理系统不在本文展开。

## 概述

DeepSeek-V4 将多个领域 specialist 的能力合并交给 multi-teacher OPD。数学、代码、agent 和 instruction-following 等 specialist 先独立进行 SFT 与 RL，再由统一学生在自己的 rollout 上接受多个教师的 full-vocabulary Reverse KL。报告把原先的 mixed RL 后训练阶段替换为 OPD，并围绕教师调度、hidden-state 缓存、logit 重建和 token-granular WAL 设计训练基础设施。

## OPD 在 V4 后训练中的作用

V4 的 OPD 目标聚焦多领域 specialist 合并，统一学生作为可部署模型。每个输入根据领域分配教师，学生自行生成轨迹，教师在这些前缀上计算完整词表分布，学生更新以 Reverse KL 为核心。多教师组合使学生能够在一次后训练阶段吸收多个领域的行为模式。

设教师集合为 $\{\pi_{E_1},\ldots,\pi_{E_N}\}$，教师权重为 $w_i$，报告给出的目标为

$$L_{OPD}(\theta)=\sum_{i=1}^{N}w_iD_{KL}(\pi_\theta\|\pi_{E_i}).$$

轨迹由学生生成，因此目标的状态分布与部署时学生行为一致。报告采用 full-vocabulary logits 计算 KL，保留教师在所有候选 token 上的分布信息。相对 sampled-token 估计，该选择增加教师前向和显存压力，同时提供更密集的梯度。

## 规模化训练条件

V4 的教师集合规模使直接缓存全词表 logits 成本很高。报告采用三项配套机制：只缓存教师最后一层 hidden states，再通过对应 output head 恢复 logits；按 teacher index 对样本排序，使一个 mini-batch 中最多只有一个 teacher head 常驻显存；通过中心化权重存储按需加载 specialist。这样把教师参数、hidden state 和 output head 的驻留生命周期分开管理。

rollout 中断还可能造成长度偏差。V4 为每个生成请求维护 token-level write-ahead log（WAL），在恢复时从中断位置继续处理，避免短轨迹更容易在截断窗口内完成的统计偏差。该机制服务于 RL/OPD 训练目标的统计一致性，具有容错和采样校正两重作用。

## 主要证据

报告将 OPD 描述为 V4 后训练中 specialist 合并的核心阶段，并给出 full-vocabulary 与多教师调度的系统设计。现有报告材料更强调训练流程和基础设施，未提供在相同教师、相同 rollout 预算下对 sampled-token、token-level KL、MiniLLM 或 ExOPD 的完整受控比较。因而可确认的是：OPD 已被用于 trillion-parameter 级多教师能力合并；单个 divergence 或缓存策略的独立因果收益仍缺少公开消融。

## 讨论与边界

DeepSeek-V4 的影响力证据来自真实规模的训练流程和多教师系统采用，重点支撑 OPD 的工程可行性。报告同时改变了模型架构、训练阶段、教师集合和系统基础设施，模型最终能力无法归因于 OPD 单一因素。full-vocabulary Reverse KL 对教师白盒访问、词表一致性、显存调度和通信带宽有较高要求，迁移到普通研究环境时需要重新评估成本。

## 可迁移设计点

1. 多教师 OPD 需要显式记录领域路由、教师权重和样本配比，避免合并结果掩盖领域间负迁移。
2. full-vocabulary 目标的系统实现应分离 hidden-state cache、output head 和教师权重的生命周期。
3. 长 rollout 训练应记录中断恢复位置，避免采样截断改变长度分布。

## 来源

- [DeepSeek-V4 Technical Report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
- 现有模型总览：[DeepSeek-V4](deepseek_v4.md)
