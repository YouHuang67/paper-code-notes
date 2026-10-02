---
title: Qwen3 报告中的 OPD：Strong-to-Weak 蒸馏主线
tags:
  - LLM Post Training
  - On-Policy Distillation
  - Knowledge Distillation
  - Industrial Adoption
category: LLM Post Training
---

# Qwen3 报告中的 OPD：Strong-to-Weak 蒸馏主线

> 论文：[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)，2025。作者：Qwen Team。本文聚焦报告中的 On-Policy Distillation（OPD）与 Strong-to-Weak Distillation。

## 概述

Qwen3 将 OPD 放在轻量模型后训练的核心位置。流程先用强教师产生 `/think` 与 `/no_think` 响应，完成 off-policy distillation；随后由学生自行生成两种模式的序列，教师在学生实际访问的前缀上提供 logits，学生以 KL divergence 更新。报告对 Qwen3-8B 的受控比较显示，在同一个 off-policy checkpoint 上，OPD 将 AIME24/AIME25 的 pass@1 提升至 74.4/65.5，LiveCodeBench v5 提升至 60.3，GPU-hours 为 1,800；直接 RL 的对应 GPU-hours 为 17,920，成绩为 67.6/55.5 和 52.9。

## 问题与训练位置

Qwen3 旗舰模型采用 Long-CoT cold start、Reasoning RL、Thinking Mode Fusion、General RL 四个阶段。为构建 0.6B、1.7B、4B、8B、14B 五个 dense 小模型及 30B-A3B MoE 模型，报告采用 Strong-to-Weak Distillation，将大模型形成的推理能力和模式控制迁移到学生。学生需要获得可验证任务上的推理能力，也需要在 `/think` 与 `/no_think` 之间稳定切换，并可用 token budget 控制思考长度。

## 方法主链

### Off-policy distillation

第一阶段从教师已经生成的响应构造训练样本。教师分别在 `/think` 和 `/no_think` 模式下回答提示，学生对固定序列执行 response distillation。`/think` 样本提供显式长链路推理，`/no_think` 样本提供低延迟回答行为；两类样本共同建立学生后续 on-policy 生成所需的初始分布。

### On-policy distillation

第二阶段从提示分布采样输入，学生自行选择并生成 `/think` 或 `/no_think` 序列。对每个学生前缀，Qwen3-32B 或 Qwen3-235B-A22B 教师计算下一 token 的 logits，学生通过 KL 对齐教师分布。训练状态来自学生自身 rollout，蒸馏信号覆盖部署时学生会访问的前缀分布。

报告没有公开 KL 的方向、token-level 与 sequence-level 的归一化方式、学习率、batch size、rollout 长度或教师 logits 缓存策略。这些缺失限制了独立复现和对单个训练因素的归因。报告也没有给出两种 thinking mode 分开训练的消融。

## 实验与证据

表 21 固定同一个 Qwen3-8B off-policy checkpoint，比较继续执行 direct RL 与 OPD。括号为 pass@64。

| 方法 | AIME24 | AIME25 | MATH500 | LiveCodeBench v5 | MMLU-Redux | GPQA-Diamond | GPU hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Off-policy Distillation | 55.0 (90.0) | 42.8 (83.3) | 92.4 | 42.0 | 86.4 | 55.6 | — |
| + Reinforcement Learning | 67.6 (90.0) | 55.5 (83.3) | 94.8 | 52.9 | 86.9 | 61.3 | 17,920 |
| + On-policy Distillation | 74.4 (93.3) | 65.5 (86.7) | 97.0 | 60.3 | 88.3 | 63.3 | 1,800 |

在 AIME24 上，OPD 相对同起点提高 19.4 个百分点；在 AIME25 上提高 22.7 个百分点；LiveCodeBench v5 提高 18.3 个百分点。pass@64 分别从 90.0 提高到 93.3、从 83.3 提高到 86.7，RL 在这两个指标上保持起点数值。OPD 的 GPU-hours 约为 RL 的十分之一，且达到更高的表内成绩。

该比较验证了 Qwen3-8B 在固定 off-policy 初始化、数学和代码查询、报告训练系统下，教师 logits 的 on-policy 监督具有质量和成本优势。它没有提供四阶段旗舰训练与 OPD 的逐项同预算对照，也没有报告其他尺寸学生的 GPU-hours。

## 机制分析与边界

off-policy 阶段解决学生初始 rollout 与教师行为差距过大的问题；OPD 阶段把学习状态转移到学生自身分布，并在每个 token 位置提供教师概率信息。pass@64 的提升支持“教师 logits 改善探索覆盖”的作者解释，但尚缺少候选熵、有效分支数或轨迹多样性的直接测量。

报告的工业影响力来自完整小模型系列的采用：同一 Strong-to-Weak 配方覆盖六个学生规格，并将 thinking control 纳入蒸馏接口。证据属于单一模型家族的报告结果；教师规模、词表兼容性、提示分布和硬件配置都会影响迁移。

## 可迁移设计点

1. 先用双模式教师响应建立可学习初始分布，再让学生 rollout，可降低早期 on-policy 样本的失配程度。
2. 将模式标记和思考预算作为蒸馏输入条件，使能力迁移与推理强度控制在同一训练链路中学习。
3. 同时报告 pass@1、pass@64、能力基准和 GPU-hours，形成质量、探索与成本的联合评估。

## 来源

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)，§4、§4.5、Table 21。
