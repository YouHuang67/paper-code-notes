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

> 论文：[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)，2025。作者：Qwen Team。本文只解读报告中的 On-Policy Distillation（OPD）与 Strong-to-Weak 蒸馏部分，不展开 Qwen3 的架构和完整模型评测。

## 概述

Qwen3 将 OPD 用作轻量模型后训练的核心阶段。流程先用教师生成的 `/think` 与 `/no_think` 响应进行 off-policy cold start，再让学生自行生成两种模式的序列，并在学生访问的前缀上对齐 Qwen3-32B 或 Qwen3-235B-A22B 的 logits。报告给出的关键系统证据是：在相同 off-policy 蒸馏起点上，Qwen3-8B 的 OPD 取得高于直接 RL 的数学、代码和通用能力，同时 GPU-hours 约为直接 RL 的十分之一。

## OPD 在 Qwen3 后训练中的位置

旗舰模型采用四阶段流程：Long-CoT cold start、Reasoning RL、Thinking Mode Fusion 和 General RL。轻量模型采用 Strong-to-Weak Distillation，目标是复用大模型已经形成的推理能力和 thinking-mode 控制能力，减少为每个小模型独立执行完整四阶段训练的成本。

报告将轻量模型蒸馏拆成两步。第一步使用教师输出进行 off-policy distillation，让学生形成基本推理行为并学习 `/think` 与 `/no_think` 的切换。第二步执行 OPD：从提示分布采样输入，学生选择 thinking mode 并生成响应，教师在相同学生前缀上提供 logits，学生以 KL divergence 进行更新。该流程直接把训练状态分布对齐到学生推理时会访问的状态。

## 方法与训练条件

报告沿用教师—学生 logits 的 KL 对齐，没有展开新的 OPD divergence 推导。教师包括 Qwen3-32B 和 Qwen3-235B-A22B，学生覆盖 Qwen3-0.6B、1.7B、4B、8B、14B 以及 Qwen3-30B-A3B。OPD 与模式控制共同训练，学生 rollout 同时覆盖 `/think` 和 `/no_think`，使蒸馏信号包含推理深度选择和 token 分布两部分。

该报告的实验设计把 OPD 与 direct RL 放在同一 off-policy 蒸馏起点上。这样比较回答的是“在已有基础能力上继续后训练时，教师 logits 的密集监督与结果奖励 RL 的成本和收益差异”，不能直接解释为所有初始化和所有任务上的普遍比例。

## 主要证据

Qwen3-8B 的表 21 给出如下结果：

| 方法 | AIME24 | AIME25 | MATH500 | LiveCodeBench v5 | MMLU-Redux | GPQA-Diamond | GPU hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Off-policy Distillation | 55.0 | 42.8 | 92.4 | 42.0 | 86.4 | 55.6 | — |
| + Reinforcement Learning | 67.6 | 55.5 | 94.8 | 52.9 | 86.9 | 61.3 | 17,920 |
| + On-policy Distillation | 74.4 | 65.5 | 97.0 | 60.3 | 88.3 | 63.3 | 1,800 |

括号中的 pass@64 也随 OPD 提升：AIME24 从 90.0 提升到 93.3，AIME25 从 83.3 提升到 86.7。报告据此认为，教师 logits 提供的逐 token 监督扩大了学生探索空间；该解释与指标变化一致，但表 21 只覆盖数学和代码查询，不能单独证明 OPD 在所有能力维度上都超过 RL。

报告还指出，直接 logits distillation 可以把轻量模型的性能和 exploration ability 同时提高，并以约十分之一 GPU-hours 达到更高结果。这个证据使 OPD 从方法论文中的训练策略，进入了多尺寸开源模型的生产式后训练流程。

## 讨论与边界

Qwen3 报告的贡献在于展示 OPD 的规模化配方：off-policy cold start 负责建立可学习的初始状态，on-policy logits 对齐负责修正学生自身 rollout，thinking-mode 目标贯穿两者。报告没有拆分 KL 方向、token-level 与 sequence-level 估计、rollout 长度和教师缓存策略的独立影响；GPU-hours 也依赖 Qwen3 的训练系统和硬件配置。因果上可以确认 OPD 阶段带来增益，具体增益来源仍需受控复现实验。

## 可迁移设计点

1. 对小模型先建立教师相容的冷启动分布，再进入 OPD，可降低早期 rollout 噪声。
2. 在同一训练配方中显式维护 thinking mode，让 OPD 同时学习能力和推理预算控制。
3. 将 GPU-hours、pass@1、pass@64 与能力基准共同报告，评估 OPD 的质量—成本曲线。

## 来源

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)，§4.5、Table 21。
