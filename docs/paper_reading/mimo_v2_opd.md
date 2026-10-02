---
title: MiMo-V2-Flash 报告中的 MOPD：多教师能力合并
tags:
  - LLM Post Training
  - On-Policy Distillation
  - Knowledge Distillation
  - Industrial Adoption
category: LLM Post Training
---

# MiMo-V2-Flash 报告中的 MOPD：多教师能力合并

> 论文：[MiMo-V2-Flash Technical Report](https://arxiv.org/abs/2601.02780)，2026。作者：Xiaomi MiMo Team。本文聚焦报告中的 Multi-Teacher On-Policy Distillation（MOPD）。

## 概述

MiMo-V2-Flash 将多教师能力整合定义为 MOPD。训练链路包含通用 SFT、领域教师训练和 MOPD 三阶段：搜索、代码、工具、数学、通用推理与安全等教师分别通过领域 RL 或 SFT 获得专长，统一学生从自身分布生成 rollout，再由输入领域对应的教师提供 token-level Reverse KL 信号。Table 7 显示，MOPD 后学生在 AIME 2025、HMMT、LiveCodeBench 和 Arena-Hard 等项目达到或超过最强教师，同时 BrowseComp、Creative Writing 等项目出现下降，说明能力合并受领域分布和教师路由影响。

## 三阶段主线

第一阶段 SFT 用高质量指令—回答对建立通用指令跟随。第二阶段从同一基础模型训练领域教师：agent 能力覆盖搜索、代码和通用工具，非 agent 能力覆盖数学、通用推理和安全；每个教师使用聚焦任务的 RL 或 SFT，并由领域奖励信号优化。

第三阶段将教师知识整合到统一学生。学生在自己的演化分布上采样，提示 `x` 的领域决定教师 `π_x^domain`。教师在学生已生成的前缀上计算下一 token 概率，所得 token 级信号参与 on-policy 更新。报告把 MOPD 与参数合并、静态离线专家数据、顺序训练区分开来，强调学生状态分布和密集 credit assignment。

## 目标函数与训练条件

设 `π_θ` 为训练引擎中的学生策略，`μ_θ` 为推理引擎中的采样策略，`π_x^domain` 为提示 `x` 对应的领域教师，`D` 为提示分布。报告先定义 Reverse KL 的采样形式：

$$L_{reverse-KL}(θ)=-\mathbb{E}_{x\sim D, y_t\sim π_θ}[\log \frac{π_x^{domain}(y_t|x,y_{<t})}{π_θ(y_t|x,y_{<t})}].$$

实际训练中采样策略与训练策略可能存在差异，报告引入 importance sampling：

$$w_t(θ)=\begin{cases}\operatorname{sg}[\frac{π_θ(y_t|x,y_{<t})}{μ_θ(y_t|x,y_{<t})}],&\epsilon_{low}\le\frac{π_θ}{μ_θ}\le\epsilon_{high},\\0,&\text{其他情况。}\end{cases}$$

随后优化

$$L_{MOPD}(θ)=-\mathbb{E}_{x\sim D,y\sim μ_θ}[\frac{1}{|y|}\sum_t w_t\hat A_{MOPD,t}\log π_θ(y_t|x,y_{<t})],$$

其中

$$\hat A_{MOPD,t}=\operatorname{sg}[\log \frac{π_x^{domain}(y_t|x,y_{<t})}{π_θ(y_t|x,y_{<t})}].$$

与 outcome reward model（ORM）联合时，报告将 `α A_ORM` 加到该优势上。这样 token-level 教师信号负责局部信用分配，ORM 保留结果级约束。报告没有公开 `ε_low`、`ε_high`、`α` 的具体数值、教师路由算法、各领域数据量和每轮采样比例。

## 实验结果

| Benchmark | 学生 Before | 最佳教师 | 学生 After | 相对最佳教师 |
| --- | ---: | ---: | ---: | ---: |
| AIME 2025 | 89.3 | 93.9 (RL) | 94.1 | +0.2 |
| HMMT Feb. 2025 | 76.9 | 82.6 (RL) | 84.4 | +1.8 |
| LiveCodeBench | 77.5 | 82.6 (RL) | 83.2 | +0.6 |
| MMLU-Pro | 84.7 | 84.7 (Self) | 84.9 | +0.2 |
| GPQA-Diamond | 84.9 | 84.9 (Self) | 84.3 | -0.6 |
| Arena-Hard (Hard Prompt) | 50.0 | 50.0 (Self) | 54.1 | +4.1 |
| Arena-Hard (Creative Writing) | 90.1 | 90.1 (Self) | 86.2 | -3.9 |
| SWE-Bench Verified | 67.8 | 74.2 (RL) | 73.4 | -0.8 |
| BrowseComp | 42.5 | 51.7 (SFT) | 45.4 | -6.3 |

Figure 6 对比 ORM、无 ORM 的 MOPD 和联合 MOPD 在 AIME 2025 与 LiveCodeBench 上的训练曲线。该图用于隔离 token-level 教师优势与 outcome reward 的组合效果；报告未给出完整曲线数据表，因此可读出的证据是收敛趋势与最终点的相对关系。

## 机制分析与边界

MOPD 的关键机制有三层：领域教师把不同能力封装成可调用策略，学生 rollout 将训练状态贴近部署分布，Reverse KL log-ratio 为每个生成 token 提供方向。importance ratio 截断控制采样策略和训练策略之间的偏差。Table 7 的负迁移表明，领域能力的统一仍受提示分类、教师覆盖和信号权重制约。

报告提出迭代共进化：MOPD 后学生可重新进入领域 RL，形成更强教师，再用于下一轮蒸馏。该循环属于设计设想，Table 7 没有验证多轮收益、稳定性或成本曲线。SGLang、partial rollout 等内容属于支撑 on-policy 训练的系统条件，报告没有将它们单独作为 MOPD 算法消融。

## 可迁移设计点

1. 以领域条件选择教师，并记录路由与样本比例，便于分析能力竞争和负迁移。
2. 使用 training–inference importance sampling 过滤陈旧 token，控制训练策略与采样策略的分布差异。
3. 在同一 rollout 上组合 token-level KL 优势和 ORM 优势，分别覆盖局部学习信号与最终结果约束。

## 来源

- [MiMo-V2-Flash Technical Report](https://arxiv.org/abs/2601.02780)，§4.1、§4.4、§4.5、Table 7、Figure 6。
