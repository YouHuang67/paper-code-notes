---
title: DistiLLM 系列：稳定且高效的策略蒸馏
tags:
  - LLM Post Training
  - On-Policy Distillation
  - Knowledge Distillation
  - Training Stability
category: LLM Post Training
---

# DistiLLM 系列：稳定且高效的策略蒸馏

> 论文：[DistiLLM: Towards Streamlined Distillation for Large Language Models](https://arxiv.org/abs/2402.03898)，ICML 2024；[DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs](https://arxiv.org/abs/2503.07067)，ICML 2025 Spotlight。作者：KAIST 等。代码：[jongwooko/distillm](https://github.com/jongwooko/distillm)。本文将两篇论文作为连续方法解读。

## 概述

DistiLLM 针对 OPD 的两个工程问题设计：纯 KL 在概率接近零时产生不稳定梯度，学生 rollout 又带来较高教师查询成本。第一篇提出 Skewed KL（SKL）与 Skewed Reverse KL（SRKL），并用自适应 off-policy 调度器和 replay buffer 控制 rollout 比例。DistiLLM-2 进一步根据序列来源选择目标：教师生成序列使用 Forward SKL，学生生成序列使用 Reverse SRKL，从而把覆盖教师分布和强化学生高质量响应分配给不同数据来源。

## 背景与问题

Forward KL 在学生概率很低的区域仍会施加较大覆盖压力，Reverse KL 对教师未覆盖区域缺少反馈。学生生成的异常前缀还可能造成噪声教师信号。DistiLLM 的设计目标是在保留学生状态访问的同时，让目标函数具有有限、可控的梯度，并以更少在线 rollout 达到相近效果。

## 方法分析

令教师分布为 p，学生分布为 q，α∈(0,1)。SKL 用混合分布

$$\tilde p=\alpha p+(1-\alpha)q$$

替换 KL 的目标分布，目标写为

$$D_{SKL}(p\|q)=D_{KL}(p\|\tilde p).$$

SRKL 交换 KL 两侧：

$$D_{SRKL}(q\|p)=D_{KL}(q\|\tilde p).$$

混合分布包含学生或教师的概率质量，降低分母接近零时的比值波动。DistiLLM 将教师生成序列、学生生成序列与固定数据放入调度器；调度概率根据验证损失更新，并把历史学生轨迹保存在 replay buffer。该流程在 on-policy 状态覆盖和教师调用成本之间建立可调折中。

DistiLLM-2 将目标与数据来源绑定。教师序列采用 Forward SKL，保留教师分布中的多个候选模式；学生序列采用 Reverse SRKL，集中修正学生当前访问状态上的高价值响应。论文还使用 α 的课程式更新，让混合比例随训练进度变化。

## 实验与证据

DistiLLM 的实验覆盖 GPT-2、OPT、OpenLLaMA 与 T5 学生，任务包括 Dolly、Self-Instruct、Super-Natural Instructions、SAMSum 和 IWSLT。SKL/SRKL 在 GPT-2 指令任务上超过标准 KLD、Reverse KLD、JSD 和 MiniLLM；自适应调度器将训练时间降至朴素 KD 的约 1.6 倍，论文报告相对其它 on-policy 方法约 2.2–3.4 倍速度提升。消融显示 α≈0.1 时梯度范数和验证表现较稳定；从预训练学生直接开始时，DistiLLM 仍保持较快收敛。

DistiLLM-2 在指令跟随、数学推理、代码生成、偏好优化和视觉问答上评测。其表 2 在三个指令数据集上比较胜率，表 3、4 分别覆盖 GSM8K/MATH 与 HumanEval/MBPP。组件消融表明，来源感知的双目标优于对所有序列使用同一 divergence；论文还报告 speculative decoding 场景下的推理加速比较。实验覆盖多个模型族和任务，方法收益仍依赖 teacher/student 配置、α 课程与数据来源划分。

## 讨论与边界

SKL 的稳定性来自混合分布下界，代价是目标已不再等价于原始 Forward 或 Reverse KL，α 成为关键超参数。自适应 replay 会引入调度状态和额外实现复杂度。DistiLLM-2 的来源划分以整条序列为粒度；同一序列内部不同 token 的熵和模式数仍可能差异很大。两篇论文主要提供白盒教师实验，黑盒 API 和极长 agent 轨迹的证据有限。

## 可迁移设计点

1. 先以 skew divergence 约束数值范围，再决定是否需要策略梯度；稳定目标有利于扩大 rollout 覆盖。
2. 将数据来源作为目标函数的条件变量，分别处理“覆盖教师模式”和“修正学生行为”两类信号。
3. 把 replay ratio、α 和教师查询成本纳入同一消融矩阵，避免只比较最终分数。

## 来源

- [DistiLLM](https://arxiv.org/abs/2402.03898)
- [DistiLLM-2](https://arxiv.org/abs/2503.07067)
