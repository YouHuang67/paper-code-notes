---
title: Rethinking OPD：可蒸馏条件与训练机制
tags:
  - LLM Post Training
  - On-Policy Distillation
  - Training Stability
  - Knowledge Distillation
category: LLM Post Training
---

# Rethinking OPD：可蒸馏条件与训练机制

> 论文：[Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe](https://arxiv.org/abs/2604.13016)，2026。作者：清华大学、上海科技大学、University of Illinois Urbana-Champaign、中国人民大学。论文标注的代码：[thunlp/OPD](https://github.com/thunlp/OPD)。本文仅解读论文。

## 概述

论文研究 OPD 何时有效以及为何失败。作者提出两个条件：学生与教师需要具有相容的思考模式，教师还要提供学生训练经历中缺少的新能力。机制分析显示，成功训练主要表现为学生与教师在高概率 overlap token 上逐步对齐；失败训练常伴随 overlap 停滞和熵差持续存在。作者提出 off-policy cold start 与 teacher-aligned prompt selection 两种恢复策略，并分析长轨迹中教师 reward 质量下降的上限。

## 背景与问题

OPD 在学生自己的 rollout 上提供密集 token feedback，理论上能覆盖推理时真实访问的状态。论文观察到更强教师可能无法改善学生，而较弱但模式相容的教师可以成功。这一现象促使作者分别考察能力分数、分布重叠、新信息和局部梯度机制，避免将教师 benchmark 分数直接等同于可蒸馏性。

## 方法与诊断指标

学生在前缀 c_t 的分布为 p_t，教师为 q_t。论文定义 top-k overlap ratio：

$$M_{overlap}=\mathbb E_t\left[\frac{|S_t^{(p)}\cap S_t^{(q)}|}{k}\right].$$

在交集 token 上，定义 overlap-token advantage：

$$A_t(v)=\bar p_t(v)(\log\bar q_t(v)-\log\bar p_t(v)),$$

其中带横线的分布是在 overlap 集合内重新归一化的学生和教师概率。训练期间同步记录 overlap、熵差、优势和梯度范数，用来判断信号是否正在被学生吸收。

冷启动策略先在教师生成轨迹上进行 off-policy SFT，再切换 OPD，以抬高初始 overlap。prompt selection 从教师后训练数据中选取更容易形成共同高概率 token 的提示；作者同时指出这种选择会降低学生熵，需要混入分布外提示保持覆盖。

## 实验与证据

论文主要使用数学推理设置，比较 R1-Distill-1.5B、JustRL-1.5B、R1-Distill-7B 和 Skywork-OR1-Math-7B 等教师—学生组合。成功运行中 overlap ratio 从约 72% 升至 91%，共同 top-k token 覆盖 97%–99% 的概率质量；失败运行 overlap 长期停滞，熵差和梯度有效性没有改善。弱到强反向蒸馏实验显示，同一模型家族、相似训练配方可能产生较高分数但较少新信息。

冷启动和 teacher-aligned prompts 均能恢复部分失败设置，并重新出现 overlap 上升、token advantage 改善和熵差收窄的动态。论文还改变最大 response length：3K–7K 的中等长度最稳定，10K–15K 出现后缀先失稳、熵和梯度范数上升。教师从学生长前缀继续生成时，准确率优势随前缀长度下降，说明密集 reward 的可靠性受轨迹深度限制。

在 reward 质量分析中，成功和失败教师的序列平均 reward 对正确/错误 rollout 都具有相近 AUROC（约 0.73 与 0.75）。作者据此提出，失败可能来自局部优化几何和位置间优势抵消；该解释仍属于待验证假设。Top-k 实验显示 Top-4、Top-16 和 Top-64 表现接近，Top-1 最不稳定，学生采样 token 在多次训练中能覆盖高概率区域。

## 讨论与边界

论文将“可蒸馏窗口”拆成模式兼容性和新能力两个条件，为教师选择、冷启动和提示筛选提供诊断语言。实验集中于数学任务，作者明确将代码、开放式生成、跨语料预训练和长程 agent 作为后续问题。overlap 指标反映候选空间相似度，不能独立证明语义能力已被吸收；框架合入或社区传播也不等同于论文机制已经获得跨团队因果验证。

## 可迁移设计点

1. 在训练前测量 top-k overlap、教师新信息和学生 pass rate，决定是否需要冷启动。
2. 用按位置的 overlap、熵和梯度曲线监控长轨迹，及时截断教师信号退化的后缀。
3. 将教师能力分数与可蒸馏性分开评估，避免以单一 benchmark 选择教师。

## 来源

- 论文正文与附录：[arXiv:2604.13016](https://arxiv.org/abs/2604.13016)
- 实现入口：[thunlp/OPD](https://github.com/thunlp/OPD)
