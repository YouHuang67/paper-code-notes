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

> 论文：[MiMo-V2-Flash Technical Report](https://arxiv.org/abs/2601.02780)，2026。作者：Xiaomi MiMo Team。本文只解读报告中的 Multi-Teacher On-Policy Distillation（MOPD），不展开模型架构与完整基准报告。

## 概述

MiMo-V2-Flash 把多教师能力整合定义为 MOPD。流程先做通用 SFT，再分别对搜索、代码、数学、工具使用和安全等领域训练专门教师，最后由统一学生从自身分布采样并接受领域教师的 token-level KL reward。MOPD 将参数合并和静态离线数据合并替换为 on-policy 的能力整合过程，并可与 outcome reward model（ORM）联合。

## 三阶段后训练

第一阶段使用高质量指令数据建立学生的通用行为。第二阶段从同一基础模型出发训练领域教师，每个教师在一个或一组任务上执行专门 RL/SFT。第三阶段 MOPD 根据输入领域选择教师，学生自己生成完整 rollout，领域教师在学生前缀上提供逐 token log-probability，学生通过 Reverse KL 方向的优势更新。

这种组织方式将能力合并分成“教师形成”和“学生吸收”两个阶段。教师可以来自 RL、SFT 或学生自身，报告强调教师接口与学生训练过程解耦，新增领域教师无需重构整个后训练管线。

## 目标函数与实现

设学生策略为 $\pi_\theta$，采样策略为 $\mu_\theta$，输入领域对应教师为 $\pi_x^{domain}$。单 token Reverse KL 的梯度写为

$$\nabla_\theta L_{reverse-KL}=-\mathbb E\left[\log\frac{\pi_x^{domain}(y_t|x,y_{<t})}{\pi_\theta(y_t|x,y_{<t})}\nabla_\theta\log\pi_\theta(y_t|x,y_{<t})\right].$$

报告使用 training–inference importance sampling：当采样策略与训练策略的比值落在 $[\epsilon_{low},\epsilon_{high}]$ 内时保留权重，否则丢弃该 token。MOPD 优势为

$$\hat A_{MOPD,t}=\operatorname{sg}\left[\log\frac{\pi_x^{domain}(y_t|x,y_{<t})}{\pi_\theta(y_t|x,y_{<t})}\right].$$

与 ORM 联合时，最终优势为

$$\hat A_t=\hat A_{MOPD,t}+\alpha A_{ORM,t}.$$

因此，教师 logits 提供密集 token 信号，ORM 提供结果级信号；二者在同一 on-policy rollout 上共同更新学生。

## 主要实验

报告的 Table 7 比较 MOPD 前后的统一学生与各领域最佳教师。AIME 2025 从学生 89.3、最佳教师 93.9 提升到 94.1；HMMT Feb. 2025 从 76.9、82.6 提升到 84.4；LiveCodeBench 从 77.5、82.6 提升到 83.2；Arena-Hard Hard Prompt 从 50.0 提升到 54.1。部分任务出现下降，例如 GPQA-Diamond 为 84.9→84.3，BrowseComp 为 51.7→45.4，Creative Writing 为 90.1→86.2。该结果说明 MOPD 能在多数领域合并能力，领域间仍存在信号竞争和分布覆盖差异。

训练曲线比较 ORM、无 ORM 的 MOPD 和联合 MOPD。在 AIME 2025 与 LiveCodeBench 上，MOPD 逐步达到或超过教师水平；无 ORM 版本用于隔离 token-level 教师信号的贡献。报告还提出迭代共进化：蒸馏后的学生重新进入领域 RL 形成新教师，再进行下一轮 MOPD。

## 讨论与边界

MOPD 的工业价值来自多教师合并、密集 credit assignment 和模块化教师接口。报告的教师、领域数据、路由策略和 ORM 均由同一训练体系控制，跨团队复现与不同教师架构的结果仍待验证。Table 7 的部分负迁移也表明，教师数量增加并不自动带来全域增益；领域采样、教师选择和 advantage 权重是关键条件。

## 可迁移设计点

1. 将领域教师选择作为输入条件，避免把所有教师分布无差别混合。
2. 用 importance ratio 过滤陈旧 rollout，控制训练策略与采样策略的偏差。
3. 同时保留 token-level KL 与 outcome reward，分别处理局部信用分配和最终结果。

## 来源

- [MiMo-V2-Flash Technical Report](https://arxiv.org/abs/2601.02780)，§4.1、§4.4、Table 7。
