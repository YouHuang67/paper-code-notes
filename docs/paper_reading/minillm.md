---
title: MiniLLM：序列级 Reverse KL 的策略蒸馏
tags:
  - LLM Post Training
  - On-Policy Distillation
  - Knowledge Distillation
  - Reinforcement Learning
category: LLM Post Training
---

# MiniLLM：序列级 Reverse KL 的策略蒸馏

> 论文：[MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)，ICLR 2024。作者：清华大学 CoAI Group、Microsoft Research。代码：[microsoft/LMOps/minillm](https://github.com/microsoft/LMOps/tree/main/minillm)。本文仅解读论文。

## 概述

MiniLLM 研究白盒大语言模型蒸馏中的序列级 Reverse KL。论文认为，Forward KL 会要求容量有限的学生覆盖教师的低概率区域，增加自由生成时的低质量输出；Reverse KL 促使学生集中到教师的主要模式。由于学生分布同时出现在采样期望和目标中，论文用 REINFORCE 推导可训练梯度，并用单步词表期望降低方差。

## 方法分析

教师分布记为 p，学生分布记为 q_θ。目标是

$$D_{KL}(q_\theta\|p)=\mathbb E_{y\sim q_\theta}[\log q_\theta(y|x)-\log p(y|x)].$$

对学生采样序列 y 应用策略梯度，论文得到

$$\nabla_\theta\mathcal L=-\mathbb E_{y\sim q_\theta}\sum_t(R_t-1)\nabla_\theta\log q_\theta(y_t|y_{<t},x),$$

其中

$$R_t=\sum_{t'=t}^{|y|}\log\frac{p(y_{t'}|y_{<t'},x)}{q_\theta(y_{t'}|y_{<t'},x)}.$$

R_t 是从当前位置开始的未来 log-ratio return；常数 −1 来源于 Reverse KL 中的熵项。该形式把教师相对学生的 log-probability 作为密集 token reward，同时保留后续 token 对当前决策的影响。

论文将 return 拆成当前 token 的 single-step quality 与未来 return。当前 token 的期望可在整个词表上闭式计算，未来部分使用采样轨迹和 clipped importance ratio；训练循环还加入语言模型预训练损失，以保持通用语言能力。算法先进行监督微调，再交替执行学生 rollout、教师评分和参数更新。

## 实验与证据

实验覆盖 GPT-2 120M/340M/760M、OPT 1.3B/2.7B/6.7B、LLaMA 7B 学生，对应更大的 GPT-2、OPT、LLaMA 教师。训练数据为 Dolly 15K 指令集，评测包括 DollyEval、Self-Instruct、VicunaEval、Super-Natural Instructions 和 Unnatural Instructions；指标为 ROUGE-L、GPT-4 评分及人工偏好，生成结果平均五个随机种子。

表 1 显示 MiniLLM 在不同模型族和规模上大多超过 SFT、token KD 与 SeqKD。例如 GPT-2 120M 在 DollyEval 的 GPT-4 评分为 44.7，SFT、KD、SeqKD 分别为 38.6、40.3、41.2；LLaMA 7B 学生在 Self-Instruct 的 ROUGE-L 为 23.2，高于三种基线的 20.8、20.2、20.8。人工评测中，LLaMA 7B 学生的 MiniLLM 响应偏好接近教师。

分析实验显示，MiniLLM 的累计暴露偏差指标增长较慢，长文本超过 150 token 后误差趋于平稳。SST2 与 BoolQ 的 ECE 也比 KD 和 SeqKD 更接近教师。教师规模从 GPT-2 340M 增加到 1.5B 时，固定 120M 学生的 MiniLLM 性能持续提高。论文同时报告了多样性、长度分组和预训练损失消融。

原文训练算法每步从指令数据采样 prompt，由学生 rollout 得到响应，再从固定数据抽取预训练批次；教师 log-prob 在学生前缀上计算。训练使用 response 长度截断、temperature=1 的学生采样和 PPO 风格 clipped importance ratio，另加预训练损失以维持语言建模能力。附录对 GPT-2、OPT、LLaMA 和 GPT-J 给出学习率、batch size、训练步数及五个随机种子结果，结论来自多模型族重复实验。

MiniLLM 的关键消融比较了单步质量项、未来 return、teacher mix-in 强度和预训练损失。只使用当前 token 的 log-ratio 会丢失后续决策影响；完整 return 能改善序列级 Reverse KL，同时带来更高方差。teacher mix-in 过强会使学生回到固定教师分布，过弱则增加 rollout 噪声；预训练损失改善通用语言能力，同时改变蒸馏目标的最优点。

## 讨论与边界

MiniLLM 的关键证据来自指令跟随和白盒教师条件，核心训练成本来自学生 rollout、词表期望和策略梯度方差。论文的 GPT-4 自动评分和人工评测均存在评测协议依赖，不能直接等同于所有生成任务的质量。Reverse KL 的 mode-seeking 性质可能削弱多样性；论文用 Dist-4 和语言模型损失观察到多样性仍被保留，但没有给出长程任务的统一保证。该方法适合解释 OPD 与 RL 的数学联系，复现时需要严格控制 clipping、baseline、长度处理和教师—学生 tokenizer 设置。

MiniLLM 的 sequence-level 目标与 token-level logits KD 具有不同偏差—方差结构。单步词表期望降低当前 token 的估计噪声，未来 return 仍由学生采样轨迹决定；长响应、低概率 token 和教师学生长度差异会改变 return 的尺度。论文的主要数据来自 Dolly 15K 及指令跟随评测，数学、工具调用和多轮 agent 任务的证据范围有限。

## 可迁移设计点

1. 将序列级目标拆成单步质量与未来回报，便于对高方差来源进行单独控制。
2. 以教师 log-prob 构造密集 reward 时，保留熵项和长度处理，避免把 Reverse KL 简化成逐 token 交叉熵。
3. 同时报告能力、暴露偏差、校准和多样性，避免只用单一 benchmark 判断蒸馏质量。

## 来源

- 论文正文与附录：[arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
