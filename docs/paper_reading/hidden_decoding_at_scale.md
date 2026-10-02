---
title: Hidden Decoding at Scale：大模型的潜在计算扩展
tags:
  - Latent Reasoning
  - Sequence-Length Scaling
  - Continued Pretraining
  - Large Language Models
category: LLM Reasoning
---

# Hidden Decoding at Scale：大模型的潜在计算扩展

> 论文：[Hidden Decoding at Scale: Latent Computation Scaling for Large Language Models](https://arxiv.org/abs/2607.08186)，Liu 等，2026。本文依据论文正文与附录解读。

## 概述

Hidden Decoding 研究固定 Transformer 主干条件下的逐 token 潜在计算扩展。方法在每个 token 位置展开 $n$ 个具有独立 embedding 的 stream，将中间 stream 的 KV 保留为后续上下文，并以 Stream-Factorized Attention 降低跨 stream 注意力成本。作者以 continued pretraining（CPT）训练 Qwen3-8B 与 WeLM-80B、617B MoE：80B 与 617B 配置在九项共同基准上均优于各自匹配基线，617B 的 GPQA Diamond 从 89.1 提升至 91.2，HLE 从 33.6 提升至 35.4。论文同时报告 stream probe 结果，显示中间 stream 的预测分布更不确定，final stream 汇总后更集中。该工作提供百亿至数百亿规模的序列长度扩展证据，训练方式明确包含 CPT，计算代价随展开长度增加。

## 问题与定位

大模型扩大参数量通常需要新一轮高成本预训练。循环深度模型通过重复 Transformer 层增加每个 token 的计算，但作者指出，重复深度计算与大规模训练中的 pipeline parallelism 配合困难。Hidden Decoding 将额外计算放在序列维度：每个 token 扩展为多个内部 stream，形成更长输入，沿用大模型训练中的序列并行设施。

论文的主张分为两部分：序列长度扩展可以在固定主干上改善模型能力；中间 stream 在预测前承载不同阶段的潜在计算。前一主张由扩展因子扫描与 80B、617B matched baseline 比较支持，后一主张由 KV retention ablation、hidden-state similarity、attention affinity 和 LM-head probes 分析。

## 多流展开与训练目标

设文本 token 序列为 $x_1,\ldots,x_T$，展开因子为 $n$。每个 token 被映射为 $n$ 个 stream embedding $e_j(x_t)$，其中 stream 索引 $j\in\{0,\ldots,n-1\}$。展开后，Transformer 按 stream 与 token 位置构造扩展序列，最终 stream 负责 next-token prediction，中间 stream 参与当前 token 的潜在计算，并将对应 KV 保留给后续位置。

与只重复 token、计算完成后丢弃中间 KV 的 Parallel Hidden Decoding Transformer 不同，本文方法在 CPT 中持续保留每个 stream 的 KV context。这个设计使后续 token 能读取前序 token 的内部 stream 状态。作者另引入 Stream-Factorized Attention：大部分层只在同一 stream 内计算注意力，少数层执行跨 stream 混合，将随 $n$ 增长的注意力开销由稠密二次形式压低至近似线性形式。不同层的 full、within-stream 与 sliding-window 配置按模型规模设定。

训练目标只对最终 stream 施加语言建模损失，中间 stream 通过后续 stream 的预测目标间接学习。论文还采用 progressive expansion，从较小展开因子逐步迁移至更大因子，控制新增 embedding 与扩展序列带来的 CPT 初期损失变化。Qwen3-8B 扩展因子实验覆盖 $n\in\{2,4,8\}$；WeLM frontier-scale 主实验使用 $n=4$。

## 训练成本与配置

Hidden Decoding 从已有 baseline checkpoint 开始继续训练，扩展开启后使用与对照模型相同的数据和训练 schedule。80B 对照共享每个训练阶段的数据；617B 的 256k 长上下文阶段，HD 模型使用 0.30T tokens，baseline 使用 0.61T tokens，因此论文将对应 HD 增益视作保守比较。HD 训练阶段覆盖全程训练 token 的约 5.3%，其余过程沿用基础训练路径。

扩大 stream 数会增加有效序列长度。实测 4 倍序列扩展使 80B 训练成本约为 5.1 倍、617B 约为 4.4 倍，接近线性参考并低于稠密注意力的 16 倍估算。推理阶段需要处理并保留扩展 stream 的 KV，论文报告了吞吐与显存代价，收益评估必须结合这些额外成本。

## Frontier-scale 结果

WeLM-HD4-80B 与 WeLM-80B 在九项共同 benchmark 上比较，HD 在九项均提高；SciCode 从 45.8 升至 50.0，PHYBench 从 69.8 升至 73.8。WeLM-HD4-617B 也在九项共同任务上全部提高，GPQA Diamond 从 89.1 升至 91.2，HLE 从 33.6 升至 35.4，FrontierMath 从 49.0 升至 51.0。对照模型与 HD 模型执行相同的 early SFT-only post-training，没有 RL 阶段。617B 长上下文 tokens 不完全匹配，解读其提升时需同时考虑这一数据差异。

在 80B progressive expansion 实验中，$n=2,4,8$ 时 MMLU 分数从 85.0 增至 86.7、87.5，Pile-test BPB 从 0.386 降至 0.378。该趋势说明更大的 stream 数在当前训练设置下继续改善语言建模和综合评测。80B 附录的十项 benchmark 中，HD 提高八项，包括 Terminal-Bench 2 的 44.9 至 58.4，以及 ARC-AGI-2 的 6.9 至 11.6。

## Stream probes 与机制证据

作者在 dense Qwen3-8B-Base 的 $n=8$ 模型上检查中间 stream。隐藏状态相似度显示各 stream 表示具有阶段差异；attention affinity 显示最终 stream 会读取中间 stream。一个小规模 $n=2$ KV retention ablation 中，保留分离 stream KV 的平均分高于共享 KV：单个 full cross-stream layer 配置为 64.23 对 63.46，四个 full cross-stream layers 配置为 64.66 对 63.90。作者说明该对照规模较小，应按定性证据理解。

LM-head probe 将各 stream 状态投影到词表空间，观察 top-1 token 和熵。$n=8$ 时，中间 stream 的 probe top-1 与最终 stream E7 不同的比例最高约 63%；最终 stream 的平均熵最低，为 2.09 bits，部分中间 stream 高于 3 bits。探针结果支持逐步收敛的计算解释：中间 stream 保留更分散的候选分布，最终 stream 输出更集中的预测。LM-head probe 读取输出头诱导的词表分布；它提供状态差异证据，不构成对离散思维内容的直接解码。

## 证据边界

论文的重要扩展证据来自 80B 与 617B MoE 的 matched CPT 比较，并提供从 $n=2$ 到 $n=8$ 的 dense 8B 扩展因子实验。训练包含额外 CPT 计算，4 倍展开的实测训练开销约 4.4–5.1 倍；推理也增加序列状态与 KV 负担。其规模结论依赖 WeLM 训练栈、层级注意力布局、长上下文 schedule 和早期 SFT 设置，不能直接换算为其他模型族的普适收益。

617B 长上下文阶段的数据量少于 baseline；frontier 对照均采用 early SFT-only，缺少成熟后训练与 RL 对照。stream probes 与 KV 消融有助于说明中间状态的作用，因果机制仍需更丰富的状态干预和跨模型验证。论文论证的是序列维度扩展路线，和以共享深度循环为主的 Looped Transformer 在实现轴上相关，在训练并行性和状态组织上各有设定。

## 总结

Hidden Decoding 将潜在计算扩展带入固定主干的大模型 CPT，以多流序列、跨 stream KV 保留和分解注意力支持 80B、617B MoE 实验。论文的规模证据、训练成本报告与中间 stream probes 为“每 token 增加连续内部计算”提供了近期重要案例；其训练范式为 CPT，评测和训练成本边界应与循环深度论文分别解读。
