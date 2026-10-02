---
title: Efficient Parallel Samplers for Recurrent-Depth Models
tags:
  - Recurrent Depth
  - Looped Transformer
  - Diffusion Language Model
  - LLM Inference
category: LLM Reasoning
---

# Efficient Parallel Samplers for Recurrent-Depth Models

> 论文：[Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models](https://arxiv.org/abs/2510.14961)，Geiping、Yang、Su 等，ICML 2026 Spotlight Posters。本文只依据论文正文与附录解读。

## 概述

本文提出面向 recurrent-depth Transformer 的 diffusion-forcing 解码器：多个未来位置并行生成与细化潜在状态，收敛位置冻结并进入共享 KV cache。ICML 官方 Spotlight Posters 列表收录该论文。Huginn-0125 的 batch-size-1 A100 实验在 GSM8K、MATH500、HumanEval、MBPP 上取得约 4.4–4.8 倍 tokens/s，准确率变化从 −2.44 到 +0.40 个百分点。方法将循环模型的宽度并行化转化为推理吞吐，效果依赖 KV 共享、潜态稳定性及硬件波前调度。

论文研究 recurrent-depth Transformer 的生成效率。模型通过重复层增加计算深度，标准自回归采样需要等待每个词元完成全部循环。作者提出 diffusion forcing sampler：未来词元保留带噪或未收敛的潜在状态，当前前向同时生成新词元并细化已有状态；达到稳定条件的词元被冻结并写入 KV cache。该调度把循环深度的额外计算转化为跨词元并行，论文在 3.5B Huginn 模型上报告约 5× 吞吐提升，准确率损失通常约 1%。

## 模型条件

实验使用 recurrent-depth 模型 Huginn-0125 及其 SWA、数学微调变体。模型约 3.5B 参数，在 800B tokens 上训练，具有前奏块、循环 Transformer 块和输出头。循环次数为每个词元的总 recurrence $r$，每次采样内部更新次数为 $r'$。论文强调采样器依赖模型能在带噪潜在输入上保持稳定；只在固定无噪循环长度上训练的模型可能不满足该条件。

## Diffusion forcing 调度

设当前序列包含已冻结 token、正在细化的候选 token 和新进入波前的 token。一次前向对所有活动位置更新其潜在嵌入，并把新 token 加入波前。满足误差阈值 $\varepsilon$ 的位置被冻结，冻结 token 的 KV 状态进入持久 cache。算法的核心状态包括当前位置、活动窗口、内部 recurrence $r'$、总 recurrence $r$、噪声日程 $\beta_t$ 与嵌入指数滑动平均系数 $\eta$。

非自适应版本每个位置执行近似 $r+r/r'$ 的前向工作，标准自回归每个 token 约执行 $r+1$ 次；共享 KV 与并行波前使多个未来位置共用一次前向。自适应版本根据状态收敛决定冻结时间，可能减少工作，也可能因晚到的状态修改引发级联更新。论文给出算法 1 和附录算法 2，说明 token 进入、状态更新、冻结和 KV 写入顺序。

每轮先投影当前文本上下文，再初始化或更新活动位置的潜态 $z$，执行 $r'$ 次循环更新并由输出头 $C(z)$ 采样候选词元。可选噪声更新为 $z\\leftarrow(1-\\beta_t)z+\\beta_tz_{noise}$。自适应算法计算相邻更新的相对差异 $\\delta_i=\\|z_i-z_{prev,i}\\|_2/\\|z_i\\|_2$；低于阈值 $\\varepsilon$ 的连续位置被冻结，对应 KV 进入缓存，已完成潜态从活动窗口裁除。条件嵌入可用 $e_t=\\eta e_{t-1}+(1-\\eta)P(y_{current})$ 平滑，以减轻新词元改变条件引起的振荡。

## 理论分析

作者把生成过程表示为随时间推进的深度与宽度状态。深度更新为 $d_{t+1}=d_t+1$；宽度由 token 进入和退出决定。理论结果比较深度扩展与宽度扩展：在相同缩放因子下，深度扩展可以表达更多递归变换；prefill 成本同时包含注意力与线性层，KV 共享使并行波前的 I/O 成本接近处理单个位置。论文还给出收敛阈值 $L^*$：较短序列和较小波前更容易从并行化获得收益，序列较长时内存访问和状态稳定性成为约束。

解码定理在相同运行时间预算下比较两类策略：若模型支持 KV 共享且波前规模不超过硬件阈值 $L^*$，diffusion forcing 可保持相同深度进度并获得更大的序列宽度。表达能力结论依赖潜态能够跨 recurrence 共享缓存、并行波前的 I/O 成本可以摊薄。阈值受硬件和内存带宽影响，理论结果本身没有给出所有模型和服务批次都能取得的加速倍数。

## 实验设置

评测包括 GSM8K、MATH500、HumanEval 和 MBPP，指标为答案准确率与 CUDA event 测得的中位 tokens/s。基线包括静态 autoregressive sampler、动态 KV 版本和经过调参的 self-speculative decoding。作者扫描内部 recurrence $r'$、退出阈值 $\varepsilon$、噪声日程和 EMA 系数，并在标准 Huginn、SWA checkpoint 与数学微调模型上复测。

## 主要结果

表 1 在 batch size 1、动态 KV cache、同一 Transformers 后端和 A100-40GB 上比较。以总 recurrence $r=32$ 的静态 AR 为参考，GSM8K 为 41.77% / 36.1 tokens/s，MATH500 为 17.60% / 6.4，HumanEval 为 22.56% / 13.5，MBPP 为 31.60% / 15.3。diffusion sampler（$r'=4,\beta_t=0$）分别为 42.08% / 157.3、18.00% / 30.3、20.12% / 64.9、31.00% / 70.2；相对速度约 4.36×、4.73×、4.81×、4.59×，准确率差异为 +0.31、+0.40、−2.44、−0.60 个百分点。作者将整体结果概括为约 5× 吞吐和约 1% 质量变化，逐任务数据表明代码任务的准确率变化更明显。

图 5 展示 $r'$ 增大时稳定性和准确率提升，同时吞吐下降，形成可调的速度—质量曲线。表 2 在 SWA checkpoint 和 MetaMath 微调 checkpoint 上复测，超参数保持相近，速度增益约 4–5×，准确率偏差在 0.5–1% 范围。噪声与 EMA 扫描显示，少量非零动量有助于稳定状态；噪声大小需要与内部 recurrence 联合调整，过高噪声会增加收敛时间。

图 2 的状态热图显示波前先快速推进，之后活动 token 的潜在表示逐步收敛。论文还报告 sampler 的表达能力在相同硬件时间预算下高于逐词自回归基线；该结论来自理论状态空间比较和实验速度—准确率曲线，适用范围依赖 recurrent-depth 模型的训练条件。

逐词自回归对照之外，论文还评估 $r=4,8,64$ 静态 recurrence、逐 token 自适应退出及调优后的 self-speculative decoding。默认 diffusion sampler 使用 $\\varepsilon=0.03,\\beta_t=0,\\eta=0.1,r'=4$，最大波前宽度 128。提高内部 recurrence 通常改善稳定性与准确率并降低吞吐；阈值扫描呈现速度—质量权衡。A100 上波前宽度 64–128 较合适，说明并行规模受具体硬件限制。

## 失败模式与边界

状态在较晚时刻仍发生明显变化时，提前冻结会把未收敛的 KV 写入缓存，造成后续 token 的误差传播。自适应 sampler 可能出现级联更新，晚到的 token 修改早期活动窗口。附录失败案例显示，大波前初始化为无信息 token 时，早期序列推进可能停滞，直至前部位置稳定后恢复。噪声较大或 recurrence 轮数较少也会延长潜态收敛；联合调节 $r'$ 与噪声计划可以改善稳定性。实验覆盖 Huginn-0125、SWA 与数学微调模型，吞吐基准采用 batch size 1、A100-40GB 和 Transformers 动态 KV，动态批处理引擎未测量。速度收益应限定在此实验配置，论文也未覆盖更大模型、不同硬件互联和长上下文服务。

## 可迁移设计点

1. 将循环深度计算组织为活动波前，让新 token 生成与旧 token 细化共享前向。
2. 用收敛阈值控制 KV 冻结，并记录未收敛状态的延迟修改。
3. 把内部 recurrence、噪声、EMA 和退出阈值作为联合速度—质量控制量。
4. 用相同模型、相同 FLOPs 和 CUDA event 同时报告吞吐与准确率。

## 总结

该工作把 recurrent-depth 模型的额外循环从逐词串行过程改造成带潜在状态细化的并行采样过程，连接了循环 Transformer 与 diffusion language model 的调度思想。它在当前路线中承担推理系统层角色，与 LoopFormer 的预算条件化架构形成互补。

## 与相邻工作的关系

该 sampler 以推理阶段的宽度并行隐藏状态细化为中心，直接处理 recurrent-depth 模型每 token 多轮计算带来的解码时延。LoopFormer 着重训练单一循环模型适应可变推理预算，PonderLM-2 将 latent thought 纳入预训练；Hidden Decoding 研究序列展开与大模型 CPT 的扩展能力。相邻工作覆盖模型训练、预算控制、frontier scaling 与解码执行等层面，当前 sampler 的速度结论限于 Huginn-0125 与 A100 batch-size-1 基准。
