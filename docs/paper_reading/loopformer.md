---
title: LoopFormer：可变深度循环 Transformer
tags:
  - Latent Reasoning
  - Looped Transformer
  - Adaptive Depth
  - Test-Time Compute
category: LLM Reasoning
---

# LoopFormer：可变深度循环 Transformer

> 论文：[LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation](https://arxiv.org/abs/2602.11451)，Jeddi、Ciccone、Taati，ICLR 2026 Poster。项目页：[loopformer.github.io](https://loopformer.github.io/)。本文只依据论文正文与附录解读。

## 概述

LoopFormer 面向循环 Transformer 的推理预算弹性，以时间和步长调制共享循环块，并通过完整轨迹与 shortcut 轨迹的一致性训练支持多种循环长度。约 1B 参数实验覆盖 The Pile 25B tokens 和 24×、12×、6×计算预算。结果显示同一模型可在不同预算下运行，任务质量会随预算降低而下降；双轨迹训练约增加 1.5× FLOPs，且论文对一致性项的正文公式与伪代码给出不同监督对象，构成复现时需要核实的证据边界。

LoopFormer 研究循环 Transformer 在不同推理预算下的稳定运行。固定循环次数训练的模型在推理时改变循环长度，表示可能停滞或漂移。LoopFormer 用时间与步长条件调制每次循环，并以 shortcut-consistency loss 对齐不同长度轨迹，使较短轨迹提供有用表示，较长轨迹继续细化。论文在约 1B 参数、24 层非循环基线和 The Pile 25B tokens 设置下评估困惑度与十项零样本任务。ICLR 官方日程将其列为 Poster；ICLR Spotlight Posters 名单未收录该论文。

## 轨迹条件化

设循环状态为 $h(t)$，一次轨迹由归一化时间 $t$ 和步长 $\Delta t$ 描述。LoopFormer 在每个循环块中以条件向量调制两个 RMSNorm 的缩放与门控参数：

$$[\alpha_{msa},\alpha_{mlp},\gamma_{msa},\gamma_{mlp}]=g(e_t,e_{\Delta t}),$$

其中 $e_t$ 与 $e_{\Delta t}$ 是时间和步长嵌入，$g$ 为 MLP。循环块接收当前状态、上下文和轨迹条件，输出下一状态。不同预算对应一组满足 $\sum_i\Delta_i=1$ 的步长序列；条件化让同一参数共享模块识别自己处于哪一段轨迹。

## Shortcut-consistency 目标

训练时对同一输入采样不同循环长度和步长安排。较短轨迹的末状态被约束接近标准完整轨迹的对应目标，长轨迹则继续优化语言建模损失。该设计让模型在提前停止时保留可用表示，同时避免每种循环长度独立训练。论文还比较了固定循环、early-exit 和 time-modulated baselines，分离时间条件、步长条件和 shortcut loss 的作用。

完整的最长循环轨迹承担 next-token 语言建模目标；另采样 shortcut 轨迹，将预测分布与完整轨迹对齐，并对循环表示施加一致性约束。正文把一致性写作 stop-gradient logits matching，Algorithm 1 的伪代码则对最长轨迹隐藏状态与 shortcut 隐藏状态计算平方误差。两种写法在监督对象上存在差异，因此本文仅将共同功能概括为跨轨迹对齐。时间 $t$ 和步长 $\\Delta t$ 条件输入 MLP，再调制 RMSNorm 的 scale 与门控，使共享循环块可以区分轨迹所处区间。该监督定义差异会影响复现时的目标实现，应以作者公开实现或正式勘误进一步核实；本报告不把两种损失视为等价形式。

## 实验设置

主实验训练约 1B 参数模型，模块层数 $k\in\{2,3\}$，循环次数 $L\in\{2,4,8,12,24\}$，数据为 The Pile 去重子集，总量约 25B tokens。评测使用困惑度和十项零样本语言/推理任务，包括 ARC、HellaSwag、PIQA、WinoGrande、OpenBookQA、BoolQ、RACE 等；作者报告归一化平均准确率。比较对象包括 vanilla Transformer、fixed-loop、TMLT 及 early-exit 变体，并按 FLOPs 匹配训练和推理预算。

## 主要结果

在 $3\otimes8$ 模型上，LoopFormer 在 24×、12×、6×三个预算中都能保持性能随计算量平滑变化。表 1 中，24×预算下 LoopFormer 的 The Pile perplexity 为 10.28、十项任务平均准确率 44.81%；相同 FLOPs 的 24 层基线为 9.49 和 45.27%，TMLT 为 10.38 和 44.69%。12×预算下 LoopFormer 为 11.12 和 43.73%，对应 12 层基线为 9.98 和 44.93%；6×预算下 LoopFormer 为 14.30 和 40.36%，6 层基线为 11.13 和 42.73%。因此模型在压低预算时保持有用表现，但 PPL 与准确率均显示预算下降会付出质量代价。

在相同 3 层参数规模下，表 2 显示 LoopFormer 从 2、4 到 8 次循环后，平均准确率依次为 40.36%、43.73%、44.81%，3 层单次基线为 40.93%；对应 The Pile PPL 为 14.30、11.12、10.28 与 12.93。额外循环逐步改善语言建模与平均任务表现，达到 8 次循环时超过浅层单次基线。

表示分析显示，LoopFormer 的曲率、各向异性和 prompt entropy 随循环推进持续变化，跨步 CKA 呈现逐步漂移；early-exit 基线的指标更平坦、CKA 更高。固定预算轨迹枚举显示同一 $3\otimes8$ 模型在 4 步预算下 PPL 跨度约 1.4、平均准确率跨度约 1.3 个百分点；$2\otimes12$ 在 6 步预算下 PPL 跨度接近 3。表现较好的步长日程偏向前期粗步、后期细步，表明预算调度本身会改变推理质量。

作者枚举不同步长安排，发现相同总预算下的轨迹顺序会改变困惑度和准确率；$3\otimes8$ 的 4 步安排准确率差异约 1.3 个百分点，$2\otimes12$ 的 6 步安排困惑度差异接近 3。该结果说明预算长度本身不足以刻画循环计算，步长调度也是模型行为的一部分。

## 附录证据

附录报告模型使用 4×H100 80GB，训练 50,000 optimizer steps，总训练量约 25B tokens；AdamW 峰值学习率 $6\times10^{-4}$，最低 $6\times10^{-5}$，4000 warmup steps。完整轨迹加一个随机 shortcut 轨迹使训练 FLOPs 约为 fixed-loop 基线的 1.5×，在该硬件设置下墙钟时间约慢 1.3×。FLOPs 匹配实验把训练步数降至约 34,000，使 LoopFormer 少看约 8B tokens；这组设置下，LoopFormer PPL 为 10.71、平均准确率 44.21%，TMLT 为 10.38、44.69%，LoopFormer 保留预算弹性，整体指标与 TMLT 接近。

consistency 项在论文正文目标式中描述为 stop-gradient logits consistency，Algorithm 1 伪代码则写成最长轨迹隐藏状态与 shortcut 状态的平方误差。论文主文与伪代码在该项的具体作用对象表述不同；本报告按共同功能解释为对齐短轨迹与完整轨迹，严格复现时需以作者实现或澄清为准。方法采用用户指定的全局序列预算，当前实验没有实现实例级或 token 级动态分配。

## 证据边界

实验集中在约 1B 参数语言模型和 The Pile，循环预算为预设的全局序列长度。论文展示了预算弹性和表示轨迹变化，尚未证明 token 级路由、超长上下文以及大规模工业模型上的同样收益。shortcut-consistency 对训练稳定性的贡献由消融支持，具体损失权重、采样分布和硬件吞吐仍依赖附录配置。主要对照与成本数值见“主要结果”和“附录证据”；一致性项的监督对象差异仍需作者实现或澄清确认。

## 可迁移设计点

1. 用时间与步长条件描述循环位置，使一个模型覆盖多种计算预算。
2. 以完整轨迹作为 shortcut 参照，约束提前结束的状态质量。
3. 评测预算顺序和步长安排，避免把循环次数视为唯一控制变量。
4. 将 CKA、曲率、各向异性和熵轨迹作为循环表示是否真正演化的诊断。

## 总结

LoopFormer 将循环 Transformer 从固定展开长度推进到预算条件化推理，并用轨迹一致性目标保持不同预算下的表示质量。它是当前 latent reasoning 主线上连接循环架构与 adaptive test-time compute 的代表工作。

## 与相邻工作的关系

LoopFormer 将循环深度的控制变量从固定迭代次数扩展为时间与步长条件，重点解决同一模型跨计算预算的轨迹质量。PonderLM-2 通过每个 token 前的 latent thought 改变预训练目标；Efficient Parallel Samplers 保留既有 recurrent-depth 模型，调整推理时的状态更新调度以提高吞吐。三者分别对应预算弹性、预训练中的潜在计算和并行解码，报告结果时应保留各自的训练条件和成本口径。
