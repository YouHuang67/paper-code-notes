---
title: PonderLM-2：连续潜在思维的预训练
tags:
  - Latent Reasoning
  - Continuous Thought
  - Recurrent Computation
  - LLM Pretraining
category: LLM Reasoning
---

# PonderLM-2：连续潜在思维的预训练

> 论文：[PonderLM-2: Pretraining LLM with Latent Thoughts in Continuous Space](https://arxiv.org/abs/2509.23184)，Zeng 等，ICML 2026 Spotlight Posters。代码：[LUMIA-Group/PonderLM-2](https://github.com/LUMIA-Group/PonderLM-2)。本文只依据论文正文与附录解读。

## 概述

PonderLM-2 将逐词元 latent thought 纳入语言模型预训练，以 Jacobi iteration 并行逼近推理时的递归隐藏状态。Pythia 模型在 The Pile 300B tokens 上的实验报告，1.4B 参数版本在九项下游任务均值上超过 2.8B Pythia；代价是训练 FLOPs 约为标准预训练的 8 倍，且每个词元增加潜在计算。该工作将潜式推理从任务微调推进到预训练目标，并以固定点一致性处理批量训练。

PonderLM-2 将每个词元生成前的隐藏状态反馈纳入语言模型预训练。模型先根据当前位置得到隐藏状态，将该状态作为下一次前向的输入嵌入，经过若干 latent thoughts 后再预测实际词元。论文的主要工程问题是训练时的左到右递归依赖；作者用 Jacobi iteration 并行更新整段序列的隐藏状态，并以固定点一致性作为训练与推理的连接。ICML 官方 Spotlight Posters 列表收录该论文。Pythia 系列实验使用 300B tokens，报告 1.4B 参数模型在多项困惑度和下游任务上超过 2.8B 标准 Pythia。

## 模型与目标

标准模型由 token embedding 序列产生隐藏状态并立即预测下一个词元。PonderLM-2 延迟词元采样，把当前位置的状态继续送回模型。对输入嵌入 $E=[e(x_1),\ldots,e(x_T)]$，第 $k$ 次并行状态为 $H^{(k)}=[h_1^{(k)},\ldots,h_T^{(k)}]$，交错序列写为

$$S^{(k)}=[e(x_1),h_1^{(k)},e(x_2),h_2^{(k)},\ldots,e(x_T),h_T^{(k)}].$$

Transformer 对 $S^{(k)}$ 计算下一轮状态 $H^{(k+1)}$。经过 $K$ 轮后，在隐藏状态位置计算预测 $x_{i+1}$ 的交叉熵：

$$\mathcal L=-\sum_{i=1}^{T}\log p_\theta(x_{i+1}\mid h_i^{(K)}).$$

训练实例随机从 $K\in\{2,3\}$ 采样，减少模型对固定轮数的依赖。推理时每个实际词元前执行一次或多次 latent recurrence；位置编码保持原词元的位置，因此上下文窗口的索引长度不因潜在步数而缩短。

## Jacobi 并行训练

直接按词元展开会产生 $h_1\rightarrow h_2\rightarrow\cdots\rightarrow h_T$ 的长串依赖。PonderLM-2 把整段隐藏状态视为待求的固定点：

$$H^{*}=\Phi(H^{*};E),\qquad H^{(k+1)}=\Phi(H^{(k)};E).$$

由于因果注意力，位置 $i$ 只依赖前缀；当前缀状态稳定后，该位置的更新也稳定。论文据此说明并行 Jacobi 更新最多在 $T$ 轮达到与顺序推理一致的固定点，并用实验验证前几轮已经快速收敛。该方法将训练并行性换成多轮整段前向，$K$ 越大，潜在计算能力与训练成本同时增加。

附录 D 将收敛误差定义为相邻迭代 RMSE $r_k=\|H^{(k)}-H^{(k-1)}\|$，并在 Pythia-410M 上拟合前约 10 轮的半对数曲线，得到 $R^2>0.95$、有效收缩系数约 $0.345$。作者报告约 4 轮可把误差降至初始值的 1%，约 6 轮降至 0.1%，与顺序推理状态的 RMSE 在约 9 轮降至 BF16 精度下限 $10^{-5}$。这些经验收敛数字来自指定模型与数据；理论上的有限步上界为 $T$，不能与少数 Jacobi 轮的经验表现混同。

## 预训练配置

论文在 Pythia-410M、1.4B 等规模上使用 The Pile 的 300B tokens 进行预训练，并补充 GPT-2 与 LLaMA-3-3B 的实验。评测包括 Pile validation、Wikitext、LAMBADA、ARC-Easy/Challenge、PIQA、SciQ、HellaSwag、WinoGrande 和 RACE；指令跟随使用 MT-Bench，推理扩展使用 GSM8K。比较对象包括官方 Pythia、PonderLM、OPT、BLOOM、TinyLLaMA 和参数量更大的标准模型。

## 实验结果

PonderLM-2-Pythia-1.26B 达到官方 Pythia-2.8B 的表现，论文将其归因于 55% 的参数量；1.4B 版本在相同 300B tokens 预算下超过 Pythia-2.8B。表格中的 0-shot 平均准确率为：Pythia-2.8B 57.3%，PonderLM-2-Pythia-1.4B 58.5%；5-shot 对应数值为 57.6% 与 59.5%。PonderLM-2-Pythia-410M 的 0-shot 平均准确率为 51.9%，相对 Pythia-410M 的 47.6% 提升 4.3 个百分点；1.4B 版本提升 4.4 个百分点。MT-Bench 的写作、推理、编码、数学等类别中，PonderLM-2 均高于对应 Pythia。

扩展实验显示，在 GSM8K 上，潜在思维与常规测试时采样可以叠加。消融比较了 Jacobi 轮数随机化、位置处理和不同 latent steps；固定轮数训练容易产生训练/推理偏差，随机取 2 或 3 轮有助于泛化。收敛图显示隐藏状态误差随迭代下降，前缀位置先稳定，后续位置随后稳定。

训练计算量是重要代价。论文将一次初始前向记为 $1\times$，每轮 Jacobi 在长度约翻倍的交错序列上运行，末尾损失前向再增加 $2\times$，总 FLOPs 近似为 $3+2K$ 倍标准预训练；$K\in\{2,3\}$ 均匀采样时平均约 $8\times$。作者因此在从头预训练之外测试对 LLaMA-3-3B 的 5B-token continual pretraining：PonderLM-2 CPT 的 0-shot 平均准确率为 66.2%，vanilla CPT 为 65.2%，5-shot 分别为 67.7% 与 66.2%。该结果显示现成底座继续训练也能获得增益，成本对照仍取决于相同数据预算下额外的迭代前向。

## 计算与证据边界

主预训练基于 Pythia 体系，在 The Pile 的 300B token 预算上训练 410M 与 1.4B 模型。Table 2 的九项下游任务平均值显示，PonderLM-2-410M 的 0-shot 为 51.9%，Pythia-410M 为 47.6%；PonderLM-2-1.4B 为 58.5%，Pythia-1.4B 为 54.1%，Pythia-2.8B 为 57.3%。5-shot 对应为 51.9%、47.6%、59.5%、54.1%、57.6%。任务包含 ARC-Easy/Challenge、PIQA、SciQ、HellaSwag、WinoGrande、RACE 等。作者还报告 Alpaca 微调后的 MT-Bench，以及 LLaMA-3-3B 上 5B-token continual pretraining：九项任务 0-shot 平均为 66.2%，vanilla CPT 为 65.2%；5-shot 为 67.7% 与 66.2%。后者支持方法可用于既有底座继续训练，数据仅覆盖一个模型规模和固定 token 预算。

PonderLM-2 把额外推理计算放进每个词元的前向路径，解码会增加单词元延迟。Jacobi 并行化训练阶段的整段状态估计，推理仍沿序列递归反馈隐藏状态。论文报告模型质量、困惑度、下游准确率和收敛诊断，没有给出跨硬件端到端成本曲线，也没有分项测量每个 latent step 的吞吐、显存和能耗。表 3 提供 LLaMA-1.4B 上相对 2× 推理 FLOPs 的比较，在线批处理与延迟分位数未报告。主规模比较覆盖 Pythia-410M 和 1.4B，不能据此推断更大模型的训练成本规律。

Jacobi 的有限步收敛结论来自因果依赖：位置 $i$ 只使用其前缀状态，前缀稳定后该位置更新随之稳定，因此最迟 $T$ 轮得到顺序解。这个结论给出迭代上界，训练时实际随机采样 2 或 3 轮。Pythia-410M 的相邻轮状态 RMSE 半对数拟合 $R^2>0.95$，有效收缩系数约 0.345，约 4 轮将误差降至初始值的 1%。训练成本近似为 $(3+2K)$ 倍标准预训练；$K$ 在 2、3 间均匀采样时约为 8 倍。该成本应与准确率提升一起评估。

## 可迁移设计点

1. 用隐藏状态固定点近似把递归训练改写成并行迭代。
2. 训练时随机化 latent iteration，建立多预算推理的鲁棒性。
3. 同时报告参数量、训练 tokens、推理 FLOPs 和质量指标，避免单一规模比较。
4. 将潜在计算与传统 test-time scaling 组合，测量二者是否提供互补增益。

## 总结

PonderLM-2 的核心推进是把 latent thought 从任务级微调机制推进到预训练目标，并以 Jacobi iteration 处理训练可扩展性。其 300B-token、多模型家族实验使该路线获得较强的近期关注度；推理成本与更大规模稳定性仍需要后续工作验证。

## 与相邻工作的关系

PonderLM-2 的关键特征是每个实际词元前执行连续隐藏状态递归，并把该机制直接纳入预训练。LoopFormer 研究不同循环长度下的轨迹一致性，承担预算适配问题；Efficient Parallel Samplers 研究 recurrent-depth 模型的并行推理调度；Hidden Decoding 以 stream 扩展将潜在计算推进至百亿与数百亿 MoE 的 CPT。PonderLM-2 的 300B-token Pythia 实验与 Hidden Decoding 的 frontier CPT 规模不能直接构成规模对照，两者的架构、训练目标和计算报告均不同。
