---
tags:
  - LLM Post Training
  - SFT
  - Knowledge Distillation
  - Data Quality
---

# The Quality-Utility Paradox: Why High-Reward Data Impairs Small Model Mathematical Reasoning

- 论文：[The Quality-Utility Paradox: Why High-Reward Data Impairs Small Model Mathematical Reasoning](https://arxiv.org/abs/2606.16152)
- 代码：[Quality-Utility-Paradox](https://github.com/Dracoqhl/Quality-Utility-Paradox)
- 团队：清华大学、Microsoft Research Asia、深圳理工大学、中国石油相关研究机构
- 提交：2026-06-18，arXiv:2606.16152v1

## 概述

论文检验一个常见假设：教师或 reward model 评分更高的推理轨迹，是否一定更适合训练小模型。作者在 Qwen2.5、LLaMA-3 和 DeepSeek 系列上发现，Oracle 精炼或合成的数据通常获得更高的感知质量评分，却可能低于学生自身生成并经 rejection sampling 选择的轨迹。

分析表明，Oracle 修复逻辑的同时会引入自身的表达分布，使训练数据偏离学生的 native reasoning distribution，增加适应成本。论文提出 Style-Aligned Refinement：保留学生原始轨迹的结构和风格，只让 Oracle 修复逻辑问题。该方法降低 perplexity 并恢复下游效用。

## 1. 数据与实验框架

论文从 NuminaMath CoT 抽取 100K 问题，让 Qwen2.5-Math-1.5B 每题以温度 1.0 生成 8 条候选；有正确解的题保留一条，不按质量再次排序，得到约 34K 个可解问题。固定这批题构造四条**主对照**数据流：NuminaMath Subset（原始 CoT）、SLM-RFT（学生正确轨迹，经保守清理）、Oracle-Refined（GPT-5.2 修补学生轨迹）和 Oracle-Synthesized（GPT-5.2 从头生成）。Style-Aligned 是后续机制验证的额外干预数据，不属于最初四路主对照。固定问题集合能控制题目难度，但各流的逻辑步骤与表达形式仍可能一起变化。

主目标模型为 Qwen2.5-Math-1.5B；论文又在 Qwen2.5-Math-7B、LLaMA-3.2-3B 与 DeepSeekMath-7B 上检查现象。训练分别用普通 SFT 和 Dynamic Fine-Tuning（DFT）：DFT 用 stop-gradient 的目标 token 概率为交叉熵加权，降低低置信 token 的更新权重。风格对齐干预分别用 Qwen2.5-Math-72B-Instruct 和 GPT-5.2；外部质量主评分器为 Qwen2.5-Math-72B-Reward，另用 Skywork 与 Nemotron reward model 交叉核验。下游评测包括 MATH-500、Minerva Math、AIME24、AMC23 和 OlympiadBench。

论文把指标分成三类：reward model 衡量外部感知质量，Global PPL 衡量目标学生对轨迹的预测负担，Avg@16 衡量 16 次独立 zero-shot CoT 采样的平均准确率。解码温度为 1.0、最大 4096 token；这里的 Avg@16 是平均准确率，不是“至少一次正确”的 pass@16 覆盖率。另用 GPT-5.2 judge 对 3,543 对轨迹评估语义保留；它检查变量定义、中间方程、假设和试错步骤等信息原子，而不把表面格式差异直接判为语义差异。

## 2. Quality-Utility Paradox

作者把训练数据的评价拆成两项：reward model 或 judge 眼中的 perceived quality，以及在目标小模型上微调后的 downstream utility。结果中，SLM-RFT 数据的 reward 分数较低，却常取得更高下游准确率；Oracle 处理的数据 reward 分数更高，却可能降低 Avg@16。

表 1 中 reward 均分按 Oracle-Synthesized、NuminaMath、Oracle-Refined、SLM-RFT 排序分别为 1.88、1.78、1.70、1.47；DFT 后的五项 Avg@16 却分别为 30.02、31.28、34.06、37.06。普通 SFT 下四者为 23.26、16.72、19.60、22.74，排序同样未遵循 reward。这个跨两种训练目标的反转，是“感知质量与目标模型训练效用错位”的直接证据；不能把所有差异都归结为 SFT 的单一优化缺陷。

以 DFT 对比为例，SLM-RFT 的 Global PPL 为 1.52、Avg@16 为 37.06%；Oracle-Refined 的 PPL 为 1.85、Avg@16 为 34.06%；Oracle-Synthesized 的 PPL 为 2.69、Avg@16 为 30.02%。Style-Aligned（Qwen）进一步把 PPL 降到 1.46，并达到 39.12%。数值说明目标模型更容易吸收 native-like 的轨迹，但 PPL 与效用的关联本身不证明因果；Style-Aligned 干预为机制解释提供了更直接的检验。

这些数字对应同一问题池和相近训练预算。作者还展示训练轨迹与学习率、batch size 扫描，SLM-RFT 的优势不是某一 checkpoint 或单一超参点的偶然值。Oracle-Synthesized 的重写幅度最大，PPL 也最高；学生原生轨迹的外部 reward 较低，却更接近学习器的 native distribution。Avg@16 是重复采样平均准确率，不能解释成答案覆盖率。

## 3. 分布漂移机制

论文把 adaptation cost 定义为目标模型处理训练轨迹时的预测负担，并用 Global PPL 及四个等长分段 $Q_1$–$Q_4$ 衡量。各数据版本的 PPL 排序在四个分段中保持一致，说明差异不是单一推理阶段造成的。

在起始分段 $Q_1$ 中，SLM-RFT 的主要损失贡献来自 “To”“the”“Let”等自然语言起始词；Oracle-Refined 则更多来自 `\\(` 等结构性符号。原生轨迹中带空格的反斜杠是常见分隔形式（频率约 2.1%），精炼后原始反斜杠频率上升约 3.6%。这说明精炼改变了模型需要预测的表述脚手架；它支持分布漂移分析，但 token 频率本身不能证明每个符号变化都会损害推理。

作者还用 GPT-5.2 对 3,543 对样本进行语义保留判断。Style-Aligned（Qwen）最接近 SLM-RFT，语义分数 4.77、平均排名 1.44，但 reward score 最低（1.37）；Oracle-Synthesized 最远，语义分数 3.91、平均排名 2.97，但 reward score 最高（1.88）。这组结果把感知质量与分布兼容性明确分开：更“漂亮”的重写可能改变问题解决路径和表述结构，保留语义并不等于保留学生可学习的形式。

## 4. Style-Aligned Refinement

Style-Aligned Refinement 要求 Oracle 对学生轨迹执行有针对性的逻辑修复，同时保留学生的句法结构、间距、列表组织和推理展开方式。它的作用是把逻辑改进和表达分布漂移解耦。

Qwen 版本的 PPL 低于原生 SLM-RFT，准确率达到 39.12，高于 Oracle-Refined 的 34.06 和 SLM-RFT 的 37.06。GPT-5.2 版本虽然与 native trajectory 的距离更大，也比标准 Oracle-Refined 恢复更多性能。论文据此认为，教师修复有益，但修复必须以目标模型易于内化的表示形式交付。

方法的操作边界是 surgical repair：prompt 要求修补逻辑错误，同时模仿学生原有的语言风格、句法组织、步骤粒度和符号习惯。它不是只凭一个风格标签重写答案；原始学生轨迹是修补对象。Qwen 版本的语义保留评分为 4.77、PPL 1.46、Avg@16 39.12；GPT-5.2 版本为 4.07、1.78、38.21。两种 Oracle 的差异说明效果受具体教师表达习惯影响，也说明“修复逻辑并保持可学形式”比单纯提高外部 reward 更贴合此实验的目标模型。

## 5. 稳健性、复现与边界

跨 reward model 的分析显示，native-distribution 数据在多个评估器中仍可能获得较低感知质量排名，因此现象不是单一评分器造成的。复现时应固定约 34K 问题集合、每题学生采样 8 次和一条正确轨迹的选择规则，并对四路数据分别用相同 SFT/DFT 设置训练；同时记录 reward、目标学生 PPL、语义保留与 Avg@16。PPL 在论文中是分布兼容性的代理，不是数据绝对质量排名。现有证据集中在数学推理和若干开源模型，Style-Aligned 主要依靠 prompt intervention；自动发现可修复片段、跨领域风格迁移和更大模型上的稳定性仍待检验。

## 6. 方法启示

1. 训练数据评价应区分外部感知质量和目标模型的可学习性，两类指标可能产生不同排序。
2. 从学生轨迹出发的修复有助于保留目标模型可预测的表示形式，同时引入教师逻辑改进。
3. Perplexity 在论文中作为分布兼容性代理，而非绝对数据质量指标；其与效用的关系需要结合下游训练验证。
4. 数据构造实验应固定问题集合并构造平行版本，以控制题目难度；内容修复与表达形式仍需额外干预才能区分。

## 来源

Qian et al., “The Quality-Utility Paradox: Why High-Reward Data Impairs Small Model Mathematical Reasoning,” arXiv:2606.16152v1, 2026. [论文](https://arxiv.org/abs/2606.16152) [代码](https://github.com/Dracoqhl/Quality-Utility-Paradox)
