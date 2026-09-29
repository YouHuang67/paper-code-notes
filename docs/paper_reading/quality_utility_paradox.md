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

论文从 NuminaMath 的约 100K 问题中筛出约 34K 个学生可解问题，然后固定同一问题集合构造四条平行数据流：SLM-RFT（学生采样并 rejection filter）、Oracle-Refined（Oracle 修复学生轨迹）、Oracle-Synthesized（Oracle 重新生成）和 Style-Aligned（保留学生风格的定向修复）。这样可以把问题难度、答案覆盖和表达分布的影响分开。实验使用 GPT-5.2 作为 Oracle，训练与评测覆盖 Qwen、LLaMA-3 和 DeepSeek 系列；下游评测包括 MATH-500、Minerva Math、AIME24、AMC23 和 OlympiadBench。

论文把指标分成三类：reward model/LLM judge 衡量外部感知质量，Global PPL 衡量目标学生对轨迹的预测负担，Avg@16 衡量模型进行 16 次 zero-shot CoT 采样后的解题覆盖。另用多个 reward model 交叉检查评分器偏差，并从 3,543 对轨迹中评估语义保留。

## 2. Quality-Utility Paradox

作者把训练数据的评价拆成两项：reward model 或 judge 眼中的 perceived quality，以及在目标小模型上微调后的 downstream utility。结果中，SLM-RFT 数据的 reward 分数较低，却常取得更高下游准确率；Oracle 处理的数据 reward 分数更高，却可能降低 Avg@16。

以论文报告的对比为例，SLM-RFT 的 Global PPL 为 1.52、准确率 37.06%；Oracle-Refined 的 PPL 为 1.85、准确率 34.06%；Oracle-Synthesized 的 PPL 为 2.69。Style-Aligned（Qwen）进一步把 PPL 降到 1.46，并达到 39.12 的准确率。数值说明目标模型更容易吸收 native-like 的轨迹，感知质量排序不能替代学习效用评测。

这些数字对应同一问题池和相近训练预算，因此差异主要来自轨迹来源与表达方式。Oracle-Synthesized 的重写幅度最大，PPL 也最高；学生原生轨迹的外部 reward 较低，却更接近学习器的 native distribution。论文同时报告 Avg@16，而不是只看一次采样，因而能观察到数据是否扩大了学生的可采样解覆盖。

## 3. 分布漂移机制

论文把 adaptation cost 定义为目标模型处理训练轨迹时的预测负担，并用 Global PPL 及四个等长分段 $Q_1$–$Q_4$ 衡量。各数据版本的 PPL 排序在四个分段中保持一致，说明差异不是单一推理阶段造成的。

在起始分段 $Q_1$ 中，SLM-RFT 的损失主要来自自然语言连接词；Oracle-Refined 则更多由结构性符号贡献。Oracle 轨迹把逻辑修复、重排和压缩绑定到教师偏好的展示方式，目标模型需要先适应新的句法脚手架，再学习逻辑内容。

作者还用 GPT-5.2 对 3,543 对样本进行语义保留判断。Style-Aligned（Qwen）最接近 SLM-RFT，语义分数 4.77、平均排名 1.44，但 reward score 最低（1.37）；Oracle-Synthesized 最远，语义分数 3.91、平均排名 2.97，但 reward score 最高（1.88）。这组结果把感知质量与分布兼容性明确分开：更“漂亮”的重写可能改变问题解决路径和表述结构，保留语义并不等于保留学生可学习的形式。

## 4. Style-Aligned Refinement

Style-Aligned Refinement 要求 Oracle 对学生轨迹执行有针对性的逻辑修复，同时保留学生的句法结构、间距、列表组织和推理展开方式。它的作用是把逻辑改进和表达分布漂移解耦。

Qwen 版本的 PPL 低于原生 SLM-RFT，准确率达到 39.12，高于 Oracle-Refined 的 34.06 和 SLM-RFT 的 37.06。GPT-5.2 版本虽然与 native trajectory 的距离更大，也比标准 Oracle-Refined 恢复更多性能。论文据此认为，教师修复有益，但修复必须以目标模型易于内化的表示形式交付。

方法的操作边界是 surgical repair：只改动被判定为逻辑错误、缺失或矛盾的局部内容，保留学生原有的句法组织、步骤粒度、符号习惯和解释顺序。它不把 Oracle 当作整条响应的写作者，因此把“纠正推理”与“替换表达分布”分离开来。

## 5. 稳健性与限制

跨 reward model 的分析显示，native-distribution 数据在多个评估器中仍可能获得较低感知质量排名，因此现象不是单一评分器造成的。论文的证据集中在数学推理和小语言模型，作者明确指出更大模型、非数学领域和混合指令微调仍需研究。Style-Aligned 主要通过 prompt intervention 实现，自动化风格迁移和 learner-aware reward model 尚未解决。

## 6. 复现与限制性解读

复现时应固定问题集合、采样次数和训练预算，并同时记录 reward、PPL、语义保留与 Avg@16；只报告单一 judge 分数会掩盖 learner adaptation cost。PPL 在论文中是分布兼容性的代理，不是数据绝对质量排名。证据集中在数学推理和若干开源模型，Style-Aligned 主要依靠 prompt intervention，自动发现可修复片段、跨领域风格迁移以及更大模型上的稳定性仍未解决。

## 7. 方法启示

1. 训练数据评价应区分外部感知质量和目标模型的可学习性，两类指标可能产生不同排序。
2. 从学生轨迹出发的修复有助于保留目标模型可预测的表示形式，同时引入教师逻辑改进。
3. Perplexity 在论文中作为分布兼容性代理，而非绝对数据质量指标；其与效用的关系需要结合下游训练验证。
4. 数据构造实验应固定问题集合并构造平行版本，以隔离内容变化与表达形式变化的影响。

## 来源

Qian et al., “The Quality-Utility Paradox: Why High-Reward Data Impairs Small Model Mathematical Reasoning,” arXiv:2606.16152v1, 2026. [论文](https://arxiv.org/abs/2606.16152) [代码](https://github.com/Dracoqhl/Quality-Utility-Paradox)
