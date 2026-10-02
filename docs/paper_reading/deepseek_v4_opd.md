---
title: DeepSeek-V4 报告中的 OPD：多教师 Full-Vocabulary 合并
tags:
  - LLM Post Training
  - On-Policy Distillation
  - Knowledge Distillation
  - Industrial Adoption
category: LLM Post Training
---

# DeepSeek-V4 报告中的 OPD：多教师 Full-Vocabulary 合并

> 论文：[DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)，2026。作者：DeepSeek-AI。本文聚焦附录 E 的 OPD 后训练部分。

## 概述

DeepSeek-V4 先按数学、代码、agent 和 instruction following 等方向训练 specialist，再用 multi-teacher OPD 将能力合并到统一学生。报告把上一代 mixed RL 后训练阶段替换为 OPD，学生在自己的 rollout 上接受多个教师的 full-vocabulary Reverse KL。为支撑这一目标，报告同时给出 hidden-state 缓存、logit 重建、教师调度、中心化权重加载、FP4 前向和 token-granular write-ahead log（WAL）等训练基础设施。

这部分报告的核心问题是多领域能力合并。每个 specialist 在单一能力上经过针对性 SFT 与 RL，统一模型需要在一次后训练中吸收这些策略，同时保持学生自身的部署分布。报告将 OPD 的贡献拆成两个层面：目标层面用学生轨迹和教师概率分布建立密集训练信号，系统层面让多个大教师能够在超长 rollout 和 full-vocabulary 计算下被调度。

## 后训练主线

第一阶段 specialist training 对不同领域分别执行 SFT 与 RL，使每个教师在局部任务上形成专长。第二阶段 multi-teacher OPD merge 由统一学生生成轨迹，并按照输入领域或教师权重吸收 specialist 分布。学生轨迹决定训练状态分布，教师集合提供目标分布；能力整合发生在学生可能实际访问的前缀上。

从训练数据流看，一条样本包含提示、学生已生成的前缀、教师索引以及教师权重。教师在该前缀上运行最后一层前向，学生则在相同位置计算自己的分布。多教师目标作用于各自负责的样本或领域，权重聚合把领域信号汇总到同一个学生更新中。报告没有给出路由器的独立网络或分类器结构，因此可确认的是按教师索引组织的调度接口，无法进一步确定领域判定是否由规则、数据标签或模型预测完成。

## 目标函数与信号形态

设教师集合为 `{π_E1,...,π_EN}`，`w_i` 为教师权重，`π_θ` 为学生策略。附录 E 给出的目标为

$$L_{OPD}(θ)=\sum_{i=1}^{N}w_iD_{KL}(π_θ\parallel π_{E_i}).$$

报告将该目标放在学生 rollout 上计算，并使用 full-vocabulary logits。每个前缀位置的教师输出覆盖整个词表，学生因此可以同时获得已采样 token 与其他候选 token 的相对概率信息。该选择要求教师与学生共享可比的输出词表和 logits 接口，也增加教师前向、通信和显存压力。

报告强调 full-vocabulary OPD 的梯度稳定性和知识保留能力，但没有提供与 sampled-token 估计、MiniLLM 或 ExOPD 在相同教师、rollout 预算和硬件条件下的受控表格。现有材料能够确认系统采用及其规模，无法单独估计 full-vocabulary 选择的因果增益。

该目标还隐含三个实现条件。第一，教师与学生必须在同一 token 空间上定义概率，或存在报告未展开的词表映射。第二，教师输出头需要与缓存的最后一层 hidden state 配套，才能在线恢复完整 logits。第三，`w_i` 的归一化和样本覆盖范围会影响不同 specialist 的梯度比例；报告只给出符号形式，未给出数值和更新规则。

## 规模化训练条件

### Hidden state 缓存与教师调度

直接缓存全词表 logits 会随序列长度、词表和教师数量快速膨胀。报告缓存教师最后一层 hidden states，再由对应 output head 在线恢复 logits。样本按 teacher index 排序，使一个 mini-batch 中最多有一个 teacher head 常驻显存；教师权重存放在中心化分布式存储中，训练需要时按教师索引加载。三个机制共同分离 hidden state、output head 和权重的驻留生命周期。

这一安排改变了缓存对象的形态：hidden state 的大小与隐藏维度相关，logits 的大小与词表相关。训练批次先完成学生 rollout 和隐藏状态记录，再按教师索引分组进行教师 head 计算，随后将恢复出的分布用于 KL 更新。排序操作增加调度步骤，同时降低多个 output head 同时驻留带来的显存峰值。报告未报告缓存命中率、教师加载延迟和通信占比，因此无法从文本估计各环节的吞吐瓶颈。

### 数值与容错

rollout 和 teacher forward 支持 FP4，以降低大规模前向的存储与带宽压力。长轨迹可能在 worker 重启或资源回收时中断；报告为每个请求记录 token 级 WAL，从中断位置恢复。若中断后从头生成，较短答案更容易落在截断窗口内，样本长度分布会发生偏移。token-granular WAL 保留已生成前缀，并让恢复过程维持 rollout 统计口径。

FP4 与 full-vocabulary OPD 的组合带来一个需要单独验证的数值问题：教师概率比值对 logit 误差敏感，低精度前向可能改变尾部 token 的相对排序。报告说明 FP4 被用于 rollout 和教师前向，同时没有给出 FP4、FP8 或更高精度教师之间的 OPD 消融，因此低精度对蒸馏梯度的影响仍属于未公开变量。

## 证据与影响范围

DeepSeek-V4 报告把 OPD 放在约 1.6T 总参数、49B activated 参数的生产级模型后训练流程中，展示多 specialist 合并和 full-vocabulary 教师访问的可运行性。这是 OPD 在超大规模系统中的采用证据，支撑重点落在系统可扩展性和训练组织方式。

报告还把 OPD 与三档 reasoning effort mode、agent rollout 和长上下文训练放在同一后训练体系中。对 OPD 本身最直接的证据是：mixed RL 阶段被 multi-teacher OPD merge 替换，且系统为 full-vocabulary 教师访问提供专门的缓存和调度路径。模型最终 benchmark 分数同时受到架构、数据、specialist RL、推理预算和 agent 工具环境影响，不能作为 OPD 的单因素效果估计。

报告没有公开 OPD 相对于 mixed RL 的同初始化受控基准，也没有披露教师数量、各 `w_i`、领域路由规则、rollout 数量、词表对齐细节和 full-vocabulary 的独立消融。因此，V4 最终模型能力不能归因于 OPD 单一因素；具体 divergence、缓存策略和 FP4 选择的收益仍需后续实验验证。

## 讨论与迁移条件

多教师 OPD 的核心接口包含四项：学生 rollout、教师路由、教师概率分布和权重聚合。迁移到较小系统时，教师调度仍需记录每条样本的教师身份与权重；否则不同领域信号会在汇总指标中相互抵消。full-vocabulary 目标还要求词表和输出头可比较，黑盒 API 只能提供有限的 sampled-token 信号。长 rollout 训练需要具备可恢复的 token 记录，以避免系统故障改变训练样本分布。

复现 V4 的 OPD 主张至少需要固定五类条件：specialist 的训练来源和能力差异、学生 rollout 的提示与长度分布、教师路由和权重、full-vocabulary logits 的数值精度、rollout 中断后的恢复策略。当前报告对前四项中的若干数值没有公开，对最后一项给出了 WAL 设计但没有提供开启与关闭的统计对照。研究环境可以先复现目标函数和教师分组，再分别测量 full-vocabulary、低精度和恢复机制的影响。

## 可迁移设计点

1. 将 hidden-state cache、output head 和教师权重拆分管理，便于多教师 full-vocabulary 训练的显存调度。
2. 把教师索引排序与权重聚合纳入训练数据结构，保证领域路由可追踪、可复核。
3. 对长 rollout 保存 token 级恢复位置，并把恢复策略视为训练统计的一部分。
4. 将 OPD 结果与 specialist 单项能力、统一学生基线和教师上界同时报告，避免把系统级模型分数直接解释为蒸馏增益。

## 来源

- [DeepSeek-V4 Technical Report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)，Appendix E。
- [DeepSeek-V4 总览](deepseek_v4.md)。
