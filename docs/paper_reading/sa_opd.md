---
tags:
  - LLM Post Training
  - On-Policy Distillation
  - Vision Language Model
  - Data Selection
---

# When Teachers Mislead: Spurious-Signal-Aware On-Policy Distillation

- 论文：[When Teachers Mislead: Spurious-Signal-Aware On-Policy Distillation](https://arxiv.org/abs/2608.03632)
- 代码：论文页面未给出官方代码链接
- 团队：浙江大学、ByteDance、上海人工智能实验室
- 提交：2026-08-04，arXiv:2608.03632v1

## 概述

SA-OPD 研究教师 token 信号与输入之间的关系。OPD 在学生自己采样的轨迹上使用教师逐 token 分布，但教师的判断可能来自与输入无关的语言先验、格式习惯或模板，而不是任务证据。这类信号可能产生大梯度，却不提供改善任务结果的方向。

论文定义 spurious signal，并提出同时依据 input-groundedness 和 distillation divergence 过滤 token：只有当 token 对输入的依赖弱、且蒸馏差异处于极端范围时才过滤。作者在 Qwen3/Qwen3.5 语言模型和视觉语言模型设置上进行实验，报告相对 Vanilla OPD 及其他 selective OPD 的稳定改进。

## 1. Spurious signal

设教师和学生在 token 位置 $t$ 的分布分别为 $q_t$ 与 $p_t$。普通 OPD 根据二者差异施加梯度；但差异本身无法说明教师信号是否由当前输入支持。语言先验会让教师在常见格式、套话和模板位置上产生高置信度，视觉任务中还可能让教师补出图像并未提供的对象或属性。

SA-OPD 将信号质量拆成两个维度：一是输入 groundedness，衡量蒸馏方向是否依赖当前输入；二是 optimization impact，以蒸馏 divergence 近似该 token 对更新的影响。低 groundedness 且高 impact 的 token 是主要过滤对象。这样只删除高影响的可疑更新，保留低影响格式变化和输入有依据的困难 token。

## 2. 算法框架

训练仍使用学生 on-policy 轨迹。对每个 token 计算输入 groundedness proxy 与教师—学生 divergence，形成二值或软过滤掩码。掩码作用于 OPD loss，学生的普通语言建模或其他训练项保持不变。方法的计算开销被设计为轻量级，重点是 token 级选择而不是重新训练教师。

具体流程是：学生先对输入采样一条轨迹；教师在同一前缀上给出 token 分布；方法估计该位置的 input-groundedness，再计算教师与学生分布的 divergence；只有同时满足“groundedness 低、divergence 高”的位置进入过滤集合，其余位置保留原始 OPD 梯度。直觉上，若对输入做轻微扰动后教师方向基本不变，该方向更像语言先验；若它同时与学生差异很大，更新量就足以造成明显漂移。联合条件因此把“影响大”与“由证据支持”分开。

设 $g_t$ 为 groundedness 分数，$d_t$ 为 teacher-student divergence，保留掩码可写为

$$m_t=\mathbf{1}[g_t\geq\tau_g\ \text{or}\ d_t\leq\tau_d],$$

并将 $m_t$ 乘到 token 级 OPD 项上；等价地，$1-m_t$ 是被过滤的低 groundedness、高 divergence 交集。阈值或连续权重由验证集调节；论文的关键是交集判据，而不是某一种固定阈值。

论文的理论分析指出，先验诱导的梯度对输入特定目标的 alignment 较弱；低信噪比 OPD 更新会造成参数漂移。SA-OPD 的联合条件把“看起来很有影响”与“确实由输入支持”区分开。

## 3. 实验设置

语言实验使用 Qwen3 和 Qwen3.5 的 non-thinking 变体，包含 Qwen3-4B-Instruct→1.7B 以及 Qwen3.5-35B-A3B→2B 等教师—学生组合。数学数据从 DeepMath 中保留难度至少为 6 的样本，并对其余数据随机抽取 30%，总量约 7K examples；视觉实验从 VERO-600K 的 Captioning & IF、Grounding、Counting & Search 子集抽取 10%，并加入 MMRL30K visual reasoning。评测覆盖五个数学基准及六个视觉/多模态基准，比较 Vanilla OPD、已有 selective OPD 和 SA-OPD。

## 4. 主要结果与消融

SA-OPD 在语言和视觉语言设置中一致优于 Vanilla OPD 与竞争选择方法。消融分别移除 groundedness 或 divergence 条件，结果显示单独使用“信号影响大”会把高梯度先验一起保留下来，单独使用 groundedness 又可能过滤掉真正困难且有价值的更新；两者联合更稳定。附录给出过滤 token 示例、详细算法和计算开销。

消融的解释重点在交集：只按 divergence 选样本会偏向教师最自信、与学生差异最大的 token，其中包含格式和先验补全；只按 groundedness 过滤则无法区分低依赖但影响很小的无害 token 与真正会改变参数的错误方向。联合判据把过滤预算集中到前者，因而在数学与视觉任务上都更稳定。

## 5. 解释与限制

论文的结论是监督质量至少包含“输入对应度”这一维度。教师更强、文本更流畅并不保证 token 更新更适合当前输入。实验主要集中在 Qwen 家族和数学/视觉语言任务；groundedness proxy 的具体形式依赖任务与模型，论文没有证明它可以直接替代所有外部 verifier。

## 6. 复现与限制

复现需要保存学生 rollout，在相同前缀上运行教师，按 token 对齐两套分布，计算 groundedness proxy 与 divergence，再把 mask 接入 OPD loss。主要成本是额外的教师前向和 groundedness 估计；论文没有提出 CUDA、Triton 或其他自定义算子，方法属于训练目标和样本路由层面的改动。groundedness proxy 依赖输入扰动或任务特定信号，迁移到新的 caption 或多模态任务时需要重新校准阈值。

## 7. 方法启示

1. 教师—学生分歧只能衡量更新差异，不能单独代表监督质量；需要同时衡量输入依赖。
2. 联合过滤条件把高影响且低输入支持的信号与普通格式差异区分开。
3. 过滤粒度可以下沉到 token，避免整条 rollout 被同一质量标签支配。
4. groundedness proxy 是任务相关的近似量，应用到新任务时需要通过独立消融验证。

## 来源

Jiang et al., “When Teachers Mislead: Spurious-Signal-Aware On-Policy Distillation,” arXiv:2608.03632v1, 2026. [论文](https://arxiv.org/abs/2608.03632)
