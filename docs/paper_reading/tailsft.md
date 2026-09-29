---
tags:
  - LLM Post Training
  - SFT
  - Data Selection
  - Coverage
---

# TailSFT: Filtered Fine-Tuning Improves Post-Training Performance

- 论文：[TailSFT: Filtered Fine-Tuning Improves Post-Training Performance](https://arxiv.org/abs/2608.25756)
- 代码：论文页面未给出官方代码链接
- 团队：University of California San Diego；部分工作在 Microsoft Research NYC 完成
- 提交：2026-08-26，arXiv:2608.25756v1

## 概述

TailSFT 研究 SFT checkpoint 作为后续强化学习初始化时应优化什么。论文的出发点是：SFT 的交叉熵会继续提高已经容易生成的目标序列概率，可能挤压模型对其他有用响应的概率质量；而 RL 的有效学习信号取决于有限 rollout 中能否采到可奖励响应。TailSFT 在序列级别过滤相对初始模型已经拟合充分的样本，把更新集中到仍未充分建模的尾部响应。

论文在受控图导航任务和 OLMo-3 7B 的数学、代码实验中验证了这一设计。TailSFT 常常牺牲 pass@1 或交叉熵，却提高 pass@16；在后续 GRPO 中，较高覆盖率的 checkpoint 也带来更高 pass@1。方法本身是轻量的 SFT 数据过滤，不需要可微的 coverage 目标。

## 1. Coverage 视角

给定 prompt $x$、响应 $y$ 和二值奖励 $R(x,y)$，模型在 $x$ 上采到奖励响应的概率为

$$p_\pi(x)=\Pr_{y\sim\pi(\cdot\mid x)}[R(x,y)=1].$$

进行 $K$ 次独立 rollout 至少得到一次奖励响应的概率为 $1-(1-p_\pi(x))^K$。当一个 prompt 的所有 rollout 都没有奖励时，该 prompt 没有直接的正向 RL 信号。由此，初始化模型的价值不仅在于单次准确率，还在于有限采样预算下能否覆盖数据策略中的有用响应。

论文用 coverage profile 描述模型相对数据生成策略 $\pi_D$ 的欠覆盖质量，并把大规模 pass@K 作为可测量代理。较大的 $K$ 对应 RL 能够使用的 rollout 预算；较高 pass@K 意味着有用响应更容易在这笔预算内出现。

## 2. TailSFT

标准 SFT 使用序列交叉熵

$$\ell_\pi(x,y)=-\log \pi(y\mid x).$$

它对每个示例持续施加增大 $\pi(y\mid x)$ 的压力，即使该响应已经足够容易生成。TailSFT 记录初始策略 $\pi_0$ 和当前策略的长度归一化序列损失，过滤相对初始策略损失下降最多的样本。保留下来的样本是相对仍未充分拟合的尾部。

过滤仅用于决定哪些序列参与训练；保留序列仍使用普通目标 token 的平均交叉熵。过滤比例 $\gamma_t$ 可以固定，也可以随训练变化。论文同时比较 absolute filtering（低于固定损失阈值停止）和 quantile filtering（每个 batch 去掉最低损失分位数），用于区分“过滤已经拟合样本”的收益和“相对初始模型过滤”的额外收益。

### 算法时序

训练开始时保存初始策略 $\pi_0$。每个 batch 计算初始策略与当前策略对同一目标序列的长度归一化损失，并用

$$d_i=\ell_{\pi_0}(x_i,y_i)-\ell_{\pi_t}(x_i,y_i)$$

表示该序列相对初始状态的拟合进度。较大的 $d_i$ 表示响应已被吸收得更多，因此进入过滤集合；其余序列继续贡献普通 SFT 梯度。过滤按序列执行，避免 token 级筛选把一条响应切成不连贯的监督片段。

### 2.1 为什么参考初始策略

在论文的理论设定中，真实奖励响应集合由初始策略重新归一化得到。初始模型因此包含奖励响应之间的相对概率结构。绝对阈值会把不同起点的响应推向同一概率门槛，标准 SFT 也可能把有限样本频率过度写入；offset filtering 依据每个响应相对初始概率的变化决定停止训练，能保留更多初始结构。论文证明，在该设定中，适当调节 offset filtering 的 coverage 不差于标准 SFT 或最优绝对阈值，并且可以严格更好。

### 2.2 受控图导航实验

每个 prompt 对应一个分层有向图，源点到目标点存在八条有效路径。预训练让模型学习通用图导航，SFT 数据为每个图确定一条奖励路径。由于奖励路径已知，pass@8 可以直接估计 coverage。

标准 SFT 获得最佳交叉熵和 pass@1，但三类过滤方法都明显提高 pass@8。该实验隔离出第一项设计依据：停止更新已经拟合的响应，能够增加有限采样下发现奖励响应的概率。论文也指出，在高度同质的图任务中，初始损失差异缺乏信息，因此 TailSFT 不一定优于其他过滤器。

## 3. 语言模型实验

作者使用 OLMo-3 7B，分别在数学和代码数据上训练，并构造 18 个“训练数据集—评测集”组合。数学使用 OpenMathInstruct-2 去重后的 350K 子集，评测遵循 OLMES，包括 2022–2025 AIME、MATH-500 最难级别和 OMEGA-500。代码使用 Magicoder、BigCode Self-OSS-Instruct 和 OpenCodeInstruct，评测 MBPP+、HumanEval+、CruxEval-I/O 和 LiveCodeBench。每个设置报告三个 seed set 的平均 pass@16。

在 18 个组合中，TailSFT 有 15 个提高 pass@16；pass@1 的变化并不一致。论文报告数学最高绝对增益约 3.1 个百分点，代码最高约 16.8 个百分点。随后将匹配的 SFT checkpoint 送入 GRPO，TailSFT 带来的更高 pass@16 通常转化为最终 pass@1 增益，最高约 3.9 个百分点。

作者还提出 coverage-ratio diagnostic：只用初始模型和一次普通 SFT 的损失统计，判断过滤是否可能改善 coverage。这个诊断用于识别 TailSFT 更有希望的训练设置，避免把过滤当成无条件增益。

### 3.1 Coverage 与交叉熵的分离

在 18 个组合中，TailSFT 的主要优势出现在 pass@16，而不是 pass@1。方法减少对已经高概率响应的继续加压，允许更多响应保持可采样概率；只看单次准确率会漏掉过滤对后续 rollout 覆盖的影响。

### 3.2 后续 GRPO

作者从标准 SFT 与 TailSFT 中选取训练步数匹配的 checkpoint，再使用相同 GRPO 配置继续训练。TailSFT checkpoint 在 rollout group 中更常产生正奖励样本，最终 pass@1 因而继续受益。这个实验单独检验了 checkpoint 作为下一阶段初始化的价值。

## 4. 结果边界

论文研究的是为后续 RL 准备初始化，直接评测主要是数学和代码；没有图像描述或多模态 caption 实验。TailSFT 的“尾部”定义依赖学生相对初始策略的损失变化，不能简单等同于教师评分最低样本。论文结果支持一种阶段感知的数据目标：中间 checkpoint 应按后续训练所需的可采样覆盖来判断。

## 5. 复现要点

实现需要保存初始模型对训练序列的 loss，在当前 checkpoint 重新计算同一 loss，并固定过滤比例或过滤调度。数学实验使用去重后的 350K 子集，代码实验使用三个独立数据源；每个设置报告三个 seed set 的 pass@16。coverage-ratio diagnostic 可先判断数据分布是否存在明显尾部，再决定是否启用过滤。

## 6. 方法启示

1. 训练样本的效用依赖训练阶段目标；为后续采样式优化准备初始化时，coverage 可比单样本交叉熵更相关。
2. 初始策略既提供起点，也包含可用于判断样本相对拟合进度的信息。
3. 过滤比例需要结合数据分布和诊断指标选择；论文结果不支持把 TailSFT 视为所有设置下的无条件改进。

## 来源

Malladi et al., “TailSFT: Filtered Fine-Tuning Improves Post-Training Performance,” arXiv:2608.25756v1, 2026. [论文](https://arxiv.org/abs/2608.25756)
