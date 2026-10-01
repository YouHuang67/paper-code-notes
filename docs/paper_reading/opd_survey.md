---
title: OPD综述：从学生轨迹到后训练主线
tags:
  - LLM Post Training
  - On-Policy Distillation
  - Knowledge Distillation
  - Reinforcement Learning
  - Survey
---

# OPD综述：从学生轨迹到后训练主线

> 论文：Mingyang Song, Mao Zheng, “A Survey of On-Policy Distillation for Large Language Models”，arXiv:2604.00626v4，2026-06-18。本文只解读综述及其引用的论文，不做代码复现。

## 先给结论

On-Policy Distillation（OPD）的核心不是“把教师答案再训练一遍”，而是让学生先生成自己的轨迹，再让教师在这些学生会遇到的状态上提供密集的 token 分布或反馈。它解决的是 off-policy 蒸馏的暴露偏差：训练时学生总在教师或数据前缀上学习，推理时却必须在自己的前缀上继续生成。

截至 2026 年，OPD 已经形成一条相对清晰的主线：GKD 把数据分布和学生分布混合起来；MiniLLM 把 reverse-KL 序列目标写成可优化的策略梯度；随后工作围绕“什么分布、什么 divergence、什么信号、怎样控制成本”展开。ExOPD/G-OPD 把 OPD 接到 KL 约束的 RL 连续体上，Veto、Entropy-Aware OPD 和 Revisiting OPD 处理目标和训练稳定性，Rethinking OPD 则给出“何时值得蒸馏”的兼容性与新能力判据。Qwen3、DeepSeek-V4、MiMo-V2-Flash、GLM-5 等报告说明它已从论文技巧变成大模型后训练的系统组件，但工业报告仍不能替代严格的跨论文因果比较。

## 1. 统一问题：学生要在自己的状态上被监督

经典 token KD 通常最小化

$$
\mathcal L_{\mathrm{off}}=\mathbb E_{x,y\sim\mathcal D}\left[\sum_tD_{\mathrm{KL}}\left(p_T(\cdot|x,y_{<t})\|p_\theta(\cdot|x,y_{<t})\right)\right].
$$

这里的前缀来自标注数据或教师生成结果。学生训练时看到的状态与推理时自己的错误、犹豫和长度变化不一致，于是小的局部误差会在长序列中累积。综述借用 DAgger 的直觉：若每一步错误概率为 $\epsilon$，on-policy 训练可以把误差累积从近似 $O(\epsilon T^2)$ 降到 $O(\epsilon T)$。这不是无条件定理：LLM 教师在严重偏离的学生前缀上可能失去校准，学生轨迹也可能已经进入教师从未覆盖的区域。

综述将 OPD 写成统一目标：

$$
\mathcal L_{\mathrm{OPD}}(\theta)=\mathbb E_{y\sim\pi_{\mathrm{mix}}(\cdot|x)}\left[\sum_tD_f\left(p_T(\cdot|x,y_{<t})\|p_\theta(\cdot|x,y_{<t})\right)\right],
$$

其中 $\pi_{\mathrm{mix}}$ 决定轨迹来自数据、教师、学生或它们的混合，$D_f$ 是由凸函数 $f$ 生成的 f-divergence，满足

$$D_f(P\|Q)=\mathbb E_{y\sim Q}\left[f\left(P(y)/Q(y)\right)\right],\quad f(1)=0.$$

因此 OPD 的研究空间可以压缩成三个轴：**轨迹分布**（谁产生前缀）、**比较目标**（用哪种 divergence）、**监督信号**（白盒 logits、黑盒反馈还是自蒸馏）。这三个轴比按应用领域分类更能解释方法差异。

## 2. 主线进展

### 2.1 GKD：把暴露偏差变成可控的轨迹混合

Generalized Knowledge Distillation（GKD，arXiv:2306.13649）是现代 OPD 的起点。它用

$$\pi_{\mathrm{mix}}=\lambda p_\theta+(1-\lambda)p_{\mathrm{data}}$$

连续调节 off-policy（$\lambda=0$）和 on-policy（$\lambda=1$），并比较 forward KL、reverse KL 与 JSD。它的重要贡献不是某个单一损失，而是把“学生生成什么前缀”提升为一等的训练变量，并证明在多个设置中学生轨迹通常比纯教师/数据轨迹更有效。JSD 在中等多样性任务上常是折中选择：既不强迫学生覆盖教师所有低概率模式，也不至于过早只追逐少数模式。

### 2.2 MiniLLM：把 reverse KL 变成序列级学习信号

MiniLLM（arXiv:2306.08543）选择 $D_{\mathrm{KL}}(p_\theta\|p_T)$，在学生采样的序列上优化。其策略梯度可写成

$$
\nabla\mathcal L=-\mathbb E_{y\sim p_\theta}\left[\sum_t(R_t-1)\nabla\log p_\theta(y_t|y_{<t})\right],
$$

$$R_t=\sum_{t'=t}^{|y|}\log\frac{p_T(y_{t'}|y_{<t'})}{p_\theta(y_{t'}|y_{<t'})}.$$

这里的 return 同时包含当前 token 质量和未来 token 的影响，$-1$ 来自熵项。MiniLLM 的实际价值在于给出了可训练的序列级 reverse-KL 路径，并用单步词表期望降低方差；代价是 rollout 和策略梯度比 token-level KD 更贵、更不稳定。

### 2.3 DistiLLM：在支持集和方差之间搭桥

DistiLLM 引入 skew KL，在教师与学生分布的混合分布上计算比值，避免某一方概率接近零时的数值和梯度问题；DistiLLM-2 进一步在教师序列和学生序列上使用不同方向的 skew divergence。它代表一类重要的“工程化中间层”：不否定 OPD 的 on-policy 状态，却用混合目标缓解 reverse KL 的零强制和序列梯度方差。

### 2.4 ExOPD/G-OPD：从蒸馏走向密集奖励的 RL 连续体

ExOPD（也称 G-OPD，arXiv:2602.12125）把教师 token 分布解释为 dense、KL 约束的 RL 信号。学生先从自身分布探索，教师提供每一步的相对偏好；当学生在教师分布之外发现更高回报的行为时，奖励外推允许它超越教师。这个方向澄清了 OPD 与 RL 的关系：OPD 提供稳定、密集、低延迟的局部信号，结果奖励提供探索和“超过教师”的方向。两者如何调度、何时从蒸馏切换到奖励优化，综述仍视为开放问题。

### 2.5 自适应目标：稳定性成为 2026 年的核心问题

固定 divergence 在不同 token 熵区间并不等价。Entropy-Aware OPD（arXiv:2603.07079）在高熵 token 偏向 forward KL、低熵 token 偏向 reverse KL：前者保留不确定区域的多种合理选择，后者在确定区域集中学习。Veto（arXiv:2601.07155）直接改写 logit-space 的目标，抑制不稳定的错误更新。Revisiting OPD（arXiv:2603.25562）则系统梳理失败模式，指出截断 reverse KL 与教师 top-k 支持对稳定训练更实用。这些工作共同把问题从“选哪种 KL”推进到“根据状态、熵和支持集动态选目标”。

### 2.6 Rethinking OPD：先判断蒸馏是否有信息增益

Rethinking OPD（arXiv:2604.13016）的关键结论是：教师更强并不自动意味着 OPD 有效。若教师与学生的思考模式高度兼容，学生生成的高概率 token 已与教师重合，蒸馏只是在重复学生已有能力；若教师提供学生没有的能力，同时两者在关键前缀上仍可比较，OPD 才有明显信息增益。论文用 teacher top-k 支持的 overlap ratio 与 token advantage 做诊断，并把它们接入 verl 的蒸馏实现。这个判据比单看最终 benchmark 更接近 OPD 的因果机制：先测“教师信号是否新”，再决定是否增加 rollout 和蒸馏预算。

## 3. 目标函数和系统取舍

Forward KL 是 mode-covering、zero-avoiding，依赖教师完整词表分布；reverse KL 是 mode-seeking、zero-forcing，适合学生采样但可能丢掉教师的可接受模式；JSD 和 alpha-divergence 提供中间点。token-level 目标方差低，却可能把错误的前缀或局部 teacher trust 当成真值；sequence-level 目标与 on-policy 状态更一致，却需要 rollout、return 估计和方差控制。KETCHUP 一类方法用有限步 Bellman 估计改善 sequence-level return，说明“序列级正确性”和“可训练性”必须一起设计。

监督信号决定了可用目标：白盒教师能给 full-vocabulary logits，适合 forward KL；黑盒 API 只能给采样、排序或 verbal feedback，通常要用 discriminator、偏好或 outcome reward；自蒸馏则用 privileged information、外部反馈或教师快照产生信号。信号源不是实现细节，而是决定 divergence 是否可估计的约束。

成本是 OPD 的硬边界。综述估计 OPD 训练成本约为 off-policy SFT 的 4–5 倍，主要来自学生 rollout、教师前向和更复杂的缓存/同步。DeepSeek-V4 报告的系统做法很有代表性：缓存教师最后一层 hidden states，用输出头重建 logits，按教师分组调度 batch，并避免所有教师同时驻留；Lightning-OPD 则在一致性假设下预计算教师 log-prob，报告约 4 倍加速。一个现实的三阶段配方是：off-policy warm-up，on-policy full-logit distillation，最后用 reward-guided refinement 收尾。

## 4. 失败模式与证据边界

综述归纳了五类主线风险：

1. **Flawed-prefix trap**：学生错误前缀让教师信号失真，on-policy 并不自动解决 OOD。
2. **Self-play saturation**：纯自蒸馏可能形成 Ouroboros 式闭环，能力没有外部增量。
3. **Diversity collapse**：reverse KL 和过强的优势更新会把可接受模式压成单一模式。
4. **Calibration-capability gap**：教师可能在最终能力上更强，却无法在学生的异常状态上可靠打分。
5. **Length inflation / multi-turn degradation**：序列级奖励和多轮 rollout 可能诱发无意义延长，教师在长对话后段也可能退化。

跨论文的“谁更好”目前证据不足。不同论文使用的基座模型、教师能力、rollout 数、上下文长度、benchmark 版本和训练预算不同；综述本身明确提醒不能把这些结果当作严格的统一排名。因此，GKD/MiniLLM 的基础性可以由后续方法持续复用来证明，2026 年方法的影响力则应分成三层：**被多个方法复用的机制**（学生轨迹、reverse-KL/自适应 divergence）、**进入通用训练框架的诊断或实现**（如 Rethinking OPD 的 overlap 指标进入 verl）、**仅在单篇论文或垂类任务中成立的增益**。第三层不能直接称为高影响力主线。

## 5. 对此前候选工作的核验

- **TRACE** 可以放在“token/sample weighting 与信号质量”支线上：它解释哪些 token 的 OPD 信号更可靠，和综述的训练动态章节相容；但目前更像主线上的加权策略，缺少独立的广泛采用证据。
- **SA-OPD** 对应不确定性和 teacher/student 信号筛选，解决的是“何时信教师”；这是有意义的稳定性问题，但尚未达到 GKD、MiniLLM 那样的基础方法地位。
- **OPDVR** 属于 RL-augmented OPD，和 ExOPD/G-OPD 的连续体一致；它可作为 reward coupling 的后续样例，不能仅凭单篇 benchmark 宣称已广泛传播。
- **SCOUT、PivotOPD** 等 agent 或垂类方法展示了 OPD 的适用面，却没有改变统一目标、轨迹分布或训练成本这三条主线，应放在应用层而非核心进展层。

## 6. 工业采用说明了什么

Qwen3、DeepSeek-V4、MiMo-V2-Flash、GLM-5 等技术报告把 OPD 放进多阶段后训练或多教师能力合并流程。这里最有价值的信号不是某个领域分数，而是系统设计开始围绕 OPD 的真实瓶颈组织：教师 logits 的获取与缓存、rollout 吞吐、混合教师调度、蒸馏与 RL 的阶段切换。DeepSeek-V4 甚至报告用多教师 OPD 承担能力整合，说明 OPD 已成为规模化训练的基础原语之一；但这些报告通常同时改变数据、模型和训练阶段，不能单独证明某个 divergence 或采样策略的因果收益。

## 7. 适合继续深挖的主线

如果只保留最值得读的核心链条，顺序应是：**GKD → MiniLLM → DistiLLM → ExOPD/G-OPD → Entropy-Aware/Veto/Revisiting → Rethinking OPD → OPD Survey**。读完这条链能回答三个实际问题：学生轨迹是否覆盖了需要修复的状态；教师在这些状态上是否提供新且可校准的分布；额外 rollout 与教师计算是否换来了可重复的能力增量。其余论文应先按“目标函数、信号来源、训练动态”三轴归位，再判断是否有独立复现、框架合入或工业采用证据。

### 来源与外部材料

- 综述正文与参考文献：[arXiv:2604.00626](https://arxiv.org/abs/2604.00626)。
- 直观解释：[Thinking Machines, On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)。
- 工具化入口：[Hugging Face TRL On-Policy Distillation](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation)。
- 诊断实现：[Rethinking-OPD](https://github.com/Thinking-Space/Rethinking-OPD)，其 README 记录了 verl PR #6469 的 overlap 指标与版本注意事项；这是传播证据，不等同于独立学术验证。
- 论文索引：[awesome-on-policy-distillation](https://github.com/chrisliu298/awesome-on-policy-distillation)。该类聚合页用于发现材料，影响力判断仍以论文、官方技术报告和框架合入为准。

## 一句话评价

OPD 的真正进展，是把蒸馏从“在教师答案上拟合分布”推进成“在学生会访问的状态上，用可选择的 divergence 和密集反馈纠正策略”；2026 年最值得关注的不是又一个垂类应用，而是何时有信息增益、怎样稳定估计以及怎样把 rollout 成本压到可规模化的系统方法。
