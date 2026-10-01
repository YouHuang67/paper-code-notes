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

> 论文：Mingyang Song, Mao Zheng, “A Survey of On-Policy Distillation for Large Language Models”，arXiv:2604.00626v4，2026-06-18。本文聚焦综述及其引用的论文。

## 综述范围与组织

作者将 OPD 定义为学生在训练期间从当前策略采样轨迹，并在这些轨迹上接受教师、反馈模型或自教师监督。综述覆盖白盒教师 logits、黑盒 API 反馈和自蒸馏，并沿目标函数、信号来源、训练动态三个设计轴组织方法。作者称其整理了两百余篇相关工作，覆盖基础目标、理论分析、失败诊断和工业训练流程。正文未给出系统综述式的数据库检索式、筛选流程或纳入排除统计，因此文献覆盖广度可作为研究地图使用，方法效果的横向统计仍须回到各论文原始实验核验。

## 概述

On-Policy Distillation（OPD）的核心流程是学生先生成自己的轨迹，教师再在这些学生会遇到的状态上提供密集的 token 分布或反馈。该流程针对 off-policy 蒸馏中的暴露偏差：训练时学生主要接触教师或数据前缀，推理时需要沿自己的前缀继续生成。

截至 2026 年，OPD 已经形成一条相对清晰的主线：GKD 把数据分布和学生分布混合起来；MiniLLM 把 reverse-KL 序列目标写成可优化的策略梯度；随后工作围绕“什么分布、什么 divergence、什么信号、怎样控制成本”展开。ExOPD/G-OPD 把 OPD 接到 KL 约束的 RL 连续体上，Veto、Entropy-Aware OPD 和 Revisiting OPD 处理目标和训练稳定性，Rethinking OPD 则给出“何时值得蒸馏”的兼容性与新能力判据。Qwen3、DeepSeek-V4、MiMo-V2-Flash、GLM-5 等报告显示 OPD 已成为大模型后训练的系统组件；工业报告提供规模化采用证据，跨论文因果比较仍需统一实验条件。

## 1. 统一问题：学生要在自己的状态上被监督

经典 token KD 通常最小化

$$
\mathcal L_{\mathrm{off}}=\mathbb E_{x,y\sim\mathcal D}\left[\sum_tD_{\mathrm{KL}}\left(p_T(\cdot|x,y_{<t})\|p_\theta(\cdot|x,y_{<t})\right)\right].
$$

off-policy 前缀来自标注数据或教师生成结果。学生训练时接触的状态与推理阶段的错误、犹豫和长度变化存在差异，小的局部误差会在长序列中累积。综述借用 DAgger 的分析直觉：若每一步错误概率为 $\epsilon$，on-policy 训练可将误差累积从近似 $O(\epsilon T^2)$ 降到 $O(\epsilon T)$。该分析依赖教师在学生前缀上的校准能力；学生轨迹进入教师覆盖不足的区域时，误差界的适用条件会减弱。

综述将 OPD 写成统一目标：

$$
\mathcal L_{\mathrm{OPD}}(\theta)=\mathbb E_{y\sim\pi_{\mathrm{mix}}(\cdot|x)}\left[\sum_tD_f\left(p_T(\cdot|x,y_{<t})\|p_\theta(\cdot|x,y_{<t})\right)\right],
$$

其中 $\pi_{\mathrm{mix}}$ 决定轨迹来自数据、教师、学生或它们的混合，$D_f$ 是由凸函数 $f$ 生成的 f-divergence，满足

$$D_f(P\|Q)=\mathbb E_{y\sim Q}\left[f\left(P(y)/Q(y)\right)\right],\quad f(1)=0.$$

据此，OPD 的研究空间可按三个维度组织：**轨迹分布**（前缀的生成来源）、**比较目标**（采用的 divergence）、**监督信号**（白盒 logits、黑盒反馈或自蒸馏）。这三个维度直接对应方法设计差异。

离散 token 采样使外层期望依赖学生参数，实际优化通常使用 score-function 或 policy-gradient 估计，并通过 stop-gradient、baseline、截断和混合轨迹控制方差。教师分布的 full-vocabulary 计算可以降低单步估计噪声；只使用 sampled token 时，目标估计更便宜，非采样 token 的分布信息也随之丢失。

## 2. 主线进展

### 2.1 GKD：把暴露偏差变成可控的轨迹混合

Generalized Knowledge Distillation（GKD，arXiv:2306.13649）是现代 OPD 的奠基工作。它用

$$\pi_{\mathrm{mix}}=\lambda p_\theta+(1-\lambda)p_{\mathrm{data}}$$

连续调节 off-policy（$\lambda=0$）和 on-policy（$\lambda=1$），并比较 forward KL、reverse KL 与 JSD。它将“学生生成什么前缀”提升为一等训练变量。综述总结，GKD 在指令跟随、摘要等设置中报告 $\lambda\geq0.5$ 的混合采样优于纯 off-policy；翻译任务中 JSD 表现较好，摘要与指令跟随上的 divergence 差别相对有限。证据支持轨迹采样分布是重要因素，具体效果仍依赖任务与实验设置。

### 2.2 MiniLLM：把 reverse KL 变成序列级学习信号

MiniLLM（arXiv:2306.08543）选择 $D_{\mathrm{KL}}(p_\theta\|p_T)$，在学生采样的序列上优化。其策略梯度可写成

$$
\nabla\mathcal L=-\mathbb E_{y\sim p_\theta}\left[\sum_t(R_t-1)\nabla\log p_\theta(y_t|y_{<t})\right],
$$

$$R_t=\sum_{t'=t}^{|y|}\log\frac{p_T(y_{t'}|y_{<t'})}{p_\theta(y_{t'}|y_{<t'})}.$$

该 return 同时包含当前 token 质量和未来 token 的影响，$-1$ 来自熵项。MiniLLM 给出了可训练的序列级 reverse-KL 路径，并用单步词表期望降低方差；rollout 和策略梯度也带来高于 token-level KD 的计算成本与训练方差。

### 2.3 DistiLLM：通过 skew KL 稳定训练

DistiLLM 引入 skew KL，在教师与学生分布的混合分布上计算比值，避免某一方概率接近零时的数值和梯度问题；DistiLLM-2 进一步在教师序列和学生序列上使用不同方向的 skew divergence。它代表一类重要的工程化中间层：保留 OPD 的 on-policy 状态，并用混合目标缓解 reverse KL 的零强制和序列梯度方差。

以 Skewed KL 为例，令 $\tilde p_t=\alpha p_T(\cdot|x,y_{<t})+(1-\alpha)p_\theta(\cdot|x,y_{<t})$，其中 $\alpha\in(0,1]$。DistiLLM 的 token 目标为

$$\mathcal L_{\mathrm{SKL}}=\mathbb E_{(x,y)\sim\mathcal D_{\mathrm{mix}}}\left[\sum_tD_{\mathrm{KL}}\left(p_T(\cdot|x,y_{<t})\|\tilde p_t\right)\right].$$

混合分布对学生概率提供下界，降低概率比值接近奇异点时的梯度爆炸风险。DistiLLM-2 按序列来源分配方向：教师序列使用 Forward SKL 传递覆盖信息，学生序列使用 Reverse SRKL 强化高质量响应；该设计把轨迹来源和 divergence 方向绑定起来。

### 2.4 ExOPD/G-OPD：从蒸馏走向密集奖励的 RL 连续体

ExOPD（也称 G-OPD，arXiv:2602.12125）把教师 token 分布解释为 dense、KL 约束的 RL 信号。学生先从自身分布探索，教师提供每一步的相对偏好；当学生在教师分布之外发现更高回报的行为时，奖励外推允许它超越教师。这个方向澄清了 OPD 与 RL 的关系：OPD 提供稳定、密集、低延迟的局部信号，结果奖励提供探索和“超过教师”的方向。两者如何调度、何时从蒸馏切换到奖励优化，综述仍视为开放问题。

### 2.5 自适应目标：稳定性成为 2026 年的核心问题

固定 divergence 在不同 token 熵区间具有不同作用。Entropy-Aware OPD（arXiv:2603.07079）在高熵 token 偏向 forward KL、低熵 token 偏向 reverse KL：前者保留不确定区域的多种合理选择，后者在确定区域集中学习。Veto（arXiv:2601.07155）直接改写 logit-space 的目标，抑制不稳定的错误更新。Revisiting OPD（arXiv:2603.25562）则系统梳理失败模式，指出截断 reverse KL 与教师 top-k 支持对稳定训练更实用。这些工作共同把问题从“选哪种 KL”推进到“根据状态、熵和支持集动态选目标”。

### 2.6 Rethinking OPD：蒸馏信息增益判定

Rethinking OPD（arXiv:2604.13016）的关键结论是：教师能力优势需要与学生可学习的新信号同时存在，OPD 才能产生有效增益。教师与学生的思考模式高度兼容时，学生生成的高概率 token 已与教师重合，蒸馏主要强化学生已有能力；教师提供学生缺少的能力，且双方在关键前缀上仍可比较时，OPD 更可能带来信息增益。论文用 teacher top-k 支持的 overlap ratio 与 token advantage 做诊断，并把它们接入 verl 的蒸馏实现。该判据直接检查教师信号的新颖程度，可用于安排 rollout 和蒸馏预算。

## 3. 目标函数和系统取舍

Forward KL 是 mode-covering、zero-avoiding，依赖教师完整词表分布；reverse KL 是 mode-seeking、zero-forcing，适合学生采样，同时存在丢失教师可接受模式的风险；JSD 和 alpha-divergence 提供中间点。token-level 目标方差低，也可能把错误的前缀或局部 teacher trust 当成真值；sequence-level 目标与 on-policy 状态更一致，同时需要 rollout、return 估计和方差控制。KETCHUP 一类方法用有限步 Bellman 估计改善 sequence-level return，说明“序列级正确性”和“可训练性”必须一起设计。

监督信号决定了可用目标：白盒教师能给 full-vocabulary logits，适合 forward KL；黑盒 API 可提供采样、排序或 verbal feedback，通常需要 discriminator、偏好或 outcome reward；自蒸馏可用 privileged information、外部反馈或教师快照产生信号。信号源属于目标设计的约束，因为它决定 divergence 是否可估计。

计算成本构成 OPD 的主要系统约束。综述估计 OPD 训练成本约为 off-policy SFT 的 4–5 倍，开销主要来自学生 rollout、教师前向和缓存/同步。DeepSeek-V4 报告采用以下系统设计：缓存教师最后一层 hidden states，通过输出头重建 logits，按教师分组调度 batch，并控制驻留显存的教师数量；Lightning-OPD 在一致性假设下预计算教师 log-prob，报告约 4 倍加速。综述提出的三阶段流程为 off-policy warm-up、on-policy full-logit distillation 和 reward-guided refinement。

### 3.1 固定 divergence 的优化含义

统一目标中的 $D_f$ 直接决定 token 概率比值 $u=p_T/p_\theta$ 的权重。Forward KL 使用 $f(u)=u\log u$，教师低概率但非零的模式仍会贡献梯度，学生因此倾向覆盖教师支持集。Reverse KL 使用 $f(u)=-\log u$，学生采样到的 token 获得更直接的更新，概率质量集中在教师高置信区域。JSD 通过混合分布限制比值的极端变化，适合教师和学生分布存在中等差异的状态。$\alpha$-divergence 提供连续插值，使目标方向成为可调变量。

这些性质带来三类工程后果。第一，Forward KL 需要教师完整词表分布，黑盒接口通常无法直接计算。第二，Reverse KL 可以利用学生已采样 token，序列级估计仍需处理回报方差和长度偏差。第三，JSD 或 skew KL 以部分极端方向的精确性换取更稳定的有限样本估计。因而 divergence 选择需要同时考虑任务分布、教师访问权限和训练预算。

### 3.2 token-level 与 sequence-level 的偏差—方差关系

token-level OPD 在每个前缀上直接比较词表分布，梯度方差较低，计算可以批量化；它依赖教师在当前前缀上的局部校准，并可能把单步匹配误差累积为整条序列的偏差。sequence-level reverse KL 以学生完整轨迹为采样对象，目标与策略分布一致，估计需要 REINFORCE 或 return 近似，方差随序列长度增长。MiniLLM 将单步词表期望与未来 return 分解，以降低方差；KETCHUP 采用有限步 Bellman 估计改善长序列回报传播。

综述的统一解释是：token-level 方法将统计效率置于首位，sequence-level 方法将目标一致性置于首位。DistiLLM、TIP、EOPD 等方法在两者之间增加混合分布、token 权重或自适应方向，形成可控的偏差—方差折中。

### 3.3 自适应与 RL 增强目标

自适应 divergence 根据教师熵、学生—教师概率比值或局部几何选择更新方向。Entropy-Aware OPD 在高熵区域保留多模态监督，在低熵区域集中学习；OPD+ 使用 f-divergence 曲率项修正 sampled-token advantage；Veto 在 logit 空间重构目标，抑制极端更新。它们共同回答“当前 token 应采用哪种监督强度”这一问题。

RL-augmented OPD 把教师 token 信号与 outcome reward 或 verifier reward 组合进策略更新。G-OPD 将 OPD 解释为 KL 约束 RL 的一种形式，并研究超出教师分布的 reward extrapolation；KDRL 在 RL 更新中加入 on-policy KL 正则；RLKD 使用生成式结构奖励模型提供序列级奖励；REOPOLD 采用 log-ratio token reward、奖励裁剪与熵自适应采样。共同机制是以教师信号提供密集的局部方向，以外部奖励提供结果级优化方向。不同方法的奖励定义和约束形式各异，权重调度依赖任务与训练阶段。

## 4. 训练动态与系统实现

OPD 的训练循环包含学生 rollout、教师评分和学生更新三个同步环节。学生参数每次更新后，旧 rollout 与当前策略产生分布差异；异步执行可提升吞吐，同时引入策略陈旧度。白盒教师还需要传输完整词表 logits 或可重建的 hidden states，通信压力随序列长度和词表规模增长。

### 4.1 样本和 token 权重

均匀平均会把低信息 token、格式 token 和教师不确定 token 与关键推理 token 等同处理。TIP 根据教师熵与学生—教师 divergence 估计 token 重要性；Rock Tokens 工作进一步显示，长期高损失 token 可能贡献较大梯度，同时与任务能力的关联有限。样本层面可按学生 pass rate、teacher-student divergence 或梯度信噪比筛选 rollout，使预算集中于学生可学习且教师信号可靠的区域。

### 4.2 难度课程与前缀截断

学生在目标提示上的通过率接近零时，rollout 几乎全部进入早期错误状态，教师反馈的有效信噪比随之下降。PACED 等课程方法把样本难度放在学生能力边界附近；FOPD 逐步延长有效前缀；Prune-OPD 根据局部 top-k overlap 和漂移预算动态截断。它们的共同抽象是控制“每次更新覆盖多长、难度多高、教师信号是否仍有区分度”。

### 4.3 full-vocabulary 与 sampled-token

full-vocabulary 蒸馏保留教师分布的完整几何信息，梯度稳定性和模式覆盖较好，代价是教师计算、显存和通信开销。sampled-token 蒸馏仅使用学生实际生成的 token，单步成本低，估计方差和支持集偏差更明显。DeepSeek-V4 通过 hidden-state 缓存与教师 head 按 batch 调度降低显存压力，Lightning-OPD 通过教师 log-prob 预计算减少在线评分，两者分别从内存和教师推理环节优化 full-vocabulary 训练。

若每个样本平均长度为 $T$、词表大小为 $|V|$，full-vocabulary 传输的 logits 规模近似为 $O(T|V|)$，sampled-token 传输规模接近 $O(T)$。前者保留所有候选 token 的相对概率，后者把教师监督压缩为已采样路径上的标量或少量 log-prob。工程选择因此取决于教师—学生部署位置、互联带宽和所需的梯度精度。

## 5. 失败模式与证据边界

综述归纳了五类主线风险：

1. **Flawed-prefix trap**：学生错误前缀让教师信号失真；缓解 OOD 需要教师在偏离状态上保持校准。
2. **Self-play saturation**：纯自蒸馏可能形成 Ouroboros 式闭环，能力增量受限于外部新信息的供给。
3. **Diversity collapse**：reverse KL 和过强的优势更新会把可接受模式压成单一模式。
4. **Calibration-capability gap**：教师可能拥有更高的最终能力，同时在学生的异常状态上缺乏可靠打分能力。
5. **Length inflation / multi-turn degradation**：序列级奖励和多轮 rollout 可能诱发无意义延长，教师在长对话后段也可能退化。

跨论文比较需要更多统一实验。不同论文使用的基座模型、教师能力、rollout 数、上下文长度、benchmark 版本和训练预算存在差异，因此综述将横向排序视为证据有限的问题。GKD/MiniLLM 的基础性可通过后续方法对其核心机制的持续复用来观察。2026 年方法的影响力可分成三层：**被多个方法复用的机制**（学生轨迹、reverse-KL/自适应 divergence）、**进入通用训练框架的诊断或实现**（如 Rethinking OPD 的 overlap 指标进入 verl）、**现阶段主要由单篇论文或垂类任务支持的增益**。第三层属于候选方向，其影响范围仍需独立复现和更多采用证据确认。

## 6. 成功条件与理论解释

综述将有效 OPD 归纳为两个必要条件。第一，教师和学生需要存在可吸收的行为重叠。Rethinking OPD 用 top-k overlap ratio 描述二者在高概率 token 上的共同支持；重叠过低时，学生难以沿教师的思考模式更新，off-policy warm-up 或教师对齐提示可缩小初始差距。第二，教师需要提供学生尚未掌握的新能力。相同训练配方和同一模型家族的教师可能与学生形成相近分布，教师规模优势转化为有限的可传递信息；教师与学生存在适度能力差距时，蒸馏信号更具信息量。

这两个条件共同定义“可蒸馏窗口”：能力差距过小时，教师信号的增量有限；差距过大时，思考模式和支持集重叠不足。实践中可以在训练前测量 top-k overlap、学生在目标提示上的 pass rate、教师在学生前缀上的校准误差以及早期输出长度变化，用来决定 warm-up、课程难度和教师评分预算。

### 6.1 OPD 与行为克隆、DAgger 和 RL 的关系

从行为克隆角度看，off-policy KD 在固定数据分布上匹配教师策略，OPD 将状态分布替换为学生或混合策略访问的状态。GKD 的 $\lambda$ 提供从行为克隆到 DAgger 风格交互式监督的连续旋钮。MiniLLM 的 sequence-level reverse KL 则把该监督写成学生策略上的期望，直接使用策略梯度。

从 RL 角度看，教师 log-prob 比值可解释为 dense shaping reward，divergence 的方向决定 KL 正则化的几何形式。G-OPD 将 OPD 表达为 KL-constrained RL，说明教师监督和 policy improvement 可以在同一目标中组合。结果奖励引入环境或验证器提供的长程信号，使学生能够探索教师分布之外的策略；KL 约束维持更新的可控范围。

### 6.2 何时选择 OPD

综述给出一套决策条件。任务具有长推理链、学生会频繁访问训练数据未覆盖的错误状态、教师可在这些状态上可靠评分时，OPD 的状态覆盖收益较大。任务输出空间高度多模态、教师只提供黑盒文本、教师在异常前缀上校准不足或 rollout 预算有限时，应优先采用混合轨迹、短前缀、skew divergence 或 off-policy warm-up。任务的核心目标是超越教师能力时，需要将 OPD 与可验证的 outcome reward 或 RL 阶段衔接。

该决策框架强调成本—质量关系。OPD 的额外计算主要换取状态覆盖和密集梯度；当 off-policy 数据已覆盖目标状态且教师信号缺少新信息时，额外 rollout 的边际收益会下降。实际部署中可先用 overlap、pass rate 和 divergence 估计筛选目标，再确定 rollout 数、教师驻留方式和 full-vocabulary 传输精度。

## 7. 工业采用与影响力证据

Qwen3、DeepSeek-V4、MiMo-V2-Flash、GLM-5 等技术报告把 OPD 放进多阶段后训练或多教师能力合并流程。更有参考价值的证据来自系统设计：教师 logits 的获取与缓存、rollout 吞吐、混合教师调度、蒸馏与 RL 的阶段切换都围绕 OPD 的真实瓶颈展开。DeepSeek-V4 报告用多教师 OPD 承担能力整合，显示 OPD 已成为规模化训练的基础原语之一；这些报告同时改变了数据、模型和训练阶段，因此单篇报告难以分离某个 divergence 或采样策略的因果收益。

影响力需要分层判断。GKD 和 MiniLLM 具有方法奠基作用：后续工作持续复用学生轨迹、reverse KL、策略梯度和混合轨迹等核心构件。DistiLLM、EOPD、Veto、ExOPD/G-OPD 与 Revisiting OPD 代表目标稳定性、分布自适应和 RL 融合的主要推进方向，影响力证据来自方法复用、统一框架中的定位和跨任务实验。Rethinking OPD 的 overlap 诊断进入 verl，提供了从论文机制到通用训练框架的传播证据。DeepSeek-V4、Qwen3 等工业报告提供了规模化系统采用证据，重点支撑 OPD 的工程可行性。单篇论文的局部 benchmark 增益属于较弱证据，通用主线地位仍需更多独立复现和跨任务采用支持。

## 8. 研究空白与核心阅读路径

### 8.1 蒸馏 scaling law

现有蒸馏 scaling law 主要研究 off-policy 场景，显示最优教师规模会随计算预算先增大后趋于饱和，教师推理成本在预算中占比上升后还可能降低最优教师规模。on-policy 场景新增学生 rollout 预算这一独立变量，且教师规模、学生容量和 rollout 数相互影响。综述提出的联合幂律形式仍属待检验假设，尚无控制变量充分的 OPD 实证定律。要建立可用的预算模型，需要分别扫描学生规模、教师规模、rollout 数和 divergence，并记录吞吐、质量与教师校准的交互效应。

### 8.2 仍待解决的主线问题

1. **教师不确定性**：token 熵衡量分布分散程度，教师在学生前缀上的 epistemic uncertainty 还需要独立估计；后续方法需把信号可靠性接入更新权重。
2. **长程 credit assignment**：单轮生成的 token-level 信号容易定位，多轮或工具轨迹要求教师评价学生访问过的环境状态，并将结果归因到局部决策。
3. **效率下界**：已有方法降低 rollout、教师评分和通信开销，达到给定质量所需的最少学生轨迹数仍未知。
4. **评测完整性**：pass@1 等指标记录精度，覆盖率、校准、分布外泛化和输出长度记录其他重要能力；OPD 评测需要同时报告这些指标。
5. **跨架构与跨模态对齐**：异构 tokenizer、隐藏空间和连续输出空间会限制词表 KL 的直接使用，通用表征对齐仍处于早期阶段。
6. **蒸馏与 RL 的联合调度**：密集教师信号和稀疏 outcome reward 的预算分配仍主要依赖经验，能力差距与不确定性驱动的动态切换值得系统研究。

### 8.3 对此前候选工作的定位

- **TRACE** 对应 token/sample weighting 与信号质量，属于训练动态轴上的选择性监督机制。它与综述主线一致，广泛采用证据仍有限。
- **SA-OPD** 对应教师信号可靠性筛选，补充 flawed-prefix 与 calibration 风险的处理方式，当前定位为稳定性方向的后续方法。
- **OPDVR** 属于 RL-augmented OPD，研究密集蒸馏信号与奖励优化的组合，可作为联合训练方向的具体实例。
- **SCOUT、PivotOPD** 聚焦 agent 场景中的轨迹结构和错误恢复，可用于观察 OPD 向长程交互扩展时的边界；当前将其归为应用侧证据。

建议按以下顺序阅读核心工作：**GKD → MiniLLM → DistiLLM → ExOPD/G-OPD → Entropy-Aware/Veto/Revisiting → Rethinking OPD → OPD Survey**。这条脉络依次呈现学生轨迹、序列级 reverse-KL、稳定性机制、RL 融合、自适应目标和蒸馏信息增益判据。阅读时可围绕三个问题评估方法：学生轨迹是否覆盖待修复状态；教师是否在这些状态上提供新且可校准的分布；额外 rollout 与教师计算是否带来可重复的能力增量。其他论文可按“目标函数、信号来源、训练动态”三个维度归类，并结合独立复现、框架合入和工业采用证据评估影响范围。

### 来源与外部材料

- 综述正文与参考文献：[arXiv:2604.00626](https://arxiv.org/abs/2604.00626)。
- 技术说明：[Thinking Machines, On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)。
- 工具化入口：[Hugging Face TRL On-Policy Distillation](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation)。
- 诊断实现：[Rethinking-OPD](https://github.com/Thinking-Space/Rethinking-OPD)，其 README 记录了 verl PR #6469 的 overlap 指标与版本注意事项；该材料可证明实现传播，学术有效性仍需独立复现评估。
- 论文索引：[awesome-on-policy-distillation](https://github.com/chrisliu298/awesome-on-policy-distillation)。该类聚合页用于发现材料，影响力判断仍以论文、官方技术报告和框架合入为准。

## 总结

OPD 将蒸馏扩展到学生实际访问的状态，并结合可选择的 divergence 与密集反馈修正策略。当前主线集中在轨迹覆盖、教师信号可靠性、散度自适应、RL 联合优化和 rollout 成本控制。其普遍优势受教师校准、师生能力差距、任务分布和计算预算共同约束；规模化应用已出现，统一的 OPD scaling law 和可比较的标准化评测仍待建立。
