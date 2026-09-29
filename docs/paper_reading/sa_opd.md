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

学生对输入 $x$ 采样 $y=(y_1,\ldots,y_L)$；冻结教师和学生分别在学生访问过的前缀 $(x,y_{<t})$ 上评分。Vanilla OPD 使用逐位置 reverse KL。对已采样 token，论文定义 $A_t=\log\pi_\theta(y_t\mid x,y_{<t})-\log\pi_T(y_t\mid x,y_{<t})$，将其作为 stop-gradient 系数时，更新方向与 $-A_t\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})$ 成正比。$|A_t|$ 因而是该 token 的优化影响代理，不是整词表 KL，也不能单独代表信号可靠性。

论文先用条件互信息 $I(X;A_t\mid Y_{<t})$ 定义“分歧有多依赖输入”，但它不可直接计算。实际代理是在**同一条学生生成前缀**上做两次教师—学生评分：一次保留原 prompt，得到 $A_t^{\mathrm{full}}$；另一次移除 prompt、保留响应前缀，得到 $A_t^{\mathrm{res}}$。二者的差 $\Delta_t^{\mathrm{IG}}=A_t^{\mathrm{full}}-A_t^{\mathrm{res}}$ 是 Input-Grounding Gap。差值小表示这条蒸馏方向在没有任务输入时仍出现，更可能来自模板或语言先验。这是论文采用的 no-prompt 对照，并非任意“轻微输入扰动”。

每个 batch 内，方法取 $\Delta_t^{\mathrm{IG}}$ 最低的 $p_1$ 分位与 $|A_t^{\mathrm{full}}|$ 最高的 $p_2$ 分位的交集 $F$，只过滤这批同时“输入依赖弱、更新影响大”的 token。为避免固定分位在不同任务中过度删除监督，作者动态调整 $p_1,p_2$，使过滤损失质量占比

$$\operatorname{FLMR}(F)=\frac{\sum_{t\in F}|A_t^{\mathrm{full}}|}{\sum_{t\in V}|A_t^{\mathrm{full}}|+\epsilon}\leq\beta,$$

其中 $V$ 是 batch 内有效响应 token。最后仅在 $V\setminus F$ 上平均 reverse-KL 损失。FLMR 约束的是被删掉的蒸馏信号总量；相同过滤 token 比例可能对应截然不同的优化影响。

论文的理论分析指出，先验诱导的梯度对输入特定目标的 alignment 较弱；低信噪比 OPD 更新会造成参数漂移。SA-OPD 的联合条件把“看起来很有影响”与“确实由输入支持”区分开。

## 3. 实验设置

语言实验使用 Qwen3 和 Qwen3.5 的 non-thinking 变体，主配对为 Qwen3-4B-Instruct→1.7B 以及 Qwen3.5-35B-A3B→2B；跨规模检验还包括 DeepSeek-R1-0528-Qwen3-8B→Qwen3-1.7B 和 Qwen3.5-9B→2B。数学数据从 DeepMath 中保留难度至少为 6 的样本，并对其余数据随机抽取 30%，总量约 7K examples；视觉理解从 VERO-600K 的 Captioning & IF、Grounding、Counting & Search 子集各抽取 10%，视觉推理从 MMRL30K 抽取 10%。

数学评测为 Math500、AMC23、AIME24/25 和 MinervaMATH；视觉理解为 EvoChart、MMIFEval、CountQA，视觉推理为 MathVision、Geo3K、MathVista。对照包括 Vanilla OPD、ExOPD、TIP 和 FiRe-OPD，论文声明在相同数据、模型和计算预算下比较。指标遵循各基准官方口径，不应将数学准确率与视觉评分直接合并。

## 4. 主要结果与消融

在 Qwen3.5-35B-A3B→2B 的视觉实验中，SA-OPD 将视觉理解三项平均分从 Vanilla OPD 的 50.5 提至 54.0，视觉推理三项从 60.4 提至 63.5。CountQA 为 26.4→33.6，Geo3K 为 67.2→72.2；SA-OPD 六项均高于对照 OPD 方法。在 Qwen3-4B-Instruct→1.7B 数学实验中，五项均分为 Vanilla OPD 28.5、TIP 29.3、SA-OPD 30.4；Math500 为 66.4→69.6。跨规模表还报告另两组教师—学生配对的正向增益，但仍集中在 Qwen 系学生。

消融在约相同过滤比例（相差约 1 个百分点）下比较随机过滤、仅按 $|A_t|$ 过滤、仅按 $\Delta_t^{\mathrm{IG}}$ 过滤，以及将 groundedness 代理替换成教师 log-probability。单维过滤虽有收益，联合判据的 Geo3K/MathVista 结果最佳。这个对照支持“两个维度都必要”，同时保留了一个边界：no-prompt gap 只是输入依赖的代理，不能证明被过滤 token 在语义上必然错误。

训练动态也有任务差异：数学 OPD 的高影响过滤信号在早期迅速衰减；视觉任务的 FLMR 在训练期间持续非零。论文据此解释视觉蒸馏收益更大，因为感知不确定性与语言先验的混合会持续制造可疑信号。这是由观测动态支持的机制解释，尚不能排除数据分布和模型架构的其他影响。

## 5. 复现与边界

复现需要保存学生 rollout，在 full-prompt 与 no-prompt 两个上下文上对齐教师和学生的逐 token log-prob，计算 $\Delta_t^{\mathrm{IG}}$、$|A_t^{\mathrm{full}}|$、交集分位集合及 FLMR，再只对保留 token 求 reverse KL。额外成本来自 no-prompt 评分前向；论文没有提出 CUDA、Triton 或其他自定义算子，方法属于训练目标和 token 过滤层面的改动。实验主要集中在 Qwen 系学生和数学/视觉语言任务；移除 prompt 可能改变 token 分布本身，因此这个代理需要随任务校准，且不能直接替代视觉事实核验或外部 verifier。

## 6. 方法启示

1. 教师—学生分歧只能衡量更新差异，不能单独代表监督质量；需要同时衡量输入依赖。
2. 联合过滤条件把高影响且低输入支持的信号与普通格式差异区分开。
3. 过滤粒度可以下沉到 token，避免整条 rollout 被同一质量标签支配。
4. groundedness proxy 是任务相关的近似量，应用到新任务时需要通过独立消融验证。

## 来源

Jiang et al., “When Teachers Mislead: Spurious-Signal-Aware On-Policy Distillation,” arXiv:2608.03632v1, 2026. [论文](https://arxiv.org/abs/2608.03632)
