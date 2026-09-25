---
tags:
  - Prompt Enhancer
  - Video Generation
  - VLM
  - Reinforcement Learning
  - GRPO
  - Post Training
---

# WanPE: 面向现代文生视频的电影化提示词增强

- 论文：[WanPE: Towards Cinematic Prompt Enhancement for Modern Text-to-Video Generation](https://arxiv.org/abs/2609.30221)
- 项目页：[wan-pe.github.io](https://wan-pe.github.io/)
- 官方仓库：[Wan-PE/Wan-PE.github.io](https://github.com/Wan-PE/Wan-PE.github.io)
- 团队：Wan Team, Alibaba Group；南京大学、中国科学技术大学、复旦大学、清华大学
- 提交：2026-09-24，arXiv:2609.30221v1

## 概述

WanPE 把 prompt enhancer 从“补充视觉细节的改写器”推进为**电影化规划器**。输入可以只是一个意图、场景或动作描述，输出则组织成面向视频生成器的层级化条件：总体设定、按时间排列的 shot、主体与动作、景别和镜头运动、光照、对白、音乐、音效及镜头间过渡。它服务于 Wan3.0 的视频生成器，核心价值是把用户意图变成可执行的短片分镜计划。

方法由两步组成。第一步是 **video-grounded reverse SFT**：先从真实视频得到带时间戳的电影化 caption，再反向重建一个自然用户请求，用“请求 → 已实现的电影化条件”训练增强器。第二步是 **Semantic-Consistency GRPO（SC-GRPO）**：对同一请求采样多份条件，用文本评估器检查风格、主体、动作、对白、声音、镜头、光照、空间关系和场景九个维度，惩罚遗漏、改写、错误绑定、事件顺序错误和跨镜头冲突。

这一路线针对的是现代 T2V 的训练条件分布：生成器接触过的 caption 往往来自已经拍成的视频，而普通 forward rewriting 只是在用户句子上继续堆描述。反向构造让增强器学习真实视频中的镜头组织和时间推进，SC-GRPO 则约束新增细节持续服从原始请求。

## 问题形式化

设用户请求为 $x$，增强器为 $\pi_\theta$，视频生成器为 $G_\phi$。推理时先采样 $y \sim \pi_\theta(\cdot\mid x)$，再由 $G_\phi(\cdot\mid y)$ 生成视频。理想条件 $y$ 同时满足请求中的约束集合 $C(x)$，并接近生成器训练时的 video-grounded caption 分布 $p_{vg}$：

$$
p^*(y\mid x) = \frac{p_{vg}(y)\,\mathbf{1}[y\in Y_{sem}(x)]}{Z(x)}.
$$

$Y_{sem}(x)$ 表示不遗漏、不改变、不矛盾地满足 $x$ 的条件集合，$Z(x)$ 是归一化项。这个定义把 prompt enhancement 的目标拆成两件事：对用户负责的语义保真，以及对下游生成器有效的条件分布对齐。

## 方法

### 1. Video-grounded reverse SFT

作者从公开或获许可的视频中筛选最多 30 秒的片段，经过技术、画质、运动和时间结构过滤，得到约 **105 万**片段，覆盖十类内容。多模态视频 captioner 为每段视频生成层级化目标 $y$：

- 视频级摘要：场景、风格、叙事视角、节奏和声音设计；
- shot 级描述：时间戳、构图、主体、动作、光照、镜头运动、转场、对白、音乐和音效；
- 类别适配：大运动关注位移和身体姿态，音乐视频关注节奏和音画同步，富文本视频关注文字及其位置，动画和广告分别强调风格与品牌呈现。

随后使用带有约 2,000 条人工请求示例的 GPT-5.4，把 $y$ 压缩成只保留视频事实的自然用户请求 $x$。这样构造的训练对满足 $y\in Y_{sem}(x)$，并以标准条件语言模型损失训练：

$$
\mathcal{L}_{\mathrm{SFT}}=-\frac{1}{N}\sum_{i=1}^{N}\log\pi_\theta(y_i\mid x_i).
$$

关键点在于目标来自已经实现的真实视频，增强器学习的是“可实现的电影化条件”及其跨镜头组织，而不是由另一个 LLM 凭空扩写的文本风格。

### 2. Semantic-Consistency GRPO

SFT 后，作者为约 **1.5 万**条人工 T2V 请求构造 RL 数据，包含广覆盖请求和由 Wan3.0 试生成后人工挑出的困难案例。对每个 $x$，策略采样 $G$ 个候选条件 $y^{(g)}$，由 Qwen3.7-Max 给出 $[0,100]$ 的语义一致性分数。评估覆盖九维：

`style / subjects / actions / dialogue / sound / camera / lighting / spatial relations / scene`。

评估器接受同义改写和兼容扩展，惩罚遗漏、弱化、改变、矛盾、主体-属性或说话人-对白错绑、动作/镜头顺序错误以及跨镜头冲突。组内标准化得到优势 $\hat A_g$，并在 SFT 参考策略上加 KL 惩罚：

$$
J(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_g\frac{1}{T_g}\sum_t
\left(\min(\rho_{g,t}\hat A_g,\operatorname{clip}(\rho_{g,t},1-\epsilon,1+\epsilon)\hat A_g)-\beta K_{g,t}\right)\right].
$$

其中 $\rho_{g,t}$ 是当前策略与旧策略的 token 概率比，$T_g$ 是候选长度，$K_{g,t}$ 是相对 SFT 参考的 KL 项。这个 reward 是文本级的语义约束，目标是保证 cinematic plan 逐镜头展开时仍然保留用户要求；它不是视频画质分数，也没有直接更新 Wan3.0 的权重。

## 评测与结果

### WanPEval

WanPEval 包含 **249** 个请求、七类内容、5/10/15/30 秒四种时长、16:9 和 9:16 两种比例，以及 intent-level、scene-level、shot-level 三种粒度。60 位编剧、导演和摄影相关专家进行匿名两两视频比较，累计约 **1.1 万**次盲评。总体 preference score $S$ 将胜出计 1、两者都好计 0.5，并报告 Bradley–Terry 分数。

### 主结果

在同一个 Wan3.0 生成器下，WanPE-397B 相对原始请求的总体 preference score 提升如下：

| 时长 | 原始请求 | + WanPE-397B | 提升 |
|---|---:|---:|---:|
| 5 秒 | 33.80 | 50.61 | +16.81 |
| 10 秒 | 33.63 | 49.91 | +16.28 |
| 15 秒 | 30.59 | 49.43 | +18.84 |
| 30 秒 | 9.38 | 60.24 | +50.86 |

5–15 秒子集的整体 score 为 **50.61**，Bradley–Terry 为 **61.85**，论文报告其在比较的商业系统中排名第一；30 秒子集的 score **60.24** 接近 Seedance 2.5 的 **59.76**。在 5–15 秒的 intent、scene、shot 三种请求粒度上，WanPE-397B 分别达到 52.44、49.26、49.32，相比原始请求提升 18.55、16.97、13.45 个百分点。

### 为什么反向构造有效

在相同 Wan3.0 生成器下，397B 版本的反向构造 SFT 得分为 **49.86**，高于 forward rewriting 的 **39.49** 和 forward-target SFT 的 **35.17**。这项消融支持论文的核心判断：真实视频 caption 提供了更贴近生成器条件分布的电影化组织。

### SC-GRPO 的作用

SC-GRPO 使四个模型规模的文本语义一致性整体提升 **18.6–23.3** 个百分点；397B 的 overall 一致性从 75.5 提升到 97.6，perfect 输出比例从 29.7 提升到 85.5，failure 比例从 36.9 降到 2.8。对应的 Wan3.0 视频偏好 score 从 397B-SFT 的 **42.70** 升到 **49.69**，七个内容类别全部改善。

### 跨生成器迁移

WanPE 输出经过 GPT-5.4 格式适配后，接入 LTX-2.5-Base 和 MiniMax-H3-Base。在论文设置下，WanPE-397B 的 preference score 分别为 **35.56** 和 **41.09**，高于各自原生 enhancer 的 21.11 和 35.92，提升 **14.45** 和 **5.17** 个百分点。这个结果说明电影化组织具有一定跨生成器迁移性；它同时说明部署仍需要针对目标生成器做格式适配。

## 网页部署与复现边界

### 官方网页能做什么

官方项目页 `wan-pe.github.io` 是一个静态展示页，不是可调用的推理服务。仓库中的网页脚本按 demo 目录加载输入 prompt 和增强后的 `pe_output.txt`，并对比两个预渲染视频：`wan30_wo_pe.mp4` 与 `wan30_wanpe_397B.mp4`。页面还展示方法图、结果表、论文链接和 5 组视频 demo；视频通过浏览器进入视口后再加载并播放。

### 当前不能做什么

- 官方仓库的 Code 链接为空，没有推理代码入口；
- 没有公开 WanPE-4B/9B/35B/397B 权重、配置或推理 API；
- demo 只提供静态 prompt、增强文本和预生成视频，不能在网页上输入任意 prompt 生成新结果；
- 论文实验使用 512 张 GPU，397B 版本的本地复现需要远超普通单卡的推理资源；
- 跨生成器实验依赖 GPT-5.4 做格式转换，转换规则和调用接口未公开。

因此当前网页部署的可复现层级是“阅读论文和观看官方前后对比 demo”。研究仓库可以复核页面源代码、文本资产、结果表和论文数字；无法仅凭公开材料复现 WanPE 的训练或在线推理。

## 与相近工作的区别

| 工作 | 增强对象 | 主要反馈 | 关键差异 |
|---|---|---|---|
| VPO | T2V prompt | 文本原则 + 视频 reward 的 DPO | 关注 harmless/accurate/helpful，WanPE 专注电影化跨镜头规划 |
| PhyPrompt | T2V prompt | 物理合理性 + 语义的 GRPO | WanPE reward 覆盖镜头、对白、声音和跨 shot 语义一致性 |
| PromptEnhancer | T2I prompt | 24 维图文对齐 reward + GRPO | 图像生成，WanPE 面向 5–30 秒视频叙事 |
| APE | T2I prompt | GRPO/GDPO | 小模型单/多 agent 图像增强，WanPE 是 4B–397B 电影化规划器 |

## 结论

WanPE 的主要贡献是把 prompt enhancement 定义成生成前的 cinematic planning：真实视频提供可实现的条件分布，reverse SFT 学习镜头组织，SC-GRPO 约束长期语义一致性。作者在 Wan3.0 上报告了显著的人类偏好提升，尤其是 30 秒场景；跨生成器结果也支持其规划结构的可迁移性。

工程上，它目前更像一项“论文 + 官方 demo”而不是可直接安装的开源模型。网页部署可以核验静态资产和结果，无法调用 WanPE 进行任意 prompt 推理。后续若官方开放代码或权重，应优先补充：模型许可、推理显存/并行配置、输入输出 schema、Wan3.0 条件格式以及跨生成器 adapter。

## 来源

1. Zhu et al., “WanPE: Towards Cinematic Prompt Enhancement for Modern Text-to-Video Generation,” arXiv:2609.30221v1, 2026-09-24. [论文](https://arxiv.org/abs/2609.30221) [PDF](https://arxiv.org/pdf/2609.30221)
2. Wan Team, official project page and demo assets. [Project page](https://wan-pe.github.io/) [Repository](https://github.com/Wan-PE/Wan-PE.github.io)
3. 本仓库证据台账：[refs/research/wanpe/](../../refs/research/wanpe/)
