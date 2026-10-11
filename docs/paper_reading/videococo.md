---
tags:
  - Video-to-Code
  - Executable Scene Reconstruction
  - Video Generation
  - Physical Consistency
  - Agent
---

# VideoCoCo：将可执行模拟作为视频生成的过程级推理

- 论文：[arXiv:2607.27380](https://arxiv.org/abs/2607.27380)
- 项目与代码：[VideoCoCo GitHub](https://github.com/micky-li-hd/VideoCoCo)
- 作者单位：CUHK、USTC、SCUT、HKU、NTU、PKU、SJTU、CMU、THU、HUST、MBZUAI。

## 概述

VideoCoCo 面向文本生成视频中的物理过程表达。系统先让编码 agent 编写 Blender 程序，执行后生成确定性的白模模拟视频；再把模拟草稿作为条件交给视频编辑模型，生成写实画面。程序承担过程级的空间和时间规划，编辑模型补足材质、光照和真实感。论文将这条链路称为 Code-as-CoT，并发布 VideoCoCo-3K 草稿—指令—目标视频三元组用于编辑模型适配。

## 方法

系统由两个引擎组成。

**可执行模拟引擎**接收文本指令，由 agent 解析物体、关系和物理过程，生成 Blender Python 程序。程序在沙箱执行，输出具有连续时序结构的草稿视频。由于每帧源自同一个场景程序，物体轨迹、接触顺序和相机视角由同一状态演化过程决定。

**视频实现引擎**接收文本与模拟草稿，将草稿转换为写实视频，同时尽量保留草稿的运动和事件关系。训练数据把草稿视频、指令和目标视频配对，优化编辑模型学习从白模模拟到真实视觉域的映射。论文比较三种训练设置：不更新编辑器、LoRA 适配、全量微调。可将其抽象为条件生成目标：

\[
\mathcal{L}_{edit}=\mathbb{E}_{(d,c,v)}[-\log p_\theta(v\mid d,c)],
\]

论文采用标准条件去噪目标：

\[
\mathcal{L}(\theta)=\mathbb{E}_{(d,e,y),t,\epsilon}\left[\|\epsilon-\epsilon_\theta(z_t,t,d,e)\|_2^2\right],\qquad \epsilon\sim\mathcal{N}(0,I).
\]

其中 \(d\) 为模拟草稿，\(e\) 为编辑指令，\(y\) 为写实目标视频，\(z_t\) 是目标视频潜变量在扩散步 \(t\) 加噪后的表示。优化目标让编辑器依据草稿和指令预测目标潜变量的噪声。底座初始化自 OmniWeaving，并比较全量微调与 LoRA。

VideoCoCo-3K 由 agent 生成 Blender 草稿与配套目标视频，构成约 3,000 组训练样本。数据构造的作用是为编辑器提供跨物理过程的草稿—写实视频监督；覆盖范围仍受模拟器可表达过程、自动生成质量和筛选机制影响。

## 实验结果

在 PhyGenBench 上，OmniWeaving 平均分从 0.475 提升至 0.558；后者超过表中最强开源基线 Wan2.2-TI2V-5B 的 0.544。分项上，VideoCoCo 在 mechanics、optics、material 分别取得 0.558、0.613、0.525；thermal 得分 0.511，低于 Wan2.2 的 0.533。相对 OmniWeaving，material 提升 0.133，thermal 提升 0.078。

在 VBench-2.0 物理合理性指标上，平均得分由 52.18% 提至 77.88%，增加 25.70 个百分点。mechanics 为 92.31%，thermotics 为 72.92%；material 为 68.42%，低于 CogVideoX-1.5 的 83.19%。这两组基准均支持模拟草稿为数据驱动生成补充物理过程线索，同时分项结果显示收益依赖现象类型。

消融将草稿生成流程固定，仅改变编辑器适配方式。直接用未经适配的 OmniWeaving 条件化草稿，平均分已从 0.475 升至 0.506；thermal 从 0.433 到 0.478，material 从 0.392 到 0.508。全量微调达到 0.535，LoRA 达到 0.558。该消融支持草稿本身提供物理动态信息，数据适配进一步增强草稿到写实视频的映射。LoRA 高于全量微调的解释是作者对有限数据过拟合的归因，结果本身并不能单独证明这一因果机制。

## 讨论与边界

代码草稿的优势在于把过程约束显式化，编辑器则负责外观实现；两阶段责任划分清楚，便于分别分析动态与视觉误差。代价是推理时增加 agent 编程与模拟环节，整体能力受 Blender 场景表达与执行稳定性约束。论文指出复杂湍流等过程仍难以零样本构造。评测结果主要来自指定基准和 OmniWeaving 编辑管线，不能直接外推到所有视频生成模型或任意物理场景。

## 训练配置与数据生成细节

数据管线先由来源 prompt 触发可执行模拟引擎，再由 instruction agent 生成编辑指令。对每个三元组，Seedance 2.0 作为 teacher editor 根据草稿和指令生成写实目标：

\[
y_i=G_T(d_i,e_i),\qquad
\mathcal{D}_{\mathrm{VideoCoCo\text{-}3K}}=\{(d_i,e_i,y_i)\}_{i=1}^{3000}.
\]

评测 prompt 及其近重复样本从训练数据排除，原始 prompt 与 Blender 程序保留为元数据。该划分让训练目标保留过程监督，同时降低 benchmark 泄漏风险。teacher 生成的目标仍会把 teacher 的视觉偏差带入监督，论文没有提供人工筛选规模或逐样本物理真值标注。

推理链路为 \(p\rightarrow c\rightarrow d\rightarrow e\rightarrow\hat v\)：编码 agent 从文本 prompt \(p\) 合成 Blender 程序 \(c\)，沙箱执行得到草稿 \(d\)，instruction agent 生成编辑指令 \(e\)，适配后的编辑器输出 \(\hat v\)。用户只提供文本，系统自动构造中间产物。草稿使用低保真白模表达逐帧状态，编辑器负责把这一状态映射到材质、光照和真实外观。

论文将 OmniWeaving 作为底座，比较 tuning-free、full-tune 和 LoRA-tune。表 3 中 tuning-free 已将平均 PhyGenBench 分数从 0.48 提升到 0.51，说明输入草稿本身提供动态约束；full-tune 为 0.54，LoRA-tune 为 0.56。LoRA 在 mechanics、optics、thermal、material 四项分别为 0.56、0.61、0.51、0.53，full tuning 对应 0.51、0.61、0.51、0.49。

## 评测口径与结果解读

PhyGenBench 使用 GPT-4o 作为多模态裁判，按 mechanics、optics、thermal、material dynamics 四类输出 0–1 一致性分数。VBench-2.0 采用预注册子集的物理维度，mechanics 覆盖重力、浮力和应力，thermotics 覆盖汽化和冻结，material 覆盖混色与溶解。VideoCoCo 在 VBench-2.0 上将 OmniWeaving 平均分从 52.18% 提到 77.88%；mechanics 为 92.31%，thermotics 为 72.92%，material 为 68.42%。

相对最强开源基线 Wan2.2-TI2V-5B，VideoCoCo 在 PhyGenBench 平均分上高出 0.014；Wan2.2 在 thermal 上为 0.533，高于 VideoCoCo 的 0.511。相对 OmniWeaving，VideoCoCo 对 material 和 thermal 的增益最大，这与代码草稿直接承载过程动态的设计相符。该解释得到分项结果支持，但仍受 benchmark prompt 分布和裁判模型影响。

## 运行代价与边界

 两阶段结构增加了代码生成、Blender 执行和视频编辑的推理延迟。模拟器支持的刚体、简单接触和有限过程表达较好，复杂湍流等现象仍受表达能力限制。LoRA 的优势来自论文当前数据规模、底座和适配策略组合，不能直接推广为所有编辑器上的普遍结论。论文评测聚焦物理合理性，视觉保真和多样性由所用视频生成管线共同决定。

## 可迁移设计点

- 可执行中间表示能把连续过程作为生成条件，并提供比文本计划和稀疏关键帧更密集的时空约束。
- 通过固定草稿、改变编辑器训练方式的消融，可以区分过程草稿与视觉适配的贡献。
- 对自动生成的配对数据，应报告筛选、过程覆盖和模拟失败情况，避免把合成覆盖误读为开放世界覆盖。
