---
tags:
  - Video-to-Code
  - Executable Scene Reconstruction
  - 4D Inverse Graphics
  - Physical Dynamics
  - Benchmark
---

# 4DCODEBENCH：动态场景逆图形的代码 Agent 基准

- 论文：[arXiv:2610.03715](https://arxiv.org/abs/2610.03715)（2026-10-05）
- 项目与基准：[4DCodeBench.com](https://4DCodeBench.com)；论文列出官方仓库 `github.com/4DCodeBench/4DCodeBench`。
- 作者单位：Johns Hopkins University、Stanford University、Max Planck Institute for Intelligent Systems、Massachusetts Institute of Technology。

## 概述

4DCODEBENCH 将动态场景理解表述为 4D 逆图形：给定视频，agent 生成可执行代码，构造随时间演化的三维几何并渲染回视频。基准包含 200 个场景，其中 100 个真实视频提供外观和复杂动态，100 个物理模拟场景提供逐帧 4D 真值。论文评测 18 个模型，并以多类互补指标区分外观、二维运动、几何和三维动态恢复。

## 任务定义

令参考视频为 \(x_{ref}=(x_0,\ldots,x_{T-1})\)，agent \(A\) 生成程序集合 \(\Gamma=(\gamma_1,\ldots,\gamma_m)\)。执行程序得到时刻 \(t\) 的几何状态 \(s_\Gamma(t)\)，渲染过程 \(R_\Gamma\) 使用程序指定的相机、光照和材质：

\[
\Gamma=A(x_{ref}),\qquad \hat{x}_k=R_\Gamma\bigl(s_\Gamma(k/r)\bigr),\quad k=0,\ldots,T-1,
\]

其中 \(r\) 是参考视频帧率。程序输出须匹配分辨率、帧数与帧率，并在生成后独立执行。基准允许关键帧、物理模拟器或自定义动态实现，评价重点是可执行场景及其随时间的状态。

## 数据与指标

200 个场景覆盖刚体与关节体、可变形固体、布料和绳索等共维结构、颗粒与流体等流动材料。真实视频侧重真实视觉和自然动态，但不具备完整 4D 真值；合成侧可逐帧比较几何与动态状态。该组合使真实画面评价与可控三维评价并行。

指标家族包括感知相似度、2D Dynamics、2.5D Geometry、3D Geometry、3D Dynamics。合成场景中，Scene 3D 以对齐后表面的对称 Chamfer 距离衡量几何：

\[
\mathrm{Scene3D}=1-\frac{D_\kappa(P,\hat P)}{\kappa},\qquad \kappa=0.5,
\]

其中 \(P\) 与 \(\hat P\) 是参考与重建表面点集，距离在阈值处截断。轨迹 DTW 对动态物质点的三维路径进行 Hungarian 一对一匹配，再计算截断后的动态时间规整距离；EMD-step 用切片 Wasserstein 距离比较逐帧位移分布，度量瞬时速度场。整体分数取五个主指标家族平均，使外观、运动和几何共同影响排名。

此外，200 场景有 1,251 个二元 VQA 问题，覆盖初末状态、关键事件和接触行为；另以参考视频为条件进行 VLM 成对比较，并用 Bradley–Terry 模型聚合 Elo。失败渲染与未通过存在性门控的问答按错误计分，确保各模型面对相同问题集合。

## 主要结果

18 个模型中，GPT-6-Astra [Max] 的综合表现领先；人工偏好 Elo 与 VLM Elo 在模型层面的 Spearman 相关系数为 0.980，人工偏好与 Overall 的相关系数为 0.96。76 名参与者完成 3,587 次成对判断；共享 916 次比较中，人类与 VLM 一致率为 89.3%。这些数据支持自动排名在模型总体比较上的有效性。

指标分解显示动态恢复是主要瓶颈。最佳模型静态感知与三维几何分数分别约为 0.91、0.92，而 2D Dynamics 与 3D Dynamics 分别约为 0.60、0.67。18 个模型整体也呈现动态分数低于感知和静态几何分数的趋势。由此，静态场景拟合能力不能充分代表模型对物体运动、接触和材料演化的恢复能力。

## 分析与边界

真实视频没有逐帧几何真值，因而依赖感知指标、问答和偏好比较；合成视频拥有精确几何状态，但模拟器覆盖受材料模型、场景设计和数值稳定性约束。两类数据提供互补证据，单一指标无法完整代表重建质量。VLM 判断与人类偏好高度相关于模型总体排名，但对同场景、表现接近的两个输出区分度较低。论文还报告有效执行率约 90.1%，程序正确性本身是端到端能力的一部分。

## 可迁移设计点

- 让 agent 输出可执行世界程序，可同时评测代码生成、三维结构恢复和动态建模。
- 真实数据与有真值的模拟数据组合，分别覆盖自然复杂度与可精确度量性。
- 把静态几何、视觉外观、二维运动和三维动态分开报告，可明确暴露能力不均衡。
