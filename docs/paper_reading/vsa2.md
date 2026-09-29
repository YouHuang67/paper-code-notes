---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
  - CUDA
  - ThunderKittens
  - CuTe
---

# Improving Video Sparse Attention with Fine-grained Router and Sparse Rebasing

- 论文：[Improving Video Sparse Attention with Fine-grained Router and Sparse Rebasing](https://arxiv.org/abs/2609.32882)
- 代码：论文页面未给出官方代码链接
- 团队：ByteDance Seed 等
- 提交：2026-09-27，arXiv:2609.32882v1

## 概述

VSA2 面向视频 DiT 的长序列自注意力。它把注意力拆成 coarse branch、fine branch 和 router：coarse branch 先低成本估计重要区域，fine branch 再在硬件友好的 block 上计算稀疏注意力。相比固定每个 query 的 topK，VSA2 使用与硬件 block 大小解耦的细粒度路由，并在序列级别分配总 topK，使困难 query 获得更多上下文、简单 query 少算冗余 KV。

论文还提出 Sparse Rebasing：从早期全注意力、低分辨率 checkpoint 继续训练，在 480p/720p 和强化学习阶段引入 VSA2；Hard-to-Easy Curriculum 则在高稀疏率训练、较低稀疏率推理。作者报告 720p 下 95% 稀疏度、注意力 8.9× 和端到端 4.62× 加速，同时保持接近或优于全注意力质量。

## 1. 从稠密注意力到 VSA2

视频序列长度随帧数和分辨率同时增长，完整 $QK^\top$ 的代价呈平方增长。非结构化 token 稀疏难以转化为真实硬件收益，因此方法把 mask 组织成大小为 $B\times B$ 的 block。传统 coarse router 的 pooling stride 通常直接等于 $B$，会把关键 token 混进大 cube；固定 per-token topK 又让所有 query 使用相同计算预算。

VSA2 用较小的 pooling 尺度 $R$ 提取 router query/key，之后以 $G=B/R$ 的窗口聚合 router score，再按硬件 block 生成 fine mask。每个序列的总 topK 固定，具体 query 可以获得不同数量的 KV。这样路由粒度和稀疏 kernel 的 block 尺寸分离，计算预算仍受控。

## 2. Kernel 实现

论文明确给出两类 GPU kernel。block-sparse attention 使用 ThunderKittens 实现，遵循硬件对 block size（如 64、128）的要求。router 侧不物化中间的 pooled score：CuTe DSL 将 $Q_rK_r^\top$、softmax 和 $G\times G$ pooling 融合为一个 kernel。

融合 kernel 分两次计算 $Q_rK_r^\top$：第一遍累积 log-sum-exp，第二遍执行 softmax、分数池化并回写。额外的一次计算换取更少的 global-memory I/O。对不能整除时空尺寸的输入，论文在 kernel 内 padding 并 mask 掉 padding token。训练系统还使用 FSDP、sequence parallelism、activation recomputation 和 `torch.compile`；这些属于训练并行与图编译，不等同于稀疏注意力 kernel。

## 3. Sparse Rebasing 与训练课程

作者从已经训练好的 256p 全注意力视频 checkpoint 开始，只在高分辨率和长视频阶段替换为 VSA2。新增加的 gating weight 初始化为零，使切换初期的输出仍由 fine branch 主导，避免从零训练稀疏模型。

Hard-to-Easy Curriculum 在训练中使用更激进的稀疏率，推理时放宽 topK。论文观察到这种结构化正则能改善运动质量，但推理时更密集的 video token 会挤压文本 token 的注意力，造成 prompt following 和美学质量下降；再进行较高 topK 的轻量 RL 可以恢复这部分能力。

## 4. 实验

训练覆盖 480p/720p、5–12 秒片段，包含 text-to-video 与 image-to-video supervision，后续还有 RL 阶段。评测使用 149 个困难 prompt 的人评，并在 480×864（约 99K token）与 720×1280（约 220K token）生成 10 秒视频。

720p 下 VSA2 达到约 95% 稀疏度，注意力相对 FlashAttention-3 加速 8.9×，端到端生成加速 4.62×。不同训练阶段的 flow-matching loss、RL reward 与全注意力相当；训练于 5–12 秒片段的模型可以直接生成 30 秒视频。人评中，top32 训练、top64 推理的设置有 22.1% 样本运动质量优于全注意力，但更高推理 topK 会带来文本遵循权衡。

## 5. 限制

结果依赖 block-sparse kernel、CuTe DSL 和 ThunderKittens 的 GPU 实现，不能把理论稀疏率直接当作任意硬件上的端到端加速。论文主要报告视频 DiT 的训练与推理，未提供可直接安装的官方代码仓库；跨硬件、不同 block size 和其他 runtime 的复现成本需要另行评估。

## 方法启示

1. 路由器的统计粒度可以小于执行 block，避免硬件约束反过来限制重要性估计。
2. 固定总计算预算、按 query 难度分配 topK，比所有 query 使用统一 topK 更适合长视频。
3. 通过 checkpoint rebasing 把稀疏注意力放到高分辨率阶段，可减少从零训练的成本。

## 来源

ByteDance Seed et al., “Improving Video Sparse Attention with Fine-grained Router and Sparse Rebasing,” arXiv:2609.32882v1, 2026. [论文](https://arxiv.org/abs/2609.32882)
