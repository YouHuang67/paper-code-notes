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

具体地，先将视频 latent 的时空 token 重新排列成 cube-major 顺序，令每个执行 block 含 $B=B_tB_hB_w$ 个 token、共有 $N=L/B$ 个 block。coarse branch 对每个 cube 的 $Q,K,V$ 求均值，以 $N\times N$ 稠密注意力得到低分辨率输出，再广播回原 token。router 则用更小的 $R=R_tR_hR_w$ 池化 $Q,K$，先在每对大 block 内保留 $G=B/R$ 个子单元的 score，经 softmax 后对 $G\times G$ 分数求均值，得到 block 级重要性。若直接先对大 cube 平均再做 softmax，因 softmax 非线性会丢失子区域差异；两次 pooling 的顺序正是细粒度 router 与旧 VSA 的关键区别。

将 $N\times N$ 路由分数展平后选出全序列 $NK$ 个 block pair，得到布尔 mask $M$；fine branch 只计算这些 $B\times B$ tile 上的注意力。coarse 输出经 $\tanh(XW_g)$ 门控后与 fine 输出相加。$W_g$ 零初始化，因此从全注意力 checkpoint 切入时，新增 coarse 分支初始不改变输出。这里的 $K$ 是每个 query block 的**平均** key-block 预算，单个 query block 的实际配额可高可低。

忽略较小的 coarse/gating 成本，论文估算 router 的相对 FLOPs 约 $1/R^2$，fine branch 约 $KB/L$，故稀疏率近似 $1-(1/R^2+KB/L)$。公式说明随 $L$ 增长 fine 成本占比下降，而 router 的相对成本有下界；它是 FLOPs 估算，不能直接等同端到端延迟。

## 2. Kernel 实现

论文明确给出两类 GPU kernel。block-sparse attention 使用 ThunderKittens 实现，遵循硬件对 block size（如 64、128）的要求。router 侧不物化中间的 pooled score：CuTe DSL 将 $Q_rK_r^\top$、softmax 和 $G\times G$ pooling 融合为一个 kernel。

融合 kernel 分两次计算 $Q_rK_r^\top$：第一遍累积 log-sum-exp，第二遍执行 softmax、分数池化并回写。额外的一次计算换取更少的 global-memory I/O。对不能整除时空尺寸的输入，论文在 kernel 内 padding 并 mask 掉 padding token。训练系统还使用 FSDP、sequence parallelism、activation recomputation 和 `torch.compile`；这些属于训练并行与图编译，不等同于稀疏注意力 kernel。

Figure 8 的 attention-level 基准在单张 H800、batch 1、20 个 attention head、head dimension 128、top64 下测量；口径含 coarse、router、fine 等图 2 的 attention 操作，排除 QKV projection。220K token 时相对 FlashAttention-3 为 8.9×；此时 router 已占 VSA2 attention 时间的 22%，到 436K token 升至 30%。因此路由融合直接影响可实现速度，不是可忽略的附加计算。

## 3. Sparse Rebasing 与训练课程

作者从已经训练好的 256p 全注意力视频 checkpoint 开始，只在高分辨率和长视频阶段替换为 VSA2。新增加的 gating weight 初始化为零，使切换初期的输出仍由 fine branch 主导，避免从零训练稀疏模型。

Hard-to-Easy Curriculum 在训练中使用更激进的稀疏率，推理时放宽 topK。论文观察到这种结构化正则能改善运动质量，但推理时更密集的 video token 会挤压文本 token 的注意力，造成 prompt following 和美学质量下降；再进行较高 topK 的轻量 RL 可以恢复这部分能力。

论文的 RL 阶段具体是 reward-feedback learning：模型预测 clean video $x_0$，VLM reward model 与 CLIP reward model 的复合分数通过可微链路回传到视频 DiT。reward 权重先在全注意力 checkpoint 上定好，再原样用于 VSA2，保证对照条件一致。它不是 PPO/GRPO，也不应把两类训练目标混称。

## 4. 实验

训练覆盖 480p/720p、5–12 秒片段，包含 text-to-video 与 image-to-video supervision，后续还有 RL 阶段。全注意力对照使用相同超参、数据和训练迭代。评测使用 149 个困难 prompt 的成对人评；每对判 good/same/bad，报告分数为 $(\#good-\#bad)/149$，不是“获胜样本比例”。480×864（约 99K token）与 720×1280（约 220K token）的测试视频均为 10 秒。

720p 下 VSA2 达到约 95% 稀疏度，注意力相对 FlashAttention-3 加速 8.9×，端到端生成加速 4.62×。不同训练阶段的 flow-matching loss、RL reward 与全注意力相当；训练于 5–12 秒片段的模型在论文示例中可以直接生成 30 秒视频。表 2 中**top64 训练、top128 推理**的 480p 设置，text-to-video 运动人评分数为 +22.1%，但 prompt following 为 -3.36%、aesthetics 为 -4.7%；top32→top128 则是运动 +7.38%、prompt following -15.4%。这些是成对净胜率口径，而不是有 22.1% 样本单独获胜。

Figure 7 的消融把硬件形状选择与路由质量分开：在全注意力下 MHA 比四组 GQA 有更低训练 loss；对相同有效 group size，邻近时空 token 共用路由优于 head 分组。让 router 与 fine QKV 共享、不给路由决策反向传播，仍优于从 coarse 输出学习路由；单独给路由 logits 引入 MoE 式梯度也未胜出。细粒度 router 的 top64 优于旧路由 top128；固定总预算的 per-sequence topK 优于每个 query 固定 topK。以上是训练 loss 和部分人评证据，并非每个设计都有独立端到端加速测量。

## 5. 限制

结果依赖 block-sparse kernel、CuTe DSL 和 ThunderKittens 的 GPU 实现，不能把理论稀疏率直接当作任意硬件上的端到端加速。论文主要报告视频 DiT 的训练与推理，未提供可直接安装的官方代码仓库；跨硬件、不同 block size 和其他 runtime 的复现成本需要另行评估。

## 方法启示

1. 路由器的统计粒度可以小于执行 block，避免硬件约束反过来限制重要性估计。
2. 固定总计算预算、按 query 难度分配 topK，比所有 query 使用统一 topK 更适合长视频。
3. 通过 checkpoint rebasing 把稀疏注意力放到高分辨率阶段，可减少从零训练的成本。

## 来源

ByteDance Seed et al., “Improving Video Sparse Attention with Fine-grained Router and Sparse Rebasing,” arXiv:2609.32882v1, 2026. [论文](https://arxiv.org/abs/2609.32882)
