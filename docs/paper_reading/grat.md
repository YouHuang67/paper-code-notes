---
tags:
  - Sparse Attention
  - Diffusion Model
  - Video Generation
---

# GRAT: Grouping First, Attending Smartly — Training-Free Acceleration for Diffusion Transformers

[arXiv 2505.14687](https://arxiv.org/abs/2505.14687) | [GitHub](https://github.com/OliverRensu/GRAT) | Johns Hopkins University

## 概述

DiT 的全量注意力 $O(N^2)$ 在高分辨率下极慢（8192×8192 图像超过 1 小时），现有稀疏方法（Neighborhood Attention、CLEAR）虽减少 FLOPs 但 GPU 实际加速有限，且远距感受野受限导致质量下降。

GRAT 提出**先分组、再智能 attend** 的两步策略，无需微调：
1. **Grouping First**：将连续 token 按空间位置分为不重叠的 $g_1 \times g_2$ 组（图像）或 $g_1 \times g_2 \times g_3$ 组（视频），组内 token 在内存中连续
2. **Attending Smartly**：同组内所有 query token 共享同一套 KV token，仅来自结构化区域——两个变体：
   - **GRAT-B**：每组 attend $(2b+1) \times (2b+1)$ 邻近 block 组，$O(b^2 g^2 N)$
   - **GRAT-X**：每组 attend 整行整列（criss-cross），$O(g H N + g W N)$

"同组共享 KV"是关键：与每 token 独立选 KV 的方法不同，组内 query 对同一 KV 集做标准 GEMM，内存访问规则，GPU 利用率高。

主要结果：
- Flux，8192×8192（262K tokens）：GRAT-B 注意力 **35.8×** 加速，GenEval 0.61（NA 0.54）；GRAT-X 11.6× 加速，GenEval **0.65**（等于全量 0.66）
- HunyuanVideo，768×1280×128f（122.9K tokens）：GRAT-B **15.8×** 加速，VBench 超 STA（80.51% vs 80.46%）；GRAT-X 2.4× 加速，VBench **83.82%**（超全量 82.71%）

## 方法

### 为什么需要分组

Flux 的注意力图分析（100 张生成图平均）：距离 query 归一化距离 ≤ 0.2 的 key token 贡献约 **69%** 注意力质量。局部性强，绝大多数注意力是稀疏且局部的。

现有局部注意力（Neighborhood Attention、CLEAR）的问题：每个 query 选不同的 KV 窗口，内存访问不规则，实际 GPU 吞吐低，理论 FLOPs 节省难以转化为实际加速（13-14× 实际加速，FLOPs 省 99%）。

### Grouping First

将序列按空间位置分组（图像 $16 \times 16$，视频 $4 \times 8 \times 8$）：

$$G^Q_{p,q} = \{Q_{i,j} \mid i // g_1 = p,\ j // g_2 = q\}$$

组内 token 在内存中相邻（coalesced access）。分组粒度对齐 GPU SM / thread block 配置，最大化并行效率。

### Attending Smartly — GRAT-B

每个 query 组 $G^Q_{p,q}$ 仅 attend 到 $(2b+1) \times (2b+1)$ 邻近组的 KV（$b=1$ 时为 $3 \times 3 = 9$ 个 KV 组）：

$$G'^K_{p,q} = \bigcup_{\{(m,n) \mid |m-p| \le b,\ |n-q| \le b\}} G^K_{m,n}$$

复杂度 $O(b^2 g^2 N)$，$g$ 和 $b$ 为常数时线性于 $N$。感受野（最远 token 距离）受限（8192×8192 下约 45），但对多数生成任务足够。

### Attending Smartly — GRAT-X

每个 query 组 attend 整行和整列的 KV（criss-cross 模式）：

$$G'^K_{p,q} = \bigcup_{\{(m,n) \mid m=p \vee n=q\}} G^K_{m,n}$$

复杂度 $O(g H N + g W N)$，高于 GRAT-B，但最远 token 距离达 512（GRAT-B 为 45，NA 为 23），提供真正的全局上下文。

**视频扩展**：分组维度扩展为 $T \times H \times W$，GRAT-B 变为 3D 邻近块，GRAT-X 变为时间轴 + 高度轴 + 宽度轴的 criss-cross。

### 实现细节

全部基于 PyTorch Flex Attention 实现，无需自定义 Triton kernel。组内 query 共享 KV 使注意力计算退化为规则的批量 GEMM，与 FlashAttention 风格的 IO-aware 分块计算完全兼容。

## 实验

### 图像生成（Flux，8192×8192，262K tokens，A100）

| 方法 | FLOPs 稀疏率 | 最远 token | 注意力延迟 | 加速比 | 推理时间(s) |
|------|------------|-----------|---------|-------|-----------|
| Full Attention | 0% | 724 | 4.081s | 1× | 5480 |
| CLEAR (r=16) | 99.50% | 16 | 0.280s | 14.6× | 812 |
| NA (w=32) | 99.42% | 23 | 0.312s | 13.0× | 842 |
| **GRAT-B** | **99.03%** | **45** | **0.114s** | **35.8×** | **598** |
| **GRAT-X** | **93.67%** | **512** | **0.353s** | **11.6×** | **898** |

### 图像质量（Flux，COCO2014 / MJHQ-30K / GenEval）

| 方法 | COCO FID↓ | COCO IR↑ | MJHQ FID↓ | GenEval↑ |
|------|---------|---------|---------|---------|
| Full Attention | 33.89 | 1.076 | 19.72 | 0.66 |
| CLEAR | 47.50 | 0.045 | 29.96 | 0.52 |
| NA | 42.62 | 0.112 | 28.65 | 0.54 |
| **GRAT-B** | **35.99** | **0.925** | **20.95** | **0.61** |
| **GRAT-X** | **34.59** | **1.068** | **20.05** | **0.65** |

GRAT-X GenEval 与全量注意力持平，Image Reward 最高。

### 视频生成（HunyuanVideo，122.9K tokens，A100）

| 方法 | FLOPs 稀疏率 | 加速比 | VBench Total↑ |
|------|------------|-------|-------------|
| Full Attention | 0% | 1× | 82.71% |
| STA | 91.6% | 10.2× | 80.46% |
| **GRAT-B** | **94.3%** | **15.8×** | **80.51%** |
| **GRAT-X** | **60.8%** | **2.4×** | **83.82%** |

GRAT-X VBench 超过全量注意力基线。

## 关键启示

- **组内共享 KV 是高效的核心**：与每 token 独立选 KV 的方法不同，所有同组 query 共享同一 KV 集，注意力计算变为规则批量矩阵乘，GPU 利用率远高于 per-token 滑窗方案。同样 FLOPs 稀疏率下，GRAT-B 比 NA 快 2.85×
- **GRAT-B vs GRAT-X 是速度-感受野的精确权衡**：B 变体局部强但远距离截断；X 变体通过 criss-cross 覆盖整行整列，实际质量优于全量注意力（视频 +1.1%）。两者组合可覆盖大多数场景
- **训练无关是关键优势**：预训练 Flux/HunyuanVideo 直接插入，无 LoRA/finetune，FLOPs 节省直接可用
- **高分辨率下加速收益超线性增长**：1024 时加速有限，8192×8192 时达 35.8× 注意力加速，因为 $O(N^2)$ 中 constant factor 随 N 线性增长，稀疏方案的绝对收益更大
- **Flex Attention 实现无需定制 kernel**：借助 PyTorch Flex Attention 框架，无需 Triton 手写 kernel 即可获得高效稀疏注意力，降低了实现门槛
