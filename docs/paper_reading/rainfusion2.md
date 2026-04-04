---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
---

# RainFusion2.0: Temporal-Spatial Awareness and Hardware-Efficient Block-wise Sparse Attention

[arXiv 2512.24086](https://arxiv.org/abs/2512.24086) | 华为技术 / 香港科技大学

## 概述

DiT 模型注意力计算成本极高（10K-80K token），现有稀疏注意力方法存在两个短板：（1）稀疏 mask 预测开销大，（2）依赖 GPU 无法泛化到 NPU/ASIC 等 AI 专用芯片。

RainFusion2.0 提出三项关键技术，同时解决效率和硬件泛化问题：
- **块均值代表 token**：用每块 Q/K 的均值计算稀疏 mask，计算量仅 $O((N/b)^2)$
- **时空感知 token 排列**：3D 窗口排列增强块内相似度，提升稀疏近似精度
- **首帧下沉（First Frame Sink）**：视频生成中固定保留与第一帧的注意力关系

实验在 NPU 上评测，80% 稀疏率下 Wan2.2 720p 加速 **1.57×**，90% 稀疏率加速 **1.80×**，视觉质量与全量注意力接近。

## 方法

### 块稀疏 Flash Attention

基础框架与标准 Flash Attention 相同，区别在于增加块级掩码矩阵 $M \in \{0,1\}^{\lceil N/b_q \rceil \times \lceil N/b_k \rceil}$：

- $M_{ij} = 0$：跳过块对 $(Q_i, K_j)$ 的 $Q_i K_j^\top$ 和 $P_{ij} V_j$ 计算
- $M_{ij} = 1$：正常计算

目标是以低开销确定哪些块对可以跳过。

### 块均值代表 token

对每个 Q block 和 K block 计算均值作为代表 token：

$$\hat{q}_i = \text{mean}(Q_i), \quad \hat{k}_j = \text{mean}(K_j)$$

用代表 token 的注意力分数 $\hat{S}_{ij} = \hat{q}_i \hat{k}_j^\top$ 来决定重要性，选 Top-N 块：

$$M_{ij} = \begin{cases} 1 & \text{if } j \in \{j \mid \hat{S}_{ij} \in \text{TopN}\} \\ 0 & \text{otherwise} \end{cases}$$

这比全量分数计算的开销低得多（从 $O(N^2)$ 降到 $O((N/b)^2)$），且块均值的计算可高效地在各种硬件上执行（GEMM 友好）。

块内 token 高相似性的前提决定了均值代表 token 的精度——这是 3D 窗口排列解决的核心问题。

### 3D 窗口排列（时空感知排列）

视频 token 默认按 [F, H, W] 顺序展平，使得 3D 空间中相邻的 token 在 1D 序列中相距甚远，导致块内 token 差异大，均值代表 token 精度低。

RainFusion2.0 在计算注意力前，将 token 重排为 3D 窗口顺序：将 [F, H, W] 空间划分为 3D 窗口，同一窗口内的 token 排列相邻，窗口间顺序展平。重排后块内 token 来自相近的时空位置，相似度大幅提升。

消融实验表明：不加 3D 排列时，Wan2.2 生成视频在某些局部区域出现明显伪影（如虚假岩石），加入后完全消除。

图像生成用类似的 2D 窗口排列。

### First Frame Sink（首帧下沉）

类比 LLM 中的 attention sink（初始 token 受到异常高的注意力），视频生成中首帧 token 对最终视频质量有关键影响：跳过涉及首帧 token 的注意力计算会导致明显质量下降。

固定规则：
- 首帧 Q tokens → attend to all K tokens（全量计算）
- All Q tokens → attend to 首帧 K tokens（全量计算）

实现时将首帧 tokens 移到序列末尾，与文本 tokens 相邻，便于二者一起参与全量注意力（有些模型将视频和文本 tokens 合并输入）。

### 硬件泛化

块均值计算可分解为标准矩阵运算（GEMM），不依赖 GPU 特有操作（如 top-k CUDA 内核），因此可原生运行在 NPU/ASIC 上。论文对比指出 SpargeAttn、SVG2 等方法与 NPU 不兼容，无法参与效率对比。

## 实验

### 主要结果

| 模型/设置 | 稀疏率 | 端到端延迟 | 加速比 | 主观一致性 | 成像质量 |
|----------|-------|-----------|-------|-----------|---------|
| Wan2.2 720p Full | 0% | 532s | – | 0.9717 | 0.6816 |
| RainFusion 80% (w/o 3D) | 80% | 339s | 1.57× | 0.9690 | 0.6791 |
| RainFusion 90% (w/o 3D) | 90% | 295s | 1.80× | 0.9643 | 0.6709 |
| RainFusion 80% (w/ 3D) | 80% | 339s | 1.57× | 0.9683 | 0.6864 |

注：加 3D 排列不增加延迟（排列为 O(N) 操作），但成像质量从 0.6791 提升到 0.6864（超过 Full Attention 的 0.6816）。

HunyuanVideo1.5 720p 80% 稀疏：1.28× 加速；Qwen-image-edit 60% 稀疏：视觉质量几乎不变。

## 关键启示

- **块均值是高效且通用的代表 token**：均值计算只需简单求和，无需 GPU 特有操作，天然适配 NPU/ASIC。精度足够的前提是块内 token 相似——窗口排列解决这一前提
- **3D 窗口排列是提升块内相似度的核心手段**：默认 [F, H, W] 展平打乱时空邻域关系；窗口排列将时空邻近 token 聚到同一块，让均值代表 token 精度大幅提升，且无额外延迟
- **First Frame Sink 是视频生成专用设计**：首帧 token 对时序一致性有全局影响，保留与首帧的完整注意力是视频质量的关键保障，类比 LLM 的 attention sink
- **硬件泛化是差异化方向**：大多数稀疏注意力方法针对 GPU 优化，对 NPU 不友好；用 GEMM 友好的均值操作替代 top-k 预测，是兼顾效率和泛化的关键选择
