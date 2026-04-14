---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
---

# Compact Attention: Exploiting Structured Spatio-Temporal Sparsity for Fast Video Generation

[arXiv 2508.12969](https://arxiv.org/abs/2508.12969) | [项目主页](https://yo-ava.github.io/Compact-Attention.github.io/) | 浙江大学, 华为

## 概述

视频扩散 Transformer（vDiT）中注意力计算占总推理时间的 68-72%，是长视频生成的核心瓶颈。Compact Attention 通过系统分析 vDiT 注意力图的结构化稀疏性，提出了一个硬件感知的稀疏注意力加速框架。

核心发现：注意力图在**逐 query token** 视角下呈现出层次化的时空模式——空间维度有 local / cross-shaped / global 三种模式，时间维度有 time-variant / time-invariant 两种模式。这些模式在不同 prompt 和 seed 下高度稳定（相似度 > 0.8），且在连续去噪步之间保持一致（94.4% 相似度），因此可以**离线预计算**稀疏掩码。

方法包含三个关键创新：
1. **自适应 tile 分组**：将 3D token 组织为时空 tile，构建可变形稀疏模式
2. **时间可变窗口**：按帧距离分组，每组独立配置稀疏参数
3. **自动化掩码搜索**：双阈值（recall τ + cost λ）约束下的边界收缩算法

主要结果：
- Hunyuan（127K token）：62.36% 稀疏率下 **2.51×** 加速，PSNR 30.08
- Wan2.1（80K token）：33.99% 稀疏率下 **1.65×** 加速，PSNR 23.73
- VBench 指标几乎无损，部分指标甚至优于全注意力基线

## 注意力模式分析

### Tile 粒度的稀疏性

逐 token 稀疏预测开销太大，无法实际加速。论文观察到注意力中的关键信息在 3D 空间中聚集，因此以 tile（时空相邻 token 块）为基本计算单元。

对比两种 flatten 策略：
- 直接按 (f, h, w) 展平为 1D 序列
- 按 3D 空间相邻性分组再展平（STA 方式）

后者在保持 0.95 recall 所需的块数上，Wan2.1 减少 1.1%，Hunyuan 减少 3.4%，同时兼容 block-wise 注意力机制。

### 五种结构化模式

**空间模式**（逐 query token 的注意力分布）：

- **Local Pattern**：紧凑邻域球形注意力场，服务于精细细节合成。形式化为 $R_{\text{local}} = \{(x,y) \mid \max(\frac{|x-x_t|}{\omega}, \frac{|y-y_t|}{\eta}) \le 1\}$
- **Cross-shaped Pattern**：沿水平/垂直轴形成连续注意力走廊，有方向敏感性。形式化为两个互补轴主导的矩形并集，约束 $(\omega_1 - \omega_2)(\eta_1 - \eta_2) < 0$
- **Global Pattern**：全空间连接或输入依赖的显著区域聚集

**时间模式**：

- **Time-Variant**：注意力权重随帧间距递减（或聚焦于特定距离的帧）
- **Time-Invariant**：帧无关的均匀分布

### 模式稳定性

- **输入/种子不变性**：不同 prompt 和 seed 下，模式区域大小的相似度 > 0.8，通过二值化掩码的 IoU 度量
- **时间步鲁棒性**：连续去噪步之间注意力配置保持稳定（94.4% 相似度区间内），支持跨步复用

## Compact Attention 框架

### Tile-Based 可变形稀疏模式

设计思路：不使用固定注意力窗口，而是让稀疏配置在时空维度上自适应。

两个核心组件：
- **Frame-Group-wise Patterns**：按与当前帧的时间距离将帧分组，每组配置独立的稀疏参数。捕获时间维度的动态性
- **Dual Attention Windows**：每组内由两个互补窗口形状组合近似观察到的注意力模式（如 cross-shaped + local），无需推理时显式分类模式类型

这一架构实现三重协同：空间自适应（tile 组合模拟多种模式）、时间感知（距离分层配置）、硬件效率（tile 处理的计算规则性）。

### 离线自动搜索算法

将掩码优化建模为边界收缩过程：
1. 从全注意力覆盖开始
2. 沿层次化维度迭代收紧窗口边界，优先收缩 recall 贡献低的区域
3. 各帧分组独立收缩

双阈值控制终止：
- **Recall 阈值 τ**：保证关键交互不丢失（如 τ = 0.9）
- **Cost 阈值 λ**：平衡计算减少与精度损失（如 λ = 0.011 for Wan, 0.04 for Hunyuan）

跨 prompt 合并策略：对多个 prompt 搜索结果取**并集**（保守策略），确保所有潜在相关注意力区域被保留。

利用时间步稳定性，掩码跨 n 个连续去噪步复用，搜索频率降低 n 倍。

### 硬件实现

基于 **ThunderKittens** 实现（参考 STA 框架），这是一个基于 CUDA 的 tile-level 注意力加速库。ThunderKittens 原生支持 tile 粒度的注意力计算，与 Compact Attention 的 tile-based 设计天然匹配。

关键实现细节：
- Tile 大小对齐 FlashAttention 的块计算粒度
- 稀疏掩码直接映射到 tile 跳过策略，跳过整个 tile 块而非逐 token 判断
- 在 H800 单卡上评测

## 实验

### 消融：稀疏模式组件贡献

| 模式组件 | Cubic Window | + Frame-Group | + Dual Window |
|---|---|---|---|
| Local | 0.726 | 0.758 | 0.766 |
| Cross | 0.385 | 0.406 | **0.516** |
| Global | 0.078 | 0.085 | 0.099 |
| Time-Variant | 0.441 | 0.472 | 0.567 |
| Time-Invariant | 0.306 | 0.317 | 0.385 |
| **Overall** | 0.361 | 0.370 | **0.459** |

Dual Window 对 cross-shaped pattern 提升最大（+13.1%），Frame-Group 对 time-variant 有额外 ~3% 提升。

### 主要结果

| Model | Method | Sparsity | PSNR↑ | Speedup |
|---|---|---|---|---|
| Wan2.1 (80K) | Full Attention | 0% | - | 1.00× |
| | SVG | 32.08% | 15.96 | 0.91× |
| | SpargeAttn | 32.27% | 20.52 | 1.02× |
| | **Compact (Ours)** | **33.99%** | **23.73** | **1.65×** |
| Hunyuan (127K) | Full Attention | 0% | - | 1.00× |
| | SVG | 50.35% | 20.43 | 1.23× |
| | SpargeAttn | 47.77% | 23.59 | 1.19× |
| | **Compact (Ours)** | **62.36%** | **30.08** | **2.51×** |

Compact Attention 在更高稀疏率下同时实现了更高的 PSNR 和加速比。Hunyuan 上由于序列更长（127K），稀疏化收益更显著。

### 早期步保护

前 15 个去噪步保持全注意力对质量至关重要（高噪声输入需要结构初始化）。仅最后 15 步用全注意力 vs 前 15 步用全注意力，PSNR 差 1.02dB。

## 关键启示

- **逐 query token 分析**是理解 vDiT 注意力稀疏性的正确粒度——比直接看全注意力图更能揭示结构化模式
- **模式稳定性**（输入/步/种子不变）使离线搜索成为可能，这比在线预测有天然的质量优势：可以用 post-softmax 的完整信息，而非 pre-softmax 的不完整近似
- **Dual Window + Frame-Group** 的组合以极少的设计复杂度逼近了多种注意力模式，避免了显式模式分类的开销
- ThunderKittens 作为 tile-level CUDA 加速后端，与结构化稀疏的 tile 跳过策略天然匹配
