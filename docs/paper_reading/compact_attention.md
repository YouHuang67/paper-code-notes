---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
  - CUDA
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

本文的核心贡献不在于提出新的数学理论，而在于：(1) 对 vDiT 注意力图建立了清晰的模式分类体系，(2) 将模式识别转化为可形式化的区域描述，(3) 设计了完整的自动搜索流程将模式描述映射到硬件友好的稀疏掩码。下文按此逻辑逐步展开全部数学 formulation。

## Tile 粒度的稀疏性

逐 token 稀疏预测开销太大，无法实际加速。论文观察到注意力中的关键信息在 3D 空间中聚集，因此以 tile（时空相邻 token 块）为基本计算单元。

FlashAttention 将 Q/K/V 沿 token 维度分为 block size 为 $b_q, b_k$ 的块 $Q_i, K_i, V_i$，通过 online softmax 增量计算每个输出块 $O_i$。这一设计使 tile 级别的稀疏跳过成为可能——整块跳过而非逐 token 判断。

关键问题：如何组织 tile？视频潜变量的 3D 结构 $(f, h, w)$ 展平为 1D 序列时，存在两种策略：
- **直接维度展平**：按 $(f, h, w)$ 层次展平为 1D 序列
- **3D 邻近分组**：先将空间相邻 token 分组为 3D tile，再展平（STA 方式）

后者利用了注意力在 3D 空间中的聚集性。实测比较：保持 0.95 recall 所需的 block 数（即 top-k% 值），3D 邻近分组在 Wan2.1 上减少 **1.1%**（58.8% vs 59.9%），在 Hunyuan 上减少 **3.4%**（47.7% vs 51.1%）。减少的 block 数直接对应减少的计算量。

## 结构化时空模式的形式化

### 观察方法：逐 query token 注意力分布

现有稀疏注意力方法（MInference、SVG 等）在 1D 全注意力图上识别 slash-line / vertical-line / block 等模式。论文指出，这种 1D 视角下的模式形态复杂且难以利用。正确的分析粒度是**逐 query token**：固定一个 query token $(x_t, y_t)$ 在第 $i$ 帧，观察其对所有 key token 的注意力分布。在这个视角下，注意力图呈现出清晰的结构化模式。

### 空间模式形式化

**Local Pattern**：注意力集中在 query 位置 $(x_t, y_t)$ 的紧凑邻域内，形成轴对齐的矩形注意力场。参数 $\omega, \eta$ 分别为水平和垂直方向的边界宽度：

$$\mathcal{R}_{\text{local}} = \left\{(x, y) \;\middle|\; \max\left(\frac{|x - x_t|}{\omega}, \frac{|y - y_t|}{\eta}\right) \le 1\right\}$$

几何解释：这定义了以 $(x_t, y_t)$ 为中心、宽 $2\omega$、高 $2\eta$ 的矩形区域。$\max$ 运算使边界为矩形而非椭圆，对齐 tile 的轴对齐计算方式。

**Cross-shaped Pattern**：注意力沿水平和垂直轴形成十字形走廊。形式化为两个互补矩形的并集：

$$\mathcal{R}_{\text{cross}} = \left\{(x, y) \;\middle|\; \bigvee_{k=1}^{2}\left(\frac{|x - x_t|}{\omega_k} \le 1 \;\wedge\; \frac{|y - y_t|}{\eta_k} \le 1\right)\right\}$$

关键约束：$(\omega_1 - \omega_2)(\eta_1 - \eta_2) < 0$，即**互补轴主导**。这意味着两个矩形一个宽而矮（$\omega_1 > \omega_2, \eta_1 < \eta_2$），另一个窄而高，组合形成十字形。如果没有这个约束，两个矩形可能都是宽而矮，退化为一个大矩形而非十字。

**Global Pattern**：注意力覆盖空间范围超过 85%（recall-based 阈值），无法稀疏化。包括两种情况：全空间均匀连接，或输入依赖的显著区域聚集（但范围仍然很大）。

### 时间模式

- **Time-Variant**：注意力权重与帧间相对距离 $|i - j|$ 强相关。典型行为包括：随距离递减（近帧主导），或聚焦于特定距离的帧（远帧主导，排除近帧）
- **Time-Invariant**：注意力权重在所有帧上均匀分布，与相对时间距离无关

### 模式分类标准

论文使用 recall-based 阈值进行分类：
- 空间覆盖 > 85% → Global Pattern
- 空间覆盖 ≤ 85% 且符合式 (1) → Local Pattern
- 空间覆盖 ≤ 85% 且符合式 (2) → Cross-shaped Pattern

### 模式稳定性的度量

**输入/种子不变性**：对二值化注意力掩码 $M_A, M_B$（不同 prompt 或 seed 生成），定义 IoU 相似度：

$$\text{Sim}(M_A, M_B) = \frac{\|M_A \odot M_B\|_1}{\|M_A + M_B - M_A \odot M_B\|_1}$$

其中 $\odot$ 为 element-wise 乘法。分子 = 交集大小，分母 = 并集大小（标准 IoU）。实测不同 prompt/seed 下模式区域大小的平均相似度 > 0.8，确认模式由 layer/head 位置决定，而非输入内容。

**时间步鲁棒性**：在连续去噪步之间，注意力配置的相似度在 94.4% 范围内保持稳定。这为**掩码跨步复用**提供了理论依据——同一掩码可在 $n$ 个连续去噪步内共享。

## Compact Attention 掩码构造

### 问题设定

设视频潜变量有 $f$ 帧，每帧空间分辨率 $h \times w$，总 token 数 $n = f \cdot h \cdot w$。token 按 3D 邻近性分组为 tile，每个 tile 对应一个注意力计算块。目标：为每个 layer-head 组合找到一个二值掩码 $M \in \{0, 1\}^{n/b_q \times n/b_k}$（tile 级别），使得 $M_{ij} = 0$ 的 tile 对被跳过，同时保持输出质量。

### Frame-Group-wise 时间分层

将所有帧按与当前 query 帧 $i$ 的时间距离分组：

$$G_d = \{j \mid |j - i| \in [\delta_d, \delta_{d+1})\}, \quad d = 0, 1, \ldots, D-1$$

其中 $\delta_0 = 0 < \delta_1 < \cdots < \delta_D = f$ 为分组边界。每个帧组 $G_d$ 配置**独立的**空间稀疏参数。

设计动机：Time-Variant 模式表明注意力权重与帧距离强相关——近帧通常需要更大的空间窗口（更多交互），远帧可以使用更小的窗口。Frame-Group-wise 分层使得不同距离的帧拥有不同的稀疏配置，而非 STA 的 cubic window（所有帧同一窗口大小）。

### Dual Attention Windows

在每个帧组 $G_d$ 内，空间掩码由**两个互补窗口**的并集构成。对 query token $(x_t, y_t)$，在帧组 $G_d$ 中的空间掩码为：

$$\mathcal{W}_d = \mathcal{R}_d^{(1)} \cup \mathcal{R}_d^{(2)}$$

其中每个 $\mathcal{R}_d^{(k)}$ 是一个轴对齐矩形：

$$\mathcal{R}_d^{(k)} = \left\{(x, y) \;\middle|\; \frac{|x - x_t|}{\omega_d^{(k)}} \le 1 \;\wedge\; \frac{|y - y_t|}{\eta_d^{(k)}} \le 1\right\}$$

参数 $(\omega_d^{(1)}, \eta_d^{(1)}, \omega_d^{(2)}, \eta_d^{(2)})$ 由自动搜索确定。当两个矩形满足互补轴主导约束时，并集近似十字形（Cross-shaped Pattern）；当两个矩形相近时，退化为单一矩形（Local Pattern）；当两个矩形都很大时，覆盖全空间（Global Pattern）。

这样一来，**无需在推理时显式判断 head 属于哪种模式**——参数本身就编码了模式信息。

### 完整掩码的参数化

对于 layer $l$、head $h$、去噪步 $t$，完整掩码由以下参数集定义：

$$\Theta_{l,h,t} = \left\{(\omega_d^{(1)}, \eta_d^{(1)}, \omega_d^{(2)}, \eta_d^{(2)})\right\}_{d=0}^{D-1}$$

总参数量 = $L \times H \times \lceil T/n \rceil \times D \times 4$，其中 $L$ 为层数，$H$ 为 head 数，$T$ 为总去噪步数，$n$ 为掩码复用步数。实际中 $D$ 和 $n$ 都不大，搜索空间可控。

## 离线自动搜索算法

### Recall 与 Cost 的定义

**Recall**：给定稀疏掩码 $M$ 和全注意力的 post-softmax 分数矩阵 $A$，recall 衡量被保留的注意力权重占比：

$$\text{Recall}(M) = \frac{\sum_{(i,j): M_{ij}=1} A_{ij}}{\sum_{i,j} A_{ij}}$$

recall = 1 表示保留了全部注意力权重（全注意力），recall = 0.9 表示丢失了 10% 的注意力权重。

**Cost**：边界收缩一步带来的 recall 损失与计算减少之比：

$$\text{Cost} = \frac{\Delta\text{Recall}}{\Delta\text{Sparsity}}$$

直观含义：收缩一个单位的窗口边界，recall 下降多少。Cost 高意味着该区域包含重要的注意力信息，不应收缩；Cost 低意味着可以安全收缩。

### 边界收缩过程

搜索建模为从全覆盖到紧凑掩码的**边界收缩**过程：

1. **初始化**：所有帧组的窗口参数设为最大值（覆盖全空间），对应全注意力
2. **迭代收缩**：对每个帧组 $G_d$，沿空间维度逐步缩减窗口边界参数 $(\omega_d^{(k)}, \eta_d^{(k)})$
3. **优先级**：优先收缩 Cost 最低的维度/帧组，即 recall 贡献最小的区域先被裁剪
4. **各帧组独立收缩**：不同距离的帧组有不同的最优窗口大小

### 双阈值终止条件

收缩过程由两个阈值联合控制：

- **Recall 阈值 $\tau$**：当 $\text{Recall}(M) < \tau$ 时停止，保证关键注意力交互不丢失
- **Cost 阈值 $\lambda$**：当 $\frac{\Delta\text{Recall}}{\Delta\text{Sparsity}} > \lambda$ 时停止，避免为微小的计算减少付出过大的质量代价

终止条件（对每个帧组独立判断）：

$$\text{停止} \iff \text{Recall}(M) < \tau \;\;\lor\;\; \frac{\Delta\text{Recall}}{\Delta\text{Sparsity}} > \lambda$$

默认参数：$\tau = 0.9$，$\lambda = 0.011$（Wan2.1）/ $\lambda = 0.04$（Hunyuan）。

### 跨 Prompt 合并

对 $P$ 个 prompt 分别搜索得到掩码 $M^{(1)}, \ldots, M^{(P)}$，取**并集**（保守策略）：

$$M_{\text{final}} = M^{(1)} \cup M^{(2)} \cup \cdots \cup M^{(P)}$$

并集操作保证所有 prompt 下潜在相关的注意力区域都被保留。由于模式稳定性，少量 prompt（论文中约 5-10 个）即可获得鲁棒的合并结果。

### 时间步掩码复用

利用时间步鲁棒性（连续步间 94.4% 相似度），同一掩码跨 $n$ 个连续去噪步复用：

$$M_t = M_{\lfloor t/n \rfloor \cdot n}, \quad t = 0, 1, \ldots, T-1$$

搜索频率降低 $n$ 倍。实测 $n$ 在合理范围内（如 $n = 5$）不影响生成质量。

## 硬件实现

基于 **ThunderKittens** 实现（参考 STA 框架），这是一个基于 CUDA 的 tile-level 注意力加速库。

关键实现细节：
- Tile 大小对齐 FlashAttention 的块计算粒度（$b_q, b_k$）
- 稀疏掩码 $M$ 直接映射到 tile 跳过策略：$M_{ij} = 0$ 时整个 tile 块的 $Q_i \cdot K_j^T$ 计算被跳过，无需逐 token 判断
- Compact Attention 的 Dual Window + Frame-Group 设计天然产生规则的矩形跳过区域，与 tile-based 计算高度匹配
- 评测硬件：H800 单卡

## 实验

### 消融：模式组件对稀疏率的贡献

在 Wan2.1 推理阶段的注意力 head 上（$\tau = 0.9, \lambda = 0.011$），按模式分类统计各组件带来的稀疏率提升：

| 模式类别 | Cubic Window | + Frame-Group | + Dual Window |
|---|---|---|---|
| Local | 0.726 | 0.758 | 0.766 |
| Cross | 0.385 | 0.406 | **0.516** |
| Global | 0.078 | 0.085 | 0.099 |
| Time-Variant | 0.441 | 0.472 | 0.567 |
| Time-Invariant | 0.306 | 0.317 | 0.385 |
| **Overall** | 0.361 | 0.370 | **0.459** |

- Dual Window 对 Cross-shaped Pattern 提升最大（+11.0%），因为十字形注意力恰好需要两个互补矩形来近似
- Frame-Group 对 Time-Variant 有额外 ~3.1% 提升，因为时间变化的模式需要按距离分层配置
- Overall 从 Cubic Window 的 0.361 提升到 0.459（+9.8%），同时保持 recall ≥ 0.9

### 主要结果

| Model | Method | Sparsity | SSIM↑ | PSNR↑ | MSE↓ | Latency (s) | Speedup |
|---|---|---|---|---|---|---|---|
| Wan2.1 (80K) | Full Attention | 0% | - | - | - | 1092.2 | 1.00× |
| | Sparse VideoGen | 32.08% | 0.529 | 15.96 | 1894.4 | 1200.1 | 0.91× |
| | SpargeAttn | 32.27% | 0.610 | 20.52 | 676.1 | 1065.8 | 1.02× |
| | **Compact (33.99%)** | **33.99%** | **0.775** | **23.73** | **351.6** | **663.8** | **1.65×** |
| | Compact (24.66%) | 24.66% | 0.815 | 25.27 | 254.2 | 758.2 | 1.44× |
| Hunyuan (127K) | Full Attention | 0% | - | - | - | 1370.7 | 1.00× |
| | Sparse VideoGen | 50.35% | 0.725 | 20.43 | 822.9 | 1117.8 | 1.23× |
| | SpargeAttn | 47.77% | 0.779 | 23.59 | 369.3 | 1148.6 | 1.19× |
| | **Compact (62.36%)** | **62.36%** | **0.904** | **30.08** | **105.2** | **546.5** | **2.51×** |
| | Compact (52.90%) | 52.90% | 0.945 | 34.55 | 35.1 | 750.2 | 1.83× |

关键观察：
- Compact Attention 在更高稀疏率下实现了显著更高的 PSNR 和加速比。Hunyuan 上序列更长（127K），稀疏化收益更显著
- SVG 在 Wan2.1 上甚至慢于全注意力（0.91×），因为其动态掩码预测的开销抵消了稀疏计算的节省
- Compact 提供两档配置（高稀疏 / 高质量），通过调节 $\tau, \lambda$ 灵活控制

### VBench 质量评估

| Model | Method | Sparsity | Subject Consist. | BG Consist. | Aesthetic | CLIPSIM | CLIP-T |
|---|---|---|---|---|---|---|---|
| Wan2.1 | Full Attention | 0% | 0.9681 | 0.9616 | 0.6486 | 0.2118 | 0.9985 |
| | Compact (33.99%) | 33.99% | 0.9659 | **0.9650** | 0.6480 | **0.2121** | 0.9985 |
| Hunyuan | Full Attention | 0% | 0.9736 | 0.9735 | 0.6542 | 0.2181 | 0.9995 |
| | Compact (62.36%) | 62.36% | 0.9716 | 0.9693 | 0.6531 | **0.2184** | 0.9995 |

Compact Attention 在 VBench 指标上几乎无损，部分指标（BG Consistency、CLIPSIM）甚至略优于全注意力。

### Recall 阈值敏感性

固定 $\lambda$，调节 $\tau$ 对稀疏率的影响（Wan2.1 $\lambda = 0.011$，Hunyuan $\lambda = 0.04$）：
- $\tau$ 从 0.95 降到 0.3 时，稀疏率单调上升
- Hunyuan 在所有 $\tau$ 下稀疏率均高于 Wan2.1（模型更小 → 注意力更稀疏）
- 当 $\tau$ 足够低时，稀疏率收敛到由 $\lambda$ 约束决定的上界——此时即使 recall 允许更大损失，Cost 约束也阻止进一步收缩

### 早期步保护

前 15 个去噪步保持全注意力对质量至关重要（高噪声输入需要结构初始化）：

| 开始稀疏化的步数 | PSNR | Latency (s) |
|---|---|---|
| Step 0 | 11.29 | 641.0 |
| Step 5 | 13.44 | 646.8 |
| Step 10 | 15.87 | 655.7 |
| Step 15 | 19.17 | 663.8 |
| Step 20 | 22.49 | 674.6 |
| 全注意力 | - | 1544.0 |

从 Step 0 到 Step 15，PSNR 从 11.29 提升到 19.17，每步增加约 2.6 dB，表明早期步的全注意力对结构初始化不可或缺。仅最后 15 步用全注意力 vs 前 15 步用全注意力，PSNR 差 1.02 dB。

## 关键启示

- **逐 query token 分析**是理解 vDiT 注意力稀疏性的正确粒度——比直接看全注意力图更能揭示结构化模式。在 1D 全注意力图中复杂的 slash/block 结构，在 per-query 3D 视角下清晰分解为 local / cross-shaped / global 三种可参数化的几何形状
- **Dual Window 的互补约束 $(\omega_1 - \omega_2)(\eta_1 - \eta_2) < 0$ 是关键设计**：这一约束使两个矩形形成十字形而非退化为单一大矩形，用 4 个参数即可近似 cross-shaped pattern。消融实验中 Cross Pattern 的稀疏率从 0.406 提升到 0.516，说明互补约束有效
- **模式稳定性**（IoU > 0.8 跨 prompt/seed，94.4% 跨步相似度）使离线搜索成为可能。这比在线预测有天然的质量优势：可以用 post-softmax 的完整注意力信息，而非 pre-softmax 的不完整近似
- **双阈值搜索的设计**通过 recall $\tau$ 和 cost $\lambda$ 两个独立旋钮分别控制质量下限和效率上限，提供了灵活的质量-速度权衡。$\tau$ 足够低时稀疏率收敛到 $\lambda$ 决定的上界，避免了过度稀疏
- **Frame-Group-wise 分层捕获了 STA cubic window 遗漏的时间动态**：同一空间位置在不同帧距离下需要不同大小的注意力窗口，这是 Overall 稀疏率从 0.361 提升到 0.370 再到 0.459 的重要驱动力
