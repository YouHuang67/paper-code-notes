---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
---

# MonarchRT: Efficient Attention for Real-Time Video Generation

[arXiv 2602.12271](https://arxiv.org/abs/2602.12271) | [GitHub](https://github.com/Infini-AI-Lab/MonarchRT) | Carnegie Mellon University / University at Buffalo / Morpheus AI

## 概述

实时视频生成（自回归 + 少步去噪）下，主流稀疏注意力方法（top-k、block local、radial）会严重失效：3D 视频注意力并非真正稀疏，某些 head 需要 48–84% 的 token 才能覆盖 95% 注意力权重，oracle top-k 在 10% 计算预算下仍产生严重几何失真。

MonarchRT 提出用 **Monarch 矩阵**参数化视频注意力，同时表示稠密周期性位置模式（rank-1 块）和稀疏语义模式。三项关键改进：

1. **空时维度对齐的块结构**：仅 6 种对齐配置（如 $(fh, w)$）能精确表示可分离位置注意力，错位配置导致严重伪影
2. **Tiled Monarch 参数化**：在基础块内引入 $(c_1, c_2)$ 分块因子，使近似误差随计算预算单调下降，突破原始 Monarch 约束 $b_1 b_2 = N$ 的非单调问题
3. **微调 + 自定义 Triton kernel**：微调将推理时迭代次数从多步降至 1 步；IO-aware kernel 利用 FlashAttention 风格分块，peak memory 通过 mini-sequence 策略控制

主要结果：
- Self-Forcing（自回归，4 步）：95% 稀疏率下 VBench 0.838 vs 稠密 0.836，质量几乎无损
- Wan 2.1-1.3B（双向，4/50 步）：90% 稀疏率超过 VSA 所有指标
- RTX 5090：720p 注意力 **11.8× 快于 FA-2**；H100：**5.6× 快于 FA-3**
- 首次实现 Self-Forcing **16 FPS** 实时视频生成

## 方法

### 3D 视频注意力的结构分析

3D 注意力矩阵 $A$ 并非稀疏，而是两个成分的叠加：

$$A_{(f_0,h_0,w_0),(f_1,h_1,w_1)} = D_{(\cdot),(\cdot)} + S_{(\cdot),(\cdot)} + \epsilon$$

$$D_{(\cdot),(\cdot)} = d_w(w_0, w_1) \cdot d_h(h_0, h_1) \cdot d_t(f_0, f_1)$$

- $D$：位置分量，三维距离函数的可分离积，形成稠密周期性带状结构
- $S$：语义分量，稀疏的跨位置语义对应

由此定理（informal）：$A$ 存在结构分解 $A = P D' + S + \epsilon$，其中 $D'$ 在某排列下为逐块 rank-1，$S$ 为稀疏矩阵。Monarch 矩阵 $M = PLP^\top R$ 恰好能同时表示这两类结构——rank-1 块捕获位置项，语义稀疏性由块内自由度覆盖。

### Monarch 参数化

标准 Monarch 将 $N \times N$ 矩阵（$N = b_1 b_2$）分解为：

$$M = P L P^\top R$$

其中 $P$ 为置换矩阵，$L$ 为 $b_2$ 个 $b_1 \times b_1$ 块对角矩阵，$R$ 为 $b_1$ 个 $b_2 \times b_2$ 块对角矩阵。等价张量形式：$M_{\ell j k i} = L_{j\ell k} R_{k j i}$。

求解 Monarch 因子 $L, R$ 的 MonarchAttention 算法利用 softmax 的变分形式 $A = \arg\max_{A_i \in \Delta_N}\langle A, QK^\top\rangle + H(A)$，约束 $A$ 为 Monarch 结构后，可对 $L$ 和 $R$ 交替极大化，无需显式构造完整注意力矩阵。

### 三大挑战与 MonarchRT 的解决方案

**挑战 1：块与空时结构不对齐**

视频 token 按 $[F, H, W]$ 展平，若块划分不尊重空时维度（如 $(b_1, b_2) = (9, 2)$ 导致时空相邻 token 分到不同块），则块内不再接近 rank-1，近似质量急剧下降，产生严重视觉伪影。

**解决**：对齐原则——$f$、$h$、$w$ 三个维度各自完全包含在 $b_1$ 或 $b_2$ 中，不允许跨维度拆分。有效对齐配置恰好 6 种：

$$(fh, w),\ (w, fh),\ (f, hw),\ (hw, f),\ (fw, h),\ (h, fw)$$

以 $(b_1, b_2) = (fh, w)$ 为例，可精确表示完全可分离位置注意力：$L_{w_0, (f_0, h_0),(f_1,h_1)} = d_t(f_0,f_1) d_h(h_0,h_1)$，$R_{(f_1,h_1),w_0,w_1} = d_w(w_0,w_1)$。

**挑战 2：计算增加不能单调降低误差**

原始 Monarch 约束 $b_1 b_2 = N$，改变块大小只是在两个因子间重新分配参数，不保证总精度单调提升。

**解决**：**Tiled Monarch 参数化**——在基础块 $(b_1, b_2)$ 内引入 $(c_1, c_2)$ 分块因子（$c_1 | b_1$，$c_2 | b_2$），将每个 Monarch 块分为 $c_1^2 c_2^2$ 个子块，每个子块独立做 rank-1 分解。

定理（严格）：$\mathcal{M}(b_1, b_2) \subset \mathcal{M}_\text{tile}(b_1, b_2; c_1, c_2)$（严格包含），Tiled Monarch 严格比标准 Monarch 表达力更强。增大 $(c_1, c_2)$ 等价于细化子块粒度，提供类似 top-k 的单调精度-效率权衡。

实践中选择 $c_1 = \frac{f}{n_f} \cdot \frac{h}{n_h}$，$c_2 = \frac{w}{n_w}$，每个 tile 对应一个大小为 $(n_f, n_h, n_w)$ 的时空邻域，tile 内 rank-1 假设成立。

**挑战 3：迭代精化开销大**

MonarchAttention 需要多步迭代精化才能达到高质量近似，推理延迟难以接受。

**解决**：微调使 1 步迭代足以匹配多步精化的视觉质量。实现上，$\beta$ 项可类 FlashAttention 分块在 SRAM 内计算，$\alpha/c$ 项须存入 HBM（维度依赖 $f_q$ 和 $f_{kv}$），用 mini-sequence 策略（按 query 帧分块处理）控制 peak memory，不影响正确性。

## 实验

### Self-Forcing（自回归，少步）— VBench

| 方法 | 稀疏率 | Quality | Semantic | Total |
|------|-------|---------|----------|-------|
| Dense | 0% | 0.844 | 0.804 | 0.836 |
| Exact top-k | 85% | 0.834 | 0.658 | 0.799 |
| RadialAttention | 85% | 0.841 | 0.718 | 0.816 |
| **MonarchRT** | **90%** | **0.847** | **0.808** | **0.839** |
| **MonarchRT (finetune)** | **95%** | **0.846** | **0.805** | **0.838** |

95% 稀疏率下 VBench 总分甚至微超稠密 baseline。

### Wan 2.1-1.3B（双向，4 步蒸馏）— 训练自由评测

| 方法 | 稀疏率 | PSNR↑ | SSIM↑ | LPIPS↓ | VBench↑ |
|------|-------|-------|-------|--------|---------|
| Dense | 0% | – | – | – | 0.846 |
| SVG2 | ~90% | 10.74 | 0.307 | 0.662 | 0.808 |
| RadialAttention | 85% | 11.43 | 0.290 | 0.711 | 0.727 |
| **MonarchRT** | **90%** | **12.66** | **0.364** | **0.585** | **0.834** |

### 注意力 Kernel 延迟（720p，81 帧，Wan 2.1）

| GPU | FA | VSA (85%) | MonarchRT (97%) | MonarchRT (98%) |
|-----|-----|-----------|----------------|----------------|
| RTX 5090 | 159.11ms | 55.51ms | 28.24ms | **13.53ms** |
| H100 | 53.29ms | 23.93ms | 18.61ms | **9.59ms** |

RTX 5090：**11.8× vs FA-2**；H100：**5.6× vs FA-3**。结合 MonarchRT，Self-Forcing 首次达到 RTX 5090 上 **16 FPS** 实时生成。

## 关键启示

- **3D 视频注意力不稀疏，稀疏近似在实时场景会失效**：周期性位置结构（稠密）+ 语义对应（稀疏）共存，使 oracle top-k 在 10% 计算预算下仍质量崩溃。实时/自回归场景比多步双向扩散更脆弱，因为每步需承载更多信息且误差累积
- **Monarch 矩阵天然契合 3D 注意力分解**：rank-1 块对应可分离位置衰减（时间 × 高度 × 宽度），稀疏语义对应由多块自由度覆盖，不依赖 token 选择
- **块对齐是 Monarch 有效的必要条件**：视频 token 的空时维度不能跨块边界，6 种合法配置各自能精确表示可分离位置注意力；错位配置虽参数量相同但质量完全崩溃
- **Tiled Monarch 提供可控精度-效率权衡**：原始约束 $b_1 b_2 = N$ 无法单调提升精度；引入 $(c_1, c_2)$ 子分块因子后，类似 top-k 的单调权衡成立，同时保持高表达力
- **微调是降低推理开销的关键**：多步迭代精化在推理时开销线性增长，微调后 1 步即可达到多步效果，与高效 Triton kernel 结合实现真正实时生成
