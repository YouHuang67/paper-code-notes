---
tags:
  - Sparse Attention
  - Diffusion Model
---

# HilbertA: Hilbert Attention for Image Generation with Diffusion Models

[arXiv 2509.26538](https://arxiv.org/abs/2509.26538) | 纽约大学计算机系

## 概述

稀疏注意力用于图像生成时存在根本矛盾：2D 局部性（相邻像素应在同一注意力块）和 GPU 内存连续性（相邻内存读写才能高效）难以同时满足。CLEAR 强制 2D 圆形感受野但产生非连续内存访问；SpargeAttention 用 Hilbert 曲线对 token 分组来提升块内相似度，但稀疏模式本身仍由动态相似度阈值决定，内存访问不连续。

HilbertA 同时满足三个目标：
1. **2D 局部性**：Hilbert 曲线重排后，序列相邻的 token 在 2D 图像中也相邻
2. **内存连续性**：注意力 tile 在重排序列中是连续内存段，无散列读写
3. **跨 tile 信息传播**：逐层滑动（固定偏移量）+ 中央共享区域实现全局感受野

Flux.1-dev 实验结果：
- 1024×1024（16 tiles，94% 稀疏率）：注意力 **2.30×** 加速，端到端 **1.10×**
- 2048×2048（16 tiles，94% 稀疏率）：注意力 **4.17×** 加速，端到端 **1.51×**

## 方法

### 曲线选择：Hilbert 优于其他空间填充曲线

衡量 2D→1D 映射质量的两个指标：

**Edge Average Stretch (EAS)**：序列中 2D 相邻 token 的平均序列距离（越低越好）

$$\text{EAS}(\pi) = \frac{1}{|E|} \sum_{(u,v) \in E} |\pi^{-1}(u) - \pi^{-1}(v)|$$

**Geometric Distortion Error (GDE)**：全局缩放后，序列距离与 2D 欧氏距离的残差（越低越好）

$$\text{GDE}(\pi) = \frac{1}{M} \sum_{(u,v) \in S} \left(\alpha(\pi) \cdot d_{1D}(u,v) - d_{2D}(u,v)\right)^2$$

对比 Serpentine（S 形）、Spiral（螺旋）、Morton（Z 形）、Moore、Hilbert 五种曲线，Hilbert 在所有网格尺寸下 GDE 最低，EAS 第二低（Moore 的 EAS 略低，但 GDE 明显更高）。Hilbert 的分形自相似性保证了任意 tile 大小都能形成 2D 连续区域。

### Tile and Slide 机制

**局部 tiling**：将 Hilbert 重排序列划分为不重叠的 $N_T$-token tile，每个 tile 内做全量注意力。tile 大小 $N_T$ 控制局部性强度（小 $N_T$ = 高稀疏率，大 $N_T$ = 更大上下文）。

**逐层滑动**：设序列长度 $N$，tile 数 $T = N/N_T$，滑动周期 $L$，每层偏移量 $\Delta = N_T / L$。第 $\ell$ 层，token $i$ 的注意力窗口为：

$$A_i^{(\ell)} = \text{tile}\left(\left\lfloor \frac{i + \ell\Delta}{N_T} \right\rfloor \bmod T\right)$$

关键性质：每次滑动引入新 tile 的 token，该 token 已在上一层聚合了其 tile 的信息，作为"信使"传递给新 tile。因此 ERF（有效感受野）每层扩大一个完整 tile：

$$|ERF_i(t)| = N_T \cdot \min(T, t)$$

$t = T$ 层后，ERF 覆盖整个序列。实现上用取模索引（序列末端绕回头部），保持内存连续性，无需内存拷贝。

### 中央共享区域（Shared Region for RoPE）

每个 tile 额外 attend 到图像中心固定区域（1024×1024 时 256 tokens，2048×2048 时 1024 tokens），两个作用：
- **全局中继**：各 tile 都能与中心区域交换信息，跨 tile 通信不依赖多层传递
- **RoPE 锚点**：RoPE 编码相对位置，tile 内 token 知道彼此的相对位置，但不知道 tile 在全图中的位置；固定中央区域为所有 tile 提供一致的位置参考点

中央区域固定（非动态、无额外参数），无需逐层 reindex 或内存复制，保持内存连续性。

### Triton Kernel 实现

- Hilbert 重排预计算好双射函数，推理时仅做 **两次 gather**（重排 + 恢复原顺序），overhead 极低
- 滑动实现为**指针偏移**，不拷贝内存
- 自定义 Triton kernel：单次 kernel launch 执行**两路并行 pass**：
  - 局部 tile 内的稀疏注意力
  - 全局前缀（文本 token + 中央共享 token）的注意力（FlashAttention 风格）
- 两路融合，无冗余 kernel launch

Reorder overhead（相对于注意力延迟）：4096 token 为 7–17%，16384 token 为更高；但 2048×2048 下相对于 attention 延迟仍可接受。

## 实验

### 主要结果（Flux.1-dev，A100）

| 方法 | 配置 | 稀疏率 | FID↓ | LPIPS↓ | CLIP-I↑ | 注意力延迟 | 端到端延迟 |
|------|------|-------|------|--------|---------|---------|---------|
| **1024×1024** | | | | | | | |
| Flux.1-dev (Dense) | – | 0% | 30.6 | 0.0 | 100.0 | 261ms (1×) | 13.85s (1×) |
| SpargeAttention | – | 17% | 28.7 | 51.5 | 91.3 | – | 14.32s (0.97×) |
| CLEAR | r=8 | 95% | 33.0 | 50.0 | 90.0 | – | 12.85s (1.06×) |
| HilbertA | 16 tiles | 94% | 31.3 | 56.3 | 87.6 | **0.62×→2.30×** | 12.57s (1.10×) |
| **2048×2048** | | | | | | | |
| Flux.1-dev (Dense) | – | 0% | 32.6 | 0.0 | 100.0 | 3299ms (1×) | 65.28s (1×) |
| CLEAR | r=8 | 99% | 43.2 | 61.6 | 80.5 | – | 47.81s (1.48×) |
| HilbertA | 16 tiles | 94% | 47.5 | 57.1 | 78.2 | **4.36×→4.17×** | 45.19s (1.51×) |

注：HilbertA 在更低稀疏率下（94% vs CLEAR 的 99%）获得更高加速——说明加速来源不只是 FLOP 减少，更关键是内存访问连续性。SpargeAttention 在两个分辨率下端到端均慢于或持平 Dense（0.97×/0.83×），证明非连续内存访问抵消了稀疏计算带来的收益。

### 训练
- LoRA 微调 Flux.1-dev 的投影权重，自收集 10K 图文对
- 1024×1024：40 GPU-hours（4×H100，9 epochs）；2048×2048 继续 40 GPU-hours
- 配置：16 tiles，4 个滑动周期
- 额外验证：LightningDiT-B/1 从零训练，收敛稳定，质量有竞争力

## 关键启示

- **真正的加速瓶颈是内存访问模式，而非 FLOP 数量**：HilbertA 在比 CLEAR 更低稀疏率下获得更高加速，正是因为 Hilbert 重排保证了内存连续读写；CLEAR 和 SpargeAttention 的非连续访问吞噬了计算节省
- **Hilbert 曲线是 2D→1D 映射的最优选择**：EAS+GDE 双指标量化验证，分形自相似性保证任意 tile 大小在 2D 中均保持连续区域，无需特殊对齐
- **Sliding + Shared Region 分工明确**：Sliding 逐层扩大 ERF（T 层后全覆盖），Shared Region 提供即时全局中继和 RoPE 位置锚点，两者都不打破内存连续性
- **固定中央区域优于动态或可学习区域**：无额外参数，无逐层 reindex，硬件友好，跨分辨率鲁棒
