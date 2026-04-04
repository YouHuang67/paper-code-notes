---
tags:
  - Sparse Attention
  - Diffusion Model
  - Video Generation
---

# PISA: Piecewise Sparse Attention Is Wiser for Efficient Diffusion Transformers

[arXiv 2602.01077](https://arxiv.org/abs/2602.01077) | 代码已开源 | 香港科技大学（广州）

## 概述

扩散 Transformer（DiT）在图像/视频生成中表现卓越，但注意力的二次复杂度严重拖慢推理速度。现有块稀疏注意力的做法是直接丢弃非关键 KV 块（"keep-or-drop"），高稀疏率下质量明显下降。

PISA 的核心观察：非关键块的预-softmax 分数分布集中在零或负值附近，具有对称钟形结构，因此可以用 **块级 Taylor 展开**精确近似，而非直接丢弃。这一"exact-or-approximate"范式让 PISA 在保持全量注意力覆盖范围的同时实现亚二次复杂度。

主要结果：
- Wan2.1-14B 视频生成：**1.91×** 加速，质量几乎无损
- Hunyuan-Video-13B：**2.57×** 加速
- FLUX.1-dev 图像生成：**1.2×** 加速，同等或更高稀疏率下显著优于 SpargeAttn

## 方法

### 核心问题

块稀疏注意力按块粒度选择关键 KV（Top-K），直接忽略剩余块。问题在于：

1. **高稀疏率下质量下降**：被丢弃的 "长尾" 非关键块仍包含有效信息
2. **短序列（如 4K token）效率差**：SpargeAttn 在短序列下甚至慢于 FlashAttention-3
3. **与预训练权重不兼容**：线性注意力/混合注意力改变了 softmax 归一化结构，必须重训练

### 关键观察

对 Wan2.1-1.3B 预-softmax 分数 $QK^\top$ 的统计分析：
- 非关键块：分数集中在 $(-\infty, 0]$，分布对称，**一阶 Taylor 展开误差极小**
- 关键块：分数值较大，分布弥散，需要精确计算

这个性质在数值稳定 "Safe-Exp" 平移后依然成立。

### PISA 分段稀疏注意力

将 KV 块划分为：选中集 $S_i$（精确计算）和未选中集 $U_i$（Taylor 近似）。

注意力输出 $o_t = N_t / D_t$，其中：

**分母**（normalization term）：
$$D_t = \underbrace{\sum_{j \in S_i} \sum_n \exp(q_t k_{j,n}^\top)}_{\text{精确稀疏项}} + \underbrace{\sum_{j \in U_i} B \cdot \exp(q_t \bar{k}_j^\top)}_{\text{块级近似}}$$

注：一阶项在分母中恰好消去（因 $\sum_n (k_{j,n} - \bar{k}_j)^\top = 0$）。

**分子**（value aggregation）：

$$N_t = \underbrace{\sum_{j \in S_i} \sum_n \exp(q_t k_{j,n}^\top) v_{j,n}}_{\text{精确稀疏}} + \underbrace{\sum_{j \in U_i} \exp(q_t \bar{k}_j^\top) \sum_n v_{j,n}}_{\text{块级零阶近似}} + \underbrace{\text{一阶修正}}_{\text{见下}}$$

块级精确一阶项为 $\sum_{j \in U_i} \exp(q_t \bar{k}_j^\top) \cdot q_t \sum_n (k_{j,n} - \bar{k}_j)^\top v_{j,n}$，但实现上内存访问是瓶颈（每块需要独立的 $d \times d$ 矩阵）。

### 混合阶近似（Hybrid Approximation）

**全局一阶修正**：将逐块的一阶项替换为全局统计量的共享修正：

$$\text{一阶修正} \approx \left(\sum_{j \in U_i} \exp(q_t \bar{k}_j^\top)\right) \cdot q_t \bar{H}$$

其中 $\bar{H} = \frac{1}{N} \sum_{j=1}^N H_j$，$H_j = \sum_n (k_{j,n} - \bar{k}_j)^\top v_{j,n} \in \mathbb{R}^{d \times d}$，在单次预扫描中预计算。

**误差界（Theorem 3.1）**：

$$\|\tilde{o}_t - o_t\|_2 \leq C_q M \frac{\rho_t}{B}$$

- $M = \max_{j \in U_i} \|H_j - \bar{H}\|_2$：块间异质性
- $\rho_t = \tau_t / D_t$：未选中块上的注意力权重占比（"尾部质量"）
- $B$：块大小（分母中出现，因 Jensen 不等式）

由于 $\rho_t$ 在高稀疏率下很小，误差自然受控。

### 协方差感知路由（Covariance-Aware Block Selection）

基于误差界，重要性分数同时考虑注意力幅度和近似误差：

$$\text{Score}_{t,j} = \text{Softmax}\left(\frac{q_t \bar{k}_j^\top}{\sqrt{d}} + \log(M_j + \epsilon)\right)$$

$M_j = \|H_j - \bar{H}\|_2$ 衡量该块的一阶矩阵偏离全局均值的程度，偏离大的块更需要精确计算。

### Kernel 实现（4 阶段流水线）

**准备阶段**：单次预扫描，计算 $\bar{Q}$、$\bar{K}$（块均值）、$\hat{V} = \sum_n v_{j,n}$（块值和）、$\bar{H}$（全局一阶统计）；用协方差感知路由选出每个 query block 的 top-K 关键块。

**Phase 1（精确注意力）**：对选中块 $j \in S_i$ 加载 $K_j, V_j$ 到 SRAM，做精确 softmax 注意力，累积 online softmax 的 rowsum 和 output。

**Phase 2（零阶近似）**：对未选中块按组扫描 $(\bar{K}, \hat{V})$，用列掩码排除已选块，高吞吐量完成尾部近似，额外累积 "tail rowsum" $\ell_i^{\text{tail}}$。

**Phase 3（全局一阶注入）**：$R_i = \text{diag}(\ell_i^{\text{tail}}) (Q_i \bar{H}) \cdot L^{-1}$，一次性注入，避免 HBM 流式读取。

关键设计：$\bar{H}$ 仅加载一次（不是逐块），消除 Phase 3 的内存瓶颈。

## 实验

### Kernel 效率

测试配置 B2-H16-D128，稀疏率 87.5%（密度 12.5%）：
- **所有序列长度（4K-32K）** 均优于 FA3 和 SpargeAttn
- SpargeAttn 在 4K 序列下退化为慢于 FA3；PISA 在此仍有优势
- 密度超过 70% 时仍全面超过 FA2

### 视频生成（带 warmup 策略）

| 模型 | 方法 | 稀疏率 | VBench | PSNR | 加速比 |
|------|------|--------|--------|------|--------|
| Wan2.1-14B | Dense | 0% | 95.98 | – | 1× |
| Wan2.1-14B | SpargeAttn | 87.5% | 95.69 | 21.47 | 1.85× |
| Wan2.1-14B | SVG2 | 80.6% | 95.39 | 22.92 | 1.77× |
| Wan2.1-14B | **PISA** | 87.5% | **95.80** | 22.69 | **1.91×** |
| Hunyuan-13B | Dense | 0% | 95.60 | – | 1× |
| Hunyuan-13B | **PISA** | 87.5% | 95.47 | 26.17 | **2.57×** |

不带 warmup 时质量优势更显著（SVG2、SpargeAttn 急剧下降，PISA 仍保持高质量）。

### 图像生成

FLUX.1-dev（85% 稀疏率）vs SpargeAttn（80% 稀疏率）：
- FID: 15.91 vs 19.20（PISA 更接近 dense 的 16.35）
- LPIPS: 0.241 vs 0.296
- 延迟: 6.87s vs 7.47s

### 消融

零阶 + 全局一阶混合 vs 仅零阶：PSNR 17.09 vs 16.10（+1dB），SSIM 0.682 vs 0.643，且 speedup 相近（1.22× vs 1.21×）。说明全局一阶修正几乎零开销但显著提升质量。

## 关键启示

- **"exact-or-approximate" 比 "keep-or-drop" 更优**：非关键块的分数具有统计规律（负值、窄分布），用 Taylor 展开近似比直接丢弃误差小得多，且理论误差界清晰。
- **全局统计量替代块级矩阵**：逐块一阶项实现困难（每块需 $d \times d$ 矩阵），用全局均值 $\bar{H}$ 替换将内存访问从 $O(N)$ 次流式读降为 1 次加载，是硬件友好性的关键。
- **协方差感知路由**：路由分数加入 $\log M_j$（一阶矩阵的异质性），将"容易被近似的块"主动分配给近似路径，减少精确计算浪费。
- **在短序列上仍有优势**：SpargeAttn 的块稀疏实现在短序列下退化，PISA 的近似扫描复用了 GEMM 高吞吐，在 4K token 下也保持速度领先。
