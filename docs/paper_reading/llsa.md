---
tags:
  - Sparse Attention
  - Diffusion Model
---

# Trainable Log-linear Sparse Attention for Efficient Diffusion Transformers

[arXiv 2512.16615](https://arxiv.org/abs/2512.16615) | 南洋理工大学 S-Lab / 北京大学

## 概述

现有 Top-K 块稀疏注意力的瓶颈在于：**选择阶段仍是 $O(N^2)$**（对压缩 token 做全对注意力分数），稀疏注意力计算阶段是 $O(NKB)$，随着序列增长，选择阶段反而主导总开销。此外，单层粗粒度代表 token 无法充分保留全局上下文，迫使 K 随序列增长而增大。

LLSA 提出分层（$O(\log N)$ 层）Top-K 选择，将选择阶段从 $O(N^2)$ 降至 $O(NK) = O(N)$，同时引入**分层 KV 增强**将各粒度层的粗粒度 token 纳入注意力，在小 K 下保全全局上下文，整体复杂度降为 $O(N \log N)$。

实验在 pixel-space DiT 上评测（无 patchification，无 VAE，直接处理像素 token）：
- FFHQ 256×256：注意力推理 **28.27×** 加速，DiT 训练 **6.09×** 加速
- ImageNet-256 PixelFlow：FID 20.41，优于 VSA（23.59）和 SLA（22.58）
- LLSA K=8 优于 baseline K=32（同时更快）

## 方法

### 背景：Top-K 块稀疏注意力的复杂度瓶颈

标准 Top-K 稀疏注意力三阶段：
1. 压缩：均值池化得压缩 token $Q', K' \in \mathbb{R}^{T \times d}$，$T = N/B$
2. 选择：计算 $S = Q'K'^\top \in \mathbb{R}^{T \times T}$，对每个 query 取 Top-K key 块 → $O(T^2 d) = O(N^2/B^2)$（主导项）
3. 稀疏注意力：仅计算选中的 K 个块对 → $O(NKBd)$

问题：步骤 2 的 $O(N^2)$ 随序列增长无法消除。

### 分层压缩

对 $Q, K, V$ 通过均值池化递归构建 $L = \lfloor \log_B N - 1 \rfloor$ 层分层表示：

$$Q^{(l)}, K^{(l)}, V^{(l)} \in \mathbb{R}^{N/B^l \times d}, \quad l = 0, 1, \ldots, L$$

$l$ 层的一个 token 是 $l-1$ 层 $B$ 个 token 的均值摘要。

### 分层 Top-K 选择

从最粗层 $L$ 开始，逐层细化：

- **最粗层**：对 $N/B^L$ 个 token 做全量相似度 $S^{(L)} = Q^{(L)} K^{(L)\top}$，取 Top-K 索引 $I^{(L)}$（此时 $N/B^L$ 极小，$O((N/B^L)^2)$ 可忽略）
- **细化层**：每层只对上一层选出的 $KB$ 个候选 key 做相似度，取 Top-K 索引 $I^{(l-1)}$

选择阶段总复杂度（等比级数收敛）：

$$\sum_{l=0}^{L-1} O\!\left(\frac{N}{B^{l+1}} \cdot KB\right) = O\!\left(NK \sum_{l=0}^{L-1} B^{-l}\right) = O(NK)$$

由于 K 为常数，选择阶段复杂度从 $O(N^2)$ 降至 $O(N)$。

### 分层 KV 增强（Hierarchical KV Enrichment）

注意力计算时，除精细层 Top-K 块外，还拼入各粗粒度层选中的 token $K^{(l)}_i, V^{(l)}_i$：

- 增强后 key/value 集大小：$K$（精细）$+ O(K \log N)$（各粗粒度层）
- 全局上下文通过各粒度层 token 保留，不需要增大 K

超参 $L_e$ 控制增强层数（默认 $L_e = L$），实验表明更多层 = 更好质量，但略微降低吞吐量。

**KV Reweighting**：粗粒度 token 覆盖范围更广，重要性应更高。将 $l$ 层 token 乘以权重 $W^{(l)} = B^l$，反映其对应的精细 token 数量：

$$K_c^{(l)} \leftarrow K^{(l)}_i \cdot B^l, \quad V_c^{(l)} \leftarrow V^{(l)}_i \cdot B^l$$

不增加训练开销，但显著提升质量（FFHQ-128 FID：有 Reweighting 24.37 vs 无 25.31）。

### 整体复杂度

$$O(NK) + O(NK \log N) = O(NK \log N) = O(N \log N)$$

低于全量注意力 $O(N^2)$ 和现有 Top-K 方法的 $O(N^2)$（后者受选择阶段主导）。

### Kernel 实现：稀疏反向传播

标准做法：将 Top-K 稀疏索引转为密集二值掩码 $T \times T$，key/value 反向时扫掩码 → $O(N^2)$ 存储和计算。

LLSA 的稀疏索引转置（算法 2，类 CSR→CSC）：
1. 统计每个 key 被多少个 query 选中，得累积偏移 $C \in \mathbb{R}^{T+1}$
2. 遍历 Top-K 索引，写入平坦查询索引数组 $I_q \in \mathbb{R}^{T \cdot K}$

用 $(I_q, C)$ 的 key-major 结构做 SpMM 反向传播，全程无密集掩码，反向复杂度为 $O(N)$。实验验证：LLSA 反向吞吐量在所有序列长度下近似恒定；baseline（密集掩码）随序列增长单调下降。

### 2D 索引重排

像素空间中用光栅顺序展平破坏局部连续性，分层池化效果差。LLSA 将尺寸为 $2^i$ 的 patch 内像素分组为连续 token，确保 2D 空间邻近像素在 1D 序列中相邻（FID：有重排 29.46 vs 无 31.19）。

## 实验

### 消融（FFHQ-128，DiT-S）

| 配置 | FID | 吞吐量 |
|------|-----|------|
| 全量注意力 | 24.91 | 188.88 |
| Top-K (L=1) | 28.21 | 483.91 |
| + KV 增强 (Le=1) | 26.09 | 302.92 |
| + KV 加权 | 24.18 | 302.92 |
| Top-K (L=2) + KV 增强 + 加权 | 24.37 | 436.40 |

$L=2$ 版本（LLSA）：FID 优于全量注意力，吞吐量 2.3×。

### K=8 vs baseline K=32

| 方法 | K | FID | 吞吐量 |
|------|---|-----|------|
| LLSA | 8 | 24.37 | 436.40 |
| Baseline | 8 | 28.21 | 483.91 |
| Baseline | 16 | 27.23 | 436.40 |
| Baseline | 32 | 25.88 | 357.95 |

LLSA K=8 优于 baseline K=32，同时更快。KV 增强通过保全全局上下文，实现"小 K 高质量"。

### FFHQ 256×256 与 ImageNet-256

| 方法 | FFHQ-256 FID | FFHQ-256 吞吐量 | ImageNet-256 FID | 吞吐量 |
|------|-------------|----------------|-----------------|------|
| 全量注意力 | 38.77 | 61.64 | – | – |
| VSA | 40.69 | 341.94 | 23.59 | 32.30 |
| SLA | 39.98 | 304.85 | 22.58 | 29.81 |
| LLSA | **39.29** | **375.34** | **20.41** | **34.16** |

LLSA 在质量和效率上双优，训练整体 6.09× 加速（256×256）。

## 关键启示

- **选择阶段 $O(N^2)$ 才是真瓶颈**：注意力计算已是线性 $O(NKB)$，但选择阶段的全对相似度 $O(N^2/B^2)$ 随序列增长主导总开销——这是之前方法被忽视的问题
- **分层 Top-K 将选择降至 $O(N)$**：等比级数求和使各层开销总和收敛，K 固定不随 N 增长，是实现 log-linear 复杂度的关键
- **KV 增强用粗粒度 token 保全全局上下文**：不依靠增大 K，而是纳入各粒度层 token，允许 K 保持小值。KV 加权（$W^{(l)} = B^l$）修正了粗粒度 token 的重要性估计
- **无密集掩码的反向传播**：CSR→CSC 风格的稀疏索引转置算法，将反向复杂度从 $O(N^2)$ 降至 $O(N)$，是训练加速的关键工程贡献
- **pixel-space DiT 是压力测试场景**：无 VAE 压缩，65K token 的序列能测出各方法的真实扩展性
