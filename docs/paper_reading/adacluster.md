---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
---

# AdaCluster: Adaptive Query-Key Clustering for Sparse Attention in Video Generation

[arXiv 2604.18348](https://arxiv.org/abs/2604.18348) | [代码](https://github.com/USTC-MLSys-Team/Adacluster) | 中科大、合肥综合国家科学中心、澳门大学、港中文

## 概述

视频 DiT 中稀疏注意力的现有聚类方法（如 SVG2）对 query 和 key 使用相同的 Euclidean 距离聚类策略，忽略了两者在注意力机制中的不同角色。AdaCluster 观察到 query/key 分布存在显著差异：
- Query 向量归一化后分布紧凑，可用角度相似性聚类实现高压缩比
- Key 向量分布在不同层间差异巨大，需逐层自适应聚类

提出 training-free 角色感知聚类框架，包含 Query 聚类（归一化 + 角度聚类）、Key 聚类（自适应多阶段 K-means）、TensorQuest 关键簇选择。在 CogVideoX-2B、HunyuanVideo、Wan-2.1 上实测 1.67×–4.31× 加速，质量损失极小。

## 背景与动机

**现有聚类方法（如 SVG2）的问题**：
- 固定 query 聚类数 100、key 聚类数 500，对所有模型和层统一
- 使用相同 Euclidean 聚类处理 query 和 key
- 聚类后基于质心注意力分数选择关键簇，但簇中心不能完全代表边界附近的关键 token

**两个核心观察**：
1. Query 向量长度不影响 query-key 分数的相对排序（^\top k_a > q^\top k_b \iff \hat{q}^\top k_a > \hat{q}^\top k_b$，其中 $\hat{q} = q / ||q||$），归一化后聚类压缩比可提升约 3.6×
2. Key 向量在不同层间分布差异显著：Wan-2.1 和 Hunyuan 模型的层间紧凑性得分（compactness score，1/MSE）变化剧烈，部分层高度集中可大幅压缩，部分层极度分散甚至不适合聚类

## Query 聚类：归一化 + 角度相似性

**理论依据**：^\top k_a > q^\top k_b \iff \hat{q}^\top k_a > \hat{q}^\top k_b$，归一化不改变 Top-K 排序。归一化后的 query 分布紧凑得多，DB 指数更低，簇内距离更小。

**实证效果**：归一化聚类只需 65 个簇即可达到未归一化 235+ 个簇的簇内紧凑度，有效压缩比 3.6×。P HunyuanVideo 上，加入 query 归一化将 PSNR 从 29.56 提升至 30.58，SSIM 从 0.763 提升至 0.835。

## Key 聚类：多阶段自适应 K-means

**紧凑性评分**：对每层 head 计算 MSE 重建误差 $\mathrm{MSE}_l^i = \frac{1}{N} \sum_{i=1}^N ||k_l^i - c(k_l^i)||_2^2$，定义紧凑性 $\mathrm{Comp}_l = 1/\mathrm{MSE}_l$。观察：不同 prompt 在同一层的紧凑性趋势相似，因此只需逐层调整，不需逐 prompt 调整。

**多阶段聚类算法**（Algorithm 1）：
- 初始阶段用适中簇数聚类
- 选出距簇中心超过阈值 $\tau$ 的 outlier token，重新聚类
- 重复至所有 token 分配到紧凑簇，或簇数超上限 {\mathrm{max}}$（此时标记为 hard-to-compress 层，退化为全注意力）
- 仅在第一个去噪步执行完整多阶段聚类，后续步复用簇中心作为初始化

**跨步复用**：相邻去噪步的 token 分布高度相似（PCA 可视化验证），因此用前一步的簇中心初始化当前步的 K-means，加速聚类。

## TensorQuest：Tensor Core 加速的关键簇选择

**动机**：Quest 方法通过 $\sum_{d=1}^D \max(q_d \cdot \max(K_d), q_d \cdot \min(K_d))$ 估计簇的注意力权重上界，但原始实现在 CUDA Core 上执行效率低。

**TensorQuest 重构**：将 Quest 分数等效变换为矩阵乘法形式：
- 提取 query 和 key 的正/负部分：^+=\max(q,0), q^-=\min(q,0), k^+=\max(k,0), k^-=\min(k,0)$
- $\mathrm{Quest}(q, K) = \mathrm{matmul}(q^+, k^+) + \mathrm{matmul}(q^-, k^-)$
- 主体计算在 Tensor Core 上执行，加速最高 5×（176.4K token 场景）

## 实验

**配置**：单卡 A40 48GB，Triton + FlashInfer 定制算子。top 15% 层使用全注意力（FlashAttention），其余层应用 AdaCluster 聚类。

**主要结果**：

| 模型 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | Speedup↑ |
|------|------|-------|-------|--------|----------|
| CogVideoX-2B 480p | SpargeAttn | 28.19 | 0.517 | 0.618 | 1.23× |
| CogVideoX-2B 480p | AdaCluster | **30.99** | **0.767** | **0.231** | **1.67×** |
| Wan 1.3B 480p | SpargeAttn | 28.29 | 0.437 | 0.599 | 1.81× |
| Wan 1.3B 480p | SVG2 | 28.23 | 0.358 | 0.679 | 1.61× |
| Wan 1.3B 480p | AdaCluster | **29.08** | **0.571** | **0.393** | **1.85×** |
| Hunyuan 720p | SpargeAttn | 28.16 | 0.490 | 0.596 | 1.33× |
| Hunyuan 720p | SVG2 | 29.32 | 0.794 | 0.308 | 1.57× |
| Hunyuan 720p | AdaCluster | **30.58** | **0.835** | **0.203** | **1.68×** |

**不同序列长度下的加速**：HunyuanVideo 在 28.3K→176.4K token 范围内加速从 1.53× 增长到 4.31×，长序列稀疏度更大，加速比更高。SVG2 在超过 101.1K token 后因内存开销太大无法运行。

**H100 验证**：Wan 14B 上 AdaCluster 1.81× vs SVG2 1.61×；HunyuanVideo 上 AdaCluster 1.67× vs SVG2 1.58×。

**消融关键发现**：
- 自适应簇数分配（AdaClus vs AvgClus）：PSNR 30.58 vs 29.01，自适应策略显著优于均匀簇数
- TensorQuest vs 均值选择（w/o Quest）：PSNR 30.58 vs 28.94，TensorQuest 对关键 token 识别至关重要
- 三组件逐步递增的完整消融显示 query 归一化贡献最大（PSNR +1.02）

## 关键启示

- **角色感知设计比统一处理更有效**：Query 和 key 在注意力中的数学作用不同（query 只需角度相似，key 需 Euclidean 相似），针对性设计带来显著压缩效率提升
- **逐层自适应比全局固定好**：不同层的 token 分布差异巨大，固定聚类参数要么压缩不足要么精度受损。用 MSE 紧凑性指导自适应分配是低开销有效方案
- **算法等效变换 + Tensor Core 是实用加速技巧**：Quest 的 CUDA Core 计算等效变换为矩阵乘法可大幅利用 Tensor Core 吞吐，这一思路在多个稀疏注意力场景中可复用
