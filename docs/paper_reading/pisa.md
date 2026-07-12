---
tags:
  - Sparse Attention
  - Diffusion Model
  - Video Generation
---

# PISA: Piecewise Sparse Attention Is Wiser for Efficient Diffusion Transformers

[arXiv 2602.01077](https://arxiv.org/abs/2602.01077) | [代码解析](../code_analysis/pisa/00_overview.md) | HKUST (Guangzhou)

## 结论先行

PISA 是一个面向 DiT 图像/视频生成的 **training-free sparse attention** 方法。它的核心不是把非 Top-K KV block 丢掉，而是把注意力拆成两部分：

- 关键 KV block：保留 token 级精确 softmax attention；
- 非关键 KV block：用块级 Taylor 展开近似，仍然参与分母归一化和 value 聚合。

因此 PISA 的范式是 **exact-or-approximate**，不是传统块稀疏的 **keep-or-drop**。在 Wan2.1-14B 上报告 1.91x 加速，在 Hunyuan-Video-13B 上报告 2.57x 加速；在 FLUX.1-dev 图像生成中也比 SpargeAttn 更稳。

官方代码当前只开源了 Triton kernel 和 FLUX 示例，Wan/Hunyuan 视频 pipeline patch 未开源。本文的实现分析基于 `refs/codes/piecewise-sparse-attention` 提交 `655648df86cafd75042dc32d6b6be78c4ea0eca8`。

## 动机

视频 DiT 的 token 数随空间分辨率、帧数和 patch granularity 快速增长，full attention 的 `O(T^2D)` 成本成为推理瓶颈。已有 training-free sparse attention 通常先按块估计重要性，再只计算 Top-K block：

$$
O_i^{sparse} =
\operatorname{Softmax}(Q_i K_{S_i}^{\top}) V_{S_i}
$$

问题是非 Top-K block 并不等于无信息。视频生成中许多背景、纹理、跨帧一致性信息分散在长尾注意力上，直接丢弃会带来闪烁、细节漂移和质量下降。

PISA 的观察是：非关键块的 pre-softmax logits 往往集中在零或负值附近，分布窄、近似对称；这类块的指数函数可以用低阶 Taylor 展开稳定近似。于是 PISA 不丢弃尾部块，而是用块统计量计算它们的近似贡献。

## 数学建模

把 token 序列切成长度 `B` 的 block。对 query token `q_t`，KV block `j` 的 key 均值是：

$$
\bar{k}_j = \frac{1}{B}\sum_{n=1}^{B} k_{j,n}
$$

选中精确计算的 block 集合为 `S_t`，未选中近似计算的集合为 `U_t`。完整 attention 输出写成：

$$
o_t = \frac{N_t}{D_t}
$$

### 分母：一阶项自然抵消

对未选块做 `k_{j,n} = \bar{k}_j + \Delta k_{j,n}`，并对 `exp(q_t k_{j,n}^{\top})` 在 `q_t \bar{k}_j^{\top}` 附近展开：

$$
\exp(q_t k_{j,n}^{\top})
\approx
\exp(q_t \bar{k}_j^{\top})
\left(1 + q_t \Delta k_{j,n}^{\top}\right)
$$

因为块均值定义保证 `\sum_n \Delta k_{j,n}=0`，所以分母的一阶项抵消：

$$
D_t =
\sum_{j \in S_t}\sum_n \exp(q_t k_{j,n}^{\top})
+
\sum_{j \in U_t} B \exp(q_t \bar{k}_j^{\top})
$$

这点很关键：近似尾部块仍然进入 softmax denominator，避免传统 sparse attention 把分母系统性低估。

### 分子：零阶 + 一阶修正

分子精确项是：

$$
\sum_{j \in S_t}\sum_n \exp(q_t k_{j,n}^{\top})v_{j,n}
$$

未选块的零阶近似是：

$$
\sum_{j \in U_t}
\exp(q_t \bar{k}_j^{\top})
\sum_n v_{j,n}
$$

如果进一步保留一阶项，会出现每个 block 一个矩阵：

$$
H_j =
\sum_n
(k_{j,n}-\bar{k}_j)^{\top} v_{j,n}
\in \mathbb{R}^{D \times D}
$$

一阶分子项为：

$$
\sum_{j \in U_t}
\exp(q_t \bar{k}_j^{\top})
q_t H_j
$$

直接实现这个式子很不划算：每个 query block 都要读取大量 `D x D` 的 `H_j`，计算量不高但 HBM 流量大，容易 memory-bound。

## Hybrid Approximation

PISA 用全局一阶矩阵替代逐块一阶矩阵：

$$
\bar{H} = \frac{1}{N}\sum_{j=1}^{N} H_j
$$

于是尾部一阶项近似成：

$$
\left(\sum_{j \in U_t}\exp(q_t \bar{k}_j^{\top})\right) q_t \bar{H}
$$

这样每个 batch-head 只需要加载一次全局 `D x D` 矩阵 tile，而不是为每个未选 block 加载一个 `H_j`。这是 PISA 从“数学近似”变成“GPU 上能跑快”的关键。

论文给出误差界。若 `||q_t|| <= C_q`，`M=max_j ||H_j-\bar{H}||_2`，`\rho_t` 是未选块注意力质量占比，则：

$$
||\tilde{o}_t-o_t||_2
\le
C_q M \frac{\rho_t}{B}
$$

直观解释：

- `\rho_t` 小：被近似的尾部块总体权重越小，误差越小；
- `B` 大：块统计更稳定，误差被块大小摊薄；
- `M` 小：各 block 的一阶矩阵越接近全局均值，全局一阶替代越准确。

## Covariance-Aware 路由

仅用 `q_t \bar{k}_j^T` 做 Top-K 会偏向注意力均值大的块，但忽略“这个块是否容易被近似”。PISA 加入近似误差 proxy：

$$
\operatorname{Score}_{t,j}
=
\operatorname{Softmax}
\left(
\frac{q_t \bar{k}_j^\top}{\sqrt{D}}
+
\log(M_j+\epsilon)
\right)
$$

其中 `M_j = ||H_j-\bar{H}||_2`。偏离全局一阶统计更大的 block 更容易被分配到 exact path。

官方代码里的 `use_bias=True` 版本用 `hc` 范数作为 bias 加到块级 score 上，再做 top-k；FLUX processor 默认开启这个分支。0th 训练版本则用更便宜的 key 方差标量 `k_var` 作为 proxy。

## 代码实现要点

详细 kernel 拆解见 [PISA 代码实现](../code_analysis/pisa/00_overview.md)。这里按执行链路概括。

### 1. 预扫描生成块统计

`fused_chunk_reduce` 的 Triton grid 是：

```python
grid = (ceil(K / BK) * ceil(V / BV), N, B * H)
```

每个 program 负责一个 `(batch-head, token block, K tile, V tile)`，生成：

- `qc`: query block 均值；
- `kc`: key block 均值；
- `vc`: value block 求和；
- `hc`: block-wise 一阶矩阵 tile。

随后 Python 侧把 `hc.sum(dim=2)` 变成全局 `h`。这个预扫描是 PISA 的准备阶段，代价是 `O(TD^2/B)` 量级的小矩阵统计和连续内存写，换来后续 attention 主 kernel 不再读取每个 block 的 `H_j`。

### 2. Top-K block 路由

入口 `piecewise_sparse_attention` 用：

```python
score = torch.einsum("bhid,bhjd->bhij", qc, kc * scale)
indices = torch.topk(score, k=top_k, dim=-1).indices
```

得到每个 query block 的精确 KV block 集合。`top_k=max(1,int(density*NT))`，因此每个 query block 的 exact path 循环次数一致，有利于 CTA 负载均衡。

### 3. 三阶段 Triton 前向

主 kernel grid 是：

```python
grid = (ceil(V / BV), NT, B * H)
```

一个 Triton program 处理一个 `(batch-head, query block, value tile)`。它把 `Q_i` 加载一次，在寄存器中维护 online softmax 的 `m_i/l_i/acc`，然后连续执行三阶段：

1. **Exact**：遍历 `NS=top_k` 个选中 KV block，做标准 64x64 tile attention；
2. **Zeroth-order**：按 `GROUP_SIZE in {32,64,128}` 扫描未选 block 的 `kc/vc`，用 `Q_i \bar{K}^T` 和 `sum(V)` 近似尾部贡献；
3. **Global first-order**：加载全局 `h` 的当前 value tile，计算 `Q_i h`，再乘 Phase 2 累计的 tail probability mass 注入 `acc`。

PISA 的 GPU 友好性来自这几个设计：

- exact path 和 approximate path 都是规则 tile dot，避免逐 token 分支；
- 每个 query block 的 `top_k` 固定，避免 sparse list 长度高度不均；
- approximate path 扫描的是连续的 `kc/vc` block statistic，访存规整；
- 一阶修正只读全局 `h`，避免逐 block `D x D` 矩阵造成 memory-bound；
- Hopper 上用 Triton TensorDescriptor/TMA allocator 优化块加载。

### 4. FLUX 接入

`FluxAttnProcessor` 在 attention processor 内替换 backend。它先做 QKV 投影、QK norm、RoPE，然后对 4096 图像 token 做空间重排：

```python
(h p1 w p2) -> (h w p1 p2)
```

这样每个 64-token block 更接近二维局部 patch，而不是原始 raster order 下横跨不自然的空间区域。`start_layer_idx` 控制从第几层开始启用 PISA，前面层仍使用 PyTorch SDPA，以降低早期层近似带来的质量风险。

## 实验结果

### 视频生成

论文在 Wan2.1-1.3B/14B 和 Hunyuan-Video-13B 上评估，指标包括 VBench、PSNR、SSIM、LPIPS 和端到端 latency。

| 模型 | 方法 | 稀疏率 | VBench | PSNR | 加速 |
|---|---:|---:|---:|---:|---:|
| Wan2.1-14B | Dense | 0% | 95.98 | - | 1.00x |
| Wan2.1-14B | SpargeAttn | 87.5% | 95.69 | 21.47 | 1.85x |
| Wan2.1-14B | SVG2 | 80.6% | 95.39 | 22.92 | 1.77x |
| Wan2.1-14B | PISA | 87.5% | 95.80 | 22.69 | 1.91x |
| Hunyuan-13B | Dense | 0% | 95.60 | - | 1.00x |
| Hunyuan-13B | PISA | 87.5% | 95.47 | 26.17 | 2.57x |

带 warmup 时 PISA 在质量和速度上都比直接 keep-or-drop 的 sparse baseline 更稳。不带 warmup 时差异更明显：SpargeAttn/SVG2 更容易质量下降，PISA 因为尾部块仍被近似计算，退化较慢。

### 图像生成

在 FLUX.1-dev 上，PISA 85% 稀疏率相对 SpargeAttn 80% 稀疏率：

- FID: 15.91 vs 19.20；
- LPIPS: 0.241 vs 0.296；
- latency: 6.87s vs 7.47s。

这说明 PISA 不是单纯牺牲质量换速度；在相近甚至更高稀疏率下，Taylor tail approximation 能减少视觉误差。

### Kernel 效率

论文在 H800 上报告，`B2-H16-D128`、密度 12.5% 时，PISA 在 4K-32K 序列长度均优于 FlashAttention-3 和 SpargeAttn。短序列 4K 下仍能超过 FA3，是因为 PISA 的 approximate path 以规则 centroid/sum tile 计算为主，而不是大量碎片化 sparse gather。

密度扫描中，PISA 在 density 超过 70% 时仍可超过 FA2；在更长序列上，density 低于 50% 时超过 FA3 更明显。

### 消融

| 模型 | 方法 | SSIM | PSNR | LPIPS | 加速 |
|---|---:|---:|---:|---:|---:|
| FLUX.1-dev | PISA-0th | 0.643 | 16.10 | 0.274 | 1.21x |
| FLUX.1-dev | PISA-1st | 0.679 | 17.04 | 0.246 | 0.96x |
| FLUX.1-dev | PISA-hyd | 0.677 | 17.01 | 0.248 | 1.24x |
| FLUX.1-dev | PISA-hyd + covariance | 0.682 | 17.09 | 0.241 | 1.22x |
| Wan2.1-14B | PISA-0th | 0.772 | 21.68 | 0.136 | 1.93x |
| Wan2.1-14B | PISA-hyd | 0.787 | 22.69 | 0.124 | 1.91x |

块级一阶 `PISA-1st` 质量接近 hybrid，但速度低于 dense baseline，说明逐块 `H_j` 的 HBM 访问确实不划算。Hybrid 几乎保留一阶质量收益，同时恢复加速。

## 和其他稀疏视频方法的区别

- 相比 SpargeAttn：PISA 不把非选中块直接置零，而是用 Taylor approximation 保留尾部质量。
- 相比 SVG2：SVG2 更强调视频 token 的空间/时间结构化选择；PISA 更强调非关键块的解析近似，两者理论上可以组合。
- 相比 PASA：PASA 可以看作 PISA 的后续扩展，进一步做动态预算、分组一阶近似和随机路由，目标是缓解视频 temporal flicker。

## 局限

- 当前开源代码没有视频模型完整接入脚本，复现实验需要自行 patch Wan/Hunyuan attention。
- Hybrid forward 没有 backward；训练或微调只能参考 0th 版本。
- 预扫描会额外产生 `hc` 临时张量，显存峰值需要关注。
- covariance-aware routing 在公开代码里是基于 `hc` norm 的简化实现，和论文完整调度策略并非完全一一对应。
- 最佳性能依赖 Hopper 和较新的 Triton；非 Hopper GPU 上 TensorDescriptor/TMA 优势会变弱。
