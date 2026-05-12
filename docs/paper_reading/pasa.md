---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
---

# Ride the Wave: Precision-Allocated Sparse Attention for Smooth Video Generation

[arXiv 2604.12219](https://arxiv.org/abs/2604.12219) | 代码未公开 | 北京邮电大学、南洋理工大学

## 概述

视频 Diffusion Transformer（DiT）中自注意力占端到端延迟 70% 以上，稀疏注意力是主流加速方案。但现有方法存在三个局限：
1. 静态均匀稀疏预算分配，忽略去噪轨迹不同阶段对计算的差异需求
2. 确定性 block 路由导致相邻帧的"选择震荡"，产生时序闪烁
3. 补偿类方法（如 PISA）用全局统计量近似非关键块，过度平滑局部纹理

PASA 是 PISA 的 training-free 扩展，提出三个改进：
- **曲率感知动态预算**：利用 flow matching 相邻步速度场的 L1 距离刻画轨迹曲率，将精确计算预算从线性阶段转移到关键语义转换阶段
- **分组一阶近似**：在 memory-coalesced group 内共享 Taylor 补偿统计量，避免全局同质化
- **随机选择偏置**：向 block 评分注入可控噪声，软化确定性路由边界，消除选择震荡

主要结果：Wan 2.1-1.3B 和 HunyuanVideo-13B 在 85% 稀疏率下达到 1.70×–2.36× 加速，Temporal Flickering 和 Motion Smoothness 优于 PISA/SVG2，甚至在部分模型上超过全注意力。

## 背景：PISA 分段稀疏注意力

PISA 将 KV 块划分为选中集 S（精确计算）和未选中集 U（Taylor 近似）。注意力输出：

4081016o_t = \sum_{j \in S} \sum_n \exp(q_t k_{j,n}^\top) v_{j,n} + \sum_{j \in U} \Psi_{t,j} \sum_n v_{j,n} + q_t \bar{G} \sum_{j \in U} \Psi_{t,j}4081016

其中 $\Psi_{t,j} = \exp(q_t \bar{k}_j^\top)$ 是块质心注意力权重，$\bar{G} = \frac{1}{N_B} \sum_{j=1}^{N_B} G_j$ 是全局一阶统计量， = \sum_n (k_{j,n} - \bar{k}_j)^\top v_{j,n}$。

块路由评分公式整合质心距离与近似误差先验：$\mathrm{Score}_{t,j} = \mathrm{softmax}(q_t \bar{k}_j^\top / \sqrt{d} + \log(||G_j - \bar{G}||_2 + \varepsilon))$

## 方法一：曲率感知动态预算

**核心洞察**：Flow matching 去噪轨迹速度场 L1 距离可视化显示，生成过程分三个不同阶段：
- 初始约 10 步：速度场变化剧烈，方差大 → 快速宏观语义构建 → 需要密集注意力
- 中间阶段：变化幅度显著降低 → 概率流接近线性 → 高计算预算冗余
- 最后约 5 步：曲率回升 → 各 prompt 独立纹理/空间细化方向

**具体做法**：
- 前 20% timestep 保持全注意力（遵循先例）
- 剩余 80% 稀疏 timestep {\mathrm{sparse}}$：使用 10 个校准 prompt 录制速度场 L1 距离 $\ell_t$，求均值曲线
- 归一化缩放因子 $\gamma_t = \ell_t / \bar{\ell}$（均值为 1）
- 有效密度 $\rho_t = \rho \cdot \gamma_t$，总精确计算量不变，仅在步间重新分配

## 方法二：随机路由消除时序闪烁

**核心洞察**：确定性评分机制下，高注意力块（主体/前景）总被选中，中低注意力块（背景/边界）总被近似计算 → 计算资源极度偏斜是时序闪烁的根本原因。随机偏置本质上起"时空多路复用"作用，让背景区域定期获得精确细粒度细化。

**具体做法**：在 block 评分注入独立采样随机偏置，软化确定性的 Top-K 选择边界。偏置在不同层和连续步间独立采样，特定中注意力块升级为精确计算的子集不断变化。

## 方法三：分组一阶近似

**核心洞察**：PISA 全局 $\bar{G}$ 虽然硬件友好（约 32KB 适合 H100 SRAM），但过度同质化非关键区域。逐块精确统计量保持局部性但触发碎片化 DRAM 流量。Triton profiling 显示 32 块/组在局部性和吞吐量间最佳。

**具体做法**：
- 将 KV 划分为 memory-coalesced group（32 块/组）
- 组内计算共享 $\bar{G}^{(g)}_{\mathrm{group}}$
- 利用 warp specialization：compute warp 执行注意力运算，load warp 预取分组统计量，overlap 访存延迟与计算
- 附录 B 给出分组残差的 Frobenius 范数上界和组内方差最优性证明

## 实验

**配置**：8× NVIDIA H800 GPU，CUDA 12.8，Triton 3.4.0。模型：Wan 2.1-T2V-1.3B/14B，HunyuanVideo-T2V-13B。

**主要结果**（85% 稀疏率）：

| 模型 | 方法 | VBench T.F.↑ | M.S.↑ | SSIM↑ | PSNR↑ | Speedup↑ |
|------|------|------------|-------|-------|-------|----------|
| Wan 1.3B | PISA | 97.00 | 98.08 | 0.729 | 19.66 | 1.70× |
| Wan 1.3B | PASA | **97.06** | **98.12** | **0.751** | **20.51** | 1.70× |
| Wan 14B | PISA | 98.45 | 98.93 | 0.779 | 21.20 | 2.23× |
| Wan 14B | PASA | **98.48** | **98.93** | **0.800** | **22.07** | 2.19× |
| Hunyuan 13B | Dense | 98.99 | 99.25 | – | – | 1.00× |
| Hunyuan 13B | PASA | **99.32** | **99.42** | 0.801 | 25.24 | 2.36× |

SVG2 在 Wan 14B 上被迫降至 79.5% 稀疏率才获得可比的 SSIM，而 PASA 严格保持 85%。

**消融关键发现**（Wan 1.3B 雷达图）：
- 分组近似单独加入 PISA 将 Background Consistency 提升至归一化最高分（1.0），移除则跌至 0
- 随机偏置单独加入大幅提升 Motion Smoothness，移除则 Motion Smoothness 崩溃 + Temporal Flickering 剧烈恶化
- 动态预算单独加入改善 Temporal Flickering 和 Aesthetic Quality，移除则 Subject Consistency 和 Temporal Flickering 受损
- 三个组件完整集成在所有维度均匀主导外围，证明组件间高度互补

## 关键启示

- **轨迹感知分配比均匀稀疏更高效**：Flow matching 速度场曲率是免费信号，不需要额外训练即可确定计算关键期。这一思路可推广到其他 diffusion-based 生成任务
- **随机化是消除确定性闪烁的有效手段**：确定性 Top-K 的"选择震荡"是视频稀疏注意力的根本质量瓶颈，注入可控噪声比增加预算更高效
- **分组统计在硬件-精度权衡中找 sweet spot**：不做全局（过粗）也不做逐块（过细），利用 GPU coalesced memory group 实现 zero-overhead 局部性保持
