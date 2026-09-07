---
tags:
  - Diffusion Model
  - Video Generation
---
# DiT 推理优化：Progressive Resolution

返回：[专题总览](overview.md)

## 抓住重点

设全分辨率 latent 为 \(z\in\mathbb R^{C\times H\times W}\)，第 \(t\) 步采用分辨率 \((H_t,W_t)\)。Progressive 在早期噪声较大时取 \(H_tW_t<HW\)，再通过频域插值恢复到全分辨率；它减少 token 数，属于质量换速度。

## 1. 机制

对 latent 做二维 DCT：\(\hat z=\mathcal D z\)。粗阶段保留低频子带 \(P_t\hat z\)，得到 \(z_t^{\rm low}=\mathcal D^{-1}(P_t\hat z)\)，在切换点用频域上采样算子 \(U\) 恢复 \(z_t^{\rm full}=\mathcal D^{-1}(UP_t\hat z)\)。若 token 数与面积近似成正比，Attention 二次项从 \(O((HW)^2)\) 降为 \(O((H_tW_t)^2)\)。`dct_rewind` 同时把 scheduler 时间状态回拨到新阶段一致；`dct` 只做频域恢复。

## 2. 阶段选择与误差

令噪声水平为 \(\sigma_t\)，阈值为 \(\delta\)。阶段策略选择最小分辨率，使被丢弃高频能量满足
\[
\frac{\lVert(I-P_t)\hat z_t\rVert_2}{\lVert\hat z_t\rVert_2+\epsilon}\le\delta.
\]
增大 \(\delta\) 会延长粗阶段并降低计算量，但扩大频率截断误差。官方示例报告约 1.5--2.8 倍 denoise 加速；该数字依赖模型、分辨率、步数和硬件，不能外推为 e2e 保证。

## 3. 组合约束

分辨率切换改变 token shape 与 RoPE/cache 状态，因此当前实现与 Ulysses/Ring SP、`torch.compile` 互斥；与 Cache-DiT 仅标记 experimental，切换时必须刷新 cache context。整模 DiT CPU offload 会为每步引入近似固定传输代价，可能吞掉 token 缩减收益，官方建议关闭。

## 附录：实现证据

依据 SGLang `progressive_resolution.mdx` 与 `runtime/pipelines_core/stages/progressive_resolution/`；Z-Image 额外处理五维 latent 的 squeeze/unsqueeze，并在切换点重算 caption/image RoPE。相关约束汇总见 [Correctness](correctness.md#2-已核验的互斥--自动降级)。

## 相关阅读

- [专题总览](overview.md) · [Memory Offload](memory_offload.md) · [Feature Cache](feature_cache.md) · [Parallelism](parallelism.md)
