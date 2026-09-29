---
tags:
  - Video Generation
  - KV Cache
  - Distributed Inference
  - Diffusion Model
---

# In-Flight KV Cache with Clean Anchors for Faster Autoregressive Video Diffusion

- 论文：[In-Flight KV Cache with Clean Anchors for Faster Autoregressive Video Diffusion](https://arxiv.org/abs/2609.32540)
- 代码：论文页面未给出官方代码链接
- 团队：Meta 等
- 提交：2026-09-27，arXiv:2609.32540v1

## 概述

FlashForward 研究少步自回归视频扩散的跨 chunk 状态契约。现有方法通常在一个 chunk 完成后再额外 forward，提取 clean KV cache；这些 cache-update-only forward 不推进视频生成，却成为少步模型的主要开销。FlashForward 直接复用当前 denoising forward 已经产生的 stage-matched KV，让它同时推进当前 chunk 并为后续 chunk 发布状态。

stage-matched history 来得早、保留近期细节，但带有噪声；论文因此增加稀疏 clean anchor。planner 预先生成 anchor latent 和 clean anchor KV，renderer 使用 clean anchor 提供长程结构，使用 stage-matched history 提供近期运动和外观。两种状态对应不同时间尺度，合起来形成可流水执行的 wavefront。

## 1. 跨 chunk 状态契约

自回归视频由多个 temporal chunk 顺序生成，每个 chunk 经过多个 denoising stage。状态契约决定一个 chunk 在何时、以什么噪声水平向后续 chunk 发布 KV，也决定哪些 forward 可以并行。

Self-Forcing 在 chunk 完成后额外构造 clean KV；HiAR 允许较早执行，但每个 denoising stage 都重新编码前序上下文。FlashForward 让普通 renderer forward 在推进输出的同时发布当前 stage 的 KV，后续 chunk 在同一 stage 读取它。因而相邻 chunk 和不同 denoising stage 形成反对角线 wavefront，避免 renderer cache-update-only forward。

## 2. Planner–Renderer 与双尺度记忆

同一个生成骨干承担 planner 和 renderer 两个角色，使用 role embedding 与 role-specific LoRA adapter 区分时间依赖。planner 以较大步长生成稀疏 clean anchor；renderer 生成所有最终输出位置，不把 planner anchor 直接拼进最终视频。

以 81 个 latent 位置为例，planner 在位置 `{0, 10, ..., 80}` 生成 9 个 anchor，分成每块 3 个的 3 个 planner block；renderer 把 81 个位置拆成 27 个、每块 3 个 latent 的 chunk。每个 renderer chunk 读取附近的三点 clean anchor，以及当前 denoising stage 最近最多 5 个 renderer chunk 的 KV。anchor 提供较慢变化的结构约束，stage-matched history 保留快速变化的局部运动。

## 3. 训练

训练分两阶段。第一阶段是 packed SFT：从真实视频直接监督 planner 和 renderer，使 planner 学会跨 anchor block 的因果依赖。论文使用 20 秒训练 clip，使一个样本包含 9 个 anchor 和 3 个 planner block。第二阶段是 self-rollout distillation：planner 先自回归生成 anchor，renderer 再在生成上下文中完成视频，并用少步 guidance-free 轨迹做分布匹配蒸馏，以减小 teacher-forcing 与部署状态的差异。

## 4. 流水与硬件

四个 denoising stage 可以分别放在四个 GPU。renderer 节点 `(i,k)` 依赖当前 chunk 的下一噪声 stage 和前一 chunk 的同 stage，因此 warm-up 后每个 pipeline round 完成一个 renderer chunk。planner 可以先在一个 GPU 上完成，也可以用 context parallelism；部署时还可以专门留一张 GPU 并行生成 planner anchor，使渲染在第一组 anchor 出现后立即开始。

论文没有提出独立 CUDA、Triton、CuTe 或 ThunderKittens attention kernel。加速来自 forward 数量减少、KV 状态复用、stage wavefront 调度和 planner/renderer 设备分工。实现中将 LoRA adapter 合并进两个完整权重模型，保证每次 forward 与基线 backbone forward 的成本可比；planner 与 renderer 权重在设备间切换时使用 pinned host memory，传输时间计入报告延迟。

## 5. 实验结果

论文在 1.3B 和 14B backbone、480p 和 720p、20/35/65 秒视频上比较 FlashForward、HiAR 和 Self-Forcing。四 GPU 设置下，相对 HiAR 加速 1.16–1.68×，相对 Self-Forcing 加速 1.42–2.92×；1.3B/480p 的 VBench Total 为 0.838，高于 Self-Forcing 的 0.805 和 HiAR 的 0.821。65 秒时质量保持稳定。

81 latent 的典型配置需要 123 次逻辑 forward，HiAR 为 134 次，Self-Forcing 为 212 次；实测加速低于简单 forward 数量比，因为 context parallel 通信、wavefront 填充和同步都有开销。14B/720p 的实测结果更接近理论上限，说明长视频、大模型和高分辨率更容易摊薄固定流水成本。

## 6. 限制

方法依赖共享 planner–renderer 模型、角色 LoRA 和少步自回归视频扩散设置。planner/renderer 权重切换、跨 GPU 通信和 wavefront 调度会影响实际收益；单 GPU 无法发挥 stage pipeline 的全部收益。论文报告的是系统级实测，不等于存在一个可移植的 fused kernel。

## 方法启示

1. 推理状态可以按时间尺度拆成“快速但嘈杂的近期状态”和“稀疏但稳定的长期状态”。
2. 让一次输出推进 forward 同时发布可复用状态，可以减少只为更新 cache 而做的额外计算。
3. 评价流水方案应同时报告逻辑 forward 数、通信/同步开销和真实 wall-clock latency。

## 来源

Meta et al., “In-Flight KV Cache with Clean Anchors for Faster Autoregressive Video Diffusion,” arXiv:2609.32540v1, 2026. [论文](https://arxiv.org/abs/2609.32540)
