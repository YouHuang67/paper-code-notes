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

论文的状态表把四种契约分开：Self-Forcing 在 chunk 完成后发布 clean history，付一次额外编码且 chunk 串行；N-C-Causal-rCM 复用最后一个去噪阶段的噪声 KV，省去额外编码但仍串行；HiAR 每阶段重编码较干净历史，形成流水但计算重复；FlashForward 在每个普通去噪 forward 后立即发布同阶段的 KV，兼得流水和零 renderer 专用 cache 更新。状态可用的时点决定后续 chunk 什么时候可以开始，不能只比较 cache 是否“干净”。

## 2. Planner–Renderer 与双尺度记忆

同一个生成骨干承担 planner 和 renderer 两个角色，使用 role embedding 与 role-specific LoRA adapter 区分时间依赖。planner 以较大步长生成稀疏 clean anchor；renderer 生成所有最终输出位置，不把 planner anchor 直接拼进最终视频。

以 81 个 latent 位置为例，planner 在位置 `{0, 10, ..., 80}` 生成 9 个 anchor，分成每块 3 个的 3 个 planner block；renderer 把 81 个位置拆成 27 个、每块 3 个 latent 的 chunk。每个 renderer chunk 读取附近的三点 clean anchor，以及当前 denoising stage 最近最多 5 个 renderer chunk 的 KV。anchor 提供较慢变化的结构约束，stage-matched history 保留快速变化的局部运动。

参数为 anchor stride $\Delta=10$、planner block 与 renderer chunk 均为 3 latent、四个去噪阶段、上下文预算 21 latent。planner 可读之前最多 6 个 planner block；renderer 在当前 stage 读最多 5 个前序 chunk。对内部 chunk，三点 anchor 窗口横跨当前区域前后，例如输出 6–8 读取 `{0,10,20}`；最终视频的所有位置都由 renderer 生成，planner anchor 仅作条件，避免直接拼接产生时间接缝。

一个 renderer forward 接收当前噪声 latent、局部 clean anchor KV 和当前 stage 的历史 KV，同时输出速度预测与自身 $K,V$。按 rectified-flow 记号 $x_\sigma=(1-\sigma)x_0+\sigma\epsilon$、目标速度 $v=\epsilon-x_0$，当前预测推进到下一噪声水平；新 $K,V$ 随即写入该 stage 的 FIFO bank，保留最近 5 个 chunk。bank 是按阶段独立的，不能将最终 clean 状态误读为所有阶段都可直接复用的 cache。

## 3. 训练

训练分两阶段。第一阶段是 packed SFT：在一次 block-causal forward 中从真实视频同时监督 planner 和 renderer；planner 看之前 block 的 clean KV，renderer 看三点 clean anchor 和相同噪声阶段的历史。随机时间偏移裁剪（time rebasing）使稀疏 anchor 覆盖不同片段位置。模型共用 Wan2.1 骨干，但用 role embedding 和独立 LoRA 学习两种时间依赖。

第二阶段是 self-rollout distillation：planner 自回归生成 anchor，renderer 在模型自己生成的上下文中完成视频；四阶段 guidance-free 学生用 distribution-matching distillation 缩小 teacher-forcing 与实际部署状态差异。训练时 renderer 将整段 clip 以 block-causal mask 一次处理，保留后块到前块、renderer 到 planner 的梯度路径；这与推理时逐 chunk 更新 KV 的执行方式不同。数据是 256K 条带生成 caption 的 Shutterstock 视频，20 秒 clip 含 81 latent/321 RGB 帧，SFT 1800 步，蒸馏 100 个 student step，global batch 128。

## 4. 流水与硬件

四个 denoising stage 可以分别放在四个 GPU。renderer 节点 `(i,k)` 依赖当前 chunk 的下一噪声 stage 和前一 chunk 的同 stage，因此 warm-up 后每个 pipeline round 完成一个 renderer chunk。planner 可以先在一个 GPU 上完成，也可以用 context parallelism；部署时还可以专门留一张 GPU 并行生成 planner anchor，使渲染在第一组 anchor 出现后立即开始。

四卡同设备对照先完成 planner 前奏，再把四卡各分配一个 renderer stage；五卡 streaming 版增加专用 planner GPU，首组 anchor 可用就开始 render，后续 anchor 与渲染并行。这两组 GPU 数不同，不能直接当作同等算力对照。论文也测 planner 用单卡或四卡 context parallelism：480p 往往单卡更合适，720p 通常 context parallel 更快；选择取决于单次 planner forward 的通信/计算比。

论文没有提出独立 CUDA、Triton、CuTe 或 ThunderKittens attention kernel。加速来自 forward 数量减少、KV 状态复用、stage wavefront 调度和 planner/renderer 设备分工。实现中将 LoRA adapter 合并进两个完整权重模型，保证每次 forward 与基线 backbone forward 的成本可比；planner 与 renderer 权重在设备间切换时使用 pinned host memory，传输时间计入报告延迟。

## 5. 实验结果

延迟矩阵覆盖 1.3B/H100 与 14B/GB300、480p/720p、5/20/35/65 秒；20 秒起在各模型—分辨率设置下，最快四卡 FlashForward 相对 HiAR 为 1.16–1.69×、相对 Self-Forcing 为 1.42–2.92×。5 秒时 planner 和流水填充尚未摊薄，论文没有声称全面领先。20 秒的 1.3B/480p 四卡实测为 FlashForward 6.32 s、HiAR 7.67 s、Self-Forcing 20.25 s；14B/720p 65 秒的最佳四卡 FlashForward-CP 为 181.25 s、HiAR 304.35 s。质量只对 1.3B/480p 作系统评测：VBench-1.0 Total 为 0.838，高于 Self-Forcing 0.805 和 HiAR 0.821；VBench-Long Total 在 20/35/65 秒分别为 0.8404/0.8401/0.8395。

81 latent 的典型配置有 3 个 planner block、27 个 renderer chunk、4 个 stage：FlashForward 的逻辑 forward 数为 $3(4+1)+27\cdot4=123$；Self-Forcing 为 $27\cdot4+26=134$；HiAR 为 $27\cdot4+26\cdot4=212$。因此 HiAR/FlashForward 的原始计算次数比为 212/123≈1.72，但四卡实测还受 context parallel 通信、planner 前奏、wavefront 填充和同步影响。14B/720p 的 1.68× 更接近这一粗略比值。该计数并非精确的 wall-clock 上界，因为每种方法的 forward 上下文和设备分配也不同。

表 3 的状态消融更直接检验两类记忆的互补性：单用 stage-matched history 的 Total/Quality 评测失败；只用 clean history 与 future 得到 Total 0.826；二者组合为 0.838。训练消融从双角色 SFT 的蒸馏后 Total 0.8184，经 time rebasing 到 0.8198，再加 role-specific embedding/LoRA 到 0.8380。两项消融分别支持稳定锚点和角色分工，不能仅由加速表推断质量来源。

## 6. 限制

方法依赖共享 planner–renderer 模型、角色 LoRA 和少步自回归视频扩散设置。planner/renderer 权重切换、跨 GPU 通信和 wavefront 调度会影响实际收益；单 GPU 无法发挥 stage pipeline 的全部收益。论文报告的是系统级实测，不等于存在一个可移植的 fused kernel。

## 方法启示

1. 推理状态可以按时间尺度拆成“快速但嘈杂的近期状态”和“稀疏但稳定的长期状态”。
2. 让一次输出推进 forward 同时发布可复用状态，可以减少只为更新 cache 而做的额外计算。
3. 评价流水方案应同时报告逻辑 forward 数、通信/同步开销和真实 wall-clock latency。

## 来源

Meta et al., “In-Flight KV Cache with Clean Anchors for Faster Autoregressive Video Diffusion,” arXiv:2609.32540v1, 2026. [论文](https://arxiv.org/abs/2609.32540)
