---
tags:
  - Video Generation
  - Image Generation
  - KV Cache
  - Diffusion Model
---

# RefAdapt-DiT: Adaptive Joint Attention for Reference-Conditioned Diffusion Transformers

- 论文：[RefAdapt-DiT: Adaptive Joint Attention for Reference-Conditioned Diffusion Transformers](https://arxiv.org/abs/2609.32415)
- 代码：论文页面未给出官方代码链接
- 团队：Tencent
- 提交：2026-09-27，arXiv:2609.32415v1

## 概述

参考条件 DiT 把 reference stream 与当前生成 stream 放在联合注意力中。reference 内容通常比 target 更稳定，却在每个 denoising step 被完整重算。RefAdapt-DiT 把参考复用变成一个运行时控制问题：根据 target query 的变化和 target→reference 的注意力暴露度，判断从哪一层开始恢复新鲜 reference K/V；较浅层复用 cache，较深层重新计算，并设置最大 cache age 强制完整 refresh。

论文在 Qwen Image Edit 和 MiniMax H3 两个系统上用真实延迟评测。8-step Qwen Image Edit 达到 2.58× 加速，4-step MiniMax H3 达到 1.564×；质量接近 full computation。与正交的分辨率压缩组合后分别达到 3.54× 和 2.097×，论文把组合结果与 standalone controller 分开报告。

## 1. 参考复用的误差来源

联合注意力包含 reference→reference、reference→target、target→reference 和 target→target 四个 score region。缓存 reference K/V 后，可以跳过 reference-side QKV 以及涉及 reference query 的区域，理论上减少约

$$\mathcal{O}\left(N_R(N_R+N_Y)\right)$$

的 score 计算，其中 $N_R$ 和 $N_Y$ 分别是 reference 与 target token 数量。

复用的难点是 stale reference K/V 对 target 输出的影响并不固定。仅观察 reference 自身变化可能高估或低估实际误差，因为只有 target query 真正读取 reference 的部分会影响当前生成。RefAdapt-DiT 因此使用两个运行时线索：target-Q 的连续 step 变化，以及上一时刻已经执行过的 target→reference 注意力质量。

## 2. Depth-Adaptive Reference Reuse

每个 denoising step 按深度顺序检查 block。边界之前的 block 使用缓存 reference K/V；第一次达到 reactivation threshold 的 block 恢复新鲜 reference hidden state，之后的深层 suffix 重新生成 reference K/V。这样每个 step 只有一个单调的“复用→刷新”切换点，避免不同 block 独立决策造成 cache 状态不一致。

若连续超过最大 cache age $\Delta_{\max}$ 没有 block-0 full refresh，控制器强制从 block 0 重算并重置年龄。控制器不需要额外完整 attention probe：target-Q 变化和已记录的上一时刻注意力质量都来自运行过程中的已有中间量。

## 3. 实现边界

RefAdapt-DiT 是训练无关的运行时 controller，复用的是模型原生 reference K/V。论文的计算核算明确区分理论 FLOPs 与实际 latency，主实验使用 NVIDIA H20 96GB 的真实执行时间。论文没有报告 CUDA、Triton、CuTe、ThunderKittens 或其他自定义 fused kernel；因此加速来源是 reference QKV/attention 区域跳过、深度 suffix 重算和 cache age 调度，而非新算子。

## 4. 实验设置与效果

Qwen Image Edit 使用固定 100-case MultiBanana 集合、1024×1024、8-step distilled checkpoint、单张 H20；RefAdapt-DiT standalone 达到 2.58×，MB Score 为 14.440，full computation 为 14.512。它在 instruction alignment、reference consistency、background-subject match、physical realism 和 visual quality 上的变化并不一致，整体优势来自更高速度与接近 full 的质量组合。

MiniMax H3 使用固定 100-case EditVerseBench、1344×768、4-step distilled checkpoint、单张 H20；standalone 达到 1.564×，EditQ 为 7.413，高于其他 standalone accelerator 的最高 7.361，CLIP/DINO temporal consistency 为 0.991/0.990。与 ResComp 组合后达到 2.097×。

在机制消融中，RefAdapt-DiT 的速度—保真度折中最好，达到 1.482× measured speedup、31.24 dB PSNR 和 0.9937 SSIM；固定 refresh baseline 为 18.00 dB/0.8823。作者还报告可选诊断 sidecar 会带来约 6.1% 额外开销，但该 sidecar 不属于生产 controller。

## 5. 限制

评测固定为两个 reference-conditioned 系统和各 100 个案例，controller 阈值与最大 cache age 仍需针对模型和硬件校准。部署收益取决于 reference/token 比例、denoising 步数和 target→reference 交互；论文未证明同一阈值可直接迁移到其他 DiT 架构。

## 方法启示

1. cache 复用条件应由“缓存内容是否变化”转向“当前 query 是否使用并放大了这类变化”。
2. 单调 depth cutoff 和 bounded-age fallback 能把自适应复用限制在可验证的执行路径内。
3. 运行时 controller 与正交分辨率压缩可以叠加，但质量和速度必须分别报告，避免把多种收益混为一个数字。

## 来源

Tencent, “RefAdapt-DiT: Adaptive Joint Attention for Reference-Conditioned Diffusion Transformers,” arXiv:2609.32415v1, 2026. [论文](https://arxiv.org/abs/2609.32415)
