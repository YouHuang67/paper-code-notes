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

论文对 reference/target token 行为先做了机制诊断。相邻 step 的相对漂移定义为

$$D_{t,b}(X)=\frac{\|X_{t,b}-X_{t-1,b}\|_F}{\|X_{t-1,b}\|_F+\epsilon},$$

并比较 matched reference 与 target 的漂移比。H3 在 block 6–25 的中位 Ref/Tgt K/V 比约 0.352，Qwen 的 K/V 比约 0.433/0.381，说明浅中层 reference 变化较慢，但深层差距缩小，固定全深度 reuse 不可靠。另一方面，target→reference directional mass 定义为 target query 对 reference key 的 attention mass；全轨迹均值约为 H3 9.51%、Qwen 14.66%，它表示暴露度而非因果影响。H3 中 target block-input change 与 reference K/V change 的 block-level Spearman 中位数为 0.675，支持使用 target-side runtime cue，但不等价于直接估计 reference 漂移。

若连续超过最大 cache age $\Delta_{\max}$ 没有 block-0 full refresh，控制器强制从 block 0 重算并重置年龄。控制器不需要额外完整 attention probe：target-Q 变化和已记录的上一时刻注意力质量都来自运行过程中的已有中间量。

控制器在 head 粒度形成分数

$$g_{t,b,h}=D_{t,b}(Q^Y_h)\,M^{Y\rightarrow R}_{t-1,b,h},$$

其中第一项是当前 target-Q 的相邻 step 漂移，第二项是上一次执行该 block 时 target→reference 的方向性质量。扫描 block 深度时，首个满足 $H^{-1}\sum_hg_{t,b,h}\geq\theta$ 的 block 设为 reactivation boundary；边界以前复用，边界处恢复缓存的 reference hidden state，边界之后连续重算 reference suffix。若没有 block 越阈值则整步复用。每个 block 保存 cached reference 输入 hidden、K/V 和上次 materialization step，因此 suffix 计算有连贯的 reference stream，而不是互不兼容的独立刷新。

## 3. 实现边界

RefAdapt-DiT 是训练无关的运行时 controller，复用的是模型原生 reference K/V。论文的计算核算明确区分理论 FLOPs 与实际 latency，主实验使用 NVIDIA H20 96GB 的真实执行时间。论文没有报告 CUDA、Triton、CuTe、ThunderKittens 或其他自定义 fused kernel；因此加速来源是 reference QKV/attention 区域跳过、深度 suffix 重算和 cache age 调度，而非新算子。

## 4. 实验设置与效果

Qwen Image Edit 使用固定 100-case MultiBanana 集合、1024×1024、8-step distilled checkpoint、单张 H20 96GB。RefAdapt-DiT standalone 为 2.58×、23.43 s，MB Score 14.440，full computation 为 1.00×、60.39 s、14.512。分项相对 full 的 IA/RC/BSM/PR/VQ 为 +0.034/-0.095/+0.212/-0.158/+0.057；因此 aggregate 接近 full 并不表示每项都无损。加入正交 ResComp 后为 3.54×，但 MB Score 降至 12.320，必须与 standalone controller 分开解读。

MiniMax H3 使用固定 100-case EditVerseBench、1344×768、4-step LightX2V distilled checkpoint、单张 H20；standalone 为 1.564×、128.38 s，EditQ 7.413，高于最快 baseline TaylorSeer 的 1.333×和最高 baseline EditQ 7.361，CLIP/DINO temporal consistency 为 0.991/0.990。与 ResComp 组合后为 2.097×、但它是另一速度—质量 operating point。

在机制消融中，RefAdapt-DiT 的速度—保真度折中最好，达到 1.482× measured speedup、31.24 dB PSNR 和 0.9937 SSIM；固定 refresh baseline 为 18.00 dB/0.8823。作者还报告可选诊断 sidecar 会带来约 6.1% 额外开销，但该 sidecar 不属于生产 controller。

机制消融使用独立的 50-step Ref2VA 轨迹，不把它当作主 4-step 部署结果。固定 R3、仅 target-Q、仅 exposure、RefAdapt-DiT 的 speedup 分别为 1.464×、1.459×、1.459×、1.482×；对应 PSNR/SSIM 为 18.00/0.8823、29.81/0.9879、32.24/0.9958、31.24/0.9937。这个表支持两种 cue 的互补性和深度执行器的连续 suffix 结构；它不证明 50-step PSNR 能直接预测 4-step perceptual quality。

## 5. 限制

评测固定为两个 reference-conditioned 系统、各 100 个案例和单一 H20 配置；controller 阈值与最大 cache age 仍需针对模型和硬件校准。主实验依赖 distilled 8-step/4-step 模型，few-step 场景后续可纠错步数少，能检验 reuse 误差恢复，但也限制了结论对长采样轨迹的外推。部署收益取决于 reference/token 比例、denoising 步数和 target→reference 交互；论文未证明同一阈值可直接迁移到其他 DiT 架构。

## 方法启示

1. cache 复用条件应由“缓存内容是否变化”转向“当前 query 是否使用并放大了这类变化”。
2. 单调 depth cutoff 和 bounded-age fallback 能把自适应复用限制在可验证的执行路径内。
3. 运行时 controller 与正交分辨率压缩可以叠加，但质量和速度必须分别报告，避免把多种收益混为一个数字。

## 来源

Tencent, “RefAdapt-DiT: Adaptive Joint Attention for Reference-Conditioned Diffusion Transformers,” arXiv:2609.32415v1, 2026. [论文](https://arxiv.org/abs/2609.32415)
