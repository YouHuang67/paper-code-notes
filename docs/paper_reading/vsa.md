---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
  - CUDA
  - Triton
---

# VSA: Faster Video Diffusion with Trainable Sparse Attention

[arXiv 2505.13389](https://arxiv.org/abs/2505.13389) | [代码](https://github.com/hao-ai-lab/FastVideo) | [代码分析](../code_analysis/fastvideo_vsa/00_overview.md)

**团队**: UC San Diego, MBZUAI, UC Berkeley

## 概述

VSA 的核心不是“给已有 dense DiT 推理时套一个稀疏 mask”，而是把视频 DiT 的自注意力改写成一个从训练开始就存在的 **两级注意力结构**：

- **coarse stage**：先把视频 latent 按 `(C_t,C_h,C_w)` 时空 cube 做均值池化，在 cube 空间上做 dense attention；
- **fine stage**：对每个 query cube，只在 coarse stage 选出的 Top-K key cube 内做 token 级精确注意力；
- **gate 融合**：把 coarse 输出和 fine 输出重新合成，让模型学会在“低频全局上下文”和“高频精细检索”之间分配信息。

这篇论文的真正贡献有三层：

- **算法层**：把 critical token 预测问题降成 block 级预测，而不是先算完整 `QK^T` 再稀疏化；
- **训练层**：通过 annealing 把 full attention checkpoint 平滑迁移到 sparse attention，而不是直接替换；
- **内核层**：稀疏模式严格贴合 block-sparse kernel，selector 也专门做了 Triton fused kernel，因此稀疏 FLOPS 可以转成真实 wall-clock speedup。

论文报告的主结论是：

- 对 16K token 训练，VSA 在 87.5% 注意力稀疏率下，与 full attention 达到几乎相同的 loss；
- attention FLOPS 下降约 `8x`，总训练 FLOPS 下降约 `2.53x`；
- 在 Wan2.1-1.3B 上，端到端推理时间从 `31s` 降到 `18s`；
- 在 Wan-14B 上，端到端时间从 `1274s` 降到 `576s`；
- 稀疏 attention 还能和 sparse distillation 同时工作。

当前官方开源实现位于 `refs/codes/FastVideo`，本仓库代码分析基于提交 `970409962f358afd529b969a378174c849665837`。我在 **2026-07-20** 额外检查了该提交到 `origin/main` 的 VSA 相关文件，核心 VSA 路径未发生实质变化，因此以下论文和代码对应关系仍然成立。

## 1. 问题设定

视频 DiT 的视频 latent 形状记作 `(T,H,W)`，拉平后序列长度为：

$$
L = T H W
$$

单头 full attention 写成：

$$
S = \frac{QK^\top}{\sqrt{d}},\qquad
A = \operatorname{Softmax}(S + M),\qquad
O = AV
$$

这里 `Q,K,V \in \mathbb{R}^{L \times d}`，`M` 是 attention mask。对普通双向自注意力，`M` 全零；代价是：

$$
\mathcal{O}(L^2 d)
$$

视频生成里这件事非常贵，原因不是抽象意义上的“二次复杂度”，而是具体序列真的很长。论文直接指出，5 秒 720p 视频展开后会超过 100K token，训练和推理都会被注意力吃掉。

VSA 想解决的问题可以表述为：

- 不显式构造完整 token-token attention；
- 仍能找到真正“承重”的 critical token 区域；
- 稀疏模式必须是硬件友好的 block 结构，而不是任意 token 稀疏。

## 2. VSA 的数学建模

### 2.1 Cube 划分不是附属细节，而是方法本体

VSA 先把 `(T,H,W)` 视频 latent 划成 cube，cube 大小记作：

$$
(C_t, C_h, C_w)
$$

每个 cube 的 token 数是：

$$
B = C_t C_h C_w
$$

论文默认使用：

$$
(C_t, C_h, C_w) = (4,4,4),\qquad B=64
$$

令 cube 网格数为：

$$
(N_t,N_h,N_w)=\left(\frac{T}{C_t},\frac{H}{C_h},\frac{W}{C_w}\right)
$$

原论文给出从三维坐标 `(t,h,w)` 到 tile-contiguous 一维顺序的映射：

$$
n =
\left(
\left\lfloor \frac{t}{C_t}\right\rfloor N_h N_w +
\left\lfloor \frac{h}{C_h}\right\rfloor N_w +
\left\lfloor \frac{w}{C_w}\right\rfloor
\right) B
+ (t \bmod C_t) C_h C_w
+ (h \bmod C_h) C_w
+ (w \bmod C_w)
$$

这个式子非常重要，因为它不是只为数学方便。它保证：

- 同一时空 cube 的 token 在内存中连续；
- 一个 cube 天然对应一个 block-sparse attention tile；
- 后续 coarse selector 和 fine sparse kernel 都能直接以块为单位执行。

### 2.2 Coarse stage：在 cube 空间预测 critical block

对每个 cube 做均值池化：

$$
q_c^{(i)}=\frac{1}{|B_i^q|}\sum_{u\in B_i^q} q_u,\quad
k_c^{(j)}=\frac{1}{|B_j^k|}\sum_{v\in B_j^k} k_v,\quad
v_c^{(j)}=\frac{1}{|B_j^k|}\sum_{v\in B_j^k} v_v
$$

注意论文和代码都明确支持 **variable block size**。边界 cube 可能不满，因此分母不是固定 `64`，而是当前块真实 token 数 `|B_i|`。

在压缩后的 cube 空间执行 dense attention：

$$
S_c = \frac{Q_c K_c^\top}{\sqrt d},\qquad
A_c = \operatorname{Softmax}(S_c),\qquad
O_c = A_c V_c
$$

然后对每一行做 Top-K：

$$
\mathcal{N}(i)=\operatorname{TopK}_j \; S_c(i,j)
$$

本质上，VSA 用 `Q_c K_c^\top` 预测“哪个 key cube 里藏着 critical token”。

### 2.3 Fine stage：只在选中的 cube 上做 token 级精算

Top-K 之后，VSA 不是在压缩 token 上继续近似，而是回到原始 token 分辨率，仅对选中的 block 做精确 sparse attention：

$$
o_f^{(u)} =
\operatorname{Softmax}
\left(
\frac{q_u K_{\mathcal{N}(i)}^\top}{\sqrt d}
\right)
V_{\mathcal{N}(i)},
\qquad u \in B_i^q
$$

这一步的关键不是公式本身，而是 mask 的结构天然是 `B \times B` block：

- coarse stage 选中的是 cube；
- 广播到 token 级后，每条边对应一个 `B \times B` 子矩阵；
- 这样 fine stage 就能直接进入 block-sparse kernel。

### 2.4 输出融合：论文是双 gate，开源实现是单 gate

论文写法是：

$$
O = O_c \odot G_c + O_f \odot G_f
$$

其中 `G_c,G_f` 都来自输入 hidden states 的线性投影。

但官方开源实现和论文正文这里存在一个很关键的差别：

- 论文方法层面保留 `G_c` 与 `G_f` 两个门；
- FastVideo 当前公开实现里，VSA kernel 实际只接收 **`compress_attn_weight`**，也就是 coarse 分支门控；
- fine 分支在 kernel 中默认系数为 `1`，即：

$$
O \approx O_f + G_c \odot O_c
$$

这和论文 §2.3 的 sparse adaptation 描述其实一致。论文为了从 dense checkpoint 迁移，会把 `G_f` 去掉，等价于固定 `G_f=1`；当前开源工程就把这条简化保留了下来。

### 2.5 复杂度

设 token 数为 `N`，block size 为 `B`，block 数为 `T=N/B`，每个 query block 选 `K` 个 key block。

则：

- coarse stage：`O(T^2 d) = O((N/B)^2 d)`
- fine stage：`O(N K B d)`

与 full attention 的 `O(N^2 d)` 相比，只要 `K << T`，总成本就会显著降低。

VSA 的优势不只来自 `K` 小，还来自 coarse 分支保留了一个低成本全局通道。也就是说它不是纯粹“砍掉大部分注意力”，而是把全局信息搬到更便宜的分辨率上。

## 3. 设计空间与消融：论文真正有价值的部分

这篇论文最好的部分不是最终公式，而是它系统地把 sparse attention 的几个关键设计变量拆开验证了。

### 3.1 Data-dependent 稀疏比固定模式更重要

论文比较了多种固定模式：

- spatial-temporal；
- spatial-full；
- strided window；
- compress KV。

结论很直接：

- 在较小训练预算下，固定模式能比 full attention 更省；
- 但当训练预算扩大时，这些固定模式优势会消失甚至反转；
- VSA 的 **data-dependent** block 选择在小预算和大预算下都更稳。

也就是说，VSA 的关键不在“有稀疏”，而在“稀疏模式是由数据决定的”。

### 3.2 Global coarse output 必要，显式 locality 反而收益很小

论文把 coarse、fine、local 几种组合都测了一遍：

- 只有 local，不够；
- 只有 fine sparse，也不够；
- coarse + fine 最稳；
- 再额外硬塞 local stage 或强制 local cube，收益很有限。

这个结论很值得重视。它说明在视频 DiT 中，真正短缺的不是局部先验，而是：

- 低成本全局检索；
- 数据相关的精确重算。

因此 VSA 最终选的是最简单的 `C & F` 结构。

### 3.3 Tile size 是表达能力与硬件效率的主旋钮

论文比较了：

- `B=256`, `(4,8,8)`
- `B=128`, `(4,8,4)`
- `B=64`, `(4,4,4)`
- `B=16`, `(2,4,2)`

结论：

- tile 越小，coarse selector 越容易更精准地定位 critical token；
- fine stage 也能只对更小、更干净的区域做精确注意力；
- 但 tile 太小会让 kernel 吞吐掉得很厉害。

作者最终选择 `B=64`，理由是：

- 表达能力已经明显优于 `128/256`；
- 相比更小的 `16`，吞吐损失可接受；
- 又能很好贴合 block-sparse kernel。

这点和代码实现是直接绑定的：FastVideo 的默认 VSA tile 就是 `(4,4,4)`。

### 3.4 Mean pooling 足够，卷积预测器反而不稳

论文还比较了：

- average pooling；
- max pooling；
- 3D convolution pooling。

结果是 average pooling 最好，卷积池化甚至会带来训练不稳定。

这说明 VSA 的 coarse stage 不需要一个复杂 predictor；简单均值池化已经足够给出高质量 block-level score。这也是为什么代码里 `fused_block_mean` 能写成一个非常直接的 reduction kernel。

## 4. 训练策略：为什么可以从 dense 平滑过渡到 sparse

### 4.1 从头训练

论文的大规模实验使用 Wan2.1 风格架构，在 `16 × 32 × 32` latent、`16384` token 上做预训练。

120M 消融模型的关键配置包括：

- `head dim = 64`
- `num heads = 12`
- `num layers = 12`
- `batch size = 1024`
- `objective = Flow Matching`
- `total FLOPs = 4.5 × 10^{20}`

作者还做了 `60M -> 1.4B` 的 scaling study，并指出 VSA 和 full attention 的 loss 曲线基本平行，因此 `2.53x` 的总 FLOPs 降幅可以稳定延续到更大模型。

### 4.2 Sparse adaptation：从 full checkpoint 迁移

如果直接把 full attention 替换成 VSA，训练会不稳定。论文认为主要有两个原因：

- coarse gate 是新加参数，初始是随机的；
- attention 结构被突然改成“两级 + 稀疏”，分布漂移太大。

解决方案是 annealing：

1. 初始化 coarse gate `G_c = 0`；
2. 去掉 fine gate，等价于 `G_f = 1`；
3. 初始令 `K = L/B`，让 VSA 退化为 full attention；
4. 训练过程中逐步减小 `K` 到目标稀疏率。

附录 C.5 给了更具体的 schedule：

- 先 full attention 训练 50 steps；
- 每 50 steps，把 attended cubes 减少 10，也就是 Top-K 减少 4；
- 直到目标 `Top-K = 32`。

这个 schedule 是理解开源代码和论文差异的关键。代码只保留 coarse gate，本质上就是为这条迁移路径服务。

### 4.3 Sparse Distill

论文还做了一个很有意思的 pilot study：

- teacher 保持 full attention；
- student 改成 VSA；
- DMD2 的蒸馏损失和超参都不改。

结果是 sparse attention 可以和少步蒸馏共存，Wan-1.3B 的 3-step 生成器能达到 `50.9x` 的 denoising 加速。

这很重要，因为它说明 VSA 不是只能单独吃“attention 稀疏化”这一个红利，而是可以继续叠加到更激进的生成加速链路里。

## 5. Kernel 视角：论文为什么能落成真实速度

### 5.1 Coarse stage 并不需要“全都塞进一个超级 FlashAttention”

论文非常务实地指出：

- coarse stage 序列长度已经被 `64x` 压缩；
- 即使 materialize `Q_c K_c^T`，内存和 FLOPs 都很小；
- 真正的额外开销主要来自 Top-K 和 mask/index 转换。

所以他们没有强行把 coarse stage 改成一版“支持 in-kernel Top-K 的 FlashAttention”，而是只做了更合算的事情：

- `block mean` fused；
- `softmax + topk + mask-to-index` 部分尽量 fused。

这也是当前 FastVideo 实现的选择。

### 5.2 Fine stage 才是主要加速来源

论文里真正的速度红利来自 fine sparse kernel：

- 87.5% 稀疏率时，fine kernel 接近理论 `8x` 上限；
- 只看 fine stage，本地 benchmark 接近 `7x` over FA3；
- 把 coarse stage 算进去，仍有 `6x+` 的 attention kernel speedup。

这解释了为什么实现上要投入这么多工程到 sparse backend：

- Triton fallback；
- Hopper ThunderKittens CUDA kernel；
- 256 tile 的 CuTe DSL / FA4 路线。

## 6. 实验结果怎么读

### 6.1 训练 scaling

在 `410M` 模型、`16384` token 设置下，论文声称：

- 与 full attention 几乎同 loss；
- attention FLOPs 约 `8x` 降低；
- 端到端训练 FLOPs 约 `2.53x` 降低。

再扩展到 `60M -> 1.4B`，VSA 与 full attention 的 scaling 曲线几乎平行。作者把这解读为：VSA 不是只在小模型或小预算上暂时占便宜，而是更优 Pareto frontier。

### 6.2 Wan2.1-1.3B sparse finetune

在 `480P` 合成数据上：

- `Top-K = 32`
- attention sparsity `91.2%`
- VBench 与 full finetune 接近
- 推理时间从 `31s` 到 `18s`

论文还把 VSA 与 training-free 的 SVG 比较，指出即使在更高稀疏率下，训练型 sparse attention 仍更受人偏好。

### 6.3 Wan-14B sparse finetune

在 `720P`、`77` 帧、`90%` 稀疏下：

- 人类偏好与官方 full model 基本接近；
- 端到端时延从 `1274s` 降到 `576s`。

这比 1.3B 实验更关键，因为它说明 VSA 的收益并不止于小模型补丁，而是能扩到真正重型视频模型。

## 7. 论文与开源代码的对应关系

当前 FastVideo 中，VSA 不再是论文 demo，而是正式 attention backend：

- `fastvideo/attention/backends/video_sparse_attn.py`
  - tile / pad / untile / metadata / 稀疏率转 Top-K
- `fastvideo-kernel/python/fastvideo_kernel/ops.py`
  - coarse branch、Top-K mask、稀疏分支调度
- `fastvideo-kernel/python/fastvideo_kernel/triton_kernels/fused_compress_topk.py`
  - Triton `fused_block_mean` 和 `fused_topk_mask`
- `fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn.py`
  - 64-token 稀疏分支的 Triton / ThunderKittens 路由
- `fastvideo-kernel/python/fastvideo_kernel/block_sparse_attn_256.py`
  - 256-token 路径的 Triton route-A / CuTe DSL 路由
- `fastvideo-kernel/csrc/attention/block_sparse_h100.cu`
  - Hopper `sm_90a` 上的 ThunderKittens CUDA kernel

实现上的几个关键事实：

- VSA 路径里 **没有 TileLang**；相关 DSL 主要是 `Triton`、`ThunderKittens CUDA`、`CuTe DSL`；
- 论文中的双 gate，在开源实现里收敛成“只门控 coarse 分支”；
- 256-tile 路径默认仍是 Triton route-A，CuTe fastpath 是 opt-in，不是默认；
- VSA 的 `block_size=64/256` 不只是算法超参，也决定具体 backend。

详细代码拆解见：

- [FastVideo VSA 代码分析：总览](../code_analysis/fastvideo_vsa/00_overview.md)
- [框架接入、tile metadata 与门控](../code_analysis/fastvideo_vsa/01_framework_and_metadata.md)
- [Triton coarse selector：fused block mean 与 Top-K mask](../code_analysis/fastvideo_vsa/02_fused_coarse_selector.md)
- [Sparse backends：Triton / ThunderKittens CUDA / CuTe DSL](../code_analysis/fastvideo_vsa/03_sparse_backends.md)
- [Kernel Execution Appendix：按源码执行顺序展开](../code_analysis/fastvideo_vsa/04_kernel_execution_appendix.md)

其中最后这页专门展开：

- `q2k/k2q` 索引压缩与反向翻转；
- Triton sparse forward/backward 的 tile、寄存器状态和内层循环；
- Hopper ThunderKittens CUDA 的 host grid、CTA 映射、TMA/WGMMA 流水和 `qo_blocks` 负载分布；
- 256-token route-A 的 `4 x 4` 物理子图展开，以及 CuTe `mask_mod` 怎样处理 partial block。

## 8. 局限

### 局限

- 当前默认 tile 固定为 `(4,4,4)`，因此 latent shape 最好可整除 4；
- 最优 Top-K 依赖序列长度、模型规模和训练预算，目前还没有统一 scaling law；
- coarse stage 尽管便宜，但仍不是零成本，尤其短序列下 Top-K runtime 更显眼；
- 256-tile 的最高性能路径依赖可选 CuTe/FA4 构建，不是所有机器都能直接跑。

## 9. 关键启示

- 对视频 DiT，attention 稀疏化不能只看“mask 够不够 sparse”，而要看是否保留了一个便宜但有效的全局通道；VSA 的 coarse stage 正是这个通道。
- 如果训练阶段不让模型看到 sparse attention，推理阶段再强行替换，质量很容易掉；VSA 的 annealing 说明“先平滑过渡再收缩预算”是更可靠的路径。
- block size 不是单纯的算子超参，而是表达能力、selector 精度和 GPU kernel 形态的共同交点；这也是为什么 `64` 会成为论文和代码同时收敛到的默认值。
- VSA 的 DSL 重心是 Triton / CUDA / CuTe，而不是 TileLang；阅读实现时应优先关注 online softmax、稀疏索引、TMA/WGMMA 与 block-sparse tensor 描述。
