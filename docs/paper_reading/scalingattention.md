---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
  - CUDA
---

# ScalingAttention: Discovering Intrinsic Sparse Attention Topology for Video Diffusion Transformers

- 论文：https://arxiv.org/abs/2606.23019
- 团队：KlingAI Research, Beijing Institute of Technology, NVIDIA, Tsinghua University
- 代码：当前未见明确官方开源仓库；本文实现分析基于论文、arXiv 源文件与补充材料重建

## 概述

ScalingAttention 是一篇很典型的“标定型 sparse attention”论文，但它比一般 calibration 方法更系统。它的核心主张是：

> 对视频 DiT 而言，attention 的具体激活值是输入相关的，但高质量 attention 区域的支撑拓扑会快速收敛到一个稳定、与 prompt 基本无关的稀疏结构。

作者把这个结构叫作 **Intrinsic Sparse Topology**。围绕这个观察，方法被拆成三部分：

1. **WEST**：Weight-Encoded Sparse Topology，离线抽取稳定的静态稀疏拓扑；
2. **FAST**：Fidelity-Aware Sensitivity Tuning，根据层/步的 fidelity 需求动态决定每个 head 的稀疏密度；
3. **crm kernel**：hardware-aligned bit-wise block-sparse kernel，把静态拓扑高效地映射到 GPU 上。

ScalingAttention 更强调“拓扑”和“稀疏度”应当解耦：

- **拓扑问题**：哪里可以看；
- **敏感度问题**：到底能剪到多 sparse。

论文在 Wan2.1-1.3B、Wan2.1-14B、HunyuanVideo 上报告最高 **1.90x 端到端加速**，并在类似 PSNR 下比 SVG2 需要更低的 attention density。

## 核心观察：Sparse Topology 是权重编码的

作者反对一个常见前提：很多 prior work 默认 attention sparsity pattern 是纯输入相关的，因此只能在线做 token/block 选择；否则静态 mask 只是粗糙启发式。

ScalingAttention 的经验观察是，对某个固定 head：

- 不同 prompt 会激活不同局部区域；
- 但这些高质量 attention 区域的并集很快收敛；
- 这个并集边界对 prompt 基本稳定；
- 还具有分辨率尺度不变性。

从论文 Figure 2 的结论可以归纳出三点：

1. **Intrinsic Structure**：union of high-mass regions 会收敛到稳定边界；
2. **Asymptotic Stability**：随着 profiling prompts 增加，union mask density 很快饱和；
3. **Scale Invariance**：在低分辨率提取的拓扑映射到高分辨率时，attention recall 依旧很高。

所以它并不把 sparse attention 看成 transient runtime phenomenon，而把它视为：

> 预训练权重中已经编码出的可复用 attention 支撑包络。

这使得视频 DiT 的稀疏化问题不再只是“怎么在线选 top-k token”，而变成：

- 先离线找出这个 head 的 **最大可用拓扑壳层**；
- 再根据当前 fidelity 要求，在这个壳层里选择多大密度。

## 方法总览

ScalingAttention 的整体流程可以写成：

```text
offline prompts
  -> WEST: threshold map
  -> FAST: fidelity-sparsity profile
  -> target density -> per-head p(l,t)
  -> threshold map instantiate static masks
  -> crm kernel execution
```

这里最关键的是：**WEST 产出的不是单个固定 mask，而是一个能支持任意目标稀疏密度的 threshold map**。因此推理前只要指定 global density target，就可以快速生成对应静态 mask，而不用重新标定拓扑。

附录 Algorithm 1 也验证了这一点。作者实际把全流程明确拆成两个离线阶段：

1. **PHASE 1: WEST**
   - 采样 calibration prompts；
   - 计算 attention block significance；
   - 聚合出 threshold map；
   - 建立后续 FAST 使用的 profile。
2. **PHASE 2: FAST**
   - 给定 global density target；
   - 为每个 `(layer, timestep)` 选出满足 fidelity 约束的最大 sparsity threshold；
   - 再结合 threshold map 实例化最终 masks。

所以 WEST 提供的是一个可裁剪的拓扑容器，FAST 则决定在当前部署预算下把这个容器裁到什么程度。

## WEST：Weight-Encoded Sparse Topology

### Block significance

给定 calibration set 中某个 head 的 attention 矩阵 `A_i`，先按 `B x B` block 聚合，论文默认 `B = 128`：

$$
S_i^{(u,v)}
=
\frac{1}{B^2}
\sum_{p \in B_u}
\sum_{q \in B_v}
A_i^{(p,q)}
$$

这和 CalibAtt 用 block energy 的思路相近，但目标不同：

- CalibAtt 是按给定阈值，直接为每个 prompt 产出最终 mask；
- WEST 想构建一个 **完整的稀疏层级索引**，支持后续不同 density 的快速实例化。

### Threshold Map

WEST 的核心对象是 Threshold Map `T`。对每个 block `(u,v)`，定义：

$$
T^{(u,v)}
=
\min\{p \mid \exists i, (u,v) \in \operatorname{Top\text{-}k}(S_i, p)\}
$$

直觉上，`T(u,v)` 表示：

- 这个 block 至少需要多高的保留密度，才会出现在某个 calibration prompt 的活跃区域中。

于是 `T` 记录的不是一个离散 mask，而是整个 sparsity hierarchy。给定任意目标密度 `p`，只要做一次比较：

$$
M_p = \mathbf 1[T \le p]
$$

就能即时恢复一个静态二值掩码。

这比直接存很多版本的 mask 更干净，因为：

- WEST 只需存一张 threshold map；
- 生成不同 density 的 mask 是 `O(1)` 风格的轻量操作；
- topology 与 density 被彻底分离。

从工程角度说，这等价于把完整 sparsity hierarchy 压到一张 threshold map 里。后续若换目标 density，不需要重新做 topology discovery，也不需要保存很多版本的二值 mask。

### WEST 的本质

WEST 可以理解成：

- 不是直接问“哪些块该保留”；
- 而是为每个块打一个“进入稀疏拓扑所需的最低优先级阈值”。

这样一来，拓扑就变成了一个可以按需裁剪的静态结构，而不是固定死的 binary pattern。

## FAST：Fidelity-Aware Sensitivity Tuning

WEST 解决的是“哪里可以看”，FAST 解决的是“每个 `(layer, timestep)` 到底能剪到多 sparse”。

### 为什么要单独建模 sensitivity

作者认为稀疏容忍度会随：

- **denoising timestep**
- **transformer layer depth**

变化。早期步和浅层通常更敏感，后期步和深层更能接受 aggressive sparsification。因此用统一密度是不合理的。

### Fidelity Score

FAST 不用 cosine similarity，而是用 Hellinger distance 度量 dense attention 分布 `P` 与 sparse approximation `\tilde P` 的差异：

$$
H(P,Q)
=
\frac{1}{\sqrt 2}
\sqrt{
\sum_i (\sqrt{p_i}-\sqrt{q_i})^2
}
$$

再定义：

$$
F(P,\tilde P) = 1 - H(P,\tilde P)
$$

作者认为它优于 cosine similarity，原因是：

- cosine 很容易过早饱和；
- heavy-tailed attention 分布中，关键结构被剪掉时 cosine 仍可能很高；
- Hellinger 对概率质量变化更敏感，又比 KL 更稳定。

### Spatial-Temporal Fidelity Allocation

FAST 对每个 `(l,t)` 定义目标 fidelity 函数：

$$
A(l,t) =
1 -
\left[
\beta
\left(
1-\frac{l+\omega t}{N_L+\omega N_T}
\right)^\gamma
+ (1-\beta)
\right]
$$

这里：

- `\omega` 平衡 layer 与 timestep 的影响；
- `\beta` 控制 baseline fidelity；
- `\gamma` 控制过渡曲率。

这不是人工拍脑袋定的超参，而是在给定 global density target 后离线优化出来，使全局稀疏预算和 fidelity 需求匹配。

对每个 head，在 WEST 给定的 topology 下，FAST 会选择满足：

$$
F(P,\tilde P) \ge A(l,t)
$$

的最大 sparsity threshold。于是最终得到：

$$
p(l,t)
$$

它表示某个层、某个扩散步允许的稀疏强度。

这一步很关键。很多方法只发现 topology，却不知道不同层/步该剪多少。FAST 本质上就是为 topology 配一个 fidelity-aware budget allocator。

从实现上看，FAST 不是在推理时在线解优化问题，而是离线把：

- fidelity target curve `A(l,t)`；
- 每个 head 的 fidelity-sparsity profile；

拼起来，最后把 `(l,t)` 直接映射成静态阈值。这样在线阶段依然只需要查表，不会引入新的 runtime adaptive control 开销。

## Stratified Sampling：让离线 profiling 可承受

如果对完整 `N x N` attention map 做 fidelity profiling，离线成本会很高。ScalingAttention 的做法是：

- 把 attention rows 分成若干块；
- 每块随机抽少量行；
- 用 block-wise stratified sampling 近似全图的 Hellinger distance。

论文声称只采样 `16` 行（约全图 `0.05%`）就足够接近完整 profiling。这个设计很重要，因为：

- WEST 已经要跨多 prompts 建拓扑；
- FAST 若再对完整 attention 图做大量精确 profiling，离线阶段会不可接受。

所以 FAST 的实现不是“高精度完整评估”，而是 **在足够稳定的 topology 先验上，使用轻量 fidelity proxy 做预算标定**。

论文还专门验证了这个近似：很低的采样率下，估计结果依然和完整 profiling 高度一致。这说明 FAST 的可行性并不建立在高成本离线暴力搜索上，而是建立在 attention 结构本身足够稳定这个前提上。

## Resolution Scalability

ScalingAttention 另一个比 CalibAtt 更完整的点，是它显式支持跨分辨率复用拓扑。

由于作者认为 topology 是权重编码而非分辨率特有结构，因此只需在一个基础分辨率上提取 threshold map，然后在目标分辨率下做 conservative bilinear interpolation：

$$
R_{target}^{(u,v)}
=
\left\lceil
\Phi\left(
R_{base},
\frac{u}{s_h},
\frac{v}{s_w}
\right)
\right\rceil
$$

这里的 ceiling rule 保证：

- 只要高分辨率块与低分辨率活跃区有重叠，就不会被错误地裁掉。

它把低分辨率 profiling 变成高分辨率部署的代理，从而降低一次性离线成本。

这里使用 conservative ceiling rule 也不是细节，而是设计原则。作者优先保证高分辨率下不漏掉低分辨率活跃区域，因此插值规则更偏 recall-preserving，而不是最瘦 mask。

## crm kernel：bit-wise block-sparse attention

### 为什么作者要自己做 kernel

论文明确指出，仅仅在算法上减少 FLOPs 不够。视频 DiT 上很多动态稀疏方法的问题在于：

- runtime search 开销高；
- 稀疏结构不规则；
- memory fragmentation 重；
- Tensor Core 利用率差。

所以作者专门 co-design 了 **crm kernel**。

### CRM 表示

crm = Compressed Row Mask。它不是 CSR 式的显式索引表，而是：

- 每个 attention row 用压缩 bitmask 表示；
- 存在对齐的 `uint32` 数组中；
- 每一位表示一个 KV block 是否激活。

因此一次 `uint32` 载入可同时解码 `32` 个 KV blocks 的开关状态。

当 block size = `128 x 128` 时，即使序列极长，mask 读取次数也很少。作者举的例子是：

- `256k` tokens 时，每行也只需 `64` 次 mask load。

这解释了为什么论文强调 bit-wise encoding，而不是常见的 index list：

- index list 更灵活，但 metadata traffic 大；
- dense bool mask 过大；
- bitmask 在静态 block topology 下是很好的折中。

这背后的系统判断是：当 topology 足够静态时，按行压成 bitmask 会比显式 block index 更划算。因为每个 block 是否激活只需要 1 bit，解码逻辑又可以完全在寄存器里完成，metadata 访问会比 CSR/COO 一类显式索引格式更规整。

### Forward iterator

论文附录提到 `CRM Forward Iterator`。这说明 kernel 的主干应当是：

1. 一个 dense FA-style query tile iterator；
2. 被一个 sparse KV iterator 替代；
3. sparse iterator 通过 bit scan / bit clear 原语遍历当前 row 中的 active blocks。

也就是说，它不是把 attention 主体改写成完全新算子，而是把 dense block traversal 替换成：

```text
load uint32 mask word
while word != 0:
    b = find_next_set_bit(word)
    clear_bit(word, b)
    process KV block b
```

在 GPU 上，这种 bitwise traversal 很轻，因为：

- metadata 体积小；
- 遍历逻辑完全在寄存器中；
- 不需要大量 global memory 间接索引。

附录的 `CRM Forward Iterator` 还透露出一个更重要的实现思路：作者不是重新发明 attention 数值核心，而是在 dense FlashAttention 的 block traversal 层做最小必要改写。在线 softmax、tile MMA、数值稳定逻辑都可以沿用成熟路径，变化集中在“如何找到下一个 active KV block”。

### 为什么这对高 GPU 利用率重要

论文的目标不是让一个 CTA 处理极其零碎的 token 级选择，而是保持：

- block size 固定为 `128 x 128`；
- 每个 active block 仍然是标准 dense tile；
- Tensor Core 计算路径尽量和 dense FlashAttention 接近。

从 CTA/warp 视角看，这样的稀疏 attention 有两个优点：

1. **负载粒度规则**：虽然不同 row 的 active blocks 数量不同，但单次计算单位始终是大块 tile；
2. **访存更连续**：KV block 是整块载入，而不是无结构 token gather。

因此 crm kernel 的设计哲学很明确：宁可在算法上牺牲一点最细粒度灵活性，也要保持 Tensor Core 友好。

论文也明确承认了这个代价：为了最大化 Tensor Core 利用率，crm kernel 采用 `128 x 128` 大块，这会限制极细粒度剪枝能力。因此在极低 density 区间，收益不会无限线性增长。

## 实现结构重建

虽然没有官方代码，但从论文文字可以比较清楚地重建整个系统：

### 1. WEST offline pass

输入 calibration prompts，计算每个 head 的 attention block significance `S_i`，聚合成 threshold map `T`。

### 2. FAST offline pass

在 WEST 拓扑先验下，对各层各步做 fidelity-sparsity 曲线 profiling，并通过目标 global density 反求每个 `(l,t)` 的最优稀疏阈值。

### 3. Mask instantiation

给定目标密度：

- 通过 FAST 得到 head-wise / layer-wise `p(l,t)`；
- 对 threshold map 做比较，生成静态 runtime mask。

### 4. Resolution remapping

若部署分辨率与标定分辨率不同，则先对 threshold map 做 conservative interpolation。

### 5. crm kernel execution

静态 mask 被编码成 CRM bitmask，在线仅执行 block-sparse attention，不再做 runtime topology discovery。

这条链路说明 ScalingAttention 的重心不是单个 kernel，而是：

> 如何把可复用的结构先验编译成硬件友好的静态执行图。

这也是这篇实现最成熟的地方：WEST、FAST、crm kernel 三者各自负责一层明确问题，而不是三个孤立技巧拼在一起：

- WEST 负责结构先验；
- FAST 负责部署预算；
- crm kernel 负责执行兑现。

## 实验结果

### 质量-效率 Pareto

论文最强调的是：在相近 PSNR 下，ScalingAttention 比 SVG2 需要更低 density。Figure 1 给出结论：

- 在可比 PSNR 下，最多可比 SVG2 少用约 **2x attention FLOPs**

其中 density 定义为 active attention blocks 的比例，也就是 `1 - sparsity`。

### 端到端加速

摘要和正文给出的代表结果是：

- **最高 1.90x end-to-end speedup**

正文还提到：

- 在 HunyuanVideo 上约 `55%` global density 可达到约 `1.73x`；
- 在 Wan2.1 上也建立了新的效率-保真度 Pareto frontier。

### Kernel 效率

论文专门对 crm kernel 做了单独 benchmark，结论是：

1. **0% sparsity 的 dense setting 下，overhead 小于 10%**
2. **随着 sparsity 上升，理论 FLOPs 降幅可以稳定转成 wall-clock speedup**
3. **高 sparsity 时可达到超过 10x 的 kernel-level acceleration**

这个结果很关键，因为它说明 crm kernel 的 metadata 处理不是性能瓶颈，否则 dense setting 的额外开销会很大。

正文还给了一个更具体的数字：在长序列 `N=262,144` 时，dense setting 下额外开销约 `4.8%`。这意味着即便完全不利用 sparsity，CRM 这套表示本身也足够轻，不会把 kernel 直接拖慢到不可用。

### WEST 与 FAST 的稳定性

论文还做了两类稳定性实验：

- WEST 对 profiling prompt 数不敏感，union mask density 大约在 `20` 个 prompts 左右开始饱和；
- FAST 在固定 WEST topology 下，甚至只用一个 profiling prompt 也能得到与 aggregate reference 很接近的结果，IoU 可达 `94%+`。

这进一步支持它的核心主张：拓扑是稳定结构，敏感度调节只需轻量 profiling。

论文还给出过离线成本拆分：WEST 大约使用 `10` 个 dense calibration prompts，FAST 只需一次 dense generation pass 做 Hellinger-profile。真正重的是 topology 采样，不是后续 mask 实例化。

## 局限

- 当前未见官方开源实现，因此 kernel 细节仍是论文重建而非代码实证；
- WEST 的“拓扑稳定”假设在更大分布偏移或极端内容上仍需验证；
- crm kernel 使用大块 `128 x 128` 粒度，虽然硬件友好，但会牺牲一部分最细粒度稀疏性；
- 方法偏向固定部署场景，对完全动态长度/结构的在线适配能力弱于 SVG2/SVOO；
- 离线 pipeline 虽然可摊销，但 WEST + FAST 仍然有一次性成本。

## 关键启示

- **视频 DiT 的 sparse attention 不该只看在线激活，还应区分“稳定拓扑”和“运行时敏感度”。**
- **如果拓扑是稳定的，最合理的系统路线不是每次在线找块，而是离线抽取完整 sparsity hierarchy。**
- **真正高效的 sparse attention 需要和 GPU 元数据格式一起设计；CRM 这种 bit-wise row mask 是非常典型的硬件导向选择。**
- **ScalingAttention 的最大贡献不是某个单独技巧，而是把 topology discovery、fidelity control 和 kernel co-design 接成了一套闭环。**
