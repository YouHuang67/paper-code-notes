---
tags:
  - CUDA
---
# CUDA 基础：分块、数据搬运与局部性

FA2 前向的第一原则不是“把 attention 算出来”，而是“尽量不要把本可在片上完成的中间数据写回 global memory”。这篇附录专门整理这个问题背后的几个基础概念：

- arithmetic intensity 为什么决定 kernel 更像 memory-bound 还是 compute-bound
- tiling 为什么会同时改变数据复用和 global memory traffic
- coalescing、predication、shared memory 布局为什么总是一起出现

## 信息源范围

本文主要依据两类本地材料交叉整理：

- 本地参考教材拆页中关于 memory efficiency、tiling、tiled matrix multiplication、memory coalescing、thread coarsening 的章节
- 本站已有的 [Flash Attention V2：总览](../flash_attention_v2/00_overview.md)、[Flash Attention V2：前向核心](../flash_attention_v2/02_forward_kernel.md)、[Flash Attention V2：参数结构与硬件配置](../flash_attention_v2/01_params_and_traits.md)

本文不会把某个具体实现技巧写成“唯一正确做法”，只讨论这些材料里反复出现的结构性原因。

## 对应 PMPP 章节

本文主要对应 PMPP 的以下内容：

- Chapter 5 `Memory architecture and data locality`
- 5.1 `Importance of memory access efficiency`
- 5.3 `Tiling for reduced memory traffic`
- 5.4 `A tiled matrix multiplication kernel`
- 5.5 `Boundary checks`
- 5.6 `Impact of memory usage on occupancy`
- Chapter 6 `Performance considerations`
- 6.1 `Memory coalescing`
- 6.2 `Hiding memory latency`
- 6.3 `Thread coarsening`

文中对 FA2 数据流、`GmemTiledCopy*`、Swizzle 和片上消费的解释，是基于上述章节概念对 FA2 的重组说明，不是对教材原文的逐段翻译。

## 1. 为什么先看数据搬运，而不是先看算式

对应 PMPP 主题：memory access efficiency、compute to global memory access ratio、arithmetic intensity。

教材在矩阵乘法的例子里给出一个很直观的算账方式：如果每次循环做 1 次乘法和 1 次加法，但要从 global memory 取两个 `float`，那么计算与 global memory 访问比只有 `2 FLOP / 8 B = 0.25 FLOP/B`。

这个例子的重要性不在于矩阵乘法本身，而在于它说明了一件事：

- 一个 kernel 即使算式不复杂，也可能被数据搬运彻底限制

FA2 的 exact attention 同样如此。如果直接物化完整的 score / probability 矩阵，再来回读写，global memory traffic 会迅速压过算术本身。因此 FA2 的核心设计从一开始就是：

- 让 Q/K/V 尽可能在片上重用
- 让 softmax 状态按块增量维护
- 避免把本可局部消费的中间结果写回再读回

## 2. Tiling 的本质：用片上复用换 global traffic

对应 PMPP 主题：tiling for reduced memory traffic。

tiling 可以理解成两个动作同时发生：

- 把输出空间切成 CTA 可独立负责的局部块
- 把输入数据按这些局部块搬到 shared memory 或寄存器里重复使用

以 FA2 前向为例，每个 CTA 固定一个 Q 行块，然后依次扫过多个 KV 块。这样做以后：

- Q 行块只需加载一次，就能和多个 K 块做 `QK^T`
- 某个 K/V 块一旦进入片上，就能在当前 CTA 内被后续计算连续消费
- softmax 的 `row_max / row_sum / acc_o` 也和这个 Q 行块绑定，可以局部累计

如果不这样做，Q、K、V 和中间 score 将频繁往返 global memory，注意力就会更像一个“访存驱动”的算法，而不是“片上融合”的算法。

## 3. 从 GEMM 视角看 FA2 的两次分块乘法

本节属于“把 PMPP 的 tiled matrix multiplication 视角代回 FA2”的归纳解释。

FA2 的前向虽然不是普通 GEMM，但它内部确实有两个强烈的 GEMM 子问题：

- `Q @ K^T`
- `P @ V`

把它们放回 tiled matrix multiplication 的视角，最重要的观察是：

- 第一段乘法的产物 `S` 不再完整写回 global memory，而是立刻进入 mask + online softmax
- 第二段乘法直接消费归一化后的局部概率块，继续在片上更新 `acc_o`

也就是说，FA2 并不是“两个 GEMM 再加几个后处理”，而是：

- 用 GEMM 的分块与复用结构
- 把本来会出现在两个 GEMM 之间的大量中间读写消掉

这正是 tiled kernel 与融合 kernel 联合设计的地方。

## 4. Shared Memory 的角色：不是缓存装饰，而是算法骨架

对应 PMPP 主题：shared memory 作为数据局部性与 tiled kernel 复用的核心机制；“算法骨架”这一表述是面向 FA2 的总结。

在教程式示例里，shared memory 常被解释成“手工管理的缓存”。这个说法没错，但放到 FA2 还不够。

对 FA2 来说，shared memory 至少承担三类职责：

- 暂存 Q/K/V tile，减少 global memory 重复加载
- 重新排布 tile，使其满足后续 `ldmatrix` / MMA / copy 的访问形状
- 在前向和反向不同阶段复用同一块片上空间

因此，shared memory 的布局设计不是附属优化，而是 kernel 结构的一部分。你在 `kernel_traits` 里看到的 `SmemLayoutQ / SmemLayoutKV / SmemLayoutPdS / Swizzle`，本质上都在回答同一个问题：

- 数据进入片上以后，怎样排布才能让下一步访问更便宜

## 5. Coalescing：减少“把数据搬上来”这一步的浪费

对应 PMPP 主题：memory coalescing。

教材把 memory coalescing 和 tiling 放在一起讲，是因为二者互补：

- tiling 解决“同一份数据能否多次复用”
- coalescing 解决“第一次把数据搬进来时浪费有多大”

对 warp 来说，更理想的 global memory 访问通常意味着：

- 相邻线程访问相邻地址
- 尽量以对齐、连续、批量的方式加载

放到 FA2 的代码里，对应的就是那些 `GmemTiledCopy*`、vectorized load、按线程切开的 `partition_S / partition_D`。这些代码看上去模板味很重，但背后的目标非常朴素：

- 让一次 warp 级加载尽量覆盖连续数据，而不是零散抓取

如果 tile 本身复用很好，但每次加载都不 coalesced，那么 global memory 带宽仍会被低效消耗。

## 6. Boundary Check 和 Predication 为什么总是跟着 Tiling 走

对应 PMPP 主题：boundary checks；对 causal/local mask 的展开是结合 FA2 做的补充。

一旦用 tile 覆盖任意长度输入，就不可避免会碰到边界块。教材在 tiled matrix multiplication 一章里把 boundary checks 单独拉出来，是因为这不是实现细枝末节，而是 tiled kernel 的常规成本。

FA2 里的边界来源至少有三类：

- `seqlen_q / seqlen_k` 不是 block 大小整数倍
- `head_dim` 不是内部 copy / MMA 粒度整数倍
- causal / local mask 让逻辑有效区域本身变成不规则形状

因此 FA2 里大量 `Is_even_MN / Is_even_K`、predicate tensor、masked copy，并不是“特殊情况补丁”，而是 tiled attention 的正常组成部分。

## 7. Thread Coarsening：每线程多做一点，换掉部分调度与访存开销

对应 PMPP 主题：thread coarsening。

教材把 thread coarsening 放在 performance 章节里，是因为它本质上是一种资源交换：

- 让每个线程负责更多元素
- 用更高的寄存器占用，换更少的控制与访存开销

在 FA2 里，这个思想也很常见。一个线程通常不会只碰一个标量，而是会持有：

- 若干 fragment 元素
- 若干行状态
- 若干 copy 临时值

这么做的收益是 CTA 内的协作粒度和 MMA 粒度可以更贴近硬件习惯；代价是寄存器压力会上升。它和 tiling 一样，不是“要不要用”的问题，而是“用到什么度更平衡”的问题。

## 8. 用这套视角回看 FA2 前向

本节是阅读 FA2 前向时的应用性总结，不属于 PMPP 原书内容本身。

如果把 FA2 前向压缩成一条数据流：

`global Q/K/V -> shared memory tile -> register fragment -> MMA -> online normalization -> acc_o -> output`

那它和朴素 attention 的最大差别不是数学公式，而是：

- 大部分中间结果都被设计成“在局部产生，在局部消费”

因此读 `02_forward_kernel.md` 时，最值得盯的不是每条模板语句，而是下面这几个问题：

- 当前这块数据为何要先落到 shared memory
- 这一步是为了提高复用，还是为了改变访问排布
- 这个 copy 是在优化 global memory transaction，还是在适配后续 MMA 布局
- 这个 predicate 是为了边界正确性，还是为了避免无效访存

把这些问题答出来，FA2 的前向就会从“模板实现细节”重新变回“分块数据流设计”。
