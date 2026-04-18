---
tags:
  - CUDA
---
# CUDA 基础：归约、Scan 与在线归一化

FA2 里最容易被低估的一部分，不是 Tensor Core GEMM，而是围绕 softmax 展开的几类并行模式：

- row-wise max / sum
- warp 内 allreduce
- `dO · O` 这类点积归约
- 不物化完整中间矩阵的在线归一化

这一页把这些模式放回 reduction 与 scan 的通用视角里整理。

## 信息源范围

本文主要依据两类本地材料交叉整理：

- 本地参考教材拆页中关于 reduction tree、minimizing divergence、hierarchical reduction、scan work efficiency、single-pass scan 的章节
- 本站已有的 [Flash Attention V2：计算原语](../flash_attention_v2/03_softmax_mask_utils.md) 与 [Flash Attention V2：反向传播](../flash_attention_v2/04_backward_kernel.md)

本文只写这些材料能够稳定支撑的结构性结论，不把某个实现细节的偶然写法扩展成一般规律。

## 对应 PMPP 章节

本文主要对应 PMPP 的以下内容：

- Chapter 10 `Reduction`
- 10.2 `Reduction trees`
- 10.4 `Minimizing control divergence`
- 10.5 `Minimizing memory divergence`
- 10.7 `Hierarchical reduction for arbitrary input length`
- Chapter 11 `Prefix sum (scan)`
- 11.3 `Speed and work efficiency consideration`
- 11.7 `Single-pass scan for memory access efficiency`

需要单独说明的是：PMPP 并不讲 online softmax。本页关于 online normalization 的内容，是把上述 reduction / scan 的并行模式拿来解释 FA2 的 softmax 实现，因此属于“基于 PMPP 概念框架的归纳”，不是教材原文摘要。

## 1. Reduction：把“很多元素变成一个摘要值”

对应 PMPP 主题：reduction 的抽象、reduction tree。

reduction 的抽象很简单：给定一个满足结合律的运算，把一组值压成一个结果。例如：

- sum reduction
- max reduction
- min reduction
- 点积中的局部求和

教材会从 reduction tree 讲起，是因为这能直接说明并行化的核心：

- 第一轮并行做局部合并
- 下一轮继续合并局部结果
- 最终得到全局摘要值

这个视角放到 FA2 上非常自然：

- `row_max` 是 max reduction
- `row_sum` 是 sum reduction
- `D_i = rowsum(dO_i \odot O_i)` 是逐行 sum reduction

也就是说，softmax 和 backward 里的这些状态，并不是零散小工具，而是一组“按行做 reduction”的变体。

## 2. 为什么教材会单独讨论 divergence

对应 PMPP 主题：minimizing control divergence、minimizing memory divergence。

reduction 看似规则，但高效实现并不自动成立。教材把控制分歧和内存分歧单独拿出来讲，原因在于：

- 线程做的工作量若不均衡，会产生控制分歧
- 中间数据若访问方式零散，会产生内存分歧

对 FA2 而言，这两个问题都真实存在：

- mask 后有效元素可能不规则
- 不同行块和列块的有效范围不同
- softmax / backward 里的行级归约要在 fragment 布局上重新解释成“按行”

因此 FA2 的做法不是追求一个“全局统一大归约”，而是拆成层次化协作：

- 先在线程本地做一轮局部归约
- 再在小范围线程组内做 allreduce
- 只在必要时做更大范围的同步

## 3. Hierarchical Reduction：先局部，再全局

对应 PMPP 主题：hierarchical reduction。

教材在 reduction 章节中强调 hierarchical reduction，是因为大输入通常不适合一次性由所有线程直接参与同一轮归约。更常见的结构是：

- 每线程先处理自己的一小块输入
- CTA 内做局部归约
- 若输入超过单 CTA 覆盖范围，再用多阶段方式合并

FA2 的 softmax 也有同样的层次感，只不过它的“层次”不是跨整段序列的统一归约，而是：

- 线程先对自己持有的 fragment 行内元素做归约
- 再通过 warp / quad 范围的 allreduce 合并
- 随着 K 块迭代推进，不断把局部统计量并入全局行状态

因此，FA2 的 online softmax 可以看成：

- 一个“分块输入 + 局部摘要 + 增量合并”的归约框架

## 4. Scan 的价值：不只是前缀和，而是“工作效率”视角

对应 PMPP 主题：scan 的 work efficiency、single-pass scan 的内存访问效率。

scan 章节最有价值的不只是 Kogge-Stone 或 Brent-Kung 这些具体算法，而是它强调了一个更广泛的问题：

- 并行算法不仅要看 step 数，还要看总工作量是否被过度放大

这对理解 FA2 的 online softmax 很重要。标准 softmax 若直接按“先全局求 max，再全局求 sum，再归一化”去做，会天然偏向多次读写完整中间结果。FA2 改成 online 形式以后，本质上是在做：

- 随着 K 块推进，维护一份足够的摘要状态
- 避免为得到最终结果而反复扫描、反复写回同一份大矩阵

这和 scan 章节讨论的 work efficiency 是同一类思路：不要为了并行而把本来可以融合的工作拆成过多轮次。

## 5. Online Softmax：把归约和状态递推绑在一起

本节不来自 PMPP 原书，而是把 PMPP 的 reduction / work-efficiency 视角用于解释 FA2 的 online softmax。

FA2 的 online softmax 可以用两层视角来看。

第一层是 reduction：

- 当前块先求每行局部最大值
- 当前块再求按新最大值重标定后的局部指数和

第二层是状态递推：

- 旧的 `row_max / row_sum / acc_o` 不是丢掉重算
- 而是按新的最大值重标定后，与当前块贡献合并

因此 online softmax 不是简单的“边算边做 softmax”，而是：

- reduction 负责为每个块提炼摘要
- 递推公式负责把不同块的摘要拼成全局正确结果

这也是为什么 FA2 里会同时看到：

- 线程内 reduce
- 小范围 allreduce
- `exp2` 重标定
- `acc_o` 的同步 rescale

这几步不是分离功能，而是同一套在线归一化逻辑的不同截面。

## 6. 为什么 FA2 的归约常常止步于 warp 或更小的线程组

本节属于结合 FA2 线程映射做的解释性归纳。

在很多 attention kernel 里，逐行统计量最终并不需要整个 CTA 所有线程都参与一次“大一统 allreduce”。更常见的做法是：

- 让同一逻辑行对应的小线程组完成本行归约
- 让不同逻辑行并行推进

这样做的好处很直接：

- 减少全 CTA 同步开销
- 保持不同逻辑行的并行独立性
- 让归约形状更贴合 MMA fragment 的实际分布

所以你在 FA2 里看到 `quad_allreduce_` 或类似 warp 子组协作时，最好把它理解成：

- 不是“只做了半个归约”
- 而是“只在真正需要共享这份行状态的线程组里归约”

## 7. 反向里的 `dO · O`：又一个按行 reduction

本节是把 PMPP 的 reduction 视角应用到 FA2 backward 预处理上。

反向预处理中的 `D_i = rowsum(dO_i \odot O_i)` 很适合拿来验证这套视角。

它并不复杂，但很典型：

- 输入是逐行向量
- 每线程先做局部乘加
- 再在负责同一行的线程组里做求和归约
- 结果写成后续 `dS = P \odot (dP - D)` 需要的行级摘要

从模式上说，它和前向的 `row_sum / row_max` 是同一家族问题。差别只是：

- 前向是 max + sum 的在线组合
- 这里是点积求和

## 8. 把 reduction / scan 视角代回 FA2

本节为阅读提示，属于基于 PMPP 框架整理出的使用方法。

带着这一页再看 FA2 的 `03_softmax_mask_utils.md` 和 `04_backward_kernel.md`，很多代码会变得更容易归类：

- `thread_reduce_` 是线程本地 reduction
- `quad_allreduce_` 是小线程组归约
- `softmax_rescale_o` 是摘要状态的增量合并
- `normalize_softmax_lse` 是最终的归一化与摘要落盘
- `dot_do_o` 是逐行点积归约

因此，读 FA2 相关代码时更好的问题不是“这一段在做哪一个 helper”，而是：

- 这里的逻辑摘要值是什么
- 这份摘要值在哪个线程层次上被共享
- 当前实现是在减少同步，还是在减少中间写回
- 这是一次普通归约，还是一次带状态递推的在线归一化

这样读，softmax 那几百行代码会从“繁琐的细节实现”重新变成“几种并行模式的组合”。
