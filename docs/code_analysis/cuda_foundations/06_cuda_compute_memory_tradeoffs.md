---
tags:
  - CUDA
---
# CUDA 基础：计算、带宽与算存权衡

FA2 最值得反复回看的一个设计决策是：前向和反向都在主动用更多计算，换更少的 memory traffic。这个选择背后需要一套更稳定的判断框架：

- 当前 kernel 更像 memory-bound 还是 compute-bound
- 哪些中间结果值得存，哪些值得重算
- 为什么同一算法在不同 GPU 上会选不同 block shape

这一页专门整理这套“算存权衡”的基础视角。

## 信息源范围

本文主要依据两类本地材料交叉整理：

- 本地参考教材拆页中关于 arithmetic intensity、Roofline、memory bandwidth vs compute throughput、deep learning kernel 与 GEMM 映射的章节
- 本站已有的 [Flash Attention V2：总览](../flash_attention_v2/00_overview.md)、[Flash Attention V2：反向传播](../flash_attention_v2/04_backward_kernel.md)、[Flash Attention V2：调度与实例化](../flash_attention_v2/05_dispatch_and_instantiation.md)

本文不追求给出某块 GPU 的精确性能预测，只给出这些材料共同支持的判断框架。

## 对应 PMPP 章节

本文主要对应 PMPP 的以下内容：

- Chapter 5 `Memory architecture and data locality`
- 5.1 `Importance of memory access efficiency`
- Chapter 16 `Deep learning`
- 16.4 `Formulating a convolutional layer as GEMM`
- Chapter 22 `Advanced practices and future evolution`
- 22.3 `Memory bandwidth and compute throughput`

文中 arithmetic intensity、Roofline 直觉、带宽与算力上限这些内容直接受上述章节支撑；将这些概念用于解释 FA2 的 recomputation 和 block shape 取舍，则属于面向 FA2 的归纳。

## 1. 先用 Arithmetic Intensity 看问题方向

对应 PMPP 主题：compute to global memory access ratio / arithmetic intensity。

判断一个 kernel 的第一步，通常不是先盯某条指令，而是先看：

- 每搬运 1 字节数据，大概能做多少有效计算

这个比值常叫 arithmetic intensity，或者 compute-to-memory ratio。它决定了一个 kernel 更容易被哪类硬件上限卡住：

- intensity 低，通常更容易 memory-bound
- intensity 高，通常更有机会接近 compute throughput 上限

这个视角对 attention 很重要，因为 attention 的朴素实现天生会产生大量中间读写：

- score 矩阵
- softmax 概率矩阵
- backward 中的各类中间梯度

如果这些中间结果全都物化到 global memory，再重新读回，kernel 很容易在 arithmetic intensity 还没上去之前就先被带宽压住。

## 2. Roofline 视角：不是看“快不快”，而是看“被谁卡住”

对应 PMPP 主题：Roofline 模型、memory bandwidth 与 compute throughput 的相对位置。

教材用 Roofline 模型强调一件事：

- 一个实现再优化，也不可能同时突破带宽上限和算力上限

因此性能分析更应该问：

- 当前离哪条上限更近
- 如果我多用一点 shared memory / registers / extra compute，能否把瓶颈从一条线推向另一条线

FA2 的很多设计都可以直接放进这个框架：

- tiling 是为了提高数据复用，减少被带宽线卡住的概率
- online softmax 是为了减少中间写回，进一步降低 memory traffic
- backward recomputation 是为了避免保存完整 `P`，把一部分负担从带宽转回计算

这三者不是独立优化技巧，而是在共同推动 attention 从“高流量中间矩阵算法”转向“更片上化的融合算法”。

## 3. 为什么 recomputation 常常是赚的

PMPP 原书不直接讨论 attention backward recomputation；这一节是用 PMPP 的带宽-算力权衡框架来解释 FA2 的设计选择。

FA2 反向最关键的权衡，是前向不保存完整 `P`，而在 backward 里从 `Q / K / LSE` 重新计算 `S -> P`。

从纯操作数的角度看，这显然增加了计算；但从系统角度看，它减少了更昂贵的东西：

- 大规模中间矩阵的 global memory 存储
- backward 再次读取这些中间矩阵的带宽开销

所以 recomputation 并不是“用更笨的方式省内存”，而是建立在一个重要判断上：

- 对现代 GPU 而言，额外做一部分片上计算，常常比把大块中间结果搬出片上再搬回来更便宜

这正是典型的 compute-for-memory tradeoff。

## 4. attention、GEMM、convolution 在这里的共同点

对应 PMPP 主题：将深度学习算子尤其是 convolution 放回 GEMM 视角理解；attention 与之的类比是结合 FA2 做的延伸说明。

教材在 deep learning 章节里强调，卷积常可重写为 GEMM。这件事对理解 FA2 很有启发，因为三者背后都在做类似权衡：

- 通过分块把大问题拆成片上可处理的小块
- 让输入 tile 尽量多次复用
- 让中间值尽量在局部消费，而不是早早落回 global memory

因此把 FA2 理解成“注意力特有技巧堆叠”并不完整。更准确的说法是：

- FA2 继承了高性能 GEMM / conv kernel 的片上复用思想
- 又额外处理了 softmax、mask、recomputation 这类 attention 特有结构

这也是为什么 FA2 的实现风格会天然靠近 CUTLASS / CuTe 的 tiled MMA 体系。

## 5. 为什么 block shape 会跟着硬件变

对应 PMPP 主题：memory bandwidth / compute throughput / 片上资源共同塑造性能上限；具体 block shape 分析则来自 FA2 实现。

如果一个 kernel 的性能真的只由数学公式决定，那么不同 GPU 上最佳 block shape 不该差那么多。但 FA2 恰恰相反：`head_dim`、架构、shared memory 上限、CTA/SM 并发数都会改写最佳配置。

原因就在于 block shape 同时影响两组东西：

- 复用与计算侧：tile 越大，通常复用越强，MMA 吞吐更容易拉高
- 资源与带宽侧：tile 越大，smem / registers 压力越高，CTA 并发可能下降

所以你在 `05_dispatch_and_instantiation.md` 里看到的那些映射规则，本质上是在不同硬件资源边界下重新找平衡点，而不是“某个固定尺寸天生最好”。

## 6. 什么时候该用“更多计算”，什么时候不该

本节是基于 PMPP 的瓶颈判断框架做的经验化整理，不是原书逐条规则。

不是所有增加计算的做法都值得。更稳的判断方式是看它是否同时满足两点：

- 额外计算主要发生在片上，能够被已有数据复用支撑
- 它换掉的是更昂贵的 global memory 流量或更重的中间存储

对 FA2 而言，下面这些通常是值得的：

- 额外的 row-wise rescale 和 normalization 计算
- backward 中重算 `S / P`
- 为了更好的数据局部性而做的布局变换、局部归约

而如果额外计算并没有换掉明显的带宽开销，只是让每个 CTA 更重、更难并发，那么收益就不一定成立。

## 7. 把这套权衡代回 FA2 的几个关键设计

本节是面向 FA2 的应用性总结，不属于 PMPP 原书内容本身。

### 7.1 前向：不物化完整 attention matrix

这一步直接减少了全局内存占用和中间读写，是 FA2 最核心的 IO 优化。

### 7.2 backward：recompute `S -> P`

这一步把 `O(N^2)` 中间状态的存储压力换成局部重算，是最典型的算换存。

### 7.3 launch 选择：宁可更小 tile 换 2 CTA/SM

这不是对 occupancy 的盲目崇拜，而是在某些硬件条件下承认：

- 如果 CTA 太重，stall 无法被隐藏
- 适度缩小 tile 带来的并发提升，可能更能接近整体吞吐平衡

### 7.4 大 `head_dim` 配置更容易受资源约束

因为 `head_dim` 上去以后，往往会同时推高：

- shared memory tile 大小
- register fragment 体积
- copy / transpose / accumulation 的中间状态

于是 kernel 更需要在“更大 tile 的复用收益”和“更少 CTA 并发的代价”之间做取舍。

## 8. 读 FA2 时最有用的几个判断问题

本节为阅读提示，属于基于 PMPP 框架整理出的使用方法。

- 这一段优化是在减少 global memory 流量，还是只是在增加本地计算
- 这份中间结果如果存下来，代价主要来自容量还是带宽
- 这个 block shape 调整，是在提高复用，还是在提高可并发 warp 数
- 当前实现更像被 memory bandwidth 卡住，还是被片上资源卡住
- 多做这一轮计算，是否真的换掉了更贵的一轮 memory round-trip

如果能稳定地用这些问题审视 FA2，前向的 online softmax、反向的 recomputation、launch 的 block shape 分支就会统一落到同一个框架里：它们都在做算力、带宽和片上资源之间的再分配。
