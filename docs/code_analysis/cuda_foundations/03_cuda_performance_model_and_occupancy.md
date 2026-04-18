---
tags:
  - CUDA
---
# CUDA 基础：性能模型与 Occupancy

本文整理阅读 FA2 这类 CUDA kernel 时最容易反复出现的四个问题：

- block / warp / SM 到底如何协作执行
- latency hiding 为什么依赖“足够多的可运行 warp”
- occupancy 真正描述的是什么
- 为什么 `kBlockM / kBlockN / kNWarps / smem / registers` 会绑在一起决定 kernel 形态

本文不复述 CuTe / CUTLASS 模板细节；相关抽象见 [CUDA 基础：CUTLASS/CuTe 编程模型](02_cuda_cutlass_cute_programming_model.md)。

## 信息源范围

本文主要依据两类本地材料交叉整理：

- 本地参考教材拆页中的 GPU 架构、block 调度、warp 调度、occupancy、memory latency 相关章节
- 本站已完成的 FA2 / CUTLASS / GEMM 笔记，尤其是 [Flash Attention V2：参数结构与硬件配置](../flash_attention_v2/01_params_and_traits.md) 与 [Flash Attention V2：调度与实例化](../flash_attention_v2/05_dispatch_and_instantiation.md)

没有在这些材料中直接看到的架构细节，本文只做结构性解释，不把经验判断写成硬结论。

## 对应 PMPP 章节

本文主要对应 PMPP 的以下内容：

- Chapter 4 `Compute architecture and scheduling`
- 4.2 `Block scheduling`
- 4.4 `Warps and SIMD hardware`
- 4.6 `Warp scheduling and latency tolerance`
- 4.7 `Resource partitioning and occupancy`

文中凡出现“放到 FA2 上看”“对 FA2 来说”这类段落，属于基于上述章节概念对 FA2 的归纳解释，而不是对教材原文的直接复述。

## 1. 从 CTA 到 SM：先看谁在和谁竞争资源

对应 PMPP 主题：block 调度、线程块与 SM 的绑定关系、block 内同步与 shared memory 的可见性。

CUDA kernel 启动后，grid 中的 block 以 CTA 为单位分配到各个 SM。一个 block 的所有线程总是在同一个 SM 上执行，这件事带来两个直接后果：

- block 内线程可以共享同一片 shared memory
- block 内线程可以通过 `__syncthreads()` 做屏障同步

但 block 之间没有这样的同步和共享机制。因此，kernel 的第一层设计问题通常不是“单个线程做什么”，而是“一个 CTA 负责哪一块输出，在哪个层次上复用数据”。

FA2 前向就是一个典型例子：每个 CTA 负责一个 Q 的行块，然后在这个 CTA 内循环遍历所有 K/V 块。这样做的本质不是代码组织偏好，而是：

- Q 这块数据可以在 CTA 内反复复用
- Online Softmax 的 `row_max / row_sum / acc_o` 状态可以局部保存
- 只有 CTA 内需要同步，避免跨 CTA 协调

## 2. Warp 调度与 Latency Hiding

对应 PMPP 主题：warp scheduling、latency tolerance。

warp 是 GPU 实际发射指令的基本单位。单个 warp 遇到长延迟操作时，例如 global memory 访问，SM 不会停下来等它，而是尽量切换到同一个 SM 上其他“已经准备好”的 warp。

所以 latency hiding 的核心不是“把一次内存访问变快”，而是：

- 让一次访问期间还有别的 warp 可以执行
- 把等待时间掩盖在别的 warp 的算术或访存工作里

这也是为什么教材会把 warp scheduling 和 occupancy 放在一起讲。两者的关系可以压缩成一句话：

- 如果某个 SM 上常驻的可运行 warp 太少，memory latency 就更容易直接暴露成 stall

但这里要注意一个常见误区：更多 warp 不是无条件更好。若为了多放几个 CTA 而把 tile 缩得过小，导致数据复用下降、global traffic 上升，最终吞吐反而可能更差。

## 3. Occupancy 是什么，不是什么

对应 PMPP 主题：resource partitioning and occupancy。

occupancy 描述的是一个 SM 上实际常驻 warp 数，相对于该 SM 可支持最大常驻 warp 数的比例。它反映的是“调度器手里有多少 warp 可以切换”，而不是“这个 kernel 一定跑得快”。

影响 occupancy 的常见约束有四类：

- 每个 CTA 的线程数
- 每个 CTA 的 shared memory 用量
- 每个线程的寄存器用量
- 架构本身对每个 SM 可同时驻留 CTA / warp / thread 的上限

在真实 kernel 里，往往不是其中一个因素单独决定结果，而是几个约束一起生效。对 FA2 来说尤其明显：

- `kBlockM / kBlockN` 变大，通常意味着 CTA 处理的数据块更大，复用更强
- 但更大的 tile 往往又需要更多 shared memory 存 Q/K/V/P/dS 等中间块
- 同时每线程 fragment 变大，寄存器压力也会上升
- 最终同一个 SM 能并发的 CTA 数下降，occupancy 可能降低

因此，occupancy 更适合当“约束诊断指标”，而不是单独的优化目标。

## 4. 资源分配如何塑造 FA2 的 kernel 形状

这一节开始明显进入“用 PMPP 的资源模型解释 FA2 配置”的部分。资源类别本身来自 PMPP，但 `kBlockM / kBlockN / kNWarps` 的落地分析来自对 FA2 实现的归纳。

读 FA2 的 `kernel_traits` 和 launch 配置时，可以把它理解成一个资源分配问题。

### 4.1 `kBlockM / kBlockN`

它们首先定义了一个 CTA 处理的 Q 行块和 KV 列块大小。这个选择同时影响：

- 一次 CTA 内可复用多少 Q/K/V 数据
- Tensor Core GEMM 的 tile 形状
- shared memory 的总占用
- 边界 predication 的复杂度

### 4.2 `kNWarps`

它定义了一个 CTA 里有多少 warp 参与协作。更多 warp 的潜在收益是：

- 更大的并行覆盖面
- 更高的访存和 MMA 吞吐

代价则是：

- CTA 线程数上升
- register 和 shared memory 的线程级分摊方式变化
- 更容易碰到“一个 CTA 太重，导致每个 SM 只能放很少 CTA”的情况

### 4.3 shared memory

FA2 的 shared memory 不是可有可无的缓存，而是算法结构本身的一部分：

- Q/K/V tile 要先落到片上
- 某些中间块要在片上布局成适合后续 MMA / copy 的形式
- 前向和反向里还有明显的内存复用与 buffer 复用

因此 shared memory 不只是“占了多少 KB”的问题，而是直接决定“这个 CTA 能不能长成这种形状”。

### 4.4 register

寄存器压力在 FA2 里主要来自三类数据：

- MMA accumulator fragment
- softmax / normalization 的行状态
- 每线程负责的加载、转置、临时累积数据

当 `head_dim`、warp 数、双缓冲层数上去以后，寄存器经常和 shared memory 一起成为 CTA 并发数的上限因素。

## 5. 为什么“更高 occupancy”不等于“更高性能”

对应 PMPP 主题：识别主导瓶颈、通过资源交换缓解约束；这里的两种情形是结合 FA2 配置做的总结。

教材里反复强调：性能调优的关键是识别主导瓶颈，再看资源交换是否真的缓解了它。这个判断放到 FA2 上尤其重要。

下面两种情形都很常见：

- 某个配置让 occupancy 变高了，但 tile 变小，global memory traffic 上升，总性能下降
- 某个配置让 occupancy 下降了，但数据复用大幅提升、Tensor Core 利用率更高，总性能反而上升

所以更合理的问题不是“occupancy 要不要尽量拉满”，而是：

- 当前 kernel 更像是被 memory traffic 卡住，还是被算力利用不足卡住
- 增大 tile 带来的复用收益，能否覆盖 occupancy 下降的代价
- 减小 tile 换来更多 CTA/SM，能否覆盖额外的 global memory 往返

## 6. 把这套模型代回 FA2

本节是面向 FA2 的解释性回扣，不是 PMPP 原书内容本身。

带着前面的视角，再看 FA2 的 launch 选择会更自然：

- `d=128` 和 `d=256` 的配置不同，不只是模板枚举问题，而是 shared memory / register / CTA 并发约束不同
- `SM80` 与其他 `SM8x` 变体选不同 block 形状，本质上是在不同共享内存容量和并发空间下换取更好的平衡
- 某些配置刻意追求 `2 CTA / SM`，不是因为这是绝对最佳值，而是因为它在当时那组资源约束下更接近吞吐平衡点

这也解释了为什么 `05_dispatch_and_instantiation.md` 里会把 block shape 选择写成一连串硬件条件分支：这不是“代码写复杂了”，而是 kernel 性能本来就被这些资源约束塑形。

## 7. 读 FA2 时建议盯住的判断问题

本节为阅读提示，属于基于 PMPP 框架整理出的使用方法。

- 这个 CTA 的输出责任边界是什么，为什么这样切
- 这个阶段的 shared memory 到底在复用什么数据
- 当前配置首先撞上的约束更像是 smem、register 还是线程数
- 这个参数变化是在换取更多复用，还是换取更多并发 warp
- 当前优化是在减少 stall，还是在提高每次数据搬运后的有效计算量

如果这些问题能回答清楚，FA2 里大多数 `kernel_traits`、launch 分支和 block shape 选择都会变成“资源分配的自然结果”，而不是一堆孤立的经验参数。
