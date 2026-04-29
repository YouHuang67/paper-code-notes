---
tags:
  - CUDA
  - LLM Inference
---
# CUDA 基础：块量化与低精度 GEMM

本文补齐 DeepSeek V4 代码分析中最关键的低精度背景：为什么输入激活按块量化、为什么权重和激活可以用不同精度、以及 kernel 里如何把 scale 和低精度 GEMM 重新拼回真实矩阵乘。

直接关联文档：

- [DeepSeek V4：模型架构](../deepseek_v4/01_model_architecture.md)
- [DeepSeek V4：Kernels 与量化](../deepseek_v4/03_kernels_and_quantization.md)
- [CUDA 基础：TileLang 编程模型](07_tilelang_programming_model.md)

## 1. 为什么是块量化而不是整张量量化

对一个张量 `x`，如果只给整个张量一个 scale：

$$
q = \mathrm{clip}\left(\frac{x}{s}, q_{\min}, q_{\max}\right)
$$

那么 `s` 必须同时适配所有行、所有列，通常会被少量大值主导，导致大部分元素有效精度不足。

块量化的思路是把最后一维切成小块，对每个块单独取 scale：

$$
s_{g} = \frac{\max_{i \in g}|x_i|}{q_{\max}}, \qquad
q_i = \mathrm{clip}\left(\frac{x_i}{s_g}, q_{\min}, q_{\max}\right)
$$

这样做的效果是：

- scale 更贴近局部动态范围
- 量化误差显著更小
- 代价只是额外存储一组小 scale

DeepSeek V4 里：

- 激活 FP8 量化按 `128` 元素一组
- 权重 FP4 量化按 `32` 元素一组

## 2. DeepSeek V4 的两类 scale

在 `model.py` 和 `kernel.py` 里，scale 不是抽象概念，而是显式张量。原始实现见：

- [model.py:L108](../deepseek_v4/src/model_py.md#__codelineno-0-108)
- [model.py:L123](../deepseek_v4/src/model_py.md#__codelineno-0-123)
- [kernel.py:L105](../deepseek_v4/src/kernel_py.md#__codelineno-0-105)
- [kernel.py:L186](../deepseek_v4/src/kernel_py.md#__codelineno-0-186)

### 激活 FP8

若输入 `x` 的最后一维长度为 `K`，block size 为 `128`，则：

- 量化后张量 `x_q` 形状仍是 `[..., K]`
- scale 张量 `s_x` 形状是 `[..., K / 128]`

即每 `128` 个连续元素共享一个 scale。

### 权重 FP4

对于逻辑形状为 `[N, K]` 的权重：

- 物理存储是 `[N, K/2]`，因为一个 `float4_e2m1fn_x2` 单元打包 2 个 FP4
- scale 张量形状是 `[N, K/32]`

所以它比 FP8 更“细块”，因为 FP4 可表示范围更小，必须用更小分组控制误差。

## 3. `act_quant`：输入激活怎么变成 FP8

DeepSeek V4 的 `act_quant` 对应以下代码路径：

- 包装函数：[kernel.py:L105-L125](../deepseek_v4/src/kernel_py.md#__codelineno-0-105)
- kernel 主体：[kernel.py:L40-L102](../deepseek_v4/src/kernel_py.md#__codelineno-0-40)

其数学过程是：

$$
a_g = \max\left(\max_{i \in g}|x_i|, \epsilon\right)
$$

$$
s_g =
\begin{cases}
2^{\lceil \log_2(a_g / q_{\max}) \rceil}, & \text{若启用 power-of-2 rounding} \\
a_g / q_{\max}, & \text{否则}
\end{cases}
$$

$$
q_i = \mathrm{clip}(x_i / s_g, -448, 448)
$$

其中 `-448 ~ 448` 对应 FP8 `e4m3` 的可用范围。

原始代码的关键结构如下：

```python
for _ in T.Pipelined(1, num_stages=num_stages):
    T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)  # 取一个 [32, 128] tile
    T.copy(x_shared, x_local)                               # shared -> fragment
    T.reduce_absmax(x_local, amax_local, dim=1)            # 每行各求一个 amax

    for i in T.Parallel(blk_m):
        amax_local[i] = T.max(amax_local[i], 1e-4)         # 防止 scale 过小
        if round_scale:
            s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
        else:
            s_local[i] = amax_local[i] * fp8_max_inv

    '''
    inplace=False:
        只输出量化值 q 和 scale

    inplace=True:
        先量化到低精度，再乘回 scale，等价于做一次 QDQ（quant-dequant）
        这不是为了省事，而是为了在 bf16 主图里模拟训练/校准时的量化误差
    '''
    ...
```

这里最关键的工程判断是：

- **真正做 GEMM 时**，要返回 `(q, s)`
- **只做量化误差模拟时**，直接 `inplace=True` 写回 BF16

后者在 DeepSeek V4 中大量用于“保持主图 BF16、但对非 RoPE 维度注入低精度误差”。

## 4. `fp4_act_quant`：为什么 indexer / compressed KV 用 FP4 模拟

相关实现：

- [kernel.py:L128-L200](../deepseek_v4/src/kernel_py.md#__codelineno-0-128)

与 FP8 相比，FP4 的差异有两点：

- 数值范围更小，代码中用 `fp4_max = 6.0`
- scale 始终走 power-of-2 rounding

对应公式：

$$
s_g = 2^{\lceil \log_2(a_g / 6) \rceil}
$$

$$
q_i = \mathrm{clip}(x_i / s_g, -6, 6)
$$

DeepSeek V4 里它主要出现在两个地方：

- `Indexer` 的 query / compressed kv 打分路径
- `Compressor(rotate=True)` 的旋转后压缩缓存路径

也就是说，作者在“索引 / 检索子系统”上刻意采用了更激进的低精度模拟。

## 5. 低精度 GEMM 不是“先反量化再普通 GEMM”

这是理解 kernel 的核心。

如果把块量化写开，矩阵乘其实是：

$$
A \approx \hat{A} \odot S_A,\qquad
B \approx \hat{B} \odot S_B
$$

其中 `S_A` / `S_B` 不是与矩阵同形，而是按块广播。

于是：

$$
C = AB^\top \approx \sum_{g}
(\hat{A}_g \hat{B}_g^\top) \cdot (s_{A,g} s_{B,g})
$$

这意味着真正高效的实现不是：

1. 先把整个 `A`、`B` 完全反量化回 BF16
2. 再做一次普通 GEMM

而是：

1. 直接对量化 tile 做低精度 MMA
2. 对每个 K 子块的部分和乘上对应 scale
3. 在高精度累加器里把所有子块加起来

DeepSeek V4 的 `fp8_gemm_kernel` / `fp4_gemm_kernel` 正是这样做的。

## 6. `fp8_gemm_kernel`：每个 K 子块单独做 scale correction

相关实现：

- [kernel.py:L203-L273](../deepseek_v4/src/kernel_py.md#__codelineno-0-203)

核心思路可直接看这段：

```python
for k in T.Pipelined(K_iters, num_stages=4):
    T.copy(A[by * block_M, k * block_K], A_shared)                 # 载入 A 的一个 K tile
    T.copy(B[bx * block_N, k * block_K], B_shared)                 # 载入 B 的一个 K tile

    Scale_B = T.Cast(FP32, scales_b[bx * block_N // group_size, k])# 这个 N tile 对应的 B scale
    for i in T.Parallel(block_M):
        Scale_C_shared[i] = T.Cast(FP32, scales_a[by * block_M + i, k]) * Scale_B

    T.gemm(A_shared, B_shared, C_local, transpose_B=True)          # 量化 tile 上直接做 GEMM

    for i, j in T.Parallel(block_M, block_N):
        C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]   # 每个 K tile 单独乘 scale 后累加
```

这段代码可以翻译成数学语言：

- `C_local` 是第 `k` 个 `K` 子块的低精度乘法结果
- `Scale_C_shared[i]` 是该子块上第 `i` 行激活 scale 与该 `N` tile 对应权重 scale 的乘积
- `C_local_accum` 是最终 FP32 累加器

所以 scale correction 不是后处理细节，而是 GEMM 主循环的一部分。

## 7. `fp4_gemm_kernel`：为什么先把 FP4 提升成 FP8

相关实现：

- [kernel.py:L441-L536](../deepseek_v4/src/kernel_py.md#__codelineno-0-441)

DeepSeek V4 这里没有单独写“FP8 x FP4 Tensor Core kernel”，而是采用：

1. 从全局内存读 FP4 weight tile
2. 在 shared memory / fragment 中把它 cast 成 FP8
3. 再复用 FP8 风格的 GEMM 主体

关键代码如下：

```python
T.copy(B[bx * block_N, k * block_K], B_fp4_shared)                # 先读 FP4 tile

'''
先把 FP4 tile 提升成 FP8 tile。
这样后续 GEMM 主体只需要处理 A_shared(FP8) x B_shared(FP8)，
避免再单独维护一条“FP8 act x FP4 weight”的 MMA 实现路径。
'''
for i, j in T.Parallel(block_N, block_K):
    B_shared[i, j] = T.Cast(FP8, T.Cast(FP32, B_fp4_shared[i, j]))

for i in T.Parallel(block_N):
    scale_b_frag[i] = T.Cast(FP32, scales_b[bx * block_N + i, k])  # weight scale 按 32 元素分组

for i in T.Parallel(block_M):
    scale_a_frag[i] = T.Cast(FP32, scales_a[by * block_M + i, k // n_sub]) # act scale 按 128 元素分组

T.gemm(A_shared, B_shared, C_local, transpose_B=True)

for i, j in T.Parallel(block_M, block_N):
    C_local_accum[i, j] += C_local[i, j] * scale_a_frag[i] * scale_b_frag[j]
```

这里最值得注意的是 **scale 粒度不一致**：

- 激活 scale：每 128 个 `K` 元素一组
- 权重 scale：每 32 个 `K` 元素一组

所以代码里有：

```python
k // n_sub
```

其中 `n_sub = 128 / 32 = 4`，表示四个权重子块共享一个激活 scale。

## 8. 为什么注意力主路径经常只量化非 RoPE 维

在 `Attention.forward` 与 `Compressor.forward` 中都能看到类似逻辑：

- [model.py:L367-L372](../deepseek_v4/src/model_py.md#__codelineno-0-367)
- [model.py:L501-L506](../deepseek_v4/src/model_py.md#__codelineno-0-501)

作者只对 `[..., :-rd]` 做量化模拟，把 RoPE 维度保留为 BF16。原因很直接：

- RoPE 维度承载显式相位信息
- 长上下文下相位误差会被放大
- 因此宁可牺牲一点算力收益，也要保住位置精度

这说明 DeepSeek V4 的量化并不是“一把梭”的统一策略，而是围绕长上下文精度做过结构化裁剪。

## 9. 这些低精度设计在架构层的作用

把上面的 kernel 设计放回模型架构里，可以得到非常清晰的分工：

- 普通线性层：
  - FP8 权重 + FP8 激活量化
- 专家权重：
  - FP4 权重 + FP8 激活
- 窗口 KV 与压缩 KV：
  - 主路径多为 BF16 存储，但在若干子路径上做 QDQ 模拟
- Indexer / 压缩检索：
  - 更激进地使用 FP4 模拟

这套分工的目标不是单纯追求最小 bit 数，而是：

- 在最贵的 GEMM 路径上降低带宽与存储
- 在最敏感的长上下文位置编码路径上保住精度
- 在检索 / 压缩子系统里接受更激进的低精度近似

## 小结

读 DeepSeek V4 的量化实现时，可以抓住三句话：

1. **量化单位不是整个张量，而是最后一维上的小块**
2. **scale correction 不是收尾动作，而是 GEMM 主循环的一部分**
3. **不同子系统使用不同精度策略，本质是在做结构化精度分配**

有了这层背景，再去读：

- [DeepSeek V4：模型架构](../deepseek_v4/01_model_architecture.md)
- [DeepSeek V4：Kernels 与量化](../deepseek_v4/03_kernels_and_quantization.md)

就不会把 `act_quant`、`fp8_gemm`、`fp4_gemm` 误看成几个孤立函数。
