---
tags:
  - LLM Inference
  - Sparse Attention
  - CUDA
---
# DeepSeek V4：Kernels 与量化

本文专门解释 `inference/kernel.py`，重点不是逐函数过目录，而是说明这些 kernel 如何服务上层推理链路：

- `act_quant` / `fp4_act_quant`
- `fp8_gemm` / `fp4_gemm`
- `sparse_attn`
- `hc_split_sinkhorn`

源码：

- [kernel.py](src/kernel_py.md)

前置：

- [TileLang 编程模型](../cuda_foundations/07_tilelang_programming_model.md)
- [块量化与低精度 GEMM](../cuda_foundations/08_blockwise_quantization_and_low_precision_gemm.md)

## 1. TileLang kernel 入口

一开头就能看出这份 kernel 的基调，[kernel.py:L1-L19](src/kernel_py.md#__codelineno-0-1)：

```python
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
```

这段配置给出了 kernel 组织方式：

- CTA tiled 风格
- 非 TMA 路线
- 用 TileLang 原语组织 GEMM / reduction / sparse gather

因此阅读顺序也应该是 TileLang 风格：

1. 看 `T.Kernel` 的输出 tile 划分
2. 看 `alloc_shared / alloc_fragment`
3. 看 `T.copy`
4. 看 `T.gemm` / `T.reduce_*`
5. 看 `T.Pipelined`

## 2. `act_quant_kernel`：量化是如何 block-wise 做的

相关源码：

- [kernel.py:L40-L102](src/kernel_py.md#__codelineno-0-40)

这是整个低精度系统的入口 kernel。

```python
@tilelang.jit(pass_configs=pass_configs)
def act_quant_kernel(...):
    M = T.symbolic("M")
    fp8_min = -448.0
    fp8_max = 448.0
    num_stages = 0 if round_scale or inplace else 2
    blk_m = 32
    group_size = block_size

    @T.prim_func
    def act_quant_kernel_(X, Y, S):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m, pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), compute_dtype)
            s_local = T.alloc_fragment((blk_m,), compute_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            '''
            一个 CTA 处理 [32, block_size] 的 tile：
            - 对 tile 的每一行单独求 amax
            - 每一行生成一个 scale
            - 再把这一行的所有元素量化
            '''
            for _ in T.Pipelined(1, num_stages=num_stages):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)

                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 1e-4)
                    if round_scale:
                        s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                    else:
                        s_local[i] = amax_local[i] * fp8_max_inv

                if inplace:
                    '''
                    QDQ 模式：
                    量化到 FP8，再乘回 scale 写回 BF16。
                    这用于在主图里模拟低精度误差，而不是返回真正的量化张量。
                    '''
                    ...
                else:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.clamp(
                            x_local[i, j] / s_local[i], fp8_min, fp8_max
                        )

                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = T.Cast(scale_dtype, s_local[i])
```

它和上层推理的关系非常直接：

- `linear()` 真正要做低精度 GEMM 时，会调用 `act_quant(..., inplace=False)` 拿到 `(x_q, s)`
- Attention / Compressor / Indexer 在只想注入量化误差时，会调用 `inplace=True`

所以 `act_quant` 同时承担了：

- 真实量化前端
- QDQ 仿真前端

## 3. `fp8_gemm_kernel`：标准低精度线性层后端

相关源码：

- [kernel.py:L203-L273](src/kernel_py.md#__codelineno-0-203)

这是“FP8 激活 x FP8 权重”的主力后端。

```python
@tilelang.jit(pass_configs=pass_configs)
def fp8_gemm_kernel(N, K, ...):
    group_size = 128
    block_M = 32
    block_N = 128
    block_K = 128

    @T.prim_func
    def fp8_gemm_kernel_(A, B, C, scales_a, scales_b):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), FP8)
            B_shared = T.alloc_shared((block_N, block_K), FP8)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            T.clear(C_local_accum)

            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=4):
                T.copy(A[by * block_M, k * block_K], A_shared)     # A 的一个 K tile
                T.copy(B[bx * block_N, k * block_K], B_shared)     # B 的一个 K tile

                Scale_B = T.Cast(FP32, scales_b[bx * block_N // group_size, k])
                for i in T.Parallel(block_M):
                    Scale_C_shared[i] = T.Cast(FP32, scales_a[by * block_M + i, k]) * Scale_B

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                '''
                低精度 GEMM 不是最后一次性反量化，而是每个 K tile:
                    先做量化乘法
                    再乘对应 scale
                    再加进总累加器
                '''
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]
                T.clear(C_local)
```

这个 kernel 对应上层几乎所有 `FP8 weight` 的线性层：

- `wq_a`
- `wq_b`
- `wo_a`
- 部分 MoE 权重之外的主干矩阵

## 4. `fp4_gemm_kernel`：专家层后端

相关源码：

- [kernel.py:L441-L536](src/kernel_py.md#__codelineno-0-441)

这条路径专门服务：

- `expert_dtype == "fp4"` 的专家权重

与 `fp8_gemm` 的差异不在于“又写一遍 GEMM”，而在于两点：

- 权重物理存储为打包 FP4
- 激活 scale 和权重 scale 的 block 粒度不同

```python
act_group_size = 128
weight_group_size = 32
block_K = 32
n_sub = act_group_size // block_K                              # 4 个 32-block 组成一个 128-block

for k in T.Pipelined(K_iters, num_stages=2):
    T.copy(A[by * block_M, k * block_K], A_shared)
    T.copy(B[bx * block_N, k * block_K], B_fp4_shared)

    '''
    没有直接做 FP8 x FP4 MMA。
    这里先把 FP4 tile 提升成 FP8 tile，再复用同样的 GEMM 结构。
    '''
    for i, j in T.Parallel(block_N, block_K):
        B_shared[i, j] = T.Cast(FP8, T.Cast(FP32, B_fp4_shared[i, j]))

    for i in T.Parallel(block_N):
        scale_b_frag[i] = T.Cast(FP32, scales_b[bx * block_N + i, k])      # 每 32 元素一个 weight scale

    for i in T.Parallel(block_M):
        scale_a_frag[i] = T.Cast(FP32, scales_a[by * block_M + i, k // n_sub])  # 每 128 元素一个 act scale

    T.gemm(A_shared, B_shared, C_local, transpose_B=True)

    for i, j in T.Parallel(block_M, block_N):
        C_local_accum[i, j] += C_local[i, j] * scale_a_frag[i] * scale_b_frag[j]
```

这条实现路径对应的工程取舍是：

- 专家层参数量巨大，FP4 带来的存储收益更大
- 但直接维护一套“FP8 x FP4 原生 MMA”实现复杂度太高
- 先在 tile 内升到 FP8，再复用 FP8 GEMM 主体，是一种非常工程化的折中

## 5. `sparse_attn_kernel`：混合记忆注意力

相关源码：

- [kernel.py:L276-L368](src/kernel_py.md#__codelineno-0-276)

这一段对应 Hybrid Attention 的 kernel 落点。

```python
@tilelang.jit(pass_configs=pass_configs)
def sparse_attn_kernel(h: int, d: int, scale=None):
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    ...

    @T.prim_func
    def sparse_attn_kernel_(q, kv, o, attn_sink, topk_idxs):
        with T.Kernel(m, b, threads=threads) as (bx, by):
            q_shared = T.alloc_shared((h, d), BF16)
            kv_shared = T.alloc_shared((block, d), BF16)
            acc_s = T.alloc_fragment((h, block), FP32)
            acc_o = T.alloc_fragment((h, d), FP32)
            scores_max = T.alloc_fragment(h, FP32)
            scores_sum = T.alloc_fragment(h, FP32)
            sum_exp = T.alloc_fragment(h, FP32)
            ...

            T.copy(q[by, bx, :, :], q_shared)                         # 一个 token 的所有 head query

            for t in T.Pipelined(num_blocks, num_stages=num_stages):
                for i in T.Parallel(block):
                    idxs[i] = T.if_then_else(
                        t * block + i < topk,
                        topk_idxs[by, bx, t * block + i], -1
                    )
                for i, j in T.Parallel(block, d):
                    kv_shared[i, j] = T.if_then_else(
                        idxs[i] != -1, kv[by, idxs[i], j], 0          # 按索引 gather KV
                    )

                T.gemm(q_shared, kv_shared, acc_s, transpose_B=True)  # QK^T
                ...
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)   # online max
                ...
                T.reduce_sum(acc_s, scores_sum, dim=1)                # online sum
                ...
                T.gemm(acc_s_cast, kv_shared, acc_o,                  # P @ V
                       policy=T.GemmWarpPolicy.FullRow)

            for i in T.Parallel(h):
                sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])     # 额外 attention sink
            for i, j in T.Parallel(h, d):
                acc_o[i, j] /= sum_exp[i]
```

这段代码的结构非常清晰：

- `topk_idxs` 决定当前 query token 访问哪些记忆位置
- kernel 按 block 分批 gather 这些位置
- 每个 block 上维护 online softmax 统计量
- 最终得到一份 head-wise attention output

这段 kernel 的计算形态是：

```text
sparse loading + dense compute + online normalization
```

### 5.1 这里的稀疏性来自哪里

不是 kernel 自己搜索稀疏模式，而是上层 `Attention.forward` 预先算好：

- 窗口索引
- 压缩缓存索引
- 两者拼接成 `topk_idxs`

所以 `sparse_attn_kernel` 是一个“**消费索引**”的 kernel，而不是“**产生索引**”的 kernel。

### 5.2 `attn_sink` 在这里怎么落地

论文里的 sink 机制，在这里体现为：

```python
sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
```

它的作用是：

- 给每个 head 增加一个额外归一化支撑项
- 避免纯稀疏检索下 softmax 过于脆弱

可以把它理解成一种“永远存在的背景记忆槽”。

## 6. `hc_split_sinkhorn_kernel`：mHC mixing

相关源码：

- [kernel.py:L371-L438](src/kernel_py.md#__codelineno-0-371)

`hc_pre` / `hc_post` 所需的 mixing 权重不是自由矩阵，而是经过约束化处理的，这部分由 `hc_split_sinkhorn_kernel` 完成。

```python
@tilelang.jit(pass_configs=pass_configs)
def hc_split_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float):
    ...
    @T.prim_func
    def hc_split_sinkhorn_kernel_(mixes, hc_scale, hc_base, pre, post, comb):
        with T.Kernel(n, threads=threads) as i:
            mixes_shared = T.alloc_shared(mix_hc, FP32)
            comb_frag = T.alloc_fragment((hc, hc), FP32)
            T.copy(mixes[i, :], mixes_shared)

            for j in T.Parallel(hc):
                pre[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
            for j in T.Parallel(hc):
                post[i, j] = 2 * T.sigmoid(mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc])
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = mixes_shared[j * hc + k + hc * 2] * hc_scale[2] + hc_base[...]

            '''
            comb 不直接使用，而是先做一轮 softmax 风格归一化，
            再交替执行行归一化 / 列归一化，逼近双随机矩阵。
            '''
            T.reduce_max(comb_frag, row_max, dim=1)
            ...
            for _ in T.serial(sinkhorn_iters - 1):
                T.reduce_sum(comb_frag, row_sum, dim=1)
                ...
                T.reduce_sum(comb_frag, col_sum, dim=0)
                ...
```

这里的重点不是它用了多少线程，而是它在数值上强制了：

- `pre` / `post` 是有界门控
- `comb` 经过 Sinkhorn 迭代后更稳定、更接近受约束的 mixing 矩阵

这和普通 residual gating 的最大差异就在这里。

## 7. `kernel.py` 服务上层结构的方式

如果从 `model.py` 回看 `kernel.py`，它们的依赖关系非常清晰：

- `linear()`
  - 依赖 `act_quant` / `fp8_gemm` / `fp4_gemm`
- `Attention.forward`
  - 依赖 `sparse_attn`
- `Block.hc_pre`
  - 依赖 `hc_split_sinkhorn`
- `Indexer` / `Compressor`
  - 依赖 `fp4_act_quant` / `act_quant`

因此 `kernel.py` 并不是“一堆底层优化”，而是直接支撑了上层四个结构设计：

- 量化线性层
- 压缩记忆检索
- 稀疏注意力
- mHC mixing

## 8. 实现取舍

从工程角度看，这份 kernel 代码有几个鲜明取舍：

- 没有追求最花哨的 Hopper 特性
  - 明确禁了 TMA lowering
- 对专家 FP4 权重走“先升到 FP8 tile 再 GEMM”的折中路线
- 稀疏注意力先由上层生成索引，kernel 只负责消费索引
- mHC mixing 单独用小 kernel 解决，而不是塞进主图的通用矩阵乘

这几条取舍对应的是一份 **围绕具体架构定制的推理实现**，不是通用库式抽象。

## 小结

`kernel.py` 的四个核心 kernel，分别把上层四个结构改造落地成了可执行算子：

- `act_quant`：把输入切成块并生成 scale
- `fp8_gemm / fp4_gemm`：把低精度线性层真正做出来
- `sparse_attn`：把混合记忆访问真正做出来
- `hc_split_sinkhorn`：把 mHC mixing 真正做出来

所以读完 `model.py` 再读 `kernel.py`，你会发现论文里的“模型设计”与“推理效率设计”在这里已经完全合并了。
