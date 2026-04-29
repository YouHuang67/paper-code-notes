---
tags:
  - CUDA
  - LLM Inference
---
# CUDA 基础：TileLang 编程模型

本文补齐 DeepSeek V4 代码分析里会反复用到的 TileLang 基础。重点不是泛讲 DSL，而是解释 `inference/kernel.py` 里真正出现的那组原语：`T.Kernel`、`T.Pipelined`、`alloc_shared`、`alloc_fragment`、`copy`、`gemm`、`reduce_*`、`use_swizzle`。

建议先有以下背景：

- [CUDA 基础：执行模型与内存访问](01_cuda_execution_model_and_memory.md)
- [CUDA 基础：CUTLASS/CuTe 编程模型](02_cuda_cutlass_cute_programming_model.md)
- [CUDA 基础：分块、数据搬运与局部性](04_cuda_tiling_data_movement_and_locality.md)
- [CUDA 基础：归约、Scan 与在线归一化](05_cuda_reduction_scan_and_online_normalization.md)

如果你是为了看 DeepSeek V4 而来，可直接把本文当成 [DeepSeek V4：Kernels 与量化](../deepseek_v4/03_kernels_and_quantization.md) 的前置页。

## 1. 它在 CUDA 生态里的位置

DeepSeek V4 的 `kernel.py` 不是 Triton，而是 **TileLang**：

- Triton 更像“以 program 为中心”的 block kernel DSL
- TileLang 更像“以 tile / pipeline / GEMM 原语为中心”的 CUDA DSL
- 它底层会进一步 lower 到 CUDA / CUTLASS 风格的实现

从阅读体验上看，TileLang 更接近“把 CTA 级 tiled kernel 写成 Python”，所以理解它时最好直接用下面这套映射：

- `T.Kernel(...)` 对应一个 CTA 级 kernel launch 网格
- `T.alloc_shared(...)` 对应 shared memory tile
- `T.alloc_fragment(...)` 对应寄存器 fragment
- `T.copy(...)` 对应一次 tile 级搬运
- `T.gemm(...)` 对应一次 tile 级矩阵乘
- `T.Pipelined(...)` 对应 K 维主循环上的多 stage 流水

## 2. 执行模型

TileLang 的 kernel 一般长成下面这样：

```python
@tilelang.jit(pass_configs=pass_configs)
def fp8_gemm_kernel(N, K, out_dtype=BF16, accum_dtype=FP32, scale_dtype=FP32):
    M = T.symbolic("M")                                     # 运行时动态维度
    group_size = 128
    block_M = 32
    block_N = 128
    block_K = 128

    @T.prim_func
    def fp8_gemm_kernel_(
        A: T.Tensor[(M, K), FP8],                           # 全局输入 A
        B: T.Tensor[(N, K), FP8],                           # 全局输入 B
        C: T.Tensor[(M, N), out_dtype],                     # 全局输出 C
        scales_a: T.Tensor[(M, T.ceildiv(K, group_size)), scale_dtype],
        scales_b: T.Tensor[(T.ceildiv(N, group_size), T.ceildiv(K, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(N, block_N),                # grid.x
                      T.ceildiv(M, block_M),                # grid.y
                      threads=128) as (bx, by):            # 单 CTA 线程数
            ...
```

这段代码对应的 CUDA 直觉是：

- 一个 CTA 负责 `C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N]`
- `threads=128` 指这个 CTA 有 128 个线程
- `T.ceildiv(N, block_N)` 和 `T.ceildiv(M, block_M)` 定义了网格尺寸
- `M = T.symbolic("M")` 表示这个维度保留为运行时参数，而不是完全静态展开

与 Triton 的 `program_id(0/1)` 很像，但 TileLang 更直接把 CTA 级 tile 形状和 tile 内原语写在同一个结构里。

## 3. 存储层级

TileLang 里最重要的是区分三类存储：

- 全局张量：函数参数里的 `T.Tensor[...]`
- shared memory：`T.alloc_shared(...)`
- 寄存器 fragment：`T.alloc_fragment(...)`

DeepSeek V4 的量化和注意力 kernel 都遵循同一个模板：

```python
with T.Kernel(..., threads=128) as (...):
    A_shared = T.alloc_shared((block_M, block_K), FP8)     # CTA 共享 tile
    B_shared = T.alloc_shared((block_N, block_K), FP8)     # CTA 共享 tile
    C_local = T.alloc_fragment((block_M, block_N), FP32)   # 寄存器累加器

    T.copy(A[...], A_shared)                               # GMEM -> SMEM
    T.copy(B[...], B_shared)                               # GMEM -> SMEM
    T.gemm(A_shared, B_shared, C_local, transpose_B=True)  # SMEM -> REG -> MMA
```

这和 [CUTLASS/CuTe 编程模型](02_cuda_cutlass_cute_programming_model.md) 的数据流是同一件事，只是 DSL 不同：

```text
Global Memory -> Shared Memory -> Register Fragment -> MMA / Reduce -> Shared / Global
```

### `alloc_shared`

`alloc_shared((tile_shape), dtype)` 的语义是：

- 为整个 CTA 分配一个共享 tile
- 这个 tile 会被 CTA 内多个线程协作读写
- 它通常承接 `T.copy` 的目标，或者 `T.gemm` 的输入

### `alloc_fragment`

`alloc_fragment((tile_shape), dtype)` 更接近：

- 一个逻辑上的寄存器 tile
- 可作为累加器、局部统计量、局部索引缓存
- 它不需要显式说明哪个线程持有哪些元素，编译器会按原语约束安排

在 DeepSeek V4 里：

- `amax_local`、`scores_max`、`sum_exp` 是 reduction 统计 fragment
- `C_local`、`C_local_accum`、`acc_o` 是 GEMM / attention 的累加 fragment

## 4. `T.copy`、`T.gemm`、`T.reduce_*`

TileLang 的优势是把最常见的 tile 级并行原语提炼成几个稳定接口。

### `T.copy`

`T.copy(src, dst)` 表示一个 tile 级搬运，不必手写每个线程的地址计算。

```python
T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)     # 全局 -> shared
T.copy(x_shared, x_local)                                  # shared -> fragment
T.copy(C_local_accum, C_shared)                            # fragment -> shared
T.copy(C_shared, C[by * block_M, bx * block_N])            # shared -> 全局
```

你可以把它理解成：

- 当源和目标在不同存储层级时，TileLang 会选择合适的 copy lowering
- 这和 CuTe 里“根据地址空间和 Atom 分发 copy”的思想是一致的

### `T.gemm`

`T.gemm(A_shared, B_shared, C_local, ...)` 不是普通 Python 矩阵乘，而是 CTA 级 tile GEMM 原语。

它意味着：

- `A_shared` / `B_shared` 作为 tile 级操作数
- `C_local` 作为寄存器累加器
- 编译器把它 lower 成 Tensor Core / MMA 友好的实现

DeepSeek V4 的 `kernel.py` 里，`T.gemm` 出现于三类场景：

- `fp8_gemm_kernel`：标准量化 GEMM
- `fp4_gemm_kernel`：先把 FP4 tile 提升到 FP8，再做 GEMM
- `sparse_attn_kernel`：先算 `QK^T`，再算 `P @ V`

### `T.reduce_absmax` / `T.reduce_max` / `T.reduce_sum`

这组接口把 block 内归约抽象成 tile 级 reduction。

例如 `act_quant_kernel`：

```python
T.reduce_absmax(x_local, amax_local, dim=1)                # 每行求绝对值最大值
```

以及 `sparse_attn_kernel`：

```python
T.reduce_max(acc_s, scores_max, dim=1, clear=False)       # 每个 head 求当前块的 row max
T.reduce_sum(acc_s, scores_sum, dim=1)                    # 每个 head 求当前块的 exp sum
```

这里的思想与 [在线归一化](05_cuda_reduction_scan_and_online_normalization.md) 完全一致：不是先物化整个矩阵再做 softmax，而是在 block 流上维护行统计量。

## 5. `T.Pipelined` 与多 stage 流水

DeepSeek V4 的 kernel 经常用：

```python
for k in T.Pipelined(K_iters, num_stages=4):
    T.copy(...)
    T.copy(...)
    T.gemm(...)
```

它的抽象含义是：

- 循环维度通常是 K 维或 sparse block 维
- `num_stages` 指流水线深度
- 编译器会尝试把“下一块搬运”和“当前块计算”重叠

对照 CUDA 直觉：

- 它相当于一个 higher-level 的 mainloop pipeline
- 和 [sgemm_sm80](../cute/09_sgemm_sm80.md) 里手写 `cp.async` / wait / stage 切换是同一个主题
- 只是 TileLang 把显式同步与分 stage 编排藏在原语层后面

## 6. `T.use_swizzle` 与局部性

在 `fp8_gemm_kernel` / `fp4_gemm_kernel` 里都能看到：

```python
T.use_swizzle(panel_size=10)
```

它对应的是一种访存布局优化提示，目标通常是：

- 改善 L2 / shared memory 局部性
- 避免 tile 布局在后续访问时形成坏的冲突模式

它的精神和 CuTe / CUTLASS 里的 swizzle 是一脉相承的，只是接口层面更高。

更底层的背景可参考：

- [CUDA 基础：分块、数据搬运与局部性](04_cuda_tiling_data_movement_and_locality.md)
- [CuTe sgemm_sm80 实战拆解](../cute/09_sgemm_sm80.md)

## 7. `pass_configs` 在这里做什么

DeepSeek V4 的 `kernel.py` 一开头就定义了：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
```

这说明作者在有意约束 lower 方式：

- 不走 warp-specialized 风格的调度
- 不启用 TMA lowering

对阅读者来说，这个设置的意义不是“性能绝对最好”，而是：

- kernel 形态更稳定
- 更接近普通 CTA tiled kernel
- 对这类以量化 GEMM 和 sparse gather 为主的实现，理解复杂度更低

## 8. 用 DeepSeek V4 的 `act_quant_kernel` 串起来看

下面这段是最典型的 TileLang 代码形态，几乎把本文讲过的原语都用了一遍。原始代码见 [kernel.py:L40](../deepseek_v4/src/kernel_py.md#__codelineno-0-40)。

```python
@tilelang.jit(pass_configs=pass_configs)
def act_quant_kernel(...):
    M = T.symbolic("M")                                     # 运行时动态行数
    ...

    @T.prim_func
    def act_quant_kernel_(X, Y, S):
        with T.Kernel(T.ceildiv(M, blk_m),                  # grid.y 上切 M
                      T.ceildiv(N, group_size),             # grid.x 上切 N
                      threads=128) as (pid_m, pid_n):      # 每个 CTA 128 线程
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), compute_dtype)
            s_local = T.alloc_fragment((blk_m,), compute_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            '''
            TileLang 版的 block-wise 量化模板：
            1. GMEM -> SMEM -> Fragment
            2. 每行做 absmax reduction
            3. 计算每行 scale
            4. 量化或量化后再反量化
            5. 把 scale 和结果写回
            '''
            for _ in T.Pipelined(1, num_stages=num_stages):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)  # 输入 tile 载入 shared
                T.copy(x_shared, x_local)                               # shared -> 寄存器 fragment
                T.reduce_absmax(x_local, amax_local, dim=1)            # 每行求 amax
                ...
                T.copy(y_local, y_shared)                              # fragment -> shared
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size]) # shared -> 全局
```

## 9. 和 Triton 的关系

如果你有 Triton 背景，可以这样做心智映射：

- Triton `program` ≈ TileLang `T.Kernel` 中的一个 CTA
- Triton `tl.load/tl.store` ≈ TileLang `T.copy`
- Triton `tl.dot` ≈ TileLang `T.gemm`
- Triton 显式张量表达更强
- TileLang 对 tiled GEMM / pipeline / fragment 的表达更直接

DeepSeek V4 之所以更像 TileLang 风格，是因为它的瓶颈原语非常集中在：

- 低精度 GEMM
- 稀疏 gather + dense GEMM
- 行级在线归一化
- 小矩阵 Sinkhorn 迭代

这些都非常适合用“tile + fragment + pipeline”来组织。

## 小结

读 DeepSeek V4 的 TileLang kernel 时，最重要的不是记住所有 API，而是记住这一句：

```text
TileLang = 用 Python 写 CTA 级 tiled CUDA kernel
```

因此阅读顺序始终可以固定成：

1. `T.Kernel` 先看一个 CTA 负责哪块输出
2. `alloc_shared / alloc_fragment` 看数据在哪一层
3. `T.copy` 看 tile 怎么搬
4. `T.gemm / T.reduce_*` 看计算主体
5. `T.Pipelined` 看主循环如何重叠搬运与计算

接下来可直接进入：

- [DeepSeek V4：推理链路](../deepseek_v4/02_inference_pipeline.md)
- [DeepSeek V4：Kernels 与量化](../deepseek_v4/03_kernels_and_quantization.md)
