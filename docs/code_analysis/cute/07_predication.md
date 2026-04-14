---
tags:
  - CUTLASS
  - CUDA
---

# CuTe 谓词：分块不整除时的处理

> **原文出处**: [NVIDIA/cutlass - media/docs/cute/0y_predication.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0y_predication.md)
> **许可证**: BSD-3-Clause, Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES

[GEMM 教程](06_gemm_tutorial.md)展示了如何通过遍历输入矩阵和输出矩阵的 tile 来计算矩阵乘法。所有示例都假设 tile 恰好整除矩阵，没有余数。如果不是这种情况怎么办？例如用 4x8 的 tile 分块一个 41x55 的矩阵，41/4 = 10 余 1，55/8 = 6 余 7。那些矩阵"剩余"的部分怎么处理？

首先注意 `logical_divide`（CuTe 的分块方式）会"向上取整"。例如 `N = 1000:1`，`B = 128:1`，则 `logical_divide(N, B)` 得到 `(128, 8):(1, 128)`，实际上将原始 shape `N = 1000` 向上取整为 `128 x 8` 矩阵（如同 `N = 1024`）。那最后 24 个不属于原始数据的元素呢？如何处理最后一个 tile 并避免越界索引？

与其他 CUDA 编程入门一样，CuTe 惯用的方式是通过**谓词化**（predication）解决。CuTe 不尝试推理"余数 tile"，而是向上取整并构造谓词使 kernel 只访问每个 tile 中矩阵内有效的数据。这与 GPU 优化方式对应良好：无 warp 分歧的分支相对较快。

## 通用谓词构造

```c++
Tensor gmem = ...     // 例如 size 1000
Tensor smem = ...     // 例如 size 128

// 为 smem 分块 gmem
Tensor gmem_tiled = logical_divide(gmem, size(smem));      // (128,8)

// 创建 gmem 的恒等 layout 并类似地分块
Layout id_layout = make_layout(shape(gmem));               // 1000:1
Layout id_tiled  = logical_divide(id_layout, size(smem));  // (128,8):(1,128)

// 创建谓词 tensor
Tensor pred = make_tensor<bool>(shape(id_tiled));          // (128,8)
for (int i = 0; i < size(pred); ++i) {
  pred(i) = id_tiled(i) < size(id_layout);  // 谓词：偏移是否在原始 shape 内？
}

// 使用：对 tile tile_i 中的元素 value_j 判断是否越界
if (pred(value_j,tile_i)) { smem(value_j) = gmem_tiled(value_j,tile_i); }
```

通用流程：

1. 创建与原始数据相同 shape 的**恒等 layout**
2. 对恒等 layout 执行与数据相同的分块/分区/切片（可能向上取整）
3. 比较参考 layout 的坐标与原始 layout 的边界，创建**谓词 tensor**
4. 用谓词 tensor 屏蔽越界元素的访问

## GEMM epilogue 谓词化示例

假设已将 mC 分区到 CTA tile 和 MMA 线程分区：

```cpp
// CTA 分区
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

// 线程分区
auto thr_mma = mma.get_slice(threadIdx.x);
Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)
Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)
```

谓词化：

```cpp
// 与 mC 相同 shape 的坐标 tensor：(m,n) -> (m,n)
Tensor cC     = make_identity_tensor(shape(mC));

// 对坐标 tensor 执行与 mC 完全相同的分区步骤
Tensor cta_cC = local_tile(cC, cta_tiler, cta_coord, Step<_1,_1, X>{});
Tensor tCcC   = thr_mma.partition_C(cta_cC);                             // (MMA,MMA_M,MMA_N) -> (m,n)

// 谓词化 axpby epilogue
for (int i = 0; i < size(tCgC); ++i) {
  if (elem_less(tCcC(i), shape(mC))) {  // 坐标在边界内
    tCgC(i) = alpha * tCrC(i) + beta * tCgC(i);
  }
}
```

坐标 tensor `tCcC` 与寄存器 fragment `tCrC` 和分区后的全局内存 tensor `tCgC` 一致，但 `tCcC` 求值时保留原始陪域——原始 tensor mC 的全局坐标。将此全局坐标与 mC 的 shape 比较即可判断有效性。

## A 和 B 加载的 m/n 谓词化

稍复杂的例子——A 和 B 加载中的 m 和 n 谓词化。创建恒等 tensor 并执行与 mA/mB 完全相同的分块和线程分区：

```c++
// 坐标 tensor
Tensor cA = make_identity_tensor(shape(mA));   // (m,k) -> (m,k)
Tensor cB = make_identity_tensor(shape(mB));   // (n,k) -> (n,k)

// CTA 分区 + 线程分区
Tensor tAcA = local_partition(local_tile(cA, ...), tA, thread_idx);  // (THR_M,THR_K,k) -> (m,k)
Tensor tBcB = local_partition(local_tile(cB, ...), tB, thread_idx);  // (THR_N,THR_K,k) -> (n,k)
```

创建谓词 tensor——只存储 m 和 n 谓词（跨 k 广播，stride-0）：

```c++
Tensor tApA = make_tensor<bool>(make_shape (size<0>(tAcA), size<1>(tAcA)),
                                make_stride(     Int<1>{},      Int<0>{}));

// 填充 m 谓词
for (int m = 0; m < size<0>(tApA); ++m) {
  tApA(m,0) = elem_less(get<0>(tAcA(m,0,0)), shape<0>(mA));
}
```

使用 `copy_if` 执行有谓词保护的拷贝：

```c++
copy_if(tApA, tAgA(_,_,k_tile), tAsA);
copy_if(tBpB, tBgB(_,_,k_tile), tBsB);
```

## 优势

此"参考恒等 tensor"/"坐标 tensor"方法的优势：

1. 不依赖被谓词化 tensor 的 layout/步长，仅依赖逻辑边界
2. 分区阶段可以是任何形式——CTA tiling、线程分区、TiledMMA、TiledCopy 都可以应用于坐标 tensor
3. 自然扩展到任意维度的谓词化
4. 是典型 CUDA 一维并行向量访问模式的自然推广

在 SIMT 编程模型中，不应修改 tensor 尺寸来避免循环越界，而应用谓词化查询原始坐标是否越界。这避免了可变/动态循环边界，有利于指令级谓词化、保持线程一致性和负载均衡。
