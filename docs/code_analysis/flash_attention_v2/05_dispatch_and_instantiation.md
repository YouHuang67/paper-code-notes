---
tags:
  - CUDA
  - CUTLASS
  - Flash Attention
---
# Flash Attention V2：调度与模板实例化

本文拆解 FA2 从 host 端到 kernel 启动的完整路径：编译期开关宏、launch template、模板实例化。

**源码**: [flash_fwd_launch_template.h](src/flash_fwd_launch_template_h.md)、[static_switch.h](src/static_switch_h.md)

## static_switch.h：编译期特性分发

[static_switch.h:L5-L125](src/static_switch_h.md#__codelineno-0-5)

FA2 的所有特性开关（causal、dropout、alibi 等）都是编译期常量，通过嵌套宏将运行时 bool 转为 `constexpr`。每个 `BOOL_SWITCH` 使二进制膨胀 2x，因此提供 `DISABLE_*` 宏来裁剪不需要的特性。

```cpp
/* BOOL_SWITCH: 核心宏, 将运行时 bool 转为编译期 constexpr */
#define BOOL_SWITCH(COND, CONST_NAME, ...)       \
  [&] {                                          \
    if (COND) {                                  \
      constexpr static bool CONST_NAME = true;   \
      return __VA_ARGS__();                      \  // 实例化 true 版本
    } else {                                     \
      constexpr static bool CONST_NAME = false;  \
      return __VA_ARGS__();                      \  // 实例化 false 版本
    }                                            \
  }()

/*
 * 特性开关宏: 可通过 FLASHATTENTION_DISABLE_* 编译期禁用
 * 禁用后始终取固定值, 避免二进制膨胀
 *   DROPOUT_SWITCH  → FLASHATTENTION_DISABLE_DROPOUT  → 始终 false
 *   ALIBI_SWITCH    → FLASHATTENTION_DISABLE_ALIBI    → 始终 false
 *   EVENK_SWITCH    → FLASHATTENTION_DISABLE_UNEVEN_K → 始终 true
 *   SOFTCAP_SWITCH  → FLASHATTENTION_DISABLE_SOFTCAP  → 始终 false
 *   LOCAL_SWITCH    → FLASHATTENTION_DISABLE_LOCAL     → 始终 false
 */
#ifdef FLASHATTENTION_DISABLE_DROPOUT
#define DROPOUT_SWITCH(COND, CONST_NAME, ...)    \
  decltype(googlemock_Eval(                      \
    [&] {                                        \
      constexpr static bool CONST_NAME = false;  \
      return __VA_ARGS__();                      \
    }                                            \
  ))
#else
#define DROPOUT_SWITCH BOOL_SWITCH               // 未禁用: 正常双分支
#endif
// ALIBI_SWITCH, EVENK_SWITCH, SOFTCAP_SWITCH, LOCAL_SWITCH 同理

/*
 * FP16_SWITCH: 选择 cutlass::half_t 或 cutlass::bfloat16_t
 * COND=true 时 elem_type = cutlass::half_t
 */
#define FP16_SWITCH(COND, ...)                   \
  [&] {                                          \
    if (COND) {                                  \
      using elem_type = cutlass::half_t;         \
      return __VA_ARGS__();                      \
    } else {                                     \
      using elem_type = cutlass::bfloat16_t;     \
      return __VA_ARGS__();                      \
    }                                            \
  }()

/*
 * HEADDIM_SWITCH: 将运行时 head_dim 映射到编译期常量
 * 支持 32/64/96/128/160/192/256, 其他值触发运行时错误
 */
#define HEADDIM_SWITCH(HEADDIM, ...)             \
  [&] {                                          \
    if (HEADDIM <= 32) {                         \
      constexpr static int kHeadDim = 32;        \
      return __VA_ARGS__();                      \
    } else if (HEADDIM <= 64) {                  \
      constexpr static int kHeadDim = 64;        \
      return __VA_ARGS__();                      \
    } else if (HEADDIM <= 96) {                  \
      constexpr static int kHeadDim = 96;        \
      return __VA_ARGS__();                      \
    } else if (HEADDIM <= 128) {                 \
      constexpr static int kHeadDim = 128;       \
      return __VA_ARGS__();                      \
    } else if (HEADDIM <= 160) {                 \
      constexpr static int kHeadDim = 160;       \
      return __VA_ARGS__();                      \
    } else if (HEADDIM <= 192) {                 \
      constexpr static int kHeadDim = 192;       \
      return __VA_ARGS__();                      \
    } else if (HEADDIM <= 256) {                 \
      constexpr static int kHeadDim = 256;       \
      return __VA_ARGS__();                      \
    }                                            \
  }()
```

## flash_fwd_launch_template.h：Kernel 启动

### Kernel 入口宏

[flash_fwd_launch_template.h:L12-L50](src/flash_fwd_launch_template_h.md#__codelineno-0-12)

```cpp
/*
 * DEFINE_FLASH_FORWARD_KERNEL: 定义 __global__ kernel 函数
 * 将 params 各字段解包为局部变量传入 device 函数
 * 目的: 避免 device 函数反复从 params 结构体读取
 */
#define DEFINE_FLASH_FORWARD_KERNEL(NAME, ...)                \
__global__ void __launch_bounds__(                            \
    Kernel_traits::kNThreads, 1)                              \  // 每 block 线程数, 最少 1 block/SM
NAME(KERNEL_PARAMS_LAUNCH) {                                  \
    /* KERNEL_PARAMS_LAUNCH 展开为 Flash_fwd_params 各字段 */ \
    static_assert(googlemock_Eval(...));                       \
    flash::compute_attn_1rowblock<                            \
        Kernel_traits, Is_dropout, Is_causal, Is_local,       \
        Has_alibi, IsEvenMNConst, IsEvenKConst,               \  // 编译期常量
        Is_softcap, ReturnSoftmaxConst                        \
    >(params, bidb, bidh, m_block, ...);                      \
}

/* 实际定义三个 kernel */
DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel, ...);           // 标准前向
DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_kernel,         // Split-KV 前向
    const int num_splits);
DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_combine_kernel, // Split-KV 合并
    ...);
```

### run_flash_fwd：Grid 配置 + 嵌套 Switch

[flash_fwd_launch_template.h:L54-L99](src/flash_fwd_launch_template_h.md#__codelineno-0-54)

```cpp
template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block =
        cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    dim3 grid(num_m_block, params.b, params.h);   // (Q行块数, batch, head)
    // kNThreads = kNWarps * 32, 通常 128 线程 = 4 warp

    /*
     * SM80 默认动态 shared memory 上限 48 KB
     * FA2 常超过此值 (如 d=128 时约 64 KB), 需显式设置
     */
    if (smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }

    /*
     * 6 层嵌套 BOOL_SWITCH: 将运行时条件转为编译期模板参数
     * 理论 2^6 = 64 变体, 实际通过互斥条件 + DISABLE 宏大幅裁减
     *   - Is_local && !Is_causal 互斥
     *   - IsEvenMNConst 只在无特殊特性时为 true
     *   - ReturnSoftmaxConst 只在有 dropout 时可能为 true
     */
    const bool is_even_MN = ...;
    const bool is_even_K  = params.d == Kernel_traits::kHeadDim;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
      EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        LOCAL_SWITCH(params.window_size_left >= 0 ||
                     params.window_size_right >= 0,
                     Is_local, [&] {
          BOOL_SWITCH(params.num_splits > 1
                      ? false : return_softmax,
                      ReturnSoftmaxConst, [&] {
            ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr,
                         Has_alibi, [&] {
              SOFTCAP_SWITCH(params.softcap > 0.0,
                             Is_softcap, [&] {
                /*
                 * 最内层: 所有模板参数已确定, 选择 kernel
                 * 条件简化: IsEvenMNConst && IsEvenKConst &&
                 *   !Is_dropout && !Is_local && !Has_alibi &&
                 *   !Is_softcap && kHeadDim <= 128
                 * → 简化路径, 减少寄存器使用
                 */
                auto kernel = &flash_fwd_kernel<Kernel_traits,
                    Is_dropout && !IsEvenMNConst,             // dropout + 非对齐才启用
                    Is_causal, Is_local, Has_alibi,
                    IsEvenMNConst, IsEvenKConst,
                    Is_softcap, ReturnSoftmaxConst>;
                kernel<<<grid, Kernel_traits::kNThreads,
                         smem_size, stream>>>(params);
              });
            });
          });
        });
      });
    });
}
```

### Split-KV 版本

[flash_fwd_launch_template.h:L101-L140](src/flash_fwd_launch_template_h.md#__codelineno-0-101)

```cpp
template<typename Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd(Flash_fwd_params &params,
                           cudaStream_t stream) {
    const int num_m_block =
        cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    /*
     * Split-KV grid: 在标准的 (Q块, batch, head) 基础上
     * 增加 num_splits 维度, 每个 split 处理 K/V 的一段
     * 最后由 combine kernel 合并各 split 的部分结果
     */
    dim3 grid(num_m_block,
              params.num_splits > 1 ? params.num_splits
                                    : params.b,
              params.num_splits > 1 ? params.b * params.h
                                    : params.h);

    // 嵌套 switch 结构与 run_flash_fwd 相同
    // 但 kernel 为 flash_fwd_splitkv_kernel, 额外传入 num_splits
    auto kernel = &flash_fwd_splitkv_kernel<...>;
    kernel<<<grid, Kernel_traits::kNThreads,
             smem_size, stream>>>(params);

    /*
     * Combine kernel: 合并各 split 的 (O_partial, LSE_partial)
     * 原理同 online softmax: 按 LSE 重新缩放各 split 的 O 再求和
     * Grid: (num_m_block * seqlen_q / kBlockM, batch, head)
     */
    if (params.num_splits > 1) {
        // combine 用独立的 kernel_traits, 不需要 KV 的 smem
        flash_fwd_splitkv_combine_kernel<<<...>>>(params);
    }
}
```

### Head Dim → Block Size 映射

[flash_fwd_launch_template.h:L140-L317](src/flash_fwd_launch_template_h.md#__codelineno-0-140)

每个 `run_mha_fwd_hdimN` 函数根据 head dim 和硬件（SM80 vs SM8x）选择最优的 `(kBlockM, kBlockN, kNWarps)` 组合。

```cpp
template<typename T, bool Is_causal>
void run_mha_fwd_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<
            Headdim, 128, 128, 4,                // kBlockM=128, kBlockN=128, 4 warps
            false, false, T>,                     // no even_K, no share_Q_K
            Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // SM86/89 causal 用 64x64 (方阵更快), 非 causal 用 128x64
        if (is_sm8x) {
            if constexpr(!Is_causal) {
                run_flash_fwd<Flash_fwd_kernel_traits<
                    Headdim, 128, 64, 4, false, false, T>,
                    Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<
                    Headdim, 64, 64, 4, false, false, T>,
                    Is_dropout, Is_causal>(params, stream);
            }
        } else {                                  // SM80 (A100)
            run_flash_fwd<Flash_fwd_kernel_traits<
                Headdim, 128, 64, 4, false, false, T>,
                Is_dropout, Is_causal>(params, stream);
        }
    });
}
// run_mha_fwd_hdim96: 与 hdim64 类似, SM8x causal 用 64x64, 其余 128x64

template<typename T, bool Is_causal>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            /*
             * SM8x 非 causal: 128x32 (48KB smem, 可 2 CTA/SM)
             * SM8x causal: 64x64 (方阵)
             * SM80: 128x64
             */
            if (is_sm8x) {
                if constexpr(!Is_causal)
                    run_flash_fwd<..., 128, 32, 4, ...>(...);
                else
                    run_flash_fwd<..., 64, 64, 4, ...>(...);
            } else {
                run_flash_fwd<..., 128, 64, 4, ...>(...); // A100/H100
            }
        } else {
            // dropout 模式统一 128x32 (smem 较小)
            run_flash_fwd<..., 128, 32, 4, ...>(...);
        }
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim192(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            // 无 dropout: 128x64, 8 warps (大 headdim 需更多寄存器)
            run_flash_fwd<..., 128, 64, 8, ...>(...);
        } else {
            // dropout: 降为 64x64, 4 warps (节省寄存器给 RNG 状态)
            run_flash_fwd<..., 64, 64, 4, ...>(...);
        }
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    /*
     * d=256 需要运行时查询 smem 容量
     * A100: max_smem_per_block=164KB, 可用 128x64 (128KB smem)
     * H100: max_smem_per_sm=228KB, 用 64x64 (96KB) 可 2 CTA/SM
     */
    cudaDeviceGetAttribute(&max_smem_per_sm,
        cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    cudaDeviceGetAttribute(&max_smem_per_block,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 2 * 256 * (128 + 128) &&
            max_smem_per_sm < 4 * 256 * (64 + 128)) {
            // A100 路径: 128x64, 8 warps
            run_flash_fwd<..., 128, 64, 8, ...>(...);
        } else {
            // H100 路径: 64x64, 4 warps (2 CTA/SM)
            run_flash_fwd<..., 64, 64, 4, ...>(...);
        }
    });
}
```

Block size 选择总结：

| head_dim | SM80 (A100) | SM8x (A6000等) causal | SM8x non-causal | warps |
|----------|------------|----------------------|-----------------|-------|
| 32 | 128×128 | 128×128 | 128×128 | 4 |
| 64 | 128×64 | 64×64 | 128×64 | 4 |
| 96 | 128×64 | 64×64 | 128×64 | 4 |
| 128 | 128×64 | 64×64 | 128×32 | 4 |
| 192 | 128×64 | 128×64 | 128×64 | 8 |
| 256 | 128×64 | 64×64 | 64×64 | 4-8 |

核心权衡：SM8x smem 较小，用更小的 block（如 64×64、128×32）以提高 occupancy（2 CTA/SM）；大 head_dim 增加 warp 数以分摊寄存器压力。

## 模板实例化：generate_kernels.py

FA2 的 74 个 `.cu` 文件由 Python 脚本生成，每个文件约 10-14 行，实例化一个特定的 `(head_dim, dtype, causal, direction)` 组合：

```cpp
// flash_fwd_hdim128_fp16_causal_sm80.cu (自动生成)
#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<cutlass::half_t, 128, true>(
    Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128<cutlass::half_t, true>(params, stream);
}
```

实例化维度：

| 维度 | 取值 | 个数 |
|------|------|------|
| head_dim | 32, 64, 96, 128, 192, 256 | 6 |
| dtype | fp16, bf16 | 2 |
| causal | true, false | 2 |
| 方向 | fwd, fwd_split, bwd | 3 |

总计约 72 个文件。每个文件独立编译，可并行 make。编译优化措施：

- **分离编译**：每个 `.cu` 独立翻译单元，`make -j` 并行
- **禁用宏**：`FLASHATTENTION_DISABLE_*` 裁剪不需要的特性分支
- **条件折叠**：launch template 中的复合条件表达式在不满足时折叠为 false，避免无效实例化

关联：CUTLASS 的模板实例化模式类似，参见 [Tensor Core 与 CUTLASS](../cutlass_gemm_blog/03_tensorcore_and_cutlass.md)。
