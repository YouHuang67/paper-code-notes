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

**源码**: [static_switch.h](src/static_switch_h.md)

FA2 的所有特性开关（causal、dropout、alibi 等）都是**编译期常量**，通过嵌套宏实现：

```cpp
'''
BOOL_SWITCH: 核心宏，将运行时 bool 转为编译期 constexpr
'''
#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \  // 实例化 true 版本
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \  // 实例化 false 版本
    }                                           \
  }()
```

每个 `BOOL_SWITCH` 使二进制膨胀 2×。FA2 使用多个变体避免不必要的膨胀：

| 宏 | 默认行为 | 禁用宏 | 禁用效果 |
|---|---|---|---|
| `DROPOUT_SWITCH` | 同 `BOOL_SWITCH` | `FLASHATTENTION_DISABLE_DROPOUT` | 始终 false |
| `ALIBI_SWITCH` | 同上 | `FLASHATTENTION_DISABLE_ALIBI` | 始终 false |
| `EVENK_SWITCH` | 同上 | `FLASHATTENTION_DISABLE_UNEVEN_K` | 始终 true |
| `SOFTCAP_SWITCH` | 同上 | `FLASHATTENTION_DISABLE_SOFTCAP` | 始终 false |
| `LOCAL_SWITCH` | 同上 | `FLASHATTENTION_DISABLE_LOCAL` | 始终 false |

**FP16_SWITCH**：选择 `cutlass::half_t` 或 `cutlass::bfloat16_t`。

**HEADDIM_SWITCH**：将运行时 head_dim 映射到编译期常量 32/64/96/128/192/256。

## flash_fwd_launch_template.h：Kernel 启动

### Grid 配置

[flash_fwd_launch_template.h:L54-L99](src/flash_fwd_launch_template_h.md#__codelineno-0-54)

```cpp
template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = ceil_div(params.seqlen_q, Kernel_traits::kBlockM);

    dim3 grid(num_m_block, params.b, params.h);  // (Q行块数, batch, head)
    // 每个 thread block 处理 1 个 Q 行块 × 1 个 (batch, head)
```

Block 配置：`kNThreads = kNWarps * 32`（通常 128 线程 = 4 warp）

### 嵌套 Switch 展开

```cpp
BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
  EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
    LOCAL_SWITCH(..., Is_local, [&] {
      BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
        ALIBI_SWITCH(..., Has_alibi, [&] {
          SOFTCAP_SWITCH(..., Is_softcap, [&] {
            auto kernel = &flash_fwd_kernel<Kernel_traits,
                Is_dropout, Is_causal, Is_local, Has_alibi,
                IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst>;
            kernel<<<grid, kNThreads, smem_size, stream>>>(params);
          });
        });
      });
    });
  });
});
```

6 层嵌套理论上产生 $2^6 = 64$ 个 kernel 变体，但代码中有多处优化减少实例化数量：

- `Is_local && !Is_causal`：local 和 causal 互斥
- `IsEvenMNConst` 只在无特殊特性时为 true
- `ReturnSoftmaxConst` 只在有 dropout 时可能为 true

### Shared Memory 配置

```cpp
if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
}
```

SM80 默认动态 shared memory 上限 48 KB，FA2 常超过此值（如 d=128 时约 64 KB），需要显式设置。

### Split-KV 版本

[flash_fwd_launch_template.h:L101+](src/flash_fwd_launch_template_h.md#__codelineno-0-101)

Grid 变为三维：`(num_m_block, num_splits, batch * head)` 或 `(num_m_block, batch, head)`，视 `num_splits > 1` 而定。Split-KV 的 combine kernel 作为独立 launch。

## 模板实例化：generate_kernels.py

FA2 的 74 个 `.cu` 文件由 Python 脚本生成，每个文件约 10-14 行：

```cpp
// flash_fwd_hdim128_fp16_causal_sm80.cu
#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<cutlass::half_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128<cutlass::half_t, true>(params, stream);
}
```

### 实例化维度

| 维度 | 取值 | 个数 |
|------|------|------|
| head_dim | 32, 64, 96, 128, 192, 256 | 6 |
| dtype | fp16, bf16 | 2 |
| causal | true, false | 2 |
| 方向 | fwd, fwd_split, bwd | 3 |

总计约 72 个文件。每个文件编译一个特定的 `(head_dim, dtype, causal, direction)` 组合。

### Head Dim → Block Size 映射

[flash_fwd_launch_template.h 中的 run_mha_fwd_hdimN 函数](src/flash_fwd_launch_template_h.md#__codelineno-0-140)

```
d=32:  kBlockM=128, kBlockN=128, kNWarps=4
d=64:  kBlockM=128, kBlockN=64,  kNWarps=4
d=96:  kBlockM=64,  kBlockN=64,  kNWarps=4
d=128: kBlockM=128, kBlockN=64,  kNWarps=4
d=192: kBlockM=64,  kBlockN=64,  kNWarps=8  (SM80 限制)
d=256: kBlockM=64,  kBlockN=64,  kNWarps=8
```

大 head dim 降低 kBlockM 和增加 kNWarps 以适配 shared memory 和寄存器压力。

## 编译优化策略

FA2 的模板膨胀是编译时间的主要瓶颈。几个缓解措施：

1. **分离编译**：每个 `.cu` 文件独立编译，可并行 make
2. **禁用宏**：`FLASHATTENTION_DISABLE_*` 系列宏可在编译时禁用不需要的特性，减少变体数量
3. **条件简化**：launch template 中的 `IsEvenMNConst && ... && kHeadDim <= 128` 表达式在不满足条件时折叠为 false，避免实例化

关联：CUTLASS 的模板实例化模式类似，参见 [Tensor Core 与 CUTLASS](../cutlass_gemm_blog/03_tensorcore_and_cutlass.md)。
