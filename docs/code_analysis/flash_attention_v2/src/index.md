---
tags:
  - CUDA
  - CUTLASS
  - Flash Attention
---
# Flash Attention V2 源码

**仓库**: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) · **路径**: `csrc/flash_attn/src/` · **解读**: [代码分析](../00_overview.md)

**参数与配置**

| 文件 | 行数 | 说明 |
|------|------|------|
| [flash.h](flash_h.md) | 194 | 前向/反向参数结构体 |
| [kernel_traits.h](kernel_traits_h.md) | 344 | MMA Atom、Smem Layout、Copy 配置 |
| [static_switch.h](static_switch_h.md) | 111 | 编译期特性分发宏 |
| [block_info.h](block_info_h.md) | 49 | 变长序列偏移计算 |

**前向 Kernel**

| 文件 | 行数 | 说明 |
|------|------|------|
| [flash_fwd_kernel.h](flash_fwd_kernel_h.md) | 1294 | 前向核心：compute_attn_1rowblock + split-KV |
| [flash_fwd_launch_template.h](flash_fwd_launch_template_h.md) | 304 | 前向 host 端调度与 kernel 启动 |

**反向 Kernel**

| 文件 | 行数 | 说明 |
|------|------|------|
| [flash_bwd_kernel.h](flash_bwd_kernel_h.md) | 841 | 反向核心：compute_dq_dk_dv_1colblock |
| [flash_bwd_launch_template.h](flash_bwd_launch_template_h.md) | 308 | 反向 host 端调度 |
| [flash_bwd_preprocess_kernel.h](flash_bwd_preprocess_kernel_h.md) | 383 | 反向预处理：dot(dO, O) |

**计算原语**

| 文件 | 行数 | 说明 |
|------|------|------|
| [softmax.h](softmax_h.md) | 189 | Online softmax（增量 max/sum/rescale） |
| [utils.h](utils_h.md) | 413 | Warp reduce、GEMM 封装、layout 转换 |
| [mask.h](mask_h.md) | 214 | Causal / local / ALiBi mask |
| [dropout.h](dropout_h.md) | 95 | Philox RNG dropout |
| [rotary.h](rotary_h.md) | 153 | RoPE 旋转位置编码 |
