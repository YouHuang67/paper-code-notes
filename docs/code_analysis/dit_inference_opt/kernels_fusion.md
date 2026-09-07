---
tags:
  - Diffusion Model
  - CUDA
  - Triton
  - LLM Inference
---
# DiT 推理优化：Kernels & Fusion

**源码**: `refs/codes/sglang/python/sglang/kernels/ops/diffusion/` @ sglang `db75dfe…`

返回：[专题总览](overview.md)

## 抓住重点

- DiT 热路径上，大量时间在 Norm / 调制 / RoPE / 残差门控等 **小算子 launch** 与 **中间 Tensor 读写**，不在「再抠 5% GEMM」。
- 融合与删临时缓冲，往往比单点 kernel 微优化更能拉动 e2e。
- 公开 `register_kernel` 的 `diffusion.*` op 少于目录文件数：许多融合以模块函数被具体模型直接调用。
- 评价融合要用 **未开 cache** 的基线；静段融合更适合塞进 [BCG](graph_runtime.md)。

## 1. 注册与后端

`diffusion/__init__.py` 通过 `register_kernel(KernelSpec(...))` 暴露公开 op，pin 上可见例如：

| Op | 后端 | 作用 |
|----|------|------|
| `diffusion.apply_group_norm_silu` | Triton | GroupNorm + SiLU |
| `diffusion.residual_gate_add` | JIT | residual + gate * update |
| `diffusion.fused_linear_gelu_tanh` | TORCH / cublasLt epilogue | Linear + tanh-GELU |
| `diffusion.fused_inplace_qknorm_rope` | JIT | in-place QK RMSNorm + RoPE |
| `diffusion.sparse_linear_attn_fwd` | Triton | 稀疏线性注意力（算法见稀疏专题） |

同目录还有 `triton/`、`cutedsl/`、`flydsl/`、`render/` 等实现树（约 40+ 相关 `.py`）。公开 `__all__` 仅导出部分符号；热路径枚举需跟到具体 DiT 实现的调用点。

## 2. 典型融合语义

| 融合 | 输入 → 输出 | 为什么有效 |
|------|-------------|------------|
| QK Norm + RoPE | Q/K + 权重 + cos/sin → in-place Q/K | 去掉多余写回与二次读 |
| Residual gate-add | residual, update, gate → residual+gate×update | DiT AdaLN 门控常见模式 |
| GroupNorm + SiLU | x + norm 参数 → 激活 | norm+act 一条龙 |
| Linear + tanh-GELU | 投影 + 激活 | 吃 epilogue，少一次全局写 |

H3 样板中 fused AdaLN / QKNorm / RoPE 的业务含义：[H3 效率主线](../minimax_h3/05_efficiency_in_sglang.md)、[DiT Runtime](../minimax_h3/07_dit_runtime_and_collectives.md)。

## 3. 「删中间 Tensor」类优化

另一类收益：去掉多余临时缓冲（量化 QKV 合并、FP8 Producer Fusion、数据重排产生的数 GB 临时 Tensor）。流程：

1. 用真实模型 profile 定位 shape 与热点  
2. 保证数值路径（bit-exact 或过 quality gate）→ [Correctness](correctness.md)  
3. 再看 e2e  

与量化路径的关系：[Quantization](quantization.md)（低 bit 若前后仍造大 BF16 Tensor，GEMM 省下的时间会被搬运吃掉）。

## 4. 与 Graph / Cache / Parallel 的关系

- 稳定、静态 shape 的融合段 → 适合 BCG segment（[Graph Runtime](graph_runtime.md)）。  
- 动态 Attention、变长、通信 → Eager。  
- Cache 跳过整段 block 时，被跳过路径上的融合收益同时消失 → 评价融合用未开 cache 基线（[Feature Cache](feature_cache.md)）。  
- 多卡通信重叠与 kernel 选择：[Parallelism](parallelism.md)。

## 5. 源码锚点

| 主题 | 路径 |
|------|------|
| 公开 registry | `kernels/ops/diffusion/__init__.py` |
| Triton / CuTeDSL / FlyDSL 树 | `kernels/ops/diffusion/{triton,cutedsl,flydsl}/` |
| H3 调用样板 | `docs/code_analysis/minimax_h3/` |

## 相关阅读

- [专题总览](overview.md) · [Graph Runtime](graph_runtime.md) · [Quantization](quantization.md) · [Correctness](correctness.md)

## 附录：仍可加深

- 按模型（FLUX / Wan / Qwen-Image / H3）枚举实际调用的 fusion 列表  
- Agent+profile 工作流与 CI kernel 回归入口  
- NVFP4 / FP8 Producer fusion 的源码级拆解
