---
tags:
  - Diffusion Model
  - CUDA
  - Triton
  - LLM Inference
---
# DiT 推理优化：Kernels & Fusion

**源码**: `refs/codes/sglang/python/sglang/kernels/ops/diffusion/` @ sglang `db75dfe…`

DiT 热路径上，大量时间不在「再抠 5% GEMM」，而在 Norm/调制/RoPE/残差门控等小算子的 launch 与中间 Tensor 读写。融合与删掉临时缓冲，往往比单点 kernel 微优化更影响端到端。

## 1. 注册与后端

`diffusion/__init__.py` 通过 `register_kernel(KernelSpec(...))` 暴露公开 op，例如：

- `diffusion.apply_group_norm_silu`（Triton）
- `diffusion.residual_gate_add`（JIT）
- `diffusion.fused_linear_gelu_tanh`（TORCH / cublasLt epilogue）
- `diffusion.fused_inplace_qknorm_rope`（JIT，in-place）
- `diffusion.sparse_linear_attn_fwd`（稀疏线性注意力；算法细节见稀疏专题）

同目录还有 `triton/`、`cutedsl/`、`flydsl/`、`render/` 等实现树（pin 上约 40+ 相关 `.py`）。公开 registry 条目少于文件数：许多融合以模块函数形式被具体 DiT 实现直接调用。

## 2. 典型融合语义

| 融合 | 输入 → 输出 | 为什么有效 |
|------|-------------|------------|
| QK Norm + RoPE | Q/K 权重与 cos/sin cache → in-place Q/K | 去掉多余写回与二次读 |
| Residual gate-add | residual, update, gate → residual+gate*update | DiT AdaLN 门控常见模式 |
| GroupNorm + SiLU | x + norm 参数 → 激活 | 标准「norm+act」一条龙 |
| Linear + tanh-GELU | 投影 + 激活 | 吃 epilogue，少一次全局写 |

H3 样板中 fused AdaLN / QKNorm / RoPE 的业务含义见 [H3 效率主线](../minimax_h3/05_efficiency_in_sglang.md) 与 [DiT Runtime](../minimax_h3/07_dit_runtime_and_collectives.md)。

## 3. 「删中间 Tensor」类优化

BBuf 强调的另一类收益：不是把算子再快 5%，而是去掉多余临时缓冲（例如量化 QKV 合并、FP8 Producer Fusion、Wan 数据重排产生的数 GB 临时 Tensor）。这类改动必须：

1. 用真实模型 profile 定位 shape 与热点  
2. 保证数值路径（最好 bit-exact 或过 quality gate）  
3. 再看 e2e  

本专题后续会按模型热路径补「融合前后 buffer 生命周期」对照；首轮以目录与 registry 为地图。

## 4. 与 Graph / Cache 的关系

- 稳定、静态 shape 的融合段更适合塞进 BCG segment（见 [Graph Runtime](graph_runtime.md)）  
- 动态 Attention、变长、通信仍宜 Eager  
- Cache 跳过整段 block 时，被跳过路径上的融合收益同时消失——评价融合要用「未开 cache」基线

## 5. 本轮缺口

- 按模型（FLUX / Wan / Qwen-Image / H3）枚举实际调用的 fusion 列表  
- Agent+profile 工作流与 CI kernel 回归入口  
- 量化 Producer fusion 与 NVFP4 路径的源码级拆解
