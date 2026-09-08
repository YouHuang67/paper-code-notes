---
tags:
  - Diffusion Model
  - Video Generation
  - LLM Inference
---
# DiT 推理优化：Quantization

**文档**: `refs/codes/sglang/docs/docs/sglang-diffusion/quantization.mdx`  
**相关代码**: `runtime/utils/quantization_utils.py`、`configs/quantization/`、`tools/build_modelopt_nvfp4_transformer.py`

返回：[专题总览](overview.md)

## 抓住重点

- 量化把权重或激活映射到有限集合。对张量 \(x\)，典型形式是 \(\hat x=s\,Q(x/s)\)，其中 \(s>0\) 为 scale，\(Q\) 为低 bit 舍入；误差 \(e=\hat x-x\) 会进入每个 DiT block。
- 低 bit 只有在 GEMM、布局转换和 scale 读取的总成本下降时才加速；量化/反量化及大 BF16 中间量可能抵消 GEMM 收益。
- 加载常拆：`--model-path`（基座）+ `--transformer-path` / `--transformer-weights-path`（量化组件）+ `--quantization`（online）+ `--kv-cache-quant`。
- 部分量化适配器会禁用不兼容的 DiT offload → [Offload](memory_offload.md)。

## 1. 量化对象与误差

对权重 \(W\) 和激活 \(x\)，分组量化可写为 \(\hat W=s_WQ(W/s_W)\)、\(\hat x=s_xQ(x/s_x)\)。以下向量范数取 \(\ell_2\)，矩阵范数取其诱导算子范数。线性层的局部扰动满足
\[
\lVert \hat W\hat x-Wx\rVert
\le \lVert W\rVert\lVert\hat x-x\rVert+
\lVert\hat W-W\rVert\lVert x\rVert+
\lVert\hat W-W\rVert\lVert\hat x-x\rVert.
\]
它说明 bit 宽、分组和 scale 共同决定误差；多步采样会把局部扰动沿 \(T\) 个去噪步骤传播，因此必须以最终输出距离而非单层误差作质量门。

## 2. 家族地图（`quant_family`）

量化家族表示权重表示、scale 布局和可用 kernel 的共同约定：

| 家族 | 特征 | 备注 |
|------|------|------|
| Online FP8 | 加载时动态量化激活/权重路径 | 无预量化 ckpt 时可用 |
| Online MXFP4 | 平台约束更强（文档绑定 ROCm 等） | 先读官方前置条件 |
| ModelOpt FP8 | `quant_method=modelopt` 的 FP8 导出 | 验证列表见文档 Validated 表 |
| ModelOpt NVFP4 | 含 FLUX.2-dev-NVFP4、Qwen-Image NVFP4 等 | Blackwell 默认 FlashInfer 等路径；1024² 上未必总更快 |
| Nunchaku / SVDQuant | `svdq-{int4\|fp4}_r{rank}-...` 权重；上游 `nunchux-ai/nunchaku` | 本仓本轮未 clone；CLI `int4` / `nvfp4`；runtime 校验限 NVIDIA Ampere+ |
| Causal KV quant | INT 低 bit KV | 适用模型与门禁见文档专节 |

加载接口、完整 checkpoint 兼容表与 runtime gate 放在附录；它们决定能否接入，不改变上面的误差模型。

## 3. 端到端成本与组合

- Loader 可能自动关闭不兼容的 offload 模式 → [Memory Offload](memory_offload.md#5-组合约束)。
- 「Token Cat + NVFP4 Quant」类融合决定量化是否真体现在 e2e → [Kernels](kernels_fusion.md#7-删中间-tensor-与纯数据搬移)。
- 量化改变数值路径，与 BCG 静段假设需实测；质量验收 → [Correctness](correctness.md)。

令原始 GEMM、量化准备、反量化/布局转换时间分别为 \(G,Q,D\)，则量化收益要求
\[
G-(G_q+Q+D)>0.
\]
同时输出误差需满足 \(d(y,\hat y)\le\varepsilon\)。因此 profile 必须同时记录 GEMM、Q/D、cat/transpose 和峰值显存；producer fusion 见 [Kernels](kernels_fusion.md)。

预量化与 online 两类路径的 scale 存储、offload 兼容性和质量门禁不同，必须按 checkpoint 元数据与 loader 校验确认。

## 附录：加载与实现证据

| 主题 | 路径 |
|------|------|
| 官方家族与加载 | `docs/.../quantization.mdx` |
| Nunchaku 配置解析 | `server_args.py` `_adjust_quant_config` |
| 构建 / 导出工具 | `tools/build_modelopt_nvfp4_transformer.py` |
| 上游索引 | https://github.com/nunchux-ai/nunchaku（未 pin clone） |

| 加载输入 | 角色 |
|------|------|
| `model-path` | 基座或完整 Diffusers 仓库 |
| `transformer-path` | 已量化 transformer 组件目录 |
| `transformer-weights-path` | 原始量化权重导出 |
| `quantization` / `kv-cache-quant` | online transformer / KV 量化选择 |

## 相关阅读

- [专题总览](overview.md) · [Kernels & Fusion](kernels_fusion.md) · [Memory Offload](memory_offload.md) · [Correctness](correctness.md)

## 附录：仍可加深

- 各 family 的 loader 类图与权重 layout  
- Nunchaku 在 SGLang 中的接线文件级 walkthrough（需 clone 上游对照）  
- Causal KV INT4/INT2 适用模型列表与正确性门禁
