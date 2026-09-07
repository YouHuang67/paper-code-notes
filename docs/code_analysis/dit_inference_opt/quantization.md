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

## 1. 加载约定

| 标志 | 用途 |
|------|------|
| `--model-path` | 基座 / 完整 Diffusers 仓库 |
| `--transformer-path` | 已量化 transformer 组件目录（含 config） |
| `--transformer-weights-path` | 原始权重导出（如 raw NVFP4、Nunchaku svdq 文件） |
| `--quantization` | 对未量化模型做 online 量化 |
| `--kv-cache-quant` | 因果视频等场景的 KV 压缩 |

官方强调：mixed override 与 full repo、raw export 的加载路径不同；以 `quantization.mdx` Quick Reference 为准。

## 2. 家族地图（`quant_family`）

`quant_family` = 共享 CLI 与 loader 约定的 checkpoint 族：

| 家族 | 特征 | 备注 |
|------|------|------|
| Online FP8 | 加载时动态量化激活/权重路径 | 无预量化 ckpt 时可用 |
| Online MXFP4 | 平台约束更强（文档绑定 ROCm 等） | 先读官方前置条件 |
| ModelOpt FP8 | `quant_method=modelopt` 的 FP8 导出 | 验证列表见文档 Validated 表 |
| ModelOpt NVFP4 | 含 FLUX.2-dev-NVFP4、Qwen-Image NVFP4 等 | Blackwell 默认 FlashInfer 等路径；1024² 上未必总更快 |
| Nunchaku / SVDQuant | `svdq-{int4\|fp4}_r{rank}-...` 权重；上游 `nunchux-ai/nunchaku` | 本仓本轮未 clone；CLI `int4` / `nvfp4`；runtime 校验限 NVIDIA Ampere+ |
| Causal KV quant | INT 低 bit KV | 适用模型与门禁见文档专节 |

完整 validated checkpoint 表很长，以 pin 内 `quantization.mdx` 为准，本篇不复制全表。

## 3. 端到端成本与组合

- Loader 可能自动关闭不兼容的 offload 模式 → [Memory Offload](memory_offload.md#5-组合约束)。
- 「Token Cat + NVFP4 Quant」类融合决定量化是否真体现在 e2e → [Kernels](kernels_fusion.md#7-删中间-tensor-与纯数据搬移)。
- 量化改变数值路径，与 BCG 静段假设需实测；质量验收 → [Correctness](correctness.md)。

令原始 GEMM、量化准备、反量化/布局转换时间分别为 \(G,Q,D\)，则量化收益要求
\[
G-(G_q+Q+D)>0.
\]
同时输出误差需满足 \(d(y,\hat y)\le\varepsilon\)。因此 profile 必须同时记录 GEMM、Q/D、cat/transpose 和峰值显存；producer fusion 见 [Kernels](kernels_fusion.md)。

`server_args` 的 loader 会依据 checkpoint 族选择预量化或 online 路径；两者的 scale 存储、offload 兼容性和质量门禁不同，不能只凭文件名推断行为。

加载阶段还决定运行时行为：预量化 transformer 通常通过 `transformer-path` 或 raw `transformer-weights-path` 接入，online quantization 则由 `quantization` 选择。两类路径的 config、offload 兼容性和质量门禁不同，不能只凭文件名判断量化格式。

## 4. 源码 / 文档锚点

| 主题 | 路径 |
|------|------|
| 官方家族与加载 | `docs/.../quantization.mdx` |
| Nunchaku 配置解析 | `server_args.py` `_adjust_quant_config` |
| 构建 / 导出工具 | `tools/build_modelopt_nvfp4_transformer.py` |
| 上游索引 | https://github.com/nunchux-ai/nunchaku（未 pin clone） |

## 相关阅读

- [专题总览](overview.md) · [Kernels & Fusion](kernels_fusion.md) · [Memory Offload](memory_offload.md) · [Correctness](correctness.md)

## 附录：仍可加深

- 各 family 的 loader 类图与权重 layout  
- Nunchaku 在 SGLang 中的接线文件级 walkthrough（需 clone 上游对照）  
- Causal KV INT4/INT2 适用模型列表与正确性门禁
