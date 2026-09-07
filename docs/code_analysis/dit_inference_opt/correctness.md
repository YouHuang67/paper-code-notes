---
tags:
  - Diffusion Model
  - Video Generation
  - LLM Inference
---
# DiT 推理优化：Correctness 与部署约束

优化数字只有在「硬件、模型、shape、精度、denoise、e2e、图像指标」说清楚时才有意义。本篇收束组合约束与 serving 相关正确性，不把专题改名为 serving。

## 1. 输出等价边界

官方用 **output-preserving** 而非承诺 bit-exact lossless：换 kernel、GPU、精度路径仍可能有微小数值差。决策边界是：优化是否 **故意** 用质量换速度。

质量敏感路径：Cache、Progressive、Quantization、近似 Attention backend。

## 2. 已核验的互斥 / 自动降级

| 约束 | 来源 |
|------|------|
| Cache-DiT ⊥ DiT layerwise offload | `server_args` 硬错误 |
| Cache-DiT ⊥ FSDP inference | 硬错误或自动关 FSDP |
| TeaCache ⊥ Spectrum | sampling_params |
| BCG 需 warmup resolutions | `server_args` |
| BCG 仅白名单模型 | 自动 disable |
| 量化适配器可能关 offload | loader adapters |

评测时若日志出现 Diffusers fallback，不能用来证明 Native Backend 速度。

## 3. Serving 相关能力（索引）

文档与代码中已存在、待后续加厚：

- Dynamic batching（兼容 shape）  
- DP replica  
- Disaggregation / Mooncake 传数据（Encoder–Denoiser–Decoder 拆分）  
- BCG warmup resolution、prompt bucket、serving signature（warmup capture 成功 ≠ 真实请求必 replay）  

H3 侧 admission / denoise 状态机：[Denoise Loop](../minimax_h3/08_denoise_loop_state_machine.md)。

## 4. 建议验收清单

1. 固定模型、分辨率、帧数、步数、GPU、精度、随机种子协议  
2. 先 output-preserving 基线（含 profile）  
3. 再开单一 quality-tradeoff，看图像/视频指标与 e2e  
4. 记录互斥开关实际生效情况（自动降级也要写进报告）  

## 5. 本轮缺口

- Fast Path `lossless` / `extra-high` / `high` 与 quality gate 源码表  
- Mooncake / disaggregation 角色与失败模式  
- 多步 BF16 rounding 误差累积的回归用例索引
