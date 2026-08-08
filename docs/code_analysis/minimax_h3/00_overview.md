---
tags:
  - Video Generation
  - Diffusion Model
  - Unified Understanding
---
# MiniMax H3 代码分析：总览

**源码仓库**: `refs/codes/minimax_h3`

**补充实现**: `refs/codes/diffusers/src/diffusers/{models,modular_pipelines,schedulers}/...`

**分析范围**:

- `README.md`
- `transformer/config.json`
- `transformer_minimax_h3.py`
- `encoders.py`
- `before_denoise.py`
- `denoise.py`
- `scheduling_minimax_h3.py`
- `autoencoder_kl_minimax_h3.py`
- `autoencoder_kl_minimax_h3_audio.py`

这组代码的核心不是“再做一个跨模态 encoder-decoder”，而是把整次生成统一改写成一个 **单流 packed multimodal diffusion sequence**：

- 文本条件由 `Qwen3-VL` 第 50 层隐藏状态提供
- 图像 / 视频 / 音频先进入各自 VAE latent 空间
- 所有 modality 被重排成同一条 1D token 序列
- 一个 50 层、33B 级的 `MiniMaxH3Transformer3DModel` 对整条序列同时预测视频与音频 velocity
- 视频与音频各自沿自己的 rectified-flow scheduler 反推回干净 latent，再分别解码

对应源码主链：

```text
encoders.py
  -> 构造 Qwen3-VL presentation / prompt embeds
  -> 视觉条件与音频条件编码成 latent

before_denoise.py
  -> 计算输出分辨率 / 帧数 / latent 形状
  -> 把 text / condition / target audio / target video 打包成一条 sequence
  -> 生成 rotary 位置、row 索引、噪声初始化、per-row timestep plan

transformer_minimax_h3.py
  -> 单流 full self-attention + AdaLN 调制
  -> 同时输出 video velocity 和 audio velocity

denoise.py + scheduling_minimax_h3.py
  -> 视频 / 音频各自按不同 shift 的 rectified-flow Euler 调度更新

decoders.py + VAE
  -> 视频 latent 解码成 24 FPS 视频
  -> 音频 latent 解码成 32 kHz 双声道音频
```

## 1. 先看清哪些东西开源了，哪些没开

`README.md` 直接说明了三段式系统：

- `H3-Context-IR`：多模态理解、整理和语义补全，未开源
- `H3-Base`：768p 音视频联合生成，开源核心就是这部分
- `H3-Regenerate-2K`：2K 再生成模块，未开源

所以这次能深入分析的“推理代码”，本质上是 **H3-Base 在 diffusers 侧的本地推理实现**，不是完整线上系统。

## 2. 这份实现最重要的四个判断

### 2.1 不是 cross-attention 架构，而是单流 full attention

`MiniMaxH3Transformer3DModel` 的说明写得非常直接：[transformer_minimax_h3.py:L374-L387](src/transformer_minimax_h3_py.md#__codelineno-0-374)

- 整个模型在一条 packed 1D sequence 上做 full self-attention
- 没有 cross-attention
- 没有 modality-specific attention / FFN block
- modality 差异只留在输入投影、AdaLN 分支和输出头

这意味着 MiniMax H3 的统一多模态不是“双塔交互”式，而是 **先把所有条件压成统一 token 时空坐标，再交给一个大 Transformer 直接建模**。

### 2.2 统一的不是原始像素，而是统一的 latent token 序列

`README.md` 和 `transformer/config.json` 给出的形状关系是：

- 视频 VAE：`f16t4d24`
- patch size：`(1, 2, 2)`
- 音频 latent 通道：`32`
- Transformer hidden size：`5376`
- 头数 `56`，每头 `128`

落到实现里，就是：

- 视频 latent 先按 `[24, T, H, W]` 压到 VAE 空间
- 再 patchify 成 `[num_video_rows, 24 * 1 * 2 * 2] = [num_video_rows, 96]`
- 音频 latent 直接按行表示成 `[num_audio_rows, 32]`
- 文本条件是 `Qwen3-VL` 的 `5120` 维隐藏状态
- 三者分别线性投影到同一 `hidden_size=5376` 残差流中，再 scatter 到同一 sequence buffer

关键实现见 [transformer_minimax_h3.py:L505-L555](src/transformer_minimax_h3_py.md#__codelineno-0-505) 与 [transformer_minimax_h3.py:L622-L634](src/transformer_minimax_h3_py.md#__codelineno-0-622)。

### 2.3 统一位置编码不是 1D，而是 3D MM-RoPE

RoPE 模块显式使用 `(t, h, w)` 三轴：[transformer_minimax_h3.py:L74-L98](src/transformer_minimax_h3_py.md#__codelineno-0-74)。

更关键的是，`before_denoise.py` 并不是简单给每个 row 一个递增位置，而是：

- 文本 token 只占用时间轴
- 视频 row 拥有完整 `(t, h, w)`
- 音频 row 没有高度轴，只被钉在 width 两端
- `ref2va` 中参考图像 / 视频 / 音频会共同推进一条共享的 rotary clock

这使得“不同模态能互相看见彼此”并不只是来自 attention，而是已经由 **同一时空坐标系** 强行对齐。

### 2.4 这版开源推理仍然是 full attention，但代码已经为更强后端留了口

`README.md` 明说训练和未来推理支持 native sparse attention，但初始开源版只放 full attention。`MiniMaxH3AttnProcessor` 当前直接调用 `dispatch_attention_fn(...)`，没有专用稀疏索引结构：[transformer_minimax_h3.py:L158-L207](src/transformer_minimax_h3_py.md#__codelineno-0-158)。

因此当前要分析的重点不是“稀疏 kernel 细节”，而是：

- packed sequence 如何降低多模块拼装成本
- per-row timestep / modality AdaLN 如何统一噪声级别不同的 token
- 双 scheduler 如何驱动同一 transformer 输出的两种模态
- 混合精度、context parallel、anchor row 冻结这些工程细节如何保证推理可落地

## 3. 文档拆分

这次按 6 篇文档组织，正文和附录分层：

- [总览](00_overview.md)
- [模型结构](01_model_architecture.md)
- [推理流程](02_inference_pipeline.md)
- [优化与实现细节](03_optimizations_and_details.md)
- [如何嵌入 SGLang 体系](04_sglang_integration.md)
- [SGLang 中的效率主线](05_efficiency_in_sglang.md)
- [效率附录](06_efficiency_appendix.md)

源码浏览页集中在：

- [源码索引](src/index.md)
- [README.md](src/readme_md.md)
- [transformer/config.json](src/transformer_config_json.md)
- [transformer_minimax_h3.py](src/transformer_minimax_h3_py.md)
- [modular_blocks_minimax_h3.py](src/modular_blocks_minimax_h3_py.md)
- [before_denoise.py](src/before_denoise_py.md)
- [encoders.py](src/encoders_py.md)
- [denoise.py](src/denoise_py.md)
- [scheduling_minimax_h3.py](src/scheduling_minimax_h3_py.md)
- [autoencoder_kl_minimax_h3.py](src/autoencoder_kl_minimax_h3_py.md)
- [autoencoder_kl_minimax_h3_audio.py](src/autoencoder_kl_minimax_h3_audio_py.md)

## 4. 推荐阅读顺序

建议按下面顺序读：

1. [模型结构](01_model_architecture.md)：先建立 packed sequence + AdaLN + 单流 Transformer 的总心智模型
2. [推理流程](02_inference_pipeline.md)：再看 `t2va` / `fl2va` / `ref2va` 如何分别走 layout、噪声和循环
3. [优化与实现细节](03_optimizations_and_details.md)：补齐 open-source diffusers 路径的工程细节
4. [如何嵌入 SGLang 体系](04_sglang_integration.md)：澄清 H3 在 SGLang 中不是 generic fallback，而是 native pipeline
5. [SGLang 中的效率主线](05_efficiency_in_sglang.md)：只抓最核心的计算加速 / 通信优化 / 算子设计
6. [效率附录](06_efficiency_appendix.md)：最后回看细节、拓扑和补充代码路径

## 小结

MiniMax H3 这份开源推理实现最值得拆的不是某个单点 trick，而是它把统一多模态生成压成了一个非常干净的执行范式：

- **统一输入形式**：所有条件都转成 row 序列
- **统一位置系统**：所有 row 都落在 `(t, h, w)` 坐标中
- **统一主干网络**：一个 dense single-stream transformer 处理全部 modality
- **统一训练目标**：视频和音频都做 velocity prediction
- **分离调度与解码**：共享主干，但 scheduler / decoder 按 modality 分开

这套设计很像把“多模态理解 + 视频 diffusion + 音频 diffusion + 条件控制”尽量折叠到一条主执行链里。
