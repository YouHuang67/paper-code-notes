---
tags:
  - Video Generation
  - Diffusion Model
  - Unified Understanding
  - LLM Inference
---
# MiniMax H3 - 如何嵌入 SGLang 体系：native pipeline / native DiT runtime / native packed-row contract

**源码仓库**:

- [MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3)
- [sgl-project/sglang](https://github.com/sgl-project/sglang)

- **H3 不是一个“外部 diffusers 模型，被 SGLang 顺手托管”的案例**
- **它已经被 SGLang 做成了 native diffusion pipeline**
- **后面所有高效率优化，都是基于这条 native pipeline 和 packed-row 执行契约展开的**

## 1. 不是 generic diffusers fallback，而是 native model family

H3 在 SGLang 注册表里被直接识别成 `MiniMaxH3Pipeline`，而不是先尝试通用 diffusers 路径再降级处理。

**源码位置**:

- [registry.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/registry.py)
- [minimax_h3_pipeline.py:L30-L35](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines/minimax_h3_pipeline.py#L30-L35)

这说明服务端在模型选择阶段就承认 H3 需要一套专门 contract，而不是“通用视频 diffusion pipeline 的一个配置变体”。原因在于 H3 的推理语义和常规 video diffusion 差异很大：

- 它是 **视频 + 音频联合 denoise**
- 它是 **CFG-distilled 单分支**
- 它依赖 **packed multimodal row layout**
- 它的 scheduler 是 **视频 / 音频分离但共用一套 DiT 主干**

如果不在注册层就切到 H3 专用 pipeline，后续的调度、缓存和分布式切分都无法对齐。

## 2. `MiniMaxH3Pipeline` 不是空壳，而是 H3 自己的 stage 链

`MiniMaxH3Pipeline` 明确声明了自己的必需组件和 stage 组织方式。

**源码位置**:

- [minimax_h3_pipeline.py:L36-L47](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines/minimax_h3_pipeline.py#L36-L47)
- [minimax_h3_pipeline.py:L101-L149](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines/minimax_h3_pipeline.py#L101-L149)

其主链是：

1. `InputValidationStage`
2. `MiniMaxH3PartitionAdmissionStage`
3. `MiniMaxH3TextEncodingStage`
4. `MiniMaxH3VisualEncodingStage`
5. `MiniMaxH3AudioEncodingStage`
6. `MiniMaxH3LatentPreparationStage`
7. `MiniMaxH3TimestepPreparationStage`
8. `MiniMaxH3DenoisingStage`
9. `MiniMaxH3DecodingStage`

这和通用 diffusers 风格的：

- 准备 latents
- for timestep:
- `unet()`
- `scheduler.step()`

不是一回事。H3 在 SGLang 中的执行单位不是一个 generic scheduler loop，而是一条 **从 request lowering 到 packed denoise loop 的原生 stage pipeline**。

## 3. `model_variant` 不是部署参数，而是权重分区契约

H3 在 SGLang 中被视作两个语义分区：

- `fl2va -> FL2VA`
- `ref2va -> Ref2VA`

**源码位置**:

- [minimax_h3_pipeline.py:L49-L64](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines/minimax_h3_pipeline.py#L49-L64)
- [minimax_h3_pipeline.py:L66-L92](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines/minimax_h3_pipeline.py#L66-L92)

这意味着：

- `fl2va` 分区统一承载 `t2va` 与 `fl2va` 任务族
- `ref2va` 也不是请求层随手开的一个条件开关

SGLang 会在加载期把：

- `--model-variant`
- 实际选中的 `model_subfolder`
- `model_index.json._minimax_h3.partition`

做严格一致性校验。

H3 的公开 release 本身就是按任务族拆分的，SGLang 接入时保留了这个边界，没有把它强行抹平成“单 checkpoint 多任务”。

## 4. H3 在 SGLang 中是 monolithic whole-loop execution

H3 pipeline 明确禁止 disaggregation，只允许 monolithic 角色。

**源码位置**:

- [minimax_h3_pipeline.py:L94-L99](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines/minimax_h3_pipeline.py#L94-L99)
- [pipeline config:L75-L89](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py#L75-L89)

这说明 H3 当前在 SGLang 中的高效路径，不是靠把不同阶段拆给不同 worker，而是靠：

- 单 pipeline 内部的 request-static 预计算
- 单机 / 单拓扑内的 packed-sequence 高效执行
- 单 whole-loop denoise 的状态复用

这也是为什么 H3 的性能分析重点必须落在：

- DiT runtime
- sequence parallel / tensor parallel
- persistent row buffer

而不是传统文本模型那种 prefill/decode 解耦。

## 5. SGLang 配置层已经把 H3 的推理边界钉死了

`MiniMaxH3PipelineConfig` 里有几个字段，本质上已经把 H3 的执行边界写成了硬契约。

**源码位置**:

- [pipeline config:L41-L89](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py#L41-L89)
- [pipeline config:L178-L197](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py#L178-L197)

最关键的是：

- `native_only_components = (...)`
- `should_use_guidance = False`
- `supports_disaggregation = False`
- `supports_cfg_parallel = False`
- `speed_mode_enable_torch_compile_by_default = False`
- `AttentionRequirements(packed_varlen=True)`

翻译成执行语义就是：

- H3 不走通用组件编排
- H3 不走 CFG 双分支
- H3 不走 CFG parallel
- H3 不走默认 compile speed path
- H3 的 attention backend 必须理解 packed varlen sequence

这已经不是“框架里多了一个模型适配器”，而是 **框架专门承认了一种新的计算 contract**。

## 6. 真正关键的嵌入点：native DiT runtime

H3 在 SGLang 里最关键的一层，不是 pipeline 类本身，而是 native DiT 实现：

- [runtime/models/dits/minimax_h3.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py)

它接收的不是通用 `hidden_states + timestep + encoder_hidden_states` 这类 diffusers 风格输入，而是一套 H3 自己的 packed contract：

- `x`
- `audio_x`
- `img_position_ids`
- `packed_seq_params`
- `refiner_packed_seq_params`
- `local_embedding_layout`
- `block_token_tags`
- `block_combined_indices`

**源码位置**:

- [minimax_h3.py:L97-L123](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L97-L123)
- [minimax_h3.py:L1465-L1537](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py#L1465-L1537)

这说明 H3 在 SGLang 中不是“沿用某个通用 DiT forward 签名”，而是 **连 forward contract 都是原生化的**。

## 7. 对“是否嵌入 SGLang”的最终判断

- **是，H3 已经深度嵌入到 SGLang 体系里**
- 嵌入点不只是 serving 命令层
- 也不只是下载 / 请求协议层
- 而是贯穿了：
  - 模型注册
  - checkpoint 分区
  - stage pipeline
  - scheduler contract
  - DiT forward contract
  - sequence-parallel runtime

也因此，真正讨论 H3 “为什么快”，不能只看 `MiniMax-AI/MiniMax-H3` 或 `diffusers` 基线，而必须看这条 SGLang native runtime。效率主线见 [MiniMax H3 在 SGLang 中的效率主线](05_efficiency_in_sglang.md)，状态机与热路径细节见 [Denoise Loop 状态机](08_denoise_loop_state_machine.md) 与 [DiT Runtime 与 Collectives](07_dit_runtime_and_collectives.md)。
