---
tags:
  - Video Generation
  - Diffusion Model
  - Unified Understanding
---
# MiniMax H3：推理流程

本文沿 `encoders.py -> before_denoise.py -> denoise.py -> scheduling_minimax_h3.py` 展开 MiniMax H3 的实际推理链。

本文聚焦三种工作流 `t2va` / `fl2va` / `ref2va` 如何被压成同一套执行框架。

## 1. 顶层流水线其实是 5 个 block

`MiniMaxH3Blocks` 的定义很清楚：[modular_blocks_minimax_h3.py:L659-L675](src/modular_blocks_minimax_h3_py.md#__codelineno-0-659)

```text
before_encode
-> text_encoder
-> vae_encoder
-> denoise
-> decode
```

三种 workflow 共享这五段，只是在每一段内部选不同子 block：

- `t2va`
  - 无参考、无 keyframe
- `fl2va`
  - 有首帧 / 尾帧 keyframe
- `ref2va`
  - 有 image / video / audio reference

也就是说，H3 以 **同一 modular pipeline 的三种 layout 策略** 覆盖这三类工作流。

## 2. 先将输入重写成 H3 的 presentation

### 2.1 文本条件来自 Qwen3-VL 第 50 层

`get_qwen3vl_prompt_embeds(...)` 会直接调用 `text_encoder.model(..., output_hidden_states=True)`，读取 `hidden_states[50]`：[encoders.py:L30-L99](src/encoders_py.md#__codelineno-0-30)。

这一步很关键，因为它说明：

- H3 使用 Qwen3-VL 的中间层特征
- 条件作用本质上是“拿大型多模态模型做高层语义表征器”

### 2.2 `t2va` 最简单，直接把 prompt 原样 tokenize

对应 [encoders.py:L139-L208](src/encoders_py.md#__codelineno-0-139)：

- 不套 chat template
- 不加 special tokens
- 没有 negative prompt
- 没有 CFG unconditional branch

这和许多 diffusers pipeline 很不一样，因为 H3 的 checkpoint 已经做了 CFG distillation。

### 2.3 `fl2va` 会把 keyframe 写成 `<Picture i>:` + vision block

见 [encoders.py:L211-L305](src/encoders_py.md#__codelineno-0-211)。

这里最值得注意的是：

- 视觉 pad token 会占一段 token span
- 这段 span 在 `text_token_tags` 中被标为 `video_tag`
- 所以即使它来自“text encoder side”，主干里仍按视频 modality 调制

这说明 H3 的“文本条件”是 **Qwen3-VL presentation 产生的 packed multimodal prompt sequence**。

### 2.4 `ref2va` 会给不同参考模态加不同标签

`MiniMaxH3Ref2VATextEncoderStep` 会构造更复杂的 presentation：[encoders.py:L378-L652](src/encoders_py.md#__codelineno-0-378)

- 图像：`<Picture i>:`
- 音频：`<Audio j>:`
- 视频：`<Video k>:`，并为每个合并时间块加 `<x.x seconds>`

这一步非常重要，因为 reference 的顺序不仅影响 prompt 文本，也会影响后面的 rotary clock。MiniMax H3 实际上把“参考顺序”视作语义的一部分。

## 3. 条件 VAE 编码：视觉条件要采样，音频条件取 posterior mean

### 3.1 视觉条件 latent 采用确定性种子的 posterior 采样

`encode_vae_condition(...)` 的流程是：[encoders.py:L102-L136](src/encoders_py.md#__codelineno-0-102)

1. ImageNet 风格归一化
2. VAE encode 得到 posterior
3. 用固定 `encode_seed=42` 对 posterior 采样
4. 再强制 round 到 float16
5. 用 `latents_mean/std` 做标准化

这很值得注意：H3 的 released recipe 为 condition latent 使用 posterior 采样。

### 3.2 音频条件 latent 则是 deterministic 的

`MiniMaxH3Ref2VAReferenceEncoderStep` 对音频参考使用 posterior `mode()`：[encoders.py:L652-L781](src/encoders_py.md#__codelineno-0-652)。

并且两个声道被当作 mono VAE 的两个 batch item 独立编码，最后 reshape 成 channel-major row layout。

所以音频和视觉条件在编码策略上是不同的：

- 视觉条件：sampled conditioning latent
- 音频条件：posterior mean conditioning latent

## 4. `before_denoise.py` 才是真正的“推理编排中心”

如果只读模型文件，很容易忽略最关键的部分其实在 `before_denoise.py`。因为 MiniMax H3 的复杂度主要不在 transformer block，而在 **如何把所有东西排成它需要的那条 sequence**。

## 5. 视频 / 音频 latent 先被排成 row

`patchify_video_latents(...)` 把 `[B, C, T, H, W]` 重排成 `[num_rows, C * patch_t * patch_h * patch_w]`：[before_denoise.py:L44-L73](src/before_denoise_py.md#__codelineno-0-44)。

视频 row 的顺序是：

- 先按时间
- 再按空间行列

音频则天然就是 row-major，只是用 channel-major 方式堆成：

```text
[left channel rows | right channel rows]
```

## 6. H3 的 RoPE 时钟比普通视频位置编码更细

`before_denoise.py` 定义了几组常量：[before_denoise.py:L36-L41](src/before_denoise_py.md#__codelineno-0-36)

- `_ROPE_FRAME_RESCALE = 5 / 3`
- `_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)`
- `_ROPE_SPATIAL_SCALE = 32`

这和 VAE 的 `17` 帧输入对应 `5` 个 latent frame 是对齐的：

- 每个 latent frame 在 rotary 时间轴上不等间隔
- 第一帧和后续 4 帧的跨度不同

因此，视频 RoPE 显式复用了 VAE 的时域压缩结构。

## 7. `t2va/fl2va` 的 packed layout

`MiniMaxH3PrepareLayoutStep.build_packed_sequence(...)` 定义了最基础布局：[before_denoise.py:L267-L371](src/before_denoise_py.md#__codelineno-0-267)

```text
[text | keyframe conditions | target audio | target video]
```

几个关键点：

- text rows 只占时间轴
- keyframe 条件可以 anchor 在首帧或尾帧
- target audio 与 target video 共用同一 rotary 时钟
- video rows 通过 `video_indices` 指出在 packed sequence 中的位置

这一层已经解释了为什么主干不需要 cross-attention：

- 条件行和生成行都在同一条序列中
- 条件保持在前面
- 生成行直接通过 full attention 读它们

## 8. `ref2va` 的 packed layout 更复杂，但本质一样

`MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(...)` 把参考块编成：[before_denoise.py:L559-L722](src/before_denoise_py.md#__codelineno-0-559)

```text
[text | reference blocks | target audio | target video]
```

其中 reference blocks 有三种：

- image block
- audio block
- video block

`video` reference 的一个很妙的实现点是：

- 参考视频自己的音频行会紧贴其视频行之前
- 两者共享同一 rotary origin

见 [before_denoise.py:L658-L686](src/before_denoise_py.md#__codelineno-0-658)。

这保证了参考视频画面和参考音频在 packed sequence 中天然时序对齐。

## 9. 帧数和输出尺寸需先对齐 VAE 约束

`MiniMaxH3PrepareLayoutStep.__call__` 会做几个重要修正：[before_denoise.py:L373-L451](src/before_denoise_py.md#__codelineno-0-373)

- `height/width` 必须是 32 的倍数
- `num_frames` 会向上对齐到 `17 * n + 5`
- duration 必须落在 5 到 15 秒
- 再由此推导：
  - `num_latent_frames`
  - `latent_height`
  - `latent_width`
  - `num_audio_latents`

这一步其实决定了所有后续张量形状。

## 10. 噪声初始化：先 video，再 audio，而且 draw order 是协议的一部分

`MiniMaxH3PrepareLatentsStep` 明确写了 draw order：[before_denoise.py:L778-L860](src/before_denoise_py.md#__codelineno-0-778)

- 若有条件噪声，先画条件噪声
- 再画视频噪声 tensor
- 最后画音频噪声 rows

这属于 reproducibility 协议的一部分。只要 draw order 变了，同一个 generator seed 的结果就会变。

## 11. Denoise loop：每一步其实只做一次主干前向

`MiniMaxH3LoopDenoiser` 的逻辑很简单：[denoise.py:L35-L131](src/denoise_py.md#__codelineno-0-35)

```text
取出当前 step 的 unique_timesteps, timestep_indices
-> 把 layout fields 塞给 transformer.forward
-> 一次前向同时得到 video noise_pred 和 audio_noise_pred
```

要点在于：

- 一次前向同时处理视频和音频
- sequence 内不同 row 可以处在不同 timestep
- `row_timestep_plan` 负责把“这一行现在该在哪个噪声级别”告诉主干

这正是 H3 能把条件 row、目标音频 row、目标视频 row 放到同一主干的关键机制。

## 12. Scheduler 更新时只写 generated rows，不碰 condition rows

`MiniMaxH3LoopSchedulerStep` 会跳过最前面的条件部分：[denoise.py:L141-L237](src/denoise_py.md#__codelineno-0-141)

```python
block_state.latents[num_condition_video_rows:] = scheduler.step(...)
block_state.audio_latents[num_condition_audio_rows:] = audio_scheduler.step(...)
```

这意味着：

- keyframe / reference 行在 loop 中是 frozen anchor
- 它们只作为上下文被读，不会被反复更新

这是一种非常干净的条件控制方式：把条件显式嵌到序列里，但通过“只更新尾部生成行”保持其恒定。

## 13. 双 scheduler：同一主干，两个时间轨

`MiniMaxH3Scheduler` 是单一模态的 scheduler 类，但 pipeline 会持有两个实例：

- `scheduler`：视频
- `audio_scheduler`：音频

README 与 scheduler 注释都强调：

- 视频默认 `shift = 12.0`
- 音频默认 `shift = 3.0`

见 [scheduling_minimax_h3.py:L15-L31](src/scheduling_minimax_h3_py.md#__codelineno-0-15) 与 [scheduling_minimax_h3.py:L60-L69](src/scheduling_minimax_h3_py.md#__codelineno-0-60)。

因此虽然主干统一，反演轨迹并不统一。可以理解为：

- 主干负责学 joint score / velocity field
- 每个 modality 再按自己更合适的 sigma schedule 去走 ODE

## 14. H3 用的是 rectified-flow velocity，而且符号和常见 diffusers flow-match 相反

Scheduler 文件开头解释得很清楚：[scheduling_minimax_h3.py:L15-L31](src/scheduling_minimax_h3_py.md#__codelineno-0-15)

最关键一点：

$$
x_0 = x_t + \sigma v
$$

常见 flow-match 实现则使用：

$$
x_0 = x_t - \sigma v
$$

所以如果直接把它当普通 `FlowMatchEulerDiscreteScheduler` 去理解，会把 velocity 方向看反。

## 15. 推理主循环的本质图

把所有实现折叠起来，一次 `t2va/fl2va/ref2va` 生成都可以抽象成：

```text
原始输入
  -> 构造 Qwen3-VL presentation
  -> Qwen3-VL hidden_states[50]
  -> 条件图像/视频/音频编码成 latent
  -> 计算输出 canvas / num_frames / latent shape
  -> 打包成 [text | condition | target audio | target video]
  -> 生成 rotary position_ids / token_tags / row indices
  -> 初始化 target rows 噪声
  -> for each denoise step:
       transformer 一次前向预测全序列 video/audio velocity
       video scheduler 更新 target video rows
       audio scheduler 更新 target audio rows
  -> video VAE decode
  -> audio VAE decode
```

## 小结

MiniMax H3 的推理复杂度主要不在 denoising math，而在“如何把 multimodal request 编排成一条 transformer 可以直接吃的 sequence”。

这也是它和很多传统视频扩散 pipeline 最大的差异：

- 条件先被排版成统一序列
- 再让一套主干和两条 scheduler 去完成联合生成
