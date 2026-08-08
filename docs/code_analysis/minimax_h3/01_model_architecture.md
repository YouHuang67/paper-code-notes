---
tags:
  - Video Generation
  - Diffusion Model
  - Unified Understanding
---
# MiniMax H3：模型结构

本文只解释 `transformer_minimax_h3.py`、`transformer/config.json` 与 `README.md` 所定义的 H3-Base 主体结构。

核心问题只有三个：

- 统一多模态序列到底长什么样
- Transformer block 怎样在不拆模态分支的前提下保留 modality-specific 行为
- 视频和音频为什么能共享一套主干

## 1. 从配置先读出结构骨架

`transformer/config.json` 已经把主干形状钉死：

- `hidden_size = 5376`
- `num_layers = 50`
- `num_attention_heads = 56`
- `attention_head_dim = 128`
- `ffn_dim = 14336`
- `num_refiner_layers = 2`
- `in_channels = 24`
- `audio_in_channels = 32`
- `patch_size = [1, 2, 2]`
- `text_dim = 5120`
- `rope_freq_dim = 16`
- `time_embed_dim = 2688`

这里最需要注意的一点是：

$$
56 \times 128 = 7168 > 5376
$$

也就是说 attention 的 `inner_dim` 明显大于 residual stream 的 `hidden_size`。这不是标准 LLaMA 那种“heads * dim = model dim”，而是更接近 **先把 QKV 提到更高的 head space，再投影回主残差流**。

源码说明也明确指出了这点：[transformer_minimax_h3.py:L393-L430](src/transformer_minimax_h3_py.md#__codelineno-0-393)。

## 2. 主干不是 encoder-decoder，而是 packed single-stream Transformer

`MiniMaxH3Transformer3DModel` 的总说明是整份代码最重要的段落：[transformer_minimax_h3.py:L374-L391](src/transformer_minimax_h3_py.md#__codelineno-0-374)。

它明确给出四个结论：

- 一次前向只处理一条 packed sequence
- sequence 同时包含 text、condition video、audio、target video
- attention 是 full self-attention
- 没有 cross-attention，也没有分模态 block

因此它不是：

- 文本 encoder + 视频 DiT decoder
- 音频分支 / 视频分支双塔
- encoder-only 理解 + decoder-only 生成

而是：

```text
所有 modality
  -> 先映射成共享 residual stream
  -> 共享 block stack
  -> 再按输出头拆回视频 / 音频
```

这套写法的收益是：**模态交互不需要显式桥接层**。只要 row 被放进同一序列，attention 天然就能访问它。

## 3. 输入层：三种 modality，三种投影，统一到同一残差流

主干构造函数的前半段就是结构总图：[transformer_minimax_h3.py:L505-L555](src/transformer_minimax_h3_py.md#__codelineno-0-505)。

```python
self.proj_in = nn.Linear(video_patch_dim, hidden_size, bias=True)
self.audio_proj_in = nn.Linear(audio_in_channels, hidden_size, bias=True)
self.context_embedder = nn.Linear(text_dim, hidden_size, bias=True)
```

这三层的语义分别是：

- `proj_in`
  - 输入是 patchify 后的视频 latent row
  - 每个 row 维度是 `24 * 1 * 2 * 2 = 96`
- `audio_proj_in`
  - 输入是音频 latent row
  - 每个 row 维度是 `32`
- `context_embedder`
  - 输入是 Qwen3-VL 第 50 层输出
  - 每个 row 维度是 `5120`

这一步本质是在做：

$$
x^{(m)}_i \in \mathbb{R}^{d_m}
\xrightarrow{W_m}
h_i \in \mathbb{R}^{5376}
$$

其中 `m` 是 modality，`W_m` 只在入口处区分模态，进入主干后就不再区分。

## 4. 文本不是直接拿来用，而是先过两层 token refiner

MiniMax H3 没有把 `Qwen3-VL` hidden state 原封不动送进主干，而是先做一个 2 层 refiner：[transformer_minimax_h3.py:L248-L314](src/transformer_minimax_h3_py.md#__codelineno-0-248)。

这个模块有几个关键特征：

- 仍然是 attention + SwiGLU FFN
- 但不带 RoPE
- 也不带 AdaLN
- 只用于 text stream

因此它更像一个 **条件投影后的轻量后处理器**，作用是把 `Qwen3-VL` 的语言/视觉条件隐藏状态重新适配到 H3 主干习惯的表示空间。

可以把它理解为：

```text
Qwen3-VL hidden_states[50]
  -> linear(context_embedder)
  -> 2-layer text-only refiner
  -> packed sequence 的 text rows
```

这一步避免直接把外部 conditioner 的分布硬塞进视频扩散主干。

## 5. AdaLN 是这套单流架构保留模态差异的关键

如果所有 row 共用同一套 attention / FFN，那么模型如何知道某一行是 text、video 还是 audio？

答案是：**AdaLN modulation table**。

`MiniMaxH3AdaLayerNormModulation` 把一个 timestep embedding 投影成 6 组参数：[transformer_minimax_h3.py:L101-L129](src/transformer_minimax_h3_py.md#__codelineno-0-101)：

- `shift_msa`
- `scale_msa`
- `gate_msa`
- `shift_mlp`
- `scale_mlp`
- `gate_mlp`

而且它不是只按 timestep 索引，而是按：

$$
\text{adaln\_index} = \text{timestep\_index} \times 3 + \text{token\_tag}
$$

对应实现：[transformer_minimax_h3.py:L642-L651](src/transformer_minimax_h3_py.md#__codelineno-0-642)。

这意味着每个 block 里，row 的归一化调制参数由两件事决定：

- 它当前处于哪个噪声级别
- 它属于哪种 modality

于是同一套注意力 / FFN 权重，就能通过不同的 shift/scale/gate，对 text/video/audio row 表现出不同偏置。

## 6. Block 结构很朴素，但调制方式不朴素

`MiniMaxH3TransformerBlock` 的主结构非常标准：[transformer_minimax_h3.py:L317-L371](src/transformer_minimax_h3_py.md#__codelineno-0-317)：

```text
RMSNorm
-> AdaLN shift/scale
-> Attention
-> residual + gate_msa * attn_out

RMSNorm
-> AdaLN shift/scale
-> SwiGLU FFN
-> residual + gate_mlp * ff_out
```

如果写成公式，大致是：

$$
\hat{h} = \mathrm{Norm}(h)
$$

$$
\tilde{h}_{attn} = (1 + s_{attn}) \odot \hat{h} + b_{attn}
$$

$$
h' = h + g_{attn} \odot \mathrm{Attn}(\tilde{h}_{attn})
$$

$$
\tilde{h}_{ffn} = (1 + s_{ffn}) \odot \mathrm{Norm}(h') + b_{ffn}
$$

$$
h'' = h' + g_{ffn} \odot \mathrm{FFN}(\tilde{h}_{ffn})
$$

因此模态差异不是通过不同 block 实现，而是通过 **同 block 内不同调制参数** 实现。

## 7. Attention 层的三个重要性质

### 7.1 full self-attention

`MiniMaxH3AttnProcessor` 直接把整个 packed sequence 的 `Q/K/V` 送给 `dispatch_attention_fn(...)`：[transformer_minimax_h3.py:L166-L207](src/transformer_minimax_h3_py.md#__codelineno-0-166)。

当前没有：

- causal mask
- cross-attention K/V
- block sparse mask

所以这是最直白的“全局读全局”。

### 7.2 QK 上有 RMSNorm

```python
query = attn.norm_q(query)
key = attn.norm_k(key)
```

见 [transformer_minimax_h3.py:L180-L189](src/transformer_minimax_h3_py.md#__codelineno-0-180)。

这有点像很多现代大模型里的 qk-norm，用来稳定长序列和多模态混合下的注意力尺度。

### 7.3 只旋转 head_dim 的一部分

RoPE 只作用在前 `2 * 3 * rope_freq_dim = 96` 个通道上，其余 head channels 直通：[transformer_minimax_h3.py:L56-L71](src/transformer_minimax_h3_py.md#__codelineno-0-56)。

这意味着 attention head 内部被分成两部分：

- 一部分承担显式 `(t,h,w)` 几何关系
- 另一部分保留为非位置化内容子空间

这对大 head dim（128）尤其合理，因为不必让所有通道都被几何约束绑定。

## 8. MM-RoPE：共享一套频率表，但位置是三轴

`MiniMaxH3RotaryPosEmbed` 只有一套 `inv_freq`，然后把 `(t,h,w)` 三轴的频率拼起来：[transformer_minimax_h3.py:L74-L98](src/transformer_minimax_h3_py.md#__codelineno-0-74)。

它的含义不是“把图像 flatten 成一维再做位置编码”，而是：

- 时间轴自己有频率
- 高度轴自己有频率
- 宽度轴自己有频率
- 三者共享频率尺度，但不共享坐标值

这就是 README 里所谓 3D MM-RoPE 的代码落点。

## 9. 输出层：共享 block stack，分头读出

主干末尾先做一层 `MiniMaxH3AdaLayerNormOut`，然后分别接两个头：[transformer_minimax_h3.py:L549-L558](src/transformer_minimax_h3_py.md#__codelineno-0-549) 与 [transformer_minimax_h3.py:L653-L662](src/transformer_minimax_h3_py.md#__codelineno-0-653)。

- `proj_out`：输出视频 row velocity
- `audio_proj_out`：输出音频 row velocity

注意这里不是只对视频 row / 音频 row 单独跑 head，而是：

1. 对完整 packed sequence 做同一层 `norm_out`
2. 两个 head 都跑一遍完整序列
3. 最后分别 `index_select` 取回视频 / 音频 row

这其实再次强调了 H3 的核心哲学：**直到最后一刻，序列仍然被视作一个整体**。

## 10. 这份结构最深的设计取舍

如果把 MiniMax H3 跟传统视频 diffusion 系统比较，它最大的结构取舍是：

- 不把“理解”和“生成主干”拆成完全不同网络
- 不给音频单独一套生成骨干
- 不让参考条件只作为 cross-attn memory

而是强制所有条件进入：

```text
统一 latent token 格式
-> 统一时空坐标系
-> 统一单流 Transformer
-> 分模态读出
```

这样做的代价是 full attention 序列会非常长；好处则是多模态交互非常直接，工程上也只维护一套大主干。

## 小结

MiniMax H3 主体结构可以压缩成一句话：

> 一个用 3D MM-RoPE 和 per-(timestep, modality) AdaLN 调制的单流 dense Transformer，在统一 packed multimodal latent sequence 上同时做视频和音频 velocity prediction。

后面看推理代码时，所有复杂性几乎都来自这句话的两个后果：

- sequence 怎么打包
- 同一 sequence 上不同 row 的噪声级别和角色怎么安排
