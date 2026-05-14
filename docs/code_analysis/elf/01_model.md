---
tags:
    - Diffusion Model
    - Flow Matching
---

# ELF 模型结构

本文按 ELF 网络前向执行顺序，逐段拆解 DiT 模型结构。

**源码位置**: [model.py](https://github.com/lillian039/ELF/blob/main/src/modules/model.py), [layers.py](https://github.com/lillian039/ELF/blob/main/src/modules/layers.py)

## 前向执行流程

```
x [B, S, D_enc] + t [B]
  │
  ├─ ① self_cond_proj: 如果输入 2*D_enc → 投影回 D_enc
  ├─ ② BottleneckTextProj: D_enc → 128 → hidden_size
  ├─ ③ prepend model mode tokens (gate-controlled)
  ├─ ④ prepend control tokens (time + CFG scale)
  ├─ ⑤ RoPE (control/mode tokens 无旋转)
  ├─ ⑥ DiT blocks × depth
  ├─ ⑦ strip prefix → 保留数据序列
  ├─ ⑧ if decoder_step_active: unembed (hidden → D_enc → vocab)
  └─ ⑨ FinalLayer: RMSNorm → Linear(D_enc, ZERO init)
```

## 1. 模型入口与 Self-Condition 投影

**源码位置**: [model.py#L75-L90](https://github.com/lillian039/ELF/blob/main/src/modules/model.py#L75-L90)

```python
@nn.compact
def __call__(self, x, t, attention_mask=None, deterministic=True,
             self_cond_cfg_scale=None, decoder_step_active=None):
    patch_size = 1
    head_dim = self.hidden_size // self.num_heads
    B = x.shape[0]

    # ① Self-conditioning 投影: [z, x̂_prev] concat → D_enc
    if x.shape[-1] == 2 * self.text_encoder_dim:
        x = nn.Dense(
            self.text_encoder_dim, use_bias=True,
            kernel_init=DEFAULT_KERNEL_INIT, bias_init=DEFAULT_BIAS_INIT,
            name='self_cond_proj',
        )(x)
```

当 `self_cond_prob > 0` 时，输入 $x$ 的最后一个维度是 $2 \times D_{enc}$——前半是含噪嵌入 $z_t$，后半是上一步的预测 $\hat{x}_{prev}$（或零向量）。`self_cond_proj` 将两者融合为 $D_{enc}$ 维。

## 2. Bottleneck 投影

**源码位置**: [model.py#L92-L95](https://github.com/lillian039/ELF/blob/main/src/modules/model.py#L92-L95), [layers.py#L89-L101](https://github.com/lillian039/ELF/blob/main/src/modules/layers.py#L89-L101)

```python
    # ② Text projection with bottleneck
    x = BottleneckTextProj(
        self.text_encoder_dim, self.hidden_size, self.bottleneck_dim,
        name='text_proj',
    )(x)
```

```python
class BottleneckTextProj(nn.Module):
    text_encoder_dim: int
    hidden_size: int
    bottleneck_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.bottleneck_dim, use_bias=False,
                     kernel_init=DEFAULT_KERNEL_INIT, name='proj1')(x)  # D_enc→128
        return nn.Dense(self.hidden_size, use_bias=True,
                        kernel_init=DEFAULT_KERNEL_INIT, bias_init=DEFAULT_BIAS_INIT,
                        name='proj2')(x)                                 # 128→hidden
```

$D_{enc} = 512$（T5-small）先压缩到 128 再膨胀到 hidden_size（768）。第一层无 bias（纯投影），中间无激活函数。128 是消融实验确定的最优 bottleneck 维度——过小（32）损失多样性，过大（512）损失生成质量。

## 3. Model Mode Tokens

**源码位置**: [model.py#L97-L111](https://github.com/lillian039/ELF/blob/main/src/modules/model.py#L97-L111)

```python
    # ③ Prepend learnable model-mode tokens (gated)
    model_mode_offset = 0
    if self.num_model_mode_tokens > 0:
        mode_tokens = jnp.tile(
            self.param('mode_tokens', NORMAL_INIT_002,
                       (1, self.num_model_mode_tokens, self.hidden_size)),
            (B, 1, 1),
        )
        active_gate = (jnp.array(False) if decoder_step_active is None
                       else decoder_step_active)
        mode_tokens = mode_tokens * active_gate.astype(mode_tokens.dtype)
        x = jnp.concatenate([mode_tokens, x], axis=1)         # [B, 4+S, C]
        model_mode_offset = self.num_model_mode_tokens
        if attention_mask is not None:
            mode_mask = jnp.ones((B, self.num_model_mode_tokens),
                                 dtype=attention_mask.dtype)
            attention_mask = jnp.concatenate([mode_mask, attention_mask], axis=1)
```

4 个可学习 token，通过 `decoder_step_active` gate 控制：
- **denoise mode** (`decoder_step_active=False`)：token 乘以 0 → 零向量，不参与 attention
- **decode mode** (`decoder_step_active=True`)：token 激活，告知网络当前是解码模式

这是共享权重 denoiser-decoder 的关键机制——同一网络通过 mode token 区分两种行为。

## 4. In-Context Control Tokens

**源码位置**: [model.py#L55-L73](https://github.com/lillian039/ELF/blob/main/src/modules/model.py#L55-L73), [model.py#L113-L121](https://github.com/lillian039/ELF/blob/main/src/modules/model.py#L113-L121)

```python
    # ④ build_context: 生成 time + CFG scale tokens
    def build_context(self, t, self_cond_cfg_scale=None):
        prefix_tokens = []
        B = t.shape[0]

        def _make_prefix(emb, n_tokens, param_name):
            tokens = self.param(param_name, NORMAL_INIT_002,
                               (1, n_tokens, self.hidden_size))
            return jnp.tile(tokens, (B, 1, 1)) + jnp.expand_dims(emb, 1)

        # 4 time tokens: sinusoidal emb → MLP → add to learnable tokens
        time_emb = TimestepEmbedder(self.hidden_size, name='t_embedder')(t)
        prefix_tokens.append(_make_prefix(time_emb, self.num_time_tokens, 't_emb_tokens'))

        # 4 CFG scale tokens (if enabled)
        if self_cond_cfg_scale is not None:
            sc_emb = TimestepEmbedder(
                self.hidden_size, name='self_cond_cfg_embedder',
            )(self_cond_cfg_scale)
            if self.num_self_cond_cfg_tokens > 0:
                prefix_tokens.append(_make_prefix(
                    sc_emb, self.num_self_cond_cfg_tokens, 'self_cond_cfg_tokens',
                ))
        return prefix_tokens

    # prepend control tokens to sequence
    prefix_len = 0
    context_prefix_tokens = self.build_context(t, self_cond_cfg_scale)
    if context_prefix_tokens:
        prefix_tokens = jnp.concatenate(context_prefix_tokens, axis=1)
        prefix_len = prefix_tokens.shape[1]
        x = jnp.concatenate([prefix_tokens, x], axis=1)        # [B, 8+S, C]
```

控制 token 的构造方式：`可学习 token + sinusoidal_emb(t)`

- Time token：标量 $t$ → sinusoidal embedding → 2 层 MLP(SiLU) → hidden_size → 加到可学习 token 上
- CFG scale token：同理，标量 $\omega$ → sinusoidal embedding → MLP → 加到可学习 token 上

这些 token 与数据序列一起做 full self-attention，模型通过 attention 感知 time 和 CFG scale 信息。相比 adaLN-Zero（需要额外的 MLP 将条件注入每层），in-context conditioning 节省参数（ELF-B: 148M → 105M）且效果略优。

**TimestepEmbedder**（[layers.py#L104-L127](https://github.com/lillian039/ELF/blob/main/src/modules/layers.py#L104-L127)）：

```python
class TimestepEmbedder(nn.Module):
    hidden_size: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        dense = partial(nn.Dense, self.hidden_size, use_bias=True,
                        kernel_init=NORMAL_INIT_002, bias_init=DEFAULT_BIAS_INIT)
        t_emb = dense(name='mlp_0')(
            self.timestep_embedding(t, self.frequency_embedding_size))
        return dense(name='mlp_2')(nn.silu(t_emb))

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = jnp.exp(
            -math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
        args = t[:, None].astype(jnp.float32) * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding
```

标准 sinusoidal embedding：频率从 $1$ 到 $10^{-4}$，half=128 个频率 → $\cos$/$\sin$ 各 128 → 拼接为 256 维 → 2 层 MLP → hidden_size。

## 5. RoPE 位置编码

**源码位置**: [model.py#L123-L127](https://github.com/lillian039/ELF/blob/main/src/modules/model.py#L123-L127), [layers.py#L31-L70](https://github.com/lillian039/ELF/blob/main/src/modules/layers.py#L31-L70)

```python
    # ⑤ RoPE
    feat_rope = TextRotaryEmbeddingFast(
        dim=head_dim, pt_seq_len=self.max_length,
        num_empty_token=prefix_len + model_mode_offset, name='feat_rope',
    )
```

RoPE 的实现区分两类 token：
- 前 `num_empty_token` 个（control + mode tokens）：$\cos=1, \sin=0$，无旋转
- 后续数据 token：标准 RoPE $f(q, m) = q \cos(m\theta) + \text{rotate\_half}(q) \sin(m\theta)$

支持 position interpolation：`pt_seq_len`（训练长度）和 `ft_seq_len`（推理长度，默认等于 pt_seq_len）可不同。

```python
class TextRotaryEmbeddingFast(nn.Module):
    dim: int
    pt_seq_len: int = 512
    ft_seq_len: Optional[int] = None
    theta: float = 10000
    num_empty_token: int = 0

    @nn.compact
    def __call__(self, t):
        ft_seq_len = self.ft_seq_len if self.ft_seq_len is not None else self.pt_seq_len
        freqs = 1. / (self.theta ** (jnp.arange(0, dim, 2)[:dim//2].astype(jnp.float32) / dim))
        pos = jnp.arange(ft_seq_len) / ft_seq_len * pt_seq_len
        freqs_main = jnp.einsum('..., f -> ... f', pos, freqs)
        freqs_main = repeat(freqs_main, '... n -> ... (n r)', r=2)

        cos_parts, sin_parts = [], []
        if self.num_empty_token > 0:
            cos_parts.append(jnp.ones((self.num_empty_token, D)))
            sin_parts.append(jnp.zeros((self.num_empty_token, D)))
        cos_parts.append(jnp.cos(freqs_main))
        sin_parts.append(jnp.sin(freqs_main))

        freqs_cos = jnp.concatenate(cos_parts, axis=0)
        freqs_sin = jnp.concatenate(sin_parts, axis=0)
        return t * freqs_cos + rotate_half(t) * freqs_sin
```

## 6. DiT Blocks

**源码位置**: [model.py#L128-L137](https://github.com/lillian039/ELF/blob/main/src/modules/model.py#L128-L137)

```python
    # ⑥ DiT blocks: 只在中间 50% 层施加 dropout
    q1, q3 = self.depth // 4, self.depth // 4 * 3
    for i in range(self.depth):
        in_drop_range = q3 > i >= q1
        block = ELFBlock(
            self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio,
            attn_drop=self.attn_drop if in_drop_range else 0.0,
            proj_drop=self.proj_drop if in_drop_range else 0.0,
            name=f'blocks_{i}',
        )
        x = block(x, rope_fn=feat_rope, attention_mask=attention_mask,
                  deterministic=deterministic)
```

DiT 架构，标准 pre-norm Transformer block：

```python
class ELFBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, rope_fn=None, attention_mask=None, deterministic=True):
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)

        # Pre-norm + Attention + residual
        x_normed = RMSNorm(self.hidden_size, eps=1e-6, name='norm1')(x)
        attn_out = Attention(
            self.hidden_size, self.num_heads, qkv_bias=True, qk_norm=True,
            name='attn',
        )(x_normed, rope_fn, attention_mask=attention_mask, deterministic=deterministic)
        x = x + attn_out

        # Pre-norm + SwiGLU FFN + residual
        x_normed = RMSNorm(self.hidden_size, eps=1e-6, name='norm2')(x)
        mlp_out = SwiGLUFFN(
            self.hidden_size, mlp_hidden_dim, name='mlp',
        )(x_normed, deterministic=deterministic)
        x = x + mlp_out
        return x
```

**Attention 层**（[layers.py#L153-L183](https://github.com/lillian039/ELF/blob/main/src/modules/layers.py#L153-L183)）：

```python
class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = True
    qk_norm: bool = True

    @nn.compact
    def __call__(self, x, rope_fn, attention_mask=None, deterministic=True):
        B, N, C = x.shape
        head_dim = self.dim // self.num_heads

        # QKV 合并投影: [B, N, C] → [B, N, 3*C] → [3, B, H, N, D]
        qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias,
                       kernel_init=DEFAULT_KERNEL_INIT,
                       bias_init=DEFAULT_BIAS_INIT, name='qkv')(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)                     # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # QK norm + RoPE
        if self.qk_norm:
            q = RMSNorm(head_dim, name='q_norm')(q)
            k = RMSNorm(head_dim, name='k_norm')(k)
        if rope_fn is not None:
            q = rope_fn(q)
            k = rope_fn(k)

        # 手工 scaled dot-product attention
        x = scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)           # [B, N, C]
        x = nn.Dense(self.dim, kernel_init=DEFAULT_KERNEL_INIT,
                     bias_init=DEFAULT_BIAS_INIT, name='proj')(x)
        return nn.Dropout(rate=self.proj_drop, deterministic=deterministic)(x)
```

QKV 投影合并 + QK norm + RoPE + 手工 attention（不用 Flax 内置，便于控制 mask 格式）。

**SwiGLU FFN**（[layers.py#L186-L201](https://github.com/lillian039/ELF/blob/main/src/modules/layers.py#L186-L201)）：

```python
class SwiGLUFFN(nn.Module):
    dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x, deterministic=True):
        hidden_dim = int(self.hidden_dim * 2 / 3)               # 从 4× 换算
        dense = partial(nn.Dense, use_bias=self.bias,
                        kernel_init=DEFAULT_KERNEL_INIT, bias_init=DEFAULT_BIAS_INIT)

        # gate/up 合并投影 → split → SiLU(gate) ⊙ up
        x12 = dense(2 * hidden_dim, name='w12')(x)              # [B, N, 2*hidden]
        x1, x2 = jnp.split(x12, 2, axis=-1)
        hidden = nn.Dropout(rate=self.drop, deterministic=deterministic)(
            nn.silu(x1) * x2)                                   # [B, N, hidden]
        return dense(self.dim, name='w3')(hidden)               # [B, N, C]
```

标准 SwiGLU：$\text{output} = (\text{SiLU}(x W_{gate}) \odot x W_{up}) W_{down}$。gate 和 up 投影合并为 `Dense(2 * hidden_dim)`，一次矩阵乘法后 split。

## 7. Strip Prefix 与 Decode Head

**源码位置**: [model.py#L139-L157](https://github.com/lillian039/ELF/blob/main/src/modules/model.py#L139-L157)

```python
    # ⑦ Strip prefix: 去掉 control + mode tokens
    x = x[:, prefix_len + model_mode_offset:]                   # [B, S, C]

    # ⑧ Factored decoder unembedding (only when decode mode)
    decoder_logits = None
    bn = self.text_encoder_dim
    proj_kernel = self.param('proj_kernel', DEFAULT_KERNEL_INIT,
                             (self.hidden_size, bn))
    proj_bias = self.param('proj_bias', DEFAULT_BIAS_INIT, (bn,))
    unembed_kernel = self.param('unembed_kernel', DEFAULT_KERNEL_INIT,
                                (bn, self.vocab_size))
    unembed_bias = self.param('unembed_bias', DEFAULT_BIAS_INIT,
                              (self.vocab_size,))
    if decoder_step_active is not None:
        decoder_logits = jax.lax.cond(
            decoder_step_active,
            lambda xi: (jax.nn.gelu(xi @ proj_kernel + proj_bias)
                        @ unembed_kernel + unembed_bias),       # [B, S, V]
            lambda xi: jnp.zeros((*xi.shape[:2], self.vocab_size), dtype=xi.dtype),
            x,
        )

    # ⑨ FinalLayer: RMSNorm → Linear(D_enc), ZERO init
    output = FinalLayer(self.hidden_size, patch_size,
                        self.text_encoder_dim, name='final_layer')(x)
    return output, decoder_logits
```

**Unembedding 的分解设计**：`hidden(768) → Dense → D_enc(512) → GELU → Dense → vocab`

直接 $768 \to V$ 需要 $768V$ 参数，分解后只需 $768 \times 512 + 512V$。对 $V \approx 32000$，节省约 20M 参数。

`jax.lax.cond` 保证 denoise mode 时不计算 unembedding（返回 zero logits），避免浪费算力。

**FinalLayer**（[layers.py#L204-L216](https://github.com/lillian039/ELF/blob/main/src/modules/layers.py#L204-L216)）：

```python
class FinalLayer(nn.Module):
    hidden_size: int
    patch_size: int
    out_channels: int

    @nn.compact
    def __call__(self, x):
        x = RMSNorm(self.hidden_size, name='norm_final')(x)
        return nn.Dense(
            self.patch_size * self.patch_size * self.out_channels,
            use_bias=True,
            kernel_init=ZERO_INIT, bias_init=ZERO_INIT, name='linear',
        )(x)
```

零初始化 ($kernel=0, bias=0$) 确保训练初期网络输出为 0——denoising 从恒等映射开始，是稳定训练的关键设计。

## 8. 参数初始化约定

| 组件 | kernel_init | bias_init |
|------|-----------|----------|
| Dense（主模型） | `xavier_uniform` | 0 |
| TimestepEmbedder MLP | `normal(0.02)` | 0 |
| Learnable tokens | `normal(0.02)` | — |
| **FinalLayer.linear** | **0** | **0** |

## 9. 条件序列处理

条件生成时，条件嵌入 prepend 在目标序列前。条件 token 由 `cond_seq_mask` 标记，在训练和推理中始终保持 clean（不加噪），目标 token 通过 bidirectional self-attention 感知条件信息。

```python
# 条件 token 始终保留 clean embedding
def restore_cond(z_updated, cond_seq, cond_seq_mask):
    mask = cond_seq_mask[..., None]  # broadcast to embedding dim
    return jnp.where(mask > 0, cond_seq, z_updated)
```
