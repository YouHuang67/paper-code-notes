---
tags:
    - Diffusion Model
    - Flow Matching
---

# ELF 训练流程

本文按 `train_step()` 的执行顺序，拆解双分支混合训练、self-conditioning mask、training-time CFG 的完整实现。

**源码位置**: [train_step.py](https://github.com/lillian039/ELF/blob/main/src/train_step.py), [train.py](https://github.com/lillian039/ELF/blob/main/src/train.py)

参考：[sampling_utils.py](https://github.com/lillian039/ELF/blob/main/src/utils/sampling_utils.py) 中的 `add_noise`, `sample_timesteps`, `sample_cfg_scale`, `net_out_to_v_x`

## 训练步执行流程

```
① RNG 分叉 → ② Label drop (条件生成) → ③ T5 Encoder: tokens → x₀
→ ④ 准备 denoising branch: sample t (logit-normal), noise, z_t = t*x₀+(1-t)*ε
→ ⑤ 准备 decoding branch: per-token p (logit-normal), z̃ = p*x₀+(1-p)*ε
→ ⑥ Bernoulli(0.2) → decoder_step_active
→ ⑦ jax.lax.cond → _denoiser_branch 或 _decoder_branch
→ ⑧ 梯度 pmean 同步 → ⑨ EMA 更新
```

## 1. 环境准备：Encoder + 噪声调度

**源码位置**: [train_step.py#L27-L101](https://github.com/lillian039/ELF/blob/main/src/train_step.py#L27-L101)

```python
def train_step(state, encoder_params, encoder_apply_fn, batch, config):
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob
    latent_mean, latent_std = config.latent_mean, config.latent_std
    decoder_prob = config.decoder_prob
    decoder_noise_scale = config.decoder_noise_scale

    # ① RNG 分叉
    new_dropout_rng, current_step_rng = jax.random.split(state.dropout_rng, 2)
    current_step_rng = jax.random.fold_in(current_step_rng,
                                          jax.lax.axis_index(axis_name="batch"))
    (t_rng, noise_rng, self_cond_mask_rng, self_cond_cfg_rng, _,
     model_dropout_rng, decoder_step_rng, decoder_rng,
     decoder_lambda_rng, decoder_noise_rng, _,
    ) = jax.random.split(current_step_rng, 11)
```

11 路 RNG 分叉——比实际需要多 1 路（第 5 和第 11 路未使用），用于保证与已发布 checkpoint 的 bit-for-bit 可复现性。

```python
    # ② Label drop: 条件 token 间 attention 随机清零
    if config.label_drop_prob > 0:
        drop = batch["label_drop_mask"][:, None, None]          # [B, 1, 1]
        cond_mask = batch["cond_seq_mask"]                       # [B, S]
        # block_mask: 1 only at (non-cond row, cond col)
        block_mask = (1 - cond_mask)[:, :, None] * cond_mask[:, None, :]
        encoder_attention_mask = encoder_attention_mask * (1 - drop * block_mask)

    # ③ T5 Encoder: tokens → 归一化嵌入
    x0 = encode_text(
        input_ids=batch["input_ids"],
        attention_mask=encoder_attention_mask,
        encoder_apply_fn=encoder_apply_fn,
        encoder_params=encoder_params,
        latent_mean=latent_mean, latent_std=latent_std,
    )                                                            # [B, S, D_enc]
```

`label_drop` 关闭条件 token 之间的 attention，但保留条件 ↔ 目标的 attention 和 目标内部的 attention。这模拟推理时的 unconditional 路径。`encode_text` 返回归一化嵌入：`(latents - mean) / std`。

```python
    # ④ Denoising branch 准备: logit-normal t + noise
    t = sample_timesteps(
        t_rng, batch_size,
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        time_schedule=config.time_schedule,
    )                                                            # [B]
    noise = jax.random.normal(noise_rng, x0.shape, dtype=x0.dtype)

    denoiser_z = add_noise(x0, noise, t, config, cond_seq_mask=cond_seq_mask)

    # ⑤ Decoding branch 准备: per-token logit-normal corruption
    decoder_step_active = jax.random.bernoulli(decoder_step_rng, decoder_prob)  # ~20%
```

**`add_noise` 实现**（[sampling_utils.py#L12-L18](https://github.com/lillian039/ELF/blob/main/src/utils/sampling_utils.py#L12-L18)）：

```python
def add_noise(x0, noise, t, config, cond_seq_mask=None):
    t_expanded = t.reshape(-1, 1, 1)                             # [B, 1, 1]
    z = t_expanded * x0 + (1 - t_expanded) * noise * config.denoiser_noise_scale
    if cond_seq_mask is not None:
        z = cond_seq_mask * x0 + (1 - cond_seq_mask) * z         # 条件 token 保持 clean
    return z
```

条件序列 mask 确保条件 token 始终为干净 $x_0$。

**Per-token corruption 实现**（[train_step.py#L93-L101](https://github.com/lillian039/ELF/blob/main/src/train_step.py#L93-L101)）：

```python
    # 每个 token 独立采样 corruption level p
    decoder_z_vals = (
        jax.random.normal(decoder_lambda_rng, (batch_size * seq_length,))
        * config.decoder_p_std + config.decoder_p_mean
    )
    decoder_lambda_t = jax.nn.sigmoid(decoder_z_vals).reshape(
        batch_size, seq_length, 1)                               # [B, S, 1]
    decoder_noise = (jax.random.normal(decoder_noise_rng, x0.shape)
                     * decoder_noise_scale)                       # [B, S, D]
    decoder_z = decoder_lambda_t * x0 + (1 - decoder_lambda_t) * decoder_noise
```

每个 token 的 $p$ 独立从 $\text{sigmoid}(\mathcal{N}(0.8, 0.8^2))$ 采样，同一序列中部分 token 几乎干净、部分高度噪声。`decoder_noise_scale = 5.0` 是 OWT 的设置（条件生成任务为 1.0）。

**Denoising 分支的 velocity target**：

```python
    t_expanded = t.reshape(-1, 1, 1)
    v_target = (x0 - denoiser_z) / jnp.maximum(1 - t_expanded, t_eps)  # [B, S, D]
```

预防 $t=1$ 时除零，`t_eps = 5e-2` 做下界裁剪。

## 2. Self-Conditioning Mask

**源码位置**: [train_step.py#L106-L138](https://github.com/lillian039/ELF/blob/main/src/train_step.py#L106-L138)

```python
    # ⑥ Bernoulli(0.5) per example
    if self_cond_prob > 0:
        use_self_cond_mask = (
            (jax.random.uniform(self_cond_mask_rng, (batch_size,)) < self_cond_prob)
            .reshape(-1, 1, 1).astype(x0.dtype)
        )                                                        # [B, 1, 1]

    # ⑦ CFG scale 采样: log-uniform [0.5, 5.0]
    if config.num_self_cond_cfg_tokens > 0:
        self_cond_cfg_scale = sample_cfg_scale(
            self_cond_cfg_rng, batch_size,
            cfg_min=config.self_cond_cfg_min, cfg_max=config.self_cond_cfg_max,
        )                                                        # [B]
```

`use_self_cond_mask` 是 Bernoulli(0.5) per-example mask——一半样本用真实 self-conditioning 预测，另一半用零向量。这确保网络能同时学习 conditional 和 unconditional 路径。

**`sample_cfg_scale`**（[sampling_utils.py#L77-L82](https://github.com/lillian039/ELF/blob/main/src/utils/sampling_utils.py#L77-L82)）：

```python
def sample_cfg_scale(rng, batch_size, cfg_min=0.0, cfg_max=3.0):
    u = jax.random.uniform(rng, (batch_size,))
    a = jnp.float32(1.0 + cfg_min)
    b = jnp.float32(1.0 + cfg_max)
    return a * jnp.exp(u * jnp.log(b / a)) - 1.0               # log-uniform
```

Log-uniform 采样偏向小值——训练时较低 CFG scale 的样本更多，与推理时常用的低 scale 一致。

**Self-condition 输入构造**（[train_step.py#L122-L138](https://github.com/lillian039/ELF/blob/main/src/train_step.py#L122-L138)）：

```python
def get_z_input(params, z, t_input, self_cond_cfg_input, x_tokens):
    if self_cond_prob == 0:
        return z
    # unconditional pass: self-cond = 0
    z_uncond = restore_cond(jnp.zeros_like(z), x_tokens, cond_seq_mask)
    z_with_zeros = jnp.concatenate([z, z_uncond], axis=-1)      # [B, S, 2*D]
    net_out_init = state.apply_fn(
        {"params": params}, z_with_zeros, t_input, deterministic=True,
        self_cond_cfg_scale=self_cond_cfg_input,
    )
    net_out_init = jax.lax.stop_gradient(net_out_init)
    _, x_pred_init = net_out_to_v_x(net_out_init, z, t_input, t_eps)
    x_pred_init = restore_cond(x_pred_init, x_tokens, cond_seq_mask)

    # 根据 mask 选择使用真实预测还是零向量
    x_pred_cond = x_pred_init * use_self_cond_mask
    x_pred_cond = restore_cond(x_pred_cond, x_tokens, cond_seq_mask)
    return jnp.concatenate([z, x_pred_cond], axis=-1)           # [B, S, 2*D]
```

这一步执行了一次额外的 forward pass 来获得 self-conditioning 的中间预测 $\hat{x}'$。梯度在此被 `stop_gradient` 截断。然后根据 `use_self_cond_mask` 决定是使用 $\hat{x}'$ 还是零向量。

## 3. Training-Time CFG

**源码位置**: [train_step.py#L145-L180](https://github.com/lillian039/ELF/blob/main/src/train_step.py#L145-L180)

这是 ELF 训练中最复杂的部分。目标不是简单地预测 $v$，而是预测 **CFG 组合后的 velocity** $v^{cfg}$，使得推理时无需额外 forward pass 就能实现 CFG。

**Conditional / Unconditional 双 pass**：

```python
def get_sc_cond_and_uncond(params, z, t, cond_mask, x_tokens):
    kwargs = {"self_cond_cfg_scale": self_cond_cfg_scale, "deterministic": True}

    if config.self_cond_prob == 0:
        net_out_uncod = state.apply_fn({"params": params}, z, t, **kwargs)
        v_uncond, _ = net_out_to_v_x(net_out_uncod, z, t, t_eps)
        return v_uncond, v_uncond

    # Unconditional: self-cond = 0, cond tokens restore
    z_uncond = restore_cond(jnp.zeros_like(z), x_tokens, cond_mask)
    z_input_uncond = jnp.concatenate([z, z_uncond], axis=-1)
    net_out_uncond = state.apply_fn(
        {"params": params}, z_input_uncond, t, **kwargs)
    v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, z, t, t_eps)
    x_uncond = restore_cond(x_uncond, x_tokens, cond_mask)

    # Conditional: self-cond = x_uncond (真实预测)
    z_input_cond = jnp.concatenate([z, x_uncond], axis=-1)
    net_out_cond = state.apply_fn(
        {"params": params}, z_input_cond, t, **kwargs)
    v_cond, _ = net_out_to_v_x(net_out_cond, z, t, t_eps)
    return v_cond, v_uncond
```

**CFG velocity target 组合**（[train_step.py#L166-L174](https://github.com/lillian039/ELF/blob/main/src/train_step.py#L166-L174)）：

```python
def get_sc_guided_v(params, z, t, base_v_target, x_tokens):
    v_cond, v_uncond = get_sc_cond_and_uncond(
        params, z, t, cond_mask=cond_seq_mask, x_tokens=x_tokens,
    )
    sc_w = self_cond_cfg_scale.reshape(batch_size, 1, 1)        # [B, 1, 1]

    # v_target = v + (1 - 1/ω) * (v_cond - v_uncond)
    sc_guidance = (1 - 1 / sc_w) * (v_cond - v_uncond)
    sc_guidance = jnp.where(
        use_self_cond_mask, sc_guidance, jnp.zeros_like(sc_guidance))
    return jax.lax.stop_gradient(base_v_target + sc_guidance)
```

核心公式（论文 Eq. 3）：

$$\nu^{target} = v + \left(1 - \frac{1}{\omega}\right) \big(v_\theta(z_t \mid c, \omega) - v_\theta(z_t \mid \emptyset, \omega)\big)$$

- $\omega$ 是 guidance scale，训练时从 [0.5, 5.0] 采样
- $\omega = 1$ 时 $\nu^{target} = v$，退化为无 CFG
- `use_self_cond_mask = 0` 时施加零 CFG guidance（self-cond 未激活，无需 CFG）
- `stop_gradient` 确保 target 不通过 CFG 组合公式回传梯度

## 4. Denoising 与 Decoding 分支

**源码位置**: [train_step.py#L182-L229](https://github.com/lillian039/ELF/blob/main/src/train_step.py#L182-L229)

```python
    def loss_fn(params):

        def _decoder_branch(_):
            # ⑧ Decoding: t=1, z̃ (per-token corrupted), CE loss
            decoder_t = jnp.ones_like(t)
            decoder_input = (
                jnp.concatenate([decoder_z, jnp.zeros_like(decoder_z)], axis=-1)
                if config.self_cond_prob > 0 else decoder_z
            )                                                    # [B, S, 2*D]
            _, decoder_logits = state.apply_fn(
                {"params": params}, decoder_input, decoder_t,
                deterministic=False,
                rngs={"dropout": model_dropout_rng},
                self_cond_cfg_scale=self_cond_cfg_scale,
                decoder_step_active=jnp.array(True),              # 激活 decode mode
            )                                                    # [B, S, V]
            log_probs = jax.nn.log_softmax(
                decoder_logits.astype(jnp.float32), axis=-1)
            ce = -jnp.take_along_axis(
                log_probs, decoder_targets[..., None], axis=-1).squeeze(-1)
            ce_loss = (ce * loss_mask).sum() / jnp.maximum(loss_mask.sum(), 1.0)
            return ce_loss, ce_loss, jnp.zeros(())

        def _denoiser_branch(_):
            # ⑧ Denoising: random t, MSE loss on velocity with CFG
            denoiser_t = t
            denoiser_input = get_z_input(
                params, denoiser_z, denoiser_t,
                self_cond_cfg_input=self_cond_cfg_scale, x_tokens=x0,
            )                                                    # [B, S, 2*D]
            net_out, _ = state.apply_fn(
                {"params": params}, denoiser_input, denoiser_t,
                deterministic=False,
                rngs={"dropout": model_dropout_rng},
                self_cond_cfg_scale=self_cond_cfg_scale,
                decoder_step_active=jnp.array(False),             # denoise mode
            )
            v_pred, _ = net_out_to_v_x(net_out, denoiser_z, denoiser_t, t_eps)
            v_final_target = get_v_target(
                params, denoiser_z, denoiser_t,
                base_v_target=v_target, x_tokens=x0,
            )
            per_dim_loss = (v_pred - v_final_target) ** 2
            l2_loss = reduce_token_loss(
                jnp.mean(per_dim_loss, axis=-1), loss_mask)
            return l2_loss, jnp.zeros(()), l2_loss

        # ⑥ jax.lax.cond: 根据 decoder_step_active 选择分支
        loss, ce_loss, l2_loss = jax.lax.cond(
            decoder_step_active, _decoder_branch, _denoiser_branch, None,
        )
        return loss, (l2_loss, ce_loss)
```

`jax.lax.cond` 是真 lazy conditional——只有活跃分支被执行，另一分支的代码完全不运行。

**Denoising 分支要点**：
- `decoder_step_active = False`：mode token 乘以 0，FinalLayer 后的 unembedding 返回 zero logits
- `v_final_target` = base velocity + CFG guidance（训练时 CFG 在 target 侧实现）
- MSE 在 velocity 空间计算，而非 x 空间——与论文 Eq.1 一致
- `reduce_token_loss` 对有效 token 取平均（排除 padding 和条件 token）

**Decoding 分支要点**：
- `decoder_step_active = True`：mode token 激活，unembedding 路径工作
- Self-cond 始终为 0（decode 不需要迭代细化）
- CE loss 只计算目标 token（`loss_mask` 已排除条件 token）

## 5. 梯度同步与 EMA

**源码位置**: [train_step.py#L231-L269](https://github.com/lillian039/ELF/blob/main/src/train_step.py#L231-L269)

```python
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (l2_loss_val, ce_loss_val)), grads = grad_fn(state.params)

    # pmean 跨设备同步梯度
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    l2_loss_val = jax.lax.pmean(l2_loss_val, axis_name="batch")
    ce_loss_val = jax.lax.pmean(ce_loss_val, axis_name="batch")

    new_state = state.apply_gradients(grads=grads, dropout_rng=new_dropout_rng)

    # Loss rescale: 按分支概率归一化，使报告值反映各自独立的 loss 水平
    decoder_prob_arr = jnp.asarray(decoder_prob, dtype=jnp.float32)
    denoiser_prob_arr = jnp.asarray(1.0 - decoder_prob, dtype=jnp.float32)
    active_ce_loss_val = jnp.where(
        decoder_prob_arr > 0.0,
        ce_loss_val / decoder_prob_arr,                          # /0.2
        jnp.zeros_like(ce_loss_val),
    )
    active_l2_loss_val = jnp.where(
        denoiser_prob_arr > 0.0,
        l2_loss_val / denoiser_prob_arr,                         # /0.8
        jnp.zeros_like(l2_loss_val),
    )
```

`/decoder_prob` 和 `/denoiser_prob` 的 rescale：因为每个 step 只有一个分支激活，直接平均会低估 loss 水平。例如 decoder 只有 20% step 激活，CE loss 需要除以 0.2 才能反映真实的 per-decoding-step 损失。

**EMA 更新**：

```python
    # 只在 optimizer step（非 grad accumulation micro-step）更新 EMA
    is_optimizer_step = (new_state.step % config.grad_accum_steps) == 0
    new_ema_params1 = jax.lax.cond(
        is_optimizer_step,
        lambda: ema_update(state.ema_params1, new_state.params, config.ema_decay1),
        lambda: state.ema_params1,
    )
```

EMA decay = 0.9999，推理时使用 `ema_params1` 而非 `params`。只在 optimizer step（而非 gradient accumulation 的 micro-step）更新 EMA，避免有效 decay 被放大。

## 6. 训练主循环

**源码位置**: [train.py#L313-L419](https://github.com/lillian039/ELF/blob/main/src/train.py#L313-L419)

```python
for epoch in range(start_epoch, config.epochs):
    train_loader = prefetch_to_device(iterator, size=4)          # 预取 4 batch

    for step_in_epoch, batch in enumerate(train_loader):
        batch = prepare_batch(batch, config, rng=batch_rng)      # tokenize + mask
        batch = shard(batch)                                     # 切分到各设备
        state, metrics = p_train_step(state, encoder_params, batch=batch)

        if global_step % config.log_freq == 0:
            # log loss / l2 / ce / lr / steps_per_sec

    if epoch % config.eval_freq == 0:
        run_generation()                                         # Gen. PPL / BLEU
    if epoch % config.save_freq == 0:
        save_checkpoint()
```

训练用 `jax.pmap` 做数据并行，`prefetch_to_device` 预取减少 GPU 等待时间。

## 7. 关键训练超参数 (ELF-B on OWT)

| 参数 | 值 |
|------|-----|
| Optimizer | Muon |
| Learning rate | 0.002 |
| Global batch size | 512 |
| Denoiser P_mean / P_std | -1.5 / 0.8 |
| Denoiser noise scale | 2.0 |
| Decoder P_mean / P_std | 0.8 / 0.8 |
| Decoder noise scale | 5.0 |
| Decoder prob | 0.2 |
| Self-cond prob | 0.5 |
| SC CFG range | [0.5, 5.0] |
| EMA decay | 0.9999 |
| t_eps | 5e-2 |
| Training epochs | 5 (~95K steps) |
