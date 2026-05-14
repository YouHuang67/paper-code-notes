---
tags:
    - Diffusion Model
    - Flow Matching
---

# ELF 采样与生成

本文按推理采样流程拆解：时间步构造 → lax.scan 迭代 → ODE/SDE 单步 → self-cond + CFG 前向 → 最终步 decode。

**源码位置**: [generation_utils.py](https://github.com/lillian039/ELF/blob/main/src/utils/generation_utils.py), [sampling_utils.py](https://github.com/lillian039/ELF/blob/main/src/utils/sampling_utils.py), [generation.py](https://github.com/lillian039/ELF/blob/main/src/generation.py)

## 推理执行流程

```
① get_sampling_steps: 构造时间网格 [t₀=0, t₁, t₂, ..., tₙ₋₁, tₙ=1]
② z₀ ~ N(0, I), x̂_prev = 0
③ for (t, t_next) in t_pairs:           # lax.scan 迭代
     ├─ ODE: z' = z + Δt * v(z, t)       # Euler step
     ├─ SDE: z_back = α*z + (1-α)*ε      # 噪声回退
     │        v = net(z_back, t_back)
     │        z' = z_back + (t_next - t_back)*v
     └─ x̂_prev = 当前预测 x̂ (传给下一步)
④ 最后步 ODE: z_final, _ = ode_step(z, tₙ₋₁, tₙ=1, ...)
⑤ decode: t=1, mode="decode", unembed → argmax → tokens
```

## 1. 时间步构造

**源码位置**: [sampling_utils.py#L53-L70](https://github.com/lillian039/ELF/blob/main/src/utils/sampling_utils.py#L53-L70)

```python
def get_sampling_steps(rng, n_steps, time_schedule, P_mean, P_std):
    if time_schedule == "uniform":
        return jnp.linspace(0.0, 1.0, n_steps + 1)              # 均匀分布
    if time_schedule == "logit_normal":
        steps = sample_timesteps(
            rng, batch_size=n_steps - 1,                         # N-1 个中间点
            P_mean=P_mean, P_std=P_std, time_schedule=time_schedule,
        )
        return jnp.concatenate(
            [jnp.array([0.0]), jnp.sort(steps), jnp.array([1.0])]  # 拼接端点
        )
```

Logit-normal schedule：从训练时相同分布采样 $N-1$ 个中间时间点，排序后两端拼接 0 和 1。效果是在 $t \approx 0$（高噪声区）步长小、采样密集；$t \approx 1$（接近干净）步长大、采样稀疏。实验证明在所有步数下优于 uniform。

`P_mean = -1.5, P_std = 0.8` 使得采样集中在中间区域（sigmoid 将 normal 样本映射到 [0,1]）。

## 2. 采样主循环：lax.scan 驱动

**源码位置**: [generation_utils.py#L108-L141](https://github.com/lillian039/ELF/blob/main/src/utils/generation_utils.py#L108-L141)

```python
def _generate_samples_single_batch(
    model_params, model_apply_fn, rng, z, t_steps,
    cond_seq, cond_seq_mask, config, sampling_config,
    cfg_scale, self_cond_cfg_scale,
):
    method = sampling_config.sampling_method
    batch_size, max_length, d_model = z.shape                        # z: [B, L, D]

    # 初始化: cond token 恢复为 clean embedding
    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(jnp.zeros_like(z), cond_seq, cond_seq_mask)  # x̂_prev = 0

    # 前 N-2 步: t_pairs = [(t₀,t₁), (t₁,t₂), ..., (tₙ₋₂,tₙ₋₁)]
    t_pairs = jnp.stack([t_steps[:-2], t_steps[1:-1]], axis=1)

    if method == "sde":
        step_fn = _sample_step_for_scan(
            sampling_config=sampling_config, rng=rng, **step_kwargs)
        # SDE carry 额外携带 step_idx 用于 per-step RNG
        (z, x_pred, _), _ = jax.lax.scan(
            step_fn, (z, x_pred, jnp.int32(0)), t_pairs)
    else:
        step_fn = _sample_step_for_scan(
            sampling_config=sampling_config, **step_kwargs)
        (z, x_pred), _ = jax.lax.scan(step_fn, (z, x_pred), t_pairs)

    # 最后一步始终用 ODE (SDE 噪声回退不适合 t=1)
    z, x_pred = _ode_step(
        z=z, t=t_steps[-2], t_next=t_steps[-1], x_pred_prev=x_pred,
        **step_kwargs,
    )
    return z                                                         # 最终嵌入
```

`jax.lax.scan` 将整个迭代循环编译为单个 XLA 操作——32 步采样的 overhead 不高于一次大 GEMM。SDE 的 `step_idx` 用于在 scan 内 `fold_in(rng, step_idx)` 生成各步的独立噪声。

**step_fn 构造**（[generation_utils.py#L61-L105](https://github.com/lillian039/ELF/blob/main/src/utils/generation_utils.py#L61-L105)）：

```python
def _sample_step_for_scan(model_apply_fn, model_params, config,
                           sampling_config, cfg_scale, self_cond_cfg_scale,
                           cond_seq, cond_seq_mask, rng=None):
    method = sampling_config.sampling_method
    base_kwargs = dict(
        model_apply_fn=model_apply_fn, model_params=model_params,
        config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )

    if method == "sde":
        sde_gamma = getattr(sampling_config, "sde_gamma", 0.0)

        def step_fn(carry, t_pair):
            z, x_pred, step_idx = carry
            t, t_next = t_pair
            step_rng = jax.random.fold_in(rng, step_idx)            # per-step RNG
            z_new, x_pred_new = _sde_step(
                z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                gamma=sde_gamma, rng=step_rng, **base_kwargs,
            )
            return (z_new, x_pred_new, step_idx + 1), None
        return step_fn

    if method == "ode":
        def step_fn(carry, t_pair):
            z, x_pred = carry
            t, t_next = t_pair
            z_new, x_pred_new = _ode_step(
                z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **base_kwargs,
            )
            return (z_new, x_pred_new), None
        return step_fn
```

## 3. ODE 单步

**源码位置**: [sampling_utils.py#L211-L226](https://github.com/lillian039/ELF/blob/main/src/utils/sampling_utils.py#L211-L226)

```python
@partial(jax.jit, static_argnums=(0, 6, 7, 8))
def _ode_step(model_apply_fn, model_params, z, t, t_next, x_pred_prev,
              config, cfg_scale, self_cond_cfg_scale,
              cond_seq, cond_seq_mask):
    t_batch = jnp.full((z.shape[0],), t)                            # [B]
    v_pred, x_pred = _forward_sample(
        model_apply_fn=model_apply_fn, model_params=model_params,
        z=z, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return z + (t_next - t) * v_pred, x_pred                        # Euler step
```

标准一阶 Euler 积分：$z_{t+\Delta t} = z_t + \Delta t \cdot v_\theta(z_t, t, \hat{x}_{prev})$。

## 4. SDE 单步

**源码位置**: [sampling_utils.py#L229-L254](https://github.com/lillian039/ELF/blob/main/src/utils/sampling_utils.py#L229-L254)

```python
@partial(jax.jit, static_argnums=(0, 6, 7, 8))
def _sde_step(model_apply_fn, model_params, z, t, t_next, x_pred_prev,
              config, cfg_scale, self_cond_cfg_scale,
              cond_seq, cond_seq_mask, gamma, rng):
    h = t_next - t                                                  # 时间步长
    alpha = jnp.clip(1.0 - gamma * h, 0.0, 1.0)                     # 信号保留比例
    t_back = alpha * t                                              # 回退时间

    # 重注噪声: z_back = α·z + (1-α)·ε
    eps = jax.random.normal(rng, z.shape) * config.denoiser_noise_scale
    z_back = restore_cond(alpha * z + (1.0 - alpha) * eps,
                          cond_seq, cond_seq_mask)

    # 在 (z_back, t_back) 处求速度
    t_batch = jnp.full((z.shape[0],), t_back)
    v_pred, x_pred = _forward_sample(
        model_apply_fn=model_apply_fn, model_params=model_params,
        z=z_back, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    # 主线步进: z_next = z_back + (t_next - t_back) * v
    return z_back + (t_next - t_back) * v_pred, x_pred
```

SDE 步的逻辑：
1. 当前时间 $t$，步长 $h = t_{next} - t$
2. 后退：$t_{back} = \alpha t$（$\alpha = 1 - \gamma h$）
3. 注入噪声：$z_{back} = \alpha z_t + (1-\alpha) \epsilon$（信号/噪声混合）
4. 在扰动后的 $(z_{back}, t_{back})$ 上求速度
5. 主线步进：$z_{next} = z_{back} + (t_{next} - t_{back}) v$

$\gamma$ 控制随机性：$\gamma = 0$ 退化为 ODE；$\gamma$ 越大噪声越强。极少数步（8-16）时使用 $\gamma = 2.0$，常规步数（32-64）使用 $\gamma = 1.0$ 或 $1.5$。

**x → v 转换**（[sampling_utils.py#L110-L121](https://github.com/lillian039/ELF/blob/main/src/utils/sampling_utils.py#L110-L121)）：

```python
def net_out_to_v_x(net_out, z, t, t_eps=5e-2):
    if isinstance(net_out, tuple):
        net_out = net_out[0]                                       # 丢弃 decoder_logits
    t_reshaped = t.reshape(-1, 1, 1)
    x = net_out                                                    # x-prediction
    v = (x - z) / jnp.maximum(1.0 - t_reshaped, t_eps)            # v = (x̂ - z) / (1-t)
    return v, x
```

`t_eps = 5e-2` 防止 $t \approx 1$ 时除零。采样时 $t$ 不会到达精确的 1（最后一步走 decode mode），所以 `t_eps` 主要防止训练时的边界情况。

## 5. Self-Conditioning 前向

**源码位置**: [sampling_utils.py#L124-L177](https://github.com/lillian039/ELF/blob/main/src/utils/sampling_utils.py#L124-L177)

```python
@partial(jax.jit, static_argnums=(0, 5, 6))
def _forward_sample_self_cond(
    model_apply_fn, model_params, z, t_batch, x_pred_prev, config,
    self_cond_cfg_scale, cond_seq, cond_seq_mask,
):
    t_eps = config.t_eps
    _restore_vx = partial(restore_vx, cond_seq=cond_seq, cond_seq_mask=cond_seq_mask)

    # Path A: in-context CFG tokens → 单次 forward (ELF 默认)
    if config.num_self_cond_cfg_tokens > 0:
        if x_pred_prev is None:
            x_pred_prev = restore_cond(jnp.zeros_like(z), cond_seq, cond_seq_mask)
        z_input_cond = jnp.concatenate([z, x_pred_prev], axis=-1)  # [B, S, 2*D]
        self_cond_scale_batch = jnp.full(
            (z.shape[0],), self_cond_cfg_scale)                    # [B]
        net_out_cond = model_apply_fn(
            {"params": model_params}, z_input_cond, t_batch,
            deterministic=True,
            self_cond_cfg_scale=self_cond_scale_batch,             # 网络内部做 CFG
        )
        v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
        return _restore_vx(v_cond, x_cond)
```

当 `num_self_cond_cfg_tokens > 0`（ELF 默认）时，只需**一次 forward pass**——CFG scale 作为 in-context token 输入，网络在训练时已学会根据 scale token 组合 velocity。这比传统 CFG（需要两次 forward）节省 50% 推理计算。

**无 in-context CFG 时的双 pass**（[sampling_utils.py#L154-L177](https://github.com/lillian039/ELF/blob/main/src/utils/sampling_utils.py#L154-L177)）：

```python
    # Path B: 无 in-context CFG → 两次 forward
    # Unconditional: self-cond = 0
    z_uncond = restore_cond(jnp.zeros_like(z), cond_seq, cond_seq_mask)
    z_input_uncond = jnp.concatenate([z, z_uncond], axis=-1)
    net_out_uncond = model_apply_fn(
        {"params": model_params}, z_input_uncond, t_batch, deterministic=True)
    v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, z, t_batch, t_eps)
    v_uncond, x_uncond = _restore_vx(v_uncond, x_uncond)

    # Conditional: self-cond = x_pred_prev (上一步预测)
    z_input_cond = jnp.concatenate([z, x_pred_prev], axis=-1)
    net_out_cond = model_apply_fn(
        {"params": model_params}, z_input_cond, t_batch, deterministic=True)
    v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
    v_cond, x_cond = _restore_vx(v_cond, x_cond)

    # CFG 组合: v_out = v_uncond + ω * (v_cond - v_uncond)
    v_out = v_uncond + self_cond_cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + self_cond_cfg_scale * (x_cond - x_uncond)
    return _restore_vx(v_out, x_out)
```

注意—self-conditioning CFG 用的是加法形式 $v_{uncond} + \omega(v_{cond} - v_{uncond})$，而不是训练时 CFG 的 $v + (1 - 1/\omega)(v_{cond} - v_{uncond})$（两者在 $\omega \to 1/\omega$ 变换下等价）。

## 6. Input CFG 前向

**源码位置**: [sampling_utils.py#L180-L208](https://github.com/lillian039/ELF/blob/main/src/utils/sampling_utils.py#L180-L208)

```python
@partial(jax.jit, static_argnums=(0, 5, 6, 7))
def _forward_sample(model_apply_fn, model_params, z, t_batch,
                     x_pred_prev, config,
                     cfg_scale, self_cond_cfg_scale,
                     cond_seq, cond_seq_mask):
    # Step 1: conditional forward (with clean cond prefix)
    v_cond, x_cond = _forward_sample_self_cond(
        model_apply_fn, model_params, z, t_batch, x_pred_prev, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    if cfg_scale == 1.0:
        return v_cond, x_cond

    # Step 2: unconditional forward (zeroed cond prefix, no self-cond)
    z_uncond = restore_cond(z, jnp.zeros_like(z), cond_seq_mask)
    x_pred_prev_uncond = (
        None if x_pred_prev is None
        else restore_cond(x_pred_prev, jnp.zeros_like(x_pred_prev), cond_seq_mask)
    )
    v_uncond, x_uncond = _forward_sample_self_cond(
        model_apply_fn, model_params, z_uncond, t_batch,
        x_pred_prev_uncond, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=jnp.zeros_like(cond_seq), cond_seq_mask=cond_seq_mask,
    )

    # CFG 组合
    v_out = v_uncond + cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + cfg_scale * (x_cond - x_uncond)
    return restore_vx(v_out, x_out, cond_seq, cond_seq_mask)
```

两层 CFG 独立控制：
- **Self-conditioning CFG**（在 `_forward_sample_self_cond` 内）：控制自条件的引导强度，调节质量-多样性
- **Input-condition CFG**（在 `_forward_sample`）：控制对输入条件（翻译/摘要的 source）的忠实度

条件生成时：`self_cond_cfg_scale = 1`（不额外引导），`cfg_scale = 2`（中等条件忠实度）。

## 7. 最终步 Decode

**源码位置**: [generation_utils.py#L144-L159](https://github.com/lillian039/ELF/blob/main/src/utils/generation_utils.py#L144-L159)

```python
def _dlm_decode_batch(z, model_params, model_apply_fn, t_final_val,
                       config, self_cond_cfg_scale):
    batch_size = z.shape[0]
    t_final = jnp.full((batch_size,), t_final_val, dtype=z.dtype)  # t = 1.0
    self_cond_cfg_scale_batch = (
        jnp.full((batch_size,), self_cond_cfg_scale, dtype=z.dtype)
        if config.num_self_cond_cfg_tokens > 0 else None
    )
    # self-cond 为 0（最终步不需要迭代细化）
    z_input = (jnp.concatenate([z, jnp.zeros_like(z)], axis=-1)
               if config.self_cond_prob > 0 else z)                 # [B, S, 2*D]

    _, decoder_logits = model_apply_fn(
        {"params": model_params}, z_input, t_final,
        deterministic=True,
        self_cond_cfg_scale=self_cond_cfg_scale_batch,
        decoder_step_active=jnp.array(True),                        # 激活 decode mode
    )                                                               # [B, S, V]
    return jnp.argmax(decoder_logits, axis=-1)                      # greedy decode
```

`decoder_step_active=True` 激活 mode token → unembedding 路径 → logits → argmax。当前只支持 greedy decoding。

## 8. 条件序列处理

条件生成的通用模式——条件 token 在所有采样步中保持 clean：

```python
def restore_cond(z_updated, cond_seq, cond_seq_mask):
    mask = cond_seq_mask                                            # [B, S]
    target_ndim = max(z_updated.ndim, cond_seq.ndim)
    while mask.ndim < target_ndim:
        mask = mask[..., None]                                     # broadcast to [B, S, 1]
    return jnp.where(mask > 0, cond_seq, z_updated)

def restore_vx(v, x, cond_seq, cond_seq_mask):
    if cond_seq is not None:
        x = restore_cond(x, cond_seq, cond_seq_mask)                # 条件位置: x = clean
        v = restore_cond(v, jnp.zeros_like(cond_seq), cond_seq_mask) # 条件位置: v = 0
    return v, x
```

条件位置的 velocity 强制为 0——因为条件序列已 clean，不需要移动。

## 9. 推理参数与性能

### 无条件生成 best config（系统级对比）

| Steps | SC CFG | γ | Gen. PPL | 熵 |
|-------|--------|---|---------|-----|
| 8 | 3 | 2.0 | 67.3 | 5.14 |
| 16 | 3 | 2.0 | 33.7 | 5.16 |
| 32 | 3 | 1.5 | 24.1 | 5.15 |

### 条件生成 best config

- 64 步 ODE, logit-normal schedule
- SC CFG scale = 1, Input CFG scale = 2

### 性能优化技巧

- `jax.lax.scan` 替代 Python for：整个采样循环编译为单个 XLA kernel
- `jax.pmap` 多设备并行生成
- `@partial(jax.jit, static_argnums=...)` 对配置常量化，避免重复编译
- RNG fold_in 生成 per-step 独立噪声
- `decoder_step_active` 用 `jax.lax.cond` 避免在 denoise 步计算 unembedding
