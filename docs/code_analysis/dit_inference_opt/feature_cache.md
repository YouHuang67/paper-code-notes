---
tags:
  - Diffusion Model
  - Video Generation
  - LLM Inference
---
# DiT 推理优化：Feature Cache

**源码**:

- Cache-DiT：`refs/codes/cache-dit`（`DBCacheConfig`、TaylorSeer、SCM）
- SGLang 集成：`multimodal_gen/runtime/cache/cache_dit_integration.py`、`teacache.py`、`spectrum.py`
- 文档：`docs/docs/sglang-diffusion/cache_dit.mdx`、`teacache.mdx`、`caching-acceleration.mdx`

Feature cache 利用相邻 denoise step 隐状态高度相似，跳过部分 block 或整步计算。属 **quality-tradeoff**：阈值与策略敏感，必须做质量回归。

## 1. Cache-DiT 三件套

来自 `BasicCacheConfig` / `DBCacheConfig`（`cache_contexts/cache_config.py`）：

| 机制 | 作用 |
|------|------|
| **DBCache** | 用前 Fn 个 block 估 residual diff；中间块可复用；后 Bn 块可再融合校正。`residual_diff_threshold` 越高越激进 |
| **TaylorSeer** | 用泰勒展开校准/预测特征，减轻「直接抄旧特征」的误差（`calibrators/taylorseer.py`） |
| **SCM** | `steps_computation_mask`：逐步强制计算(1)或走 cache(0)；可用 `steps_mask` 策略名（如 fast）生成 |

其它旋钮：`max_warmup_steps`、`warmup_interval`、`max_cached_steps`、`max_continuous_cached_steps`、`enable_separate_cfg`、`force_refresh_step_hint` 等。

SGLang 启用：`SGLANG_CACHE_DIT_ENABLED=true`，或 Diffusers backend 下 `--cache-dit-config` YAML。集成层会 patch 并行下的 similarity 归约（`cache_dit_integration.py`），使 SP/TP 组上的 diff 判定一致。

## 2. TeaCache 与 Spectrum

- **TeaCache**：基于 timestep embedding 调制输入的相对差异，决定是否复用 residual（经典 training-free 路径）  
- **Spectrum**：SGLang 另一缓存后端  
- 采样参数校验：`enable_teacache` 与 `enable_spectrum` **互斥**

## 3. 组合约束

- **不可**与 DiT layerwise offload 同开（SGLang 硬校验）  
- **不可**与 FSDP inference 同开（或自动关 FSDP）  
- 与 BCG / 静态 graph：实务上避免同开  
- 阈值不能跨模型照搬（Qwen-Image 上好用的值换 Wan 可能崩画质）

官方建议：先有 lossless 风格基线与验收标准，再开 cache。

## 4. 本轮缺口

- DBCache Fn/Bn 前向数据流逐步图解（对照 `cache_blocks/`）  
- TaylorSeer 阶数与误差项  
- SCM 各 policy 掩码生成算法  
- TeaCache 多项式校准系数在 SGLang 中的挂载点
