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
- 文档：`cache_dit.mdx`、`teacache.mdx`、`caching-acceleration.mdx`、`spectrum.mdx`

返回：[专题总览](overview.md)

## 抓住重点

- Feature cache 利用相邻 denoise step 隐状态相似，跳过中间 block 或整步计算 → **quality-tradeoff**。
- Cache-DiT 主轴是 **DBCache（Fn/Bn + residual diff）**，可叠 **TaylorSeer** 校准与 **SCM** 逐步掩码。
- SGLang 另有 TeaCache、Spectrum；二者互斥。
- **不可**与 DiT layerwise offload、FSDP 同开；实务上勿与 BCG 同开。阈值不能跨模型照搬。

## 1. 为什么能跳

相邻 timestep 的 residual / hidden 往往缓慢变化。若前若干 block 算出的 residual 相对差低于阈值，中间 block 可复用缓存结果，只在尾部再算少量块做校正。跳过的是整段 DiT block 计算，因此 e2e 加速可观，但阈值与 warmup 直接决定画质。

## 2. Cache-DiT 三件套

配置来自 `BasicCacheConfig` / `DBCacheConfig`（`cache_contexts/cache_config.py`）。

### DBCache（Dual Block Cache）

单步内把 transformer 分成三段：

```text
[ Fn 必算块 ] → 估 residual L1 diff → [ 中间可 cache ] → [ Bn 再算校正 ]
```

| 旋钮 | 默认（库） | 含义 |
|------|------------|------|
| `Fn_compute_blocks` | 8 | 前 Fn 块必算，用于稳定 L1 diff |
| `Bn_compute_blocks` | 0 | 后 Bn 块再算，融合近似 hidden |
| `residual_diff_threshold` | 0.08 | 越高越激进（更快、更糙） |
| `max_warmup_steps` | 8 | warmup 内不 cache（或按 interval） |
| `max_continuous_cached_steps` | -1 | 连续 cache 步数上限，防漂移 |
| `enable_separate_cfg` | None | Wan / Qwen-Image 等分 CFG 步要单独设 |

SGLang env 默认更激进（示例）：`SGLANG_CACHE_DIT_FN=1`、`RDT=0.24`、`WARMUP=4`、`MC=3`（见 `envs.py`）。**以目标模型实测为准，勿直接抄默认。**

### TaylorSeer

`calibrators/taylorseer.py` + `TaylorSeerCalibratorConfig`：用泰勒展开预测/校准特征，减轻「直接抄旧 residual」的误差。SGLang：`SGLANG_CACHE_DIT_TAYLORSEER`、`SGLANG_CACHE_DIT_TS_ORDER`（1 或 2）。

### SCM（Steps Computation Mask）

`steps_computation_mask`：长度 = `num_inference_steps` 的 0/1 列表；1 = 本步必算，0 = 走 dynamic/static cache。可用 `steps_mask` / preset（SGLang：`SGLANG_CACHE_DIT_SCM_PRESET` = none/slow/medium/fast/ultra，或自定义 compute/cache bins）。掩码会覆盖其它「是否算本步」决策。

### SGLang 集成要点

路径：`cache_dit_integration.py`。

- 启用：`SGLANG_CACHE_DIT_ENABLED=true`，或 Diffusers backend 下 `--cache-dit-config` YAML。  
- `_patch_cache_dit_similarity`：在 SP/TP 并行下对 similarity 做组归约，保证各 rank 对「是否 cache」判定一致。  
- 支持 secondary transformer（如 Wan2.2 low-noise expert）一组独立 Fn/Bn/RDT env。

## 3. TeaCache 与 Spectrum

| 后端 | 判据 | 互斥 |
|------|------|------|
| TeaCache | timestep embedding 调制输入的相对差异，决定是否复用 residual | 与 Spectrum 互斥 |
| Spectrum | SGLang 另一缓存后端（`spectrum.py` / `spectrum.mdx`） | 与 TeaCache 互斥 |

`sampling_params.py`：`enable_teacache and enable_spectrum` → ValueError。  
Wan2.2 注释：TeaCache 系数未校准前可能 silent no-op，可改用 Cache-DiT。

## 4. 组合约束

| 组合 | 行为 | 链接 |
|------|------|------|
| Cache-DiT + DiT layerwise | 硬错误 | [Offload §4](memory_offload.md#4-组合约束必须先读) |
| Cache-DiT + FSDP | 硬错误或自动关 FSDP | 同上 |
| TeaCache + Spectrum | 硬错误 | 上文 |
| Cache-DiT + BCG | CLI：互斥，BCG 优先 | [Graph §4](graph_runtime.md#4-与-compile--cache--offload) |
| Cache-DiT + Progressive | 文档称 experimental（换分辨率会 refresh context） | [Progressive §3](progressive_resolution.md#3-组合约束) |

官方建议：先有 lossless 风格基线与验收标准，再开 cache → [Correctness](correctness.md#4-建议验收清单)。

## 5. 源码锚点

| 主题 | 路径 |
|------|------|
| DBCache 配置 | `cache-dit/.../cache_config.py` |
| Cache manager / similarity | `cache-dit/.../cache_manager.py` |
| SGLang 集成 + similarity patch | `sglang/.../cache/cache_dit_integration.py` |
| Env 默认 | `sglang/.../envs.py`（`SGLANG_CACHE_DIT_*`） |
| TeaCache/Spectrum 互斥 | `configs/sample/sampling_params.py` |

## 相关阅读

- [专题总览](overview.md) · [Memory Offload](memory_offload.md) · [Parallelism](parallelism.md)（并行下 similarity） · [Correctness](correctness.md)

## 附录：仍可加深

- DBCache 前向对照 `cache_blocks/` 的逐步数据流图  
- TaylorSeer 阶数与误差项公式  
- SCM 各 preset 掩码生成算法细节  
- TeaCache 多项式校准系数在各模型 SamplingParam 中的挂载点
