---
tags:
  - Diffusion Model
  - Video Generation
  - LLM Inference
  - CUDA
---
# DiT 推理优化：Parallelism

**文档**: `refs/codes/sglang/docs/docs/sglang-diffusion/parallelism.mdx`  
**相关**: `ring_sp_performance.mdx`、`encoder_parallel.mdx`

返回：[专题总览](overview.md)

## 抓住重点

- 多卡公式：`num_gpus = cfg × tp × sp`，其中 `sp = ulysses × ring`（或用 KV-Gather 占同一 SP 槽位）。
- **Ring 与 KV-Gather 互斥争槽**：都回答「本地 Q 行如何看见远程 K/V」。
- 拓扑：Ulysses 吃 NVLink 连续 rank；Ring 邻居可跨慢互联。错映射仍正确但变慢。
- Encoder 并行是另一轴，见 [Encoder & VAE](encoder_vae.md)；与 Progressive 的 SP 互斥见 [Progressive](progressive_resolution.md#3-组合约束)。

## 1. 合成公式

```text
num_gpus = cfg_parallel_degree × tp_size × sp_degree
sp_degree = ulysses_degree × ring_degree   # 或由 kv_gather_degree 占用 SP 槽
```

| 策略 | 切什么 | 通信 | 标志 |
|------|--------|------|------|
| CFG parallel | guidance 分支 | 每步 combine | `--cfg-parallel-size` |
| TP | 权重与头 | 每 block all-reduce | `--tp-size` |
| Ulysses SP | 序列 ↔ 头 | 两次 all-to-all | `--ulysses-degree` |
| Ring SP | 序列行 | 邻居 K/V 旋转 + 计算重叠 | `--ring-degree` |
| KV-Gather CP | 序列行 | 一次 K/V all-gather | `--kv-gather-degree` |
| DP | 请求 | 副本间无 | `--dp-size` |

默认行为（`server_args`）：`sp_degree=2` 时常自动落到 `kv_gather_degree=2`；更高 SP 默认偏向 Ulysses。`kv_gather_degree > 1` 时不与 Ulysses/Ring 组合。

## 2. 形状与约束

Ulysses 输入 all-to-all 后，Ring 在「同一组头、不同行」上做 online softmax merge。关键整除：

- `num_heads % tp == 0`  
- `(num_heads / tp) % ulysses == 0`（TP **本地**头数）  
- 序列（含模型 packing）能被 `ulysses × ring` 整除  
- Ring 需要 backend 声明 `supports_ring_rotation()`（如 fa、sage_attn）

拓扑建议（官方并行文档）：

- Ulysses 组：连续 rank，吃满 NVLink bisection  
- Ring 组：跨步 rank，邻居跳走慢互联  

稀疏 Attention 与 USP/Ring 的已知限制见并行文档与稀疏专题；本篇不展开算法 → [overview 稀疏边界](overview.md#4-与稀疏-attention-的边界)。

## 3. 实践选择

| 目标 | 优先尝试 |
|------|----------|
| True-CFG | CFG Parallel |
| 节点内长序列 | Ulysses |
| 跨节点扩展 | Ring（或测 KV-Gather） |
| DiT 上 TP | 实测；不少模型易被通信拖住 |
| 编码阶段闲置 | [Encoder Parallel](encoder_vae.md) |

H3 上 Ulysses/Ring 与 late gather 数据流：[DiT Runtime 与 Collectives](../minimax_h3/07_dit_runtime_and_collectives.md)。

Cache-DiT YAML 也可声明 1D/2D/3D parallelism（`cache_dit.mdx`）；SGLang 集成下 similarity 需组归约 → [Feature Cache](feature_cache.md#2-cache-dit-三件套)。

## 4. 源码 / 文档锚点

| 主题 | 路径 |
|------|------|
| 合成公式与拓扑 | `docs/.../parallelism.mdx` |
| Ring 性能专题 | `docs/.../ring_sp_performance.mdx` |
| SP/KV-Gather 校验 | `server_args.py` `_validate_parallelism` / kv_gather 自动设定 |
| Encoder 并行 | `docs/.../encoder_parallel.mdx` |

## 相关阅读

- [专题总览](overview.md) · [Encoder & VAE](encoder_vae.md) · [Feature Cache](feature_cache.md) · [Graph Runtime](graph_runtime.md) · [Correctness](correctness.md)

## 附录：仍可加深

- Packed QKV、2 卡 IPC All-to-All、comm-compute overlap 的实现文件级拆解  
- Encoder Parallel Folding 与 SP Group 复用细节（见 Encoder 篇附录）
