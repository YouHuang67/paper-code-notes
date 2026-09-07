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

多卡把一次 DiT forward 的不同维切开。轴互相正交时可以相乘；切同一语义维的则互斥。

## 1. 合成公式

```text
num_gpus = cfg_parallel_degree × tp_size × sp_degree
sp_degree = ulysses_degree × ring_degree
```

| 策略 | 切什么 | 通信 | 标志 |
|------|--------|------|------|
| CFG parallel | guidance 分支 | 每步 combine | `--cfg-parallel-size` |
| TP | 权重与头 | 每 block all-reduce | `--tp-size` |
| Ulysses SP | 序列 ↔ 头 | 两次 all-to-all | `--ulysses-degree` |
| Ring SP | 序列行 | 邻居 K/V 旋转 + 计算重叠 | `--ring-degree` |
| KV-Gather CP | 序列行 | 一次 K/V all-gather | `--kv-gather-degree` |
| DP | 请求 | 副本间无 | `--dp-size` |

Ring 与 KV-Gather 竞争 **同一 SP 槽位**（都回答「本地 Q 行如何看见远程 K/V」），不是互补。默认：`sp_degree=2` 时常落到 `kv_gather`；更高度数默认 Ulysses。

## 2. 形状与约束

Ulysses 输入 all-to-all 后，ring 在「同一组头、不同行」上做 online softmax merge。关键整除：

- `num_heads % tp == 0`  
- `(num_heads / tp) % ulysses == 0`（注意是 TP **本地**头数）  
- 序列（含模型 packing）能被 `ulysses × ring` 整除  
- Ring 需要 backend 声明 `supports_ring_rotation()`（如 fa、sage_attn）

拓扑：Ulysses 组连续 rank（吃满 NVLink bisection），Ring 组跨步 rank（邻居跳走慢互联）。错映射仍正确但变慢。

## 3. 实践选择

- True-CFG：先试 CFG Parallel  
- 节点内：优先 Ulysses  
- 跨节点：再考虑 Ring  
- 不少 DiT 上 TP 易被通信拖住，需实测  

稀疏 Attention 与 USP/Ring 的已知限制（例如部分 masked/varlen 路径）见并行文档与稀疏专题；本篇不展开算法。

H3 上 Ulysses/Ring 与 late gather 的具体数据流：[DiT Runtime 与 Collectives](../minimax_h3/07_dit_runtime_and_collectives.md)。

## 4. 本轮缺口

- Packed QKV、2 卡 IPC All-to-All、comm-compute overlap 的实现文件级拆解  
- Encoder Parallel Folding 与 SP Group 复用（见 [Encoder & VAE](encoder_vae.md)）
