---
tags:
  - Diffusion Model
  - Video Generation
  - LLM Inference
---
# DiT 推理优化：Feature Cache

返回：[专题总览](overview.md)

## 抓住重点

- 特征缓存以相邻步的局部变化预测“重算是否值得”。它用受控近似 \(\tilde f_\theta\) 替换 \(f_\theta\)，属于质量换速度。
- DBCache 的关键不是“跳步”：先计算锚点 block 得到误差信号，再选择复用中段、尾部校正或完整计算。
- SCM 是强制计算日程，TaylorSeer 是缓存特征的局部外推；二者分别约束决策与近似值。
- 阈值必须在目标模型、采样器、步数和条件分布上校准；当前集成与层间 offload/FSDP 存在硬约束。

## 1. 为什么能跳

设 \(h_t^\ell\) 为第 \(t\) 个去噪步、第 \(\ell\) 个 block 后的隐状态，\(r_t^\ell=h_t^\ell-h_t^{\ell-1}\) 为该 block 的 residual。缓存的可检验假设是锚点集合 \(A\) 上

\[
\delta_t=\frac{\sum_{\ell\in A}\lVert r_t^\ell-r_{t'}^\ell\rVert_1}
{\sum_{\ell\in A}\lVert r_{t'}^\ell\rVert_1+\eta}\leq\rho,
\]

其中 \(t'\) 是最近完整计算步，\(\eta>0\) 防止分母为零，\(\rho\) 是模型专属阈值。小 \(\delta_t\) 只说明锚点局部变化小；尾部校正和连续缓存上限用于限制其沿层、沿时间的累积误差。

## 2. Cache-DiT 三件套

### DBCache（Dual Block Cache）

令 \(L\) 是 block 数，\(a\) 和 \(b\) 分别是前、后重算 block 数。一条候选步按下列结构执行：

```text
[ 1,\ldots,a：锚点重算 ] → 判定 \(\delta_t\) → [ a+1,\ldots,L-b：复用或预测 ] → [ L-b+1,\ldots,L：校正 ]
```

| 旋钮 | 默认（库） | 含义 |
|------|------------|------|
| \(a\) | 库默认 8 | 产生判定 \(\delta_t\) 的锚点长度 |
| \(b\) | 库默认 0 | 尾部重算长度 |
| \(\rho\) | 库默认 0.08 | 放宽后缓存概率上升，误差预算同步变紧 |
| warmup | 库默认 8 步 | 在轨迹未稳定阶段禁用或稀疏缓存 |
| 连续上限 | 库默认 -1 | 限制连续近似步数 |

不同运行时默认值只是工作负载假设，不能作为质量承诺。配置应报告 \((a,b,\rho)\)、warmup 和最大连续缓存数。

### TaylorSeer

对缓存对象 \(u_t\)，TaylorSeer 用历史差分近似其时间导数，形成一阶或二阶预测 \(\tilde u_t=u_{t'}+\Delta t\,\hat u'_{t'}+\tfrac12\Delta t^2\hat u''_{t'}\)。它的作用是减小“直接复制 \(u_{t'}\)”的局部截断误差；阶数提高也会增加状态和校准假设，不能单凭阶数判断质量更高。

### SCM（Steps Computation Mask）

令 \(m_t\in\{0,1\}\) 是长度 \(T\) 的掩码。\(m_t=1\) 强制完整计算，\(m_t=0\) 才允许动态相似度判定或静态复用。故 SCM 优先级高于 \(\delta_t\)：它把质量预算显式分配给时间轴，而不是让阈值在所有步上自由放行。

### SGLang 集成要点

在并行组 \(\mathcal P\) 中，每个 rank 的局部 \(\delta_t^{(p)}\) 必须约化成同一全局判据（例如加和分子、分母后求比）。否则各 rank 会在不同步数进入不同分支，后续 collective 无法对齐。多 transformer 专家应维护独立的 \((a,b,\rho)\)，因为它们的 residual 标度未必相同。

## 3. TeaCache 与 Spectrum

| 后端 | 判据 | 互斥 |
|------|------|------|
| TeaCache | 用时间条件的调制量变化作为代理误差，决定 residual 是否复用 | 与 Spectrum 互斥 |
| Spectrum | 另一条特征复用策略 | 与 TeaCache 互斥 |

二者对应不同状态与判据，当前参数校验禁止同时启用。TeaCache 的代理量需要模型校准；缺少系数时它可能不产生有效跳过。

## 4. 组合约束

| 组合 | 行为 | 链接 |
|------|------|------|
| Cache-DiT + DiT layerwise | 硬错误 | [Offload §5](memory_offload.md#5-组合约束) |
| Cache-DiT + FSDP | 硬错误或自动关 FSDP | 同上 |
| TeaCache + Spectrum | 硬错误 | 上文 |
| Cache-DiT + BCG | CLI：互斥，BCG 优先 | [Graph §4](graph_runtime.md#4-与-compile--cache--offload) |
| Cache-DiT + Progressive | 文档称 experimental（换分辨率会 refresh context） | [Progressive §3](progressive_resolution.md#3-组合约束) |

官方建议：先有 lossless 风格基线与验收标准，再开 cache → [Correctness](correctness.md#4-建议验收清单)。

## 4.1 从判定到复用

每步完整计算比例记为 \(q\)，中段平均重算比例为 \(r\)。理想 DiT block 计算成本从 \(L\) 降至 \(qL+(1-q)[a+b+r(L-a-b)]\)。实际加速还要扣除判定、缓存读写和未被隐藏的同步开销。缓存后 block 总量变化，不能用某个 kernel 的占比变化代表端到端收益。

## 附录：实现证据与参数

| 主题 | 路径 |
|------|------|
| DBCache 配置 | `cache-dit/.../cache_config.py` |
| Cache manager / similarity | `cache-dit/.../cache_manager.py` |
| SGLang 集成 + similarity patch | `sglang/.../cache/cache_dit_integration.py` |
| Env 默认 | `sglang/.../envs.py`（`SGLANG_CACHE_DIT_*`） |
| TeaCache/Spectrum 互斥 | `configs/sample/sampling_params.py` |

## 相关阅读

- [专题总览](overview.md) · [Memory Offload](memory_offload.md) · [Parallelism](parallelism.md)（并行下 similarity） · [Correctness](correctness.md)

pin：Cache-DiT 的 `cache_contexts/cache_config.py`、`cache_manager.py` 与 `calibrators/taylorseer.py`；SGLang 的 `cache_dit_integration.py`、`envs.py`、`sampling_params.py`。互斥的显存原因见 [Offload](memory_offload.md#5-组合约束)。
