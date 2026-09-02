---
tags:
  - Sparse Attention
  - Video Generation
  - Diffusion Model
  - CUDA
---

# CalibAtt: Accelerating Text-to-Video Generation with Calibrated Sparse Attention

- 论文：https://arxiv.org/abs/2603.05503
- 团队：Apple, Tel Aviv University
- 代码：https://github.com/apple/ml-calibatt（本地 `refs/codes/calibatt`）

## 概述

CalibAtt 是一个面向视频扩散 Transformer 的 **training-free sparse attention** 方法。它的核心观点不是“当前输入的 attention 是稀疏的”，而是更进一步：

- 很多 block-level attention 连接在不同 prompt / noise 下都长期接近 0；
- 有些 attention head 还存在更强的空间重复结构，很多 query 行之间几乎是重复的。

因此它把 attention 加速拆成两部分：

1. **离线标定 block 稀疏结构**：为每个 `(timestep, layer, head)` 生成一个数据无关的块掩码；
2. **离线标定空间重复 head**：为重复性很强的 head 只算少量 anchor query rows，再把结果广播给邻近行。

在线阶段它不做 query-dependent block search，而是直接读取预编译的 skip list 和重复头字典，把 attention 送进定制 CUDA kernel。论文报告在 Wan 2.1 14B、Mochi 1、LightX2V 上达到最高 **1.58x 端到端加速**，同时保持 VBench 质量和文本对齐。

## 核心动机

CalibAtt 的问题设定非常明确：许多 training-free sparse attention 方法虽然能减少 attention FLOPs，但在线仍要额外做 selector、聚类或压缩 attention 估计，这会带来两类代价：

- **在线开销**：需要为每次推理、每层、每步重新决定哪些块该算；
- **执行碎片化**：即使理论上很 sparse，如果稀疏结构在运行时高度不规则，GPU 上也难以转化成真实速度。

CalibAtt 的回答是：如果大量无效块本来就跨 prompt 稳定，那就没有必要反复在线识别。应当在少量 calibration prompts 上一次性把这些结构统计出来，编译成：

- 每个 `(t,l,h)` 的 block mask；
- 每个 `(t,l,h)` 是否属于 spatial repetition head；
- 每个 query block-row 对应的 key block 连续区间 skip list。

这条路线本质上是 **offline compile, online execute**。

## 四个关键观察

论文的方法建立在四个现象上。

### 1. attention 在 token 级和 block 级都很稀疏

作者先观察 full attention map，发现大量 token-to-token 连接长期接近零；进一步把 attention 聚合到 `B x B` block 后，这个稀疏性仍然明显。于是可行的加速粒度不是 token，而是 FlashAttention 友好的 block。

### 2. 稀疏模式在不同层、头、步之间差异很大

同一模型中，不同 layer/head/timestep 的 attention 结构差异明显，因此不能使用统一静态 mask。CalibAtt 的基本单位是：

$$
(t, l, h)
$$

即每个 diffusion step、transformer layer、attention head 都单独标定。

### 3. 稀疏模式跨 prompt 大体稳定

这是整个方法成立的关键。作者发现，对固定 `(t,l,h)`，虽然具体 attention 值会随 prompt / latent noise 变化，但“哪些 block 常被保留、哪些 block 常可跳过”在跨 prompt 上是稳定的，而且统计分布通常呈双峰：

- 一批块几乎总是要算；
- 一批块几乎总是可跳过；
- 中间模糊块比例相对小。

这正适合做离线 aggregation。

### 4. 一些 head 在空间行维度上高度重复

对某些 `(t,l,h)`，同一帧内不同 spatial rows 的 query attention pattern 几乎相同。于是可以只算少数 anchor rows，再把输出复制给相邻行。这和 block sparsity 不是同一类冗余，反而往往互补：空间重复强的头，块稀疏未必高；块稀疏高的头，空间重复未必强。

## 数学建模

### Block energy

给定 post-softmax attention 矩阵：

$$
P = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}\right)
$$

把 `P` 按 `B x B` 划成块。对第 `r` 个 query block-row 和第 `c` 个 key block-column，定义 block energy：

$$
E_{r,c}
=
\frac{1}{B}
\sum_{i \in \mathcal I_r}
\sum_{j \in \mathcal J_c}
P_{i,j}
$$

这里的平均方式很重要：对每个 query block-row，整行 block energy 和为 1。于是对每个 query block-row，可以做一个最小覆盖问题：

$$
\min_{\mathcal S_r} |\mathcal S_r|
\quad
\text{s.t.}
\quad
\sum_{c \in \mathcal S_r} E_{r,c} \ge \epsilon(t)
$$

即保留最少的 key blocks，使累计注意力质量达到阈值。

### timestep-dependent 能量阈值

CalibAtt 不用固定阈值，而是对 diffusion timestep 使用指数型 schedule：

$$
\epsilon(t) = A + (C-A)\exp(-kt/T)
$$

其中 `t=0` 对应高噪声阶段。论文给出的直觉是：早期步更敏感，不能太激进；后期步 attention 更集中，可以更稀疏。实际在 Wan 上，`\epsilon(t)` 依然较高，大约落在 `0.99` 到 `0.84` 区间，说明它的策略是“保守地跳过低质量尾部”，而不是极端稀疏。

### 跨 prompt 聚合

对每个 prompt `p`，用上面的能量覆盖策略得到一个二值 block mask：

$$
M_p^{(t,l,h)} \in \{0,1\}^{N_B \times N_B}
$$

然后跨 calibration prompt 集合 `\mathcal D` 做平均：

$$
\bar M^{(t,l,h)}
=
\frac{1}{|\mathcal D|}
\sum_{p \in \mathcal D} M_p^{(t,l,h)}
$$

`\bar M_{r,c}` 表示该块在不同 prompt 中被保留的频率。最后再用 agreement threshold `\rho` 做二值化：

$$
M^{(t,l,h)}_{r,c} =
\mathbf 1 \left[\bar M^{(t,l,h)}_{r,c} \ge \rho \right]
$$

这个 `\rho` 控制的是“跨 prompt 共识”。若 `\rho` 更高，只保留几乎所有 prompt 都需要的块，会更 sparse 但也更冒险；若 `\rho` 更低，则更保守。论文默认使用 `\rho=0.5`。

## Spatial Repetition 机制

CalibAtt 的第二条加速线不是块稀疏，而是 query 压缩。

设一帧有 `H` 行、每行 `W` 个 tokens。对 frame `f` 的第 `i` 个 spatial row，记它的全部 attention pattern 展平后为：

$$
P^{(f,i)} = \operatorname{flatten}(P[\mathcal I^{(f,i)}, :])
$$

若同一帧内不同空间行满足：

$$
P^{(f,i)} \approx P^{(f,j)}
$$

就可以只选 `k` 条 anchor rows，计算它们对所有 keys 的 attention，再把结果广播给最近的空间行。这样单帧 query token 数从 `HW` 降到 `kW`，对应稀疏率：

$$
1 - \frac{k}{H}
$$

论文中用跨 prompt 的平均余弦相似度定义一个 repetition score `s^{(t,l,h)}`，若它超过阈值 `\gamma`，就把该 head 标成 spatially repetitive。默认设置是：

- `k = 5`
- `\gamma = 0.87`

这意味着空间重复路径本身也是经过离线标定后，在在线阶段作为硬分支使用。

补充材料里的这组参数也说明，作者并没有把 repetition 当成一个很激进的近似分支。`k=5` 仍然保留了每帧多个 anchor rows，本质上是在 query 维做温和降采样，而不是极限压缩；`\gamma=0.87` 也意味着只有相似度足够高的 heads 才会进入这条路径。

## 实现结构

官方实现现已开源（`apple/ml-calibatt`）。从论文正文与补充材料仍可还原其实现结构：CalibAtt 不是一组孤立 kernel，而是一条从标定到执行的编译式流水线：

1. **离线统计层**
   - 运行 dense attention；
   - 计算 block energy；
   - 为每个 `(t,l,h)` 输出 per-prompt block mask；
   - 计算 spatial repetition score。
2. **离线编译层**
   - 对 per-prompt masks 做均值聚合与阈值化；
   - 生成最终 mask dictionary；
   - 把每个 query block-row 的 active key block 转成 skip list；
   - 生成 repetition dictionary。
3. **在线运行层**
   - 根据 `(t,l,h)` 读取 mask 或 repetition 标志；
   - 若是普通 head，走 block-sparse kernel；
   - 若是重复 head，走 reduced-query FA3 + broadcast。

也就是说，CalibAtt 的系统边界比 `SVOO` 更窄：它不试图在线估计结构，而是尽量把一切可能静态化的内容都静态化。

如果按论文 Figure 5 把这条流水线再写得更细一点，可以理解成：

```text
dense calibration attention
  -> block energy
  -> per-prompt block masks
  -> cross-prompt aggregation
  -> mask dictionary
  -> skip-list compilation

dense calibration attention
  -> row similarity statistics
  -> repetition dictionary

inference
  -> lookup by (t, l, h)
  -> block-sparse path or anchor-row path
```

这说明 CalibAtt 并不是一篇“提出一种 sparse pattern”的论文，而是一个完整的离线编译式部署方案。

## CUDA 实现细节重建

### 1. 标定阶段的 block energy kernel

论文明确提到：离线 calibration 的 block energy 统计用的是 **custom CUDA kernel**，而且它在 block granularity 上直接累加所需统计量，避免显式物化完整 `P`。

这个实现意图非常清晰。若直接保存 `N x N` attention map 再做块聚合：

- 显存开销是二次的；
- HBM 写回巨大；
- 对 calibration 这种“要跑很多 prompt”的任务尤其低效。

更合理的实现方式应当类似 FlashAttention 的在线 softmax 路径：以 query block 为主，逐块扫描 key blocks，在计算 softmax 的同时直接累加：

$$
E_{r,c} = \frac{1}{B}\sum_{i \in \mathcal I_r}\sum_{j \in \mathcal J_c} P_{i,j}
$$

因此从 GPU 角度看，这个 kernel 的自然设计会是：

- 一个 CTA 对应一个 `(batch/head, query block-row)`，或一个 `(batch/head, query block-row, key tile group)`；
- CTA 内先算 `Q_r K_c^\top` tile；
- 用在线 softmax 维护 query row 的分母；
- 同时把属于当前 `key block-column` 的 softmax 质量累加到共享/寄存器中的 block energy 缓冲；
- 最后只把 `N_B x N_B` 大小的 block energy 写回。

这类 kernel 的关键收益不是减少 FLOPs，而是 **不落完整 attention matrix**。否则 calibration 成本会被显存和 IO 放大得很厉害。

结合论文给出的 calibration 预算，这个 kernel 的意义会更具体。Wan2.1 14B 720p 在完整配置下的离线成本可到 `89.6` H100 GPU-hours`；即使把 prompt 数缩到 `16`，仍有 `13.7` H100 GPU-hours` 量级。若 block energy 统计还要显式保存完整 attention matrix，这个一次性成本会进一步恶化。

### 2. skip list 编译

论文正文说明，推理 kernel 不直接读稠密 mask，而是读 **precomputed read-only skip lists**。具体做法是：

- 对每个 `(t,l,h)` 的二值 block mask；
- 对每个 query block-row `r`；
- 把所有为 1 的 key block-columns 编码成若干连续区间。

这一步的意义非常大。因为 GPU 稀疏执行最怕的是：

- 为每个 active block 单独解码索引；
- 稀疏元数据本身比算子还耗带宽。

如果把激活块压成连续 ranges，那么 kernel 遍历时更像：

```text
for each query block-row:
    for each active key interval [c0, c1):
        iterate contiguous KV blocks
```

这样：

- metadata 体积变小；
- key/value 的块访问更连续；
- 更利于 shared memory / TMA / Tensor Core 密集计算。

这和 `SVOO`/`FlashInfer variable block sparse` 的 CSR-like 思路相似，只是 CalibAtt 这里的结构更静态，适合预先压缩成 skip list。

补充材料还给了一个很实际的工程信号：skip-list 存储本身是个大问题。Wan-T2V 14B 720p 的原始结构占用很高，经过压缩后才降到约 `6.3 GB` 且 sparsity 几乎不受影响。换句话说，这篇方法不只是“把 mask 预先算好”，而是连这些 mask 的部署格式都认真做了系统设计。

### 3. 基于 FlashAttention3 的 block-sparse kernel

论文说推理 kernel “based on FlashAttention3 and prior block-sparse kernels”，并针对 **pre-computed masks varying per timestep / layer / head** 做了优化。

从这个描述可以推断：

- 它不会自己重写一整套 attention 数值逻辑；
- 更可能是保留 FA3 的 block iterator / online softmax 主干；
- 只把 dense KV block iterator 换成由 skip list 驱动的 sparse iterator。

因此更合理的 kernel 结构应当是：

1. 复用 dense FA3 的 query tile 装载与在线 softmax 状态；
2. 用 skip list 逐段遍历当前 query block-row 对应的 active KV block intervals；
3. 对每个区间内的 block，仍使用规则 dense tile 的 MMA 路径。

这个结构的核心价值在于：全局虽然是 sparse 的，但局部计算单元仍是规则大块，因此 Tensor Core 不会因为 token-level 不规则稀疏而大幅掉利用率。

因此执行主线应当是：

1. 一个 CTA 或一个 warp-group 负责一个 query tile；
2. query tile 常驻寄存器/shared memory；
3. 通过 skip list 依次访问当前 row 对应的 active KV block intervals；
4. 对每个 active block 调用与 dense FA3 类似的 QK / softmax / PV 流程；
5. 用同样的在线 softmax 状态合并所有 selected blocks。

这种设计有两个直接好处：

- 稀疏版本复用 dense FA3 的数值稳定与 Tensor Core 高效路径；
- 由于块是预编译出来的连续区间，CTA 内不需要做复杂分支判断。

### 4. Spatial repetition 路径

对于被标成 repetitive 的 heads，作者没有用自定义稀疏 query kernel，而是直接：

- 构造只含 anchor rows 的 reduced query set；
- 调标准 FA3；
- 再把结果广播到邻近 spatial rows。

这说明作者做了很现实的权衡：重复头的结构已经足够规则，未必要再叠加 block sparse kernel。因为如果同时叠加两种压缩：

- 工程复杂度高；
- 广播/索引恢复逻辑会变复杂；
- CTA 负载也更难平衡。

直接使用 dense FA3 on reduced queries，本质上是在 query 维做规则降采样，通常比在 KV 维做更极端的不规则稀疏更容易吃满 GPU。

更重要的是，这条分支几乎不引入新的元数据格式：它只是在前端减少 query rows，再把结果 broadcast 回去。因此它继承的是标准 FA3 的稳定执行路径，而不是另起一套复杂 sparse query kernel。

### 5. 内存开销与工程代价

CalibAtt 的一个现实问题是 mask storage。论文明确给出：

- Wan2.1 14B 720p 下，mask memory overhead 约 **21.5 GB**；
- 可通过 skip-list 矩阵压缩把 footprint 降到约 **6.3 GB**，且 sparsity 损失很小。

这说明它的设计是典型的 **以存换算**：

- 在线不再做路由；
- 但要提前存好所有 `(t,l,h)` 的结构。

因此 CalibAtt 更适合：

- 推理配置固定；
- 重复生成量大；
- 可以摊销一次性 calibration 成本；
- GPU 显存较充足，或者愿意做更激进的离线压缩。

## 为什么它能快

CalibAtt 的加速不是来自单一来源，而是三部分叠加：

1. **省掉在线 selector**  
   推理阶段没有额外的 runtime route construction。
2. **skip list 让 block-sparse 执行更规整**  
   每个 query block-row 的 active KV blocks 已压成连续区间，避免 metadata 开销太大。
3. **对重复头直接减少 query 数**  
   这不是只减少 key/value 访问，而是连 query-side tile 数也变少。

从 GPU 角度看，CalibAtt 的目标不是“最高理论 sparsity”，而是让稀疏结构变得足够静态、足够规则，从而真正映射到高利用率内核。

## 实验结果

### 高步数视频生成

论文在 Wan 2.1 14B、Mochi 1 等高 timestep 场景与 `Dense FA3`、`RadialAttention`、`SVG2`、`SpargeAttention` 比较。结论是：

- CalibAtt 获得最高或接近最高的平均 attention sparsity；
- 端到端 latency 最优或接近最优；
- VBench 的 `Semantic / Quality / Total` 维持在 dense 基线附近。

摘要中给出的最强结果是：

- **最高 1.58x 端到端加速**

同时在可视化比较中，Wan2.1 14B：

- 720p 例子约 `62%` attention sparsity；
- 480p 例子约 `68%` attention sparsity。

附录里的更多结果显示，Mochi 大约能达到 `69%` sparsity，LightX2V 480p/720p 则能达到 `70%` / `74%`。这意味着 CalibAtt 的静态标定并不只适用于高步数、大模型配置，对 distilled few-step 场景同样有明显效果。

### Few-step distilled 模型

在 LightX2V 这类 4-step distilled 模型上，CalibAtt 依然有效。这点很重要，因为很多 training-free 方法在 few-step regime 中更容易被在线选择开销抵消，而 CalibAtt 因为在线几乎只剩查表和执行，通常更稳定。

### calibration budget ablation

作者对 calibration prompt 数和 agreement threshold 做了消融，结论是：

- 稀疏-质量曲线会很快稳定；
- 默认 `64` prompts 是稳妥配置；
- 若预算受限，降到 `16` prompts` 也只带来很小的质量和稀疏差异；
- 对 Wan2.1 14B 720p，calibration 成本可从约 `89.6` H100 GPU-hours 降到 `13.7` H100 GPU-hours。

这说明它的 cross-prompt topology 稳定性确实很强，否则不可能用这么少的校准样本把静态 mask 编译出来。

这组实验还有一个更深的系统含义：CalibAtt 的离线成本虽然不小，但所有中间产物都是可复用的部署资产：

- mask dictionary
- repetition dictionary
- skip lists

只要模型和推理配置不变，这些结构就能被长期复用，因此这篇方法更像一个前置编译成本换在线吞吐的部署优化方案。

## 局限

- 没有在线适应能力，可能错过特定 prompt 的稀疏机会；
- 标定成本不低，且一次性离线成本要靠重复推理摊销；
- mask / skip-list 存储开销显著；
- 结构固定在 block 粒度，不能像在线聚类方法那样细致适配内容；
- 官方代码已开源（`apple/ml-calibatt`），本文笔记仍以论文重建为主，未做逐行代码分析。

## 关键启示

- **如果稀疏结构跨 prompt 稳定，就应该离线编译，而不是每次在线搜索。**
- **真正的系统收益来自“结构静态化 + skip list 压缩 + 规则执行”，不是单纯提高 sparsity 百分比。**
- **空间重复与块稀疏是两种不同冗余来源，联合使用比只盯一种结构更划算。**
- **CalibAtt 的价值不在于发明新 attention 近似，而在于把 calibration 做成了可落地的推理编译流程。**
