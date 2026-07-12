# PISA 代码实现：Piecewise Sparse Attention

对应论文：[PISA: Piecewise Sparse Attention Is Wiser for Efficient Diffusion Transformers](../../paper_reading/pisa.md)

官方代码位于 `refs/codes/piecewise-sparse-attention`，当前解析提交：
`655648df86cafd75042dc32d6b6be78c4ea0eca8`。公开仓库主要包含 Triton attention kernel 和 FLUX 接入示例；论文中的 Wan2.1 / Hunyuan-Video 推理脚本在当前提交中没有开源，因此视频端工程集成只能从论文描述和通用 DiT attention 替换路径推断。

## 实现结构

PISA 的代码不是一个完整视频生成框架，而是一个可插拔 attention backend：

1. `piecewise_sparse_attn_hyd.py` 是论文主版本，对应 hybrid approximation：精确块 + 零阶块近似 + 全局一阶修正。它只实现前向推理。
2. `piecewise_sparse_attn_0th.py` 是更轻的零阶版本，保留 forward/backward，适合训练或微调场景，但没有论文主方法的全局一阶矩阵修正。
3. `flux_processor.py` 把 sparse attention 接到 Diffusers FLUX attention processor：完成 QKV 投影、RoPE、图像 token 重排、按层选择 sparse/dense backend。

主调用链是：

`FluxAttnProcessor.__call__` -> `piecewise_sparse_attention` -> `fused_chunk_reduce` -> `torch.topk` 路由 -> `piecewise_sparse_attention_fwd_kernel`。

源码交叉引用：

- hyd 入口与路由：[piecewise_sparse_attn_hyd.py#L303-L347](https://github.com/xie-lab-ml/piecewise-sparse-attention/blob/655648df86cafd75042dc32d6b6be78c4ea0eca8/piecewise_attn/kernels/piecewise_sparse_attn_hyd.py#L303-L347)
- hyd 预扫描 kernel：[piecewise_sparse_attn_hyd.py#L96-L183](https://github.com/xie-lab-ml/piecewise-sparse-attention/blob/655648df86cafd75042dc32d6b6be78c4ea0eca8/piecewise_attn/kernels/piecewise_sparse_attn_hyd.py#L96-L183)
- hyd 前向 Triton kernel：[piecewise_sparse_attn_hyd.py#L196-L299](https://github.com/xie-lab-ml/piecewise-sparse-attention/blob/655648df86cafd75042dc32d6b6be78c4ea0eca8/piecewise_attn/kernels/piecewise_sparse_attn_hyd.py#L196-L299)
- 0th forward/backward 版本：[piecewise_sparse_attn_0th.py#L106-L263](https://github.com/xie-lab-ml/piecewise-sparse-attention/blob/655648df86cafd75042dc32d6b6be78c4ea0eca8/piecewise_attn/kernels/piecewise_sparse_attn_0th.py#L106-L263)
- FLUX attention processor：[flux_processor.py#L37-L169](https://github.com/xie-lab-ml/piecewise-sparse-attention/blob/655648df86cafd75042dc32d6b6be78c4ea0eca8/piecewise_attn/models/flux/flux_processor.py#L37-L169)

## 数据布局与块统计

输入张量按 `[B, H, T, D]` 组织。PISA 先把序列切成长度 `BT=block_size` 的 token block，默认论文和代码都使用 64。如果 `T` 不能整除 64，最后一块通过 `BLOCK_SIZE = min(BT, T - i_t * BT)` 或 `current_lens` 单独处理。

`fused_chunk_reduce` 一次预扫描生成四类统计量：

- `qc`: 每个 query block 的 Q 均值，用于块级路由；
- `kc`: 每个 KV block 的 K 均值，用于零阶近似；
- `vc`: 每个 KV block 的 V 求和，用于把 `B * exp(q kbar)` 变成一次 `prob @ vc`；
- `h`: 全局一阶矩阵，代码里先产生每块 `hc[j] = (K_j - kbar_j)^T V_j`，再在 Python 侧 `hc.sum(dim=2)` 得到每个 batch-head 的全局 `H`。

预扫描 kernel 的 Triton grid 是：

```python
grid = (ceil(K / BK) * ceil(V / BV), N, B * H)
```

`program_id(0)` 同时编码 K tile 和 V tile，`program_id(1)` 是 token block，`program_id(2)` 是 batch-head。这个布局的好处是：

- `qc/kc/vc/hc` 的每个 tile 独立写出，没有跨 CTA 同步；
- `BK/BV` 上限为 128，匹配常见 head_dim 64/128，能把 `b_k - b_kc` 与 `b_v` 的小矩阵乘放进单个 Triton program；
- `hc` 只在预扫描中临时按 block 保存，真正前向 kernel 只读全局 `h`，避免每个 query block 对所有未选 KV block 反复读取 `d x d` 矩阵。

## 路由与近似选择

主入口先用块均值打分：

```python
score = einsum(qc, kc * scale)
top_k = max(1, int(density * NT))
indices = topk(score, k=top_k, dim=-1).indices
```

当 `use_bias=True` 时，代码会把 `hc` 的范数作为 `bias` 加入 `score + log(bias + eps)` 后 softmax，再取 top-k。这对应论文的 covariance-aware selection：不仅选择注意力均值大的块，也优先选择全局一阶近似误差可能更大的块。FLUX 接入默认调用 `use_bias=True`。

`sink_idx` 可以强制首块或尾块进入精确路径，适合保留文本/特殊 token 或近邻块的稳定性。当前公开实现没有逐 step 动态 warmup 调度；论文视频实验中的 early layer / early step dense 策略需要在外层推理框架接入。

## 前向 kernel 的 CTA / program 布局

`piecewise_sparse_attention_fwd_kernel` 的 grid 是：

```python
grid = (ceil(V / BV), NT, B * H)
```

一个 Triton program 负责一个 `(batch-head, query block, value tile)`。对应变量：

- `i_v = program_id(0)`: value/head_dim tile；
- `i_t = program_id(1)`: query token block；
- `i_bh = program_id(2)`: batch 和 head 合并维。

这种划分本质上类似 FlashAttention 的“一个 CTA 处理一块 query 行”，但把 KV 维拆成两条路径：少量精确 KV block 做 dense tile attention，大量非关键 KV block 只扫块级统计量。`b_q` 在 program 开始加载一次并驻留，`acc/l_i/m_i` 在寄存器中维护 online softmax，所有 exact 和 approximate 项共享同一套归一化状态。

### Phase 1：选中块精确注意力

代码对 `NS=top_k` 个 block 做固定次数循环。每轮：

1. 从 `indices[i_bh, i_t, i]` 取 KV block id；
2. 加载 `K_j [BT, BK]` 和 `V_j [BT, BV]`；
3. 计算 `Q_i K_j^T`；
4. 用 online softmax 更新 `m_i/l_i/acc`。

这里的 exact path 仍然是标准 64x64 tile attention，计算密度高，`Q` 复用充分。因为每个 query block 都有相同的 `NS`，所以 exact 部分循环次数一致，不会出现某些 CTA 只算 1 个块、某些 CTA 算几十个块的严重负载不均。

### Phase 2：未选块零阶近似扫描

未选块不读原始 token 级 K/V，而是按 `GROUP_SIZE in {32,64,128}` 扫描 `kc/vc`：

```python
for start_n in range(0, NT, GROUP_SIZE):
    b_kc = load(kc[start_n : start_n + GROUP_SIZE])
    b_s_mean = dot(b_q, b_kc.T)
    mask selected blocks
    b_vc = load(vc[start_n : start_n + GROUP_SIZE])
    acc += exp(score) @ b_vc
```

关键点是 `vc` 已经是块内 value sum，因此零阶分子项不需要遍历 block 内 64 个 token。分母项通过 `current_lens` 乘回块长度，修正最后一个短块。选中块通过 `loaded_indices` 在 GROUP 内做 mask，保证 exact 和 approximate 不重复计入。

从 GPU 利用率看，Phase 2 虽然扫描所有 block centroid，但每轮是规则的 `[BT, GROUP_SIZE] x [GROUP_SIZE, BV]` 小 GEMM，访问 `kc/vc` 连续，适合 coalesced load；它牺牲少量块级扫描开销，换掉了 token 级 `O(T^2)` 的 K/V 读写和 dot。

### Phase 3：全局一阶修正

论文中逐块一阶项需要读取每个未选块的 `H_j in R^{D x D}`，这会变成典型 memory-bound：每个 query block 都要流式扫大量 `D x D` 矩阵，算术强度低。

代码用全局 `h=sum_j H_j` 替代逐块 `H_j`：

```python
b_h = load(h[i_bh, :, i_v])
b_r = dot(b_q, b_h)
correction_scale = g_l * (1 / T) * scale
acc += b_r * correction_scale
```

`g_l` 是 Phase 2 累计的 tail probability mass。也就是说，一阶修正只需要每个 `(batch-head, value tile)` 加载一次全局矩阵 tile，而不是对每个未选 KV block 加载一次 `H_j`。这是 PISA kernel 能保持高吞吐的核心：理论上保留一阶信息，工程上把最贵的随机/流式矩阵访问压缩成一次固定读。

### 归一化一致性

PISA 没有把近似项作为后处理直接加到 output，而是把 exact、zero-order、first-order 都纳入同一个 online softmax 状态。Phase 1/2 共同更新 `m_i/l_i/acc`，Phase 3 更新分子后，最终执行：

```python
l_i += g_l
acc /= l_i[:, None]
```

因此输出仍然近似 softmax attention 的归一化结果，而不是“精确 sparse attention + 一个未归一化残差”。

## 负载均衡与 GPU 利用率

PISA 的负载均衡设计可以概括为“固定大循环 + 有界小分支”：

- exact path 的循环次数固定为 `NS=top_k`，由 density 决定；
- approximate path 对所有 query block 都扫描 `ceil(NT / GROUP_SIZE)` 个 centroid group；
- 每个 program 只处理一个 query block 和一个 value tile，`B*H*NT*ceil(V/BV)` 提供足够并行度；
- 对 head_dim 大于 `BV` 的场景，`i_v` 维继续拆分，避免一个 CTA 独占过多 value 维寄存器；
- 用 Triton TensorDescriptor/TMA allocator 在 Hopper 上优化规则块加载，非 Hopper 会有 warning，性能预期下降。

这和很多稀疏注意力 kernel 的风险点不同：如果每行非零块数高度不均，CTA 会因为不同 row 的 sparse list 长度不同而尾部拖慢；PISA 的每个 query block 都做相同数量的 exact top-k 和相同数量的 centroid group scan，因此调度更规整。真正的动态性主要体现在 `indices` 对应的 K/V block 地址不同，影响缓存命中，但不会大幅改变 CTA 工作量。

## 0th 版本与 backward

`piecewise_sparse_attn_0th.py` 提供 forward/backward，但方法上只做 exact + zero-order，没有 hyd 的全局一阶注入。

它的路由额外计算 `k_var = E[||k||^2] - ||E[k]||^2`，用 `mean_logits + log(k_var)` 作为近似误差 proxy。这个实现比论文主版本的 `||H_j - Hbar||` 更便宜，因为只需要每个 KV block 一个标量方差，而不是 `D x D` 一阶矩阵。

backward 分成三类 kernel：

- `bwd_dq`: 对 exact block 和 approximate centroid 都回传到 Q；
- `bwd_approx_dkdv`: 处理零阶近似路径对 centroid/sum 的梯度；
- `bwd_exact_dkdv`: 对 top-k exact block 写回 token 级 dK/dV，并把近似路径的 centroid 梯度分摊回 token。

这个版本更像“可训练稀疏近似 attention”的工程原型；论文主推的视频/图像推理加速则依赖 hyd forward。

## FLUX 接入细节

`FluxAttnProcessor` 做了两件和 sparse attention 强相关的事：

1. 对 `T=4096` 的图像 token，按 `p1=p2=sqrt(block_size)` 重排，把空间邻近 patch 放进同一个连续 token block。这样 64-token block 更接近二维局部区域，top-k 路由更稳定。
2. 通过 `processors_id >= start_layer_idx` 决定从哪一层开始替换为 PISA，前面层仍走 PyTorch SDPA。这样能保留早期层的全局建模稳定性，类似论文视频实验中的 warmup 思路。

需要注意当前代码里 `transpose(1, 2).contiguous()` 后注释写成 `B H D T`，实际张量是 `[B, H, T, D]`；kernel 入口也按 `[B,H,T,D]` 解释。

## 工程限制

- hyd kernel 只有 forward，没有 backward；
- 公开仓库没有 Wan2.1/Hunyuan-Video pipeline patch，视频实验复现需要自行替换对应 DiT attention processor；
- covariance-aware selection 在代码里通过 `hc` 范数作为 `bias`，并非完整暴露论文中的所有 warmup / layer schedule；
- 最佳性能依赖 Hopper + 新版 Triton；README 要求 `torch >= 2.10`、`triton >= 3.6`，这对普通环境较激进。
