# MInference 代码实现：Dynamic Sparse Attention Prefill

对应论文：[MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention](../../paper_reading/minference.md)

官方代码位于 `refs/codes/MInference`，当前解析提交：
`a4eb395f949ea39e871f9bc586d683390692c6be`。MInference 是长上下文 LLM prefill 加速方法，不是视频生成 DiT 方法；它在本仓库中作为 training-free sparse attention 的模式发现、标定流程和稀疏 kernel 工程参考。

## 先明确它解决的工程问题

LLM 长上下文 prefill 的瓶颈是一次性计算整段 prompt 的 causal attention：

$$
O = \operatorname{Softmax}\left(\frac{QK^T}{\sqrt{d}} + M_{causal}\right)V
$$

如果上下文长度是 `T`，计算和中间 attention 访问都随 `T^2` 增长。MInference 的目标不是训练一个新模型，也不是写一个固定稀疏 mask，而是：

1. 离线判断每个 layer/head 更适合哪种稀疏结构；
2. 推理时根据当前输入动态生成具体 token index；
3. 把这些 index 转成 CUDA/Triton kernel 能高效消费的 metadata；
4. 用稀疏 attention kernel 只计算被保留的位置。

所以它的核心工程链路是：

```text
offline search
  -> best_pattern JSON
    -> patch HuggingFace attention
      -> prefill per-layer/per-head dispatch
        -> dynamic index construction
          -> metadata conversion
            -> sparse attention kernel
```

这条链路缺一环都不完整：只有 pattern 没有 index 不可执行，只有 index 没有 kernel 不会快，只有 kernel 没有标定会掉质量。

## 实现分层

MInference 的代码可以分成五层，而不是简单理解成一个 attention 函数：

| 层级 | 代表代码 | 作用 |
|---|---|---|
| 配置层 | `MInferenceConfig` | 解析 `attn_type/kv_type/config_path/is_search/starting_layer` |
| Patch 层 | `new_patch`, `attn_forward` | 替换模型 attention forward，并把 prefill/decode 分流 |
| 标定层 | `search_pattern`, `search_pattern_v2` | 为每个 layer/head 选 pattern 和预算 |
| 索引层 | `vertical_and_slash_kernel`, `_build_block_index` | 对当前输入生成 sparse token/block index |
| Kernel 层 | CUDA converter, Triton fallback, SGLang/vLLM kernel | 把 index 转成 GPU metadata 并执行 sparse attention |

源码交叉引用集中放在附录 A，正文按数据流解释。

## 配置与 patch：把算法接进模型

`MInferenceConfig` 做两件事：

1. 决定 attention 类型：`minference`、`dense`、`a_shape`、`tri_shape`、`flexprefill` 等。
2. 决定 KV cache/decode 类型：`dense`、`quest`、`snapkv`、`kivi` 等。

这说明 MInference 工程上把 **prefill sparse attention** 和 **decode KV cache 策略** 分开。论文主线是 prefill；decode 只是可以组合其他方法。

`new_patch` 从模型中拿到 attention module 类型和 decoder layer 类型，然后把 attention forward 替换为 `attn_forward` 的 partial：

```text
model attention.forward
  -> attn_forward(
       prefill_forward=prefill_forwards[attn_type],
       decoding_forward=decoding_forwards[kv_type],
       attn_forward_config=...
     )
```

`attn_forward` 内部仍然先做标准 transformer attention 前处理：

1. Q/K/V projection；
2. RoPE；
3. KV cache update；
4. GQA/MQA 的 `repeat_kv`；
5. 判断当前是 prefill 还是 decode。

判断逻辑是：

```python
if not use_cache or q_len == past_key_value.get_seq_length(layer_idx):
    prefill_forward(...)
else:
    decoding_forward(...)
```

这保证 MInference 不改模型权重，也不改 attention 前后的 hidden state contract。它只替换 prefill 阶段的 attention kernel。

## best pattern：离线标定输出到底是什么

MInference 的离线搜索结果是一个 per-layer/per-head 的 JSON。每个 head 的条目近似为：

```json
["vertical_and_slash", 100, 750, 0.98]
```

四个字段分别表示：

1. pattern 类型：`stream_llm`、`vertical_and_slash` 或 `block_sparse`；
2. vertical/global 预算；
3. slash/local/block 预算；
4. 搜索阶段得到的质量分数或误差分数。

线上执行时，`minference_prefill_kernel` 读取：

```text
config["best_pattern"][layer_idx][head_id]
```

然后选择对应 kernel。也就是说，离线标定只决定“这个 head 用什么稀疏形状、保留多少条线或多少块”；它不保存具体 token id。具体 token id 必须在当前 prompt 上重新算。

## 标定流程：为什么叫 kernel-aware

代码里有两套搜索：

- `search_pattern`: 用 full attention 权重召回作为 proxy；
- `search_pattern_v2`: 用 dense FlashAttention output 作 reference，比对 sparse output 的误差。

候选空间不是任意 mask，而是 kernel 能执行得好的形状：

```text
stream_llm:        (n_init, n_local)
vertical_slash:    (vertical_size, slash_size)
block_sparse:      top_k_blocks
```

这就是 kernel-aware 的含义。一个数学上稀疏率很低的 mask，如果随机分布在注意力矩阵上，kernel 只能做大量 gather，未必比 dense 快。MInference 只搜索 A-shape、Vertical-Slash、Block-Sparse 这类可被压缩成连续范围或规则 block 的结构。

标定命令在 `experiments/infinite_bench` 下运行，典型流程是：

```text
load model
enable is_search
for selected calibration examples:
    for layer:
        for head:
            run candidate sparse patterns
            compare recall/output error
            write best pattern to JSON
```

标定不是训练：权重不变，没有梯度更新。但它依赖模型、任务样本、上下文长度和 kernel 实现。换模型或换部署 kernel 后，best pattern 未必仍然最优。

## Prefill 执行：逐 head dispatch 的意义和代价

`minference_prefill_forward` 的输入是标准 attention 中间态：

```text
query_states: [B, H, T, D]
key_states:   [B, H, T, D]
value_states: [B, H, T, D]
```

它逐 head 循环：

```text
for head in heads:
    q = query_states[:, head:head+1]
    k = key_states[:, head:head+1]
    v = value_states[:, head:head+1]
    output[:, head] = minference_prefill_kernel(q,k,v,head,layer)
```

这个设计的好处是每个 head 可以使用不同 pattern。代价是 Python 层循环和很多小 kernel launch，在短上下文下会吃掉收益。README 的 latency 表中 1K/10K context 下 MInference 不一定比 FlashAttention 快，原因就在这里：稀疏计算省下的 `T^2` 还不够抵消 index building 和 dispatch overhead。

在长上下文下，attention 主计算占比迅速上升，逐 head overhead 被摊薄，MInference 才进入优势区间。

## Vertical-Slash：从注意力模式到 token index

Vertical-Slash 是 MInference 最重要的 pattern。它把 attention map 拆成两类结构：

- **vertical line**：很多 query 共同关注的 key column，常对应开头 token、分隔符、检索事实；
- **slash line**：沿对角线或固定相对位移的局部依赖，常对应近邻上下文。

在线构造 index 时，并不计算完整 `T x T` attention。代码只取最后 `last_q=min(64,T)` 个 query：

```text
Q_last = Q[:, :, -last_q:, :]
score = Q_last @ K^T / sqrt(D)
score = causal_mask(score)
prob = softmax(score)
```

然后：

```text
vertical_score[j] = sum_i prob[i, j]
slash_score[delta] = sum_i prob[i, i - delta]
```

`vertical_topk` 选出全局 key column，`slash` 选出若干相对位移。只用最后 64 个 query 是一个非常实际的折中：

- index 构造成本是 `O(64*T*D)`，不是 `O(T^2*D)`；
- 最后的 query 往往最能反映当前 prompt 尾部对历史信息的需求；
- 对 causal LLM prefill，后续更长位置的注意力结构通常和尾部 query 更接近。

这个假设不是数学恒真，所以 MInference 需要离线 search 和 downstream 验证来兜底。

## Index converter：为什么还要 CUDA 预处理

Vertical-Slash 的 `vertical_topk` 和 `slash` 还不是 sparse kernel 最想要的格式。主 kernel 不应该一边计算 attention 一边解释“这条 slash line 对当前 row block 覆盖哪些 KV token”，否则会把整数逻辑塞进热路径。

因此 `convert_vertical_slash_indexes` 先把 line pattern 转成四个 metadata 张量：

| metadata | 形状 | 含义 |
|---|---:|---|
| `block_count` | `[B,H,N_ROWS]` | 每个 query row block 有多少段连续 KV block |
| `block_offset` | `[B,H,N_ROWS,NNZ_S]` | 每段连续 slash block 的起始 token/block offset |
| `column_count` | `[B,H,N_ROWS]` | 每个 query row block 有多少个 vertical column |
| `column_index` | `[B,H,N_ROWS,NNZ_V]` | 需要 gather 的 vertical key column |

CUDA converter 的 grid：

```cpp
dimBlock(64)
dimGrid(N_HEADS, BATCH_SIZE, ceil(N_ROWS / 64))
```

一个 CUDA thread 处理一个 64-row query block。它做三件事：

1. 根据当前 row block 的 `[start_m,end_m)` 把 slash offset 转成 causal 范围；
2. 将相邻或重叠的 slash 范围合并成连续 64-token block；
3. 扫描 vertical index，如果某个 vertical column 已经落入 slash range，则不写入 `column_index`，避免重复计算。

这个预处理把复杂 pattern 解释从 attention 热路径中移出去。它本身主要是整数逻辑，计算量受 `NNZ_S + NNZ_V` 限制；真正决定吞吐的是后续 sparse attention 中每个 row block 的非零 block/column 数。

## Mixed sparse Triton kernel：主计算怎么跑

如果安装了 SGLang 或 vLLM，代码优先调用它们的 `sparse_attn_func`。没有这些依赖时，走本地 Triton fallback `_triton_mixed_sparse_attn_fwd_kernel`。

fallback 的 grid 是：

```python
grid = (ceil(N_CTX / BLOCK_M), B * H, 1)
```

一个 Triton program 负责一个 `(batch-head, query row block)`。默认 `BLOCK_M=64`、`BLOCK_N=64`，内部状态和 FlashAttention 类似：

```text
q     = load Q row block
m_i   = -inf
l_i   = 0
acc   = 0
```

然后按两类稀疏项顺序累积：

### 1. Slash ranges：连续 block 路径

```text
for each block_offset:
    cols = start_n + [0..BLOCK_N)
    K,V = contiguous load
    qk = q @ K
    apply causal mask
    online_softmax_update
```

这是最接近 FlashAttention 的路径：K/V 访问连续，tile 规则，coalescing 好。Slash line 被 converter 转成 block range，就是为了让这部分尽量像 block sparse attention 而不是离散 gather。

### 2. Vertical columns：离散 gather 路径

```text
for columns in column_index chunks:
    cols = gather column ids
    K,V = gather load
    qk = q @ K
    online_softmax_update
```

Vertical columns 表达全局重要 token，但访存更离散。MInference 用预算限制 `NNZ_V`，并把已经被 slash range 覆盖的 vertical column 去掉，降低 gather 成本。

### online softmax 保证语义正确

两类稀疏项不是分别 softmax 后相加。kernel 对 slash 和 vertical 共享 `m_i/l_i/acc`，因此最终输出是：

$$
\tilde{O}_i =
\operatorname{Softmax}
\left(
Q_i K_{\Omega_i}^{T}
\right)V_{\Omega_i}
$$

其中 `\Omega_i` 是 slash ranges 与 vertical columns 的并集。这保持了 sparse attention 的 softmax 语义。

## Block-Sparse 路径：规则但表达力有限

Block-Sparse 路径先构造 block index：

```text
Q_pool = mean(Q over 64-token blocks)
K_pool = mean(K over 64-token blocks)
score_block = Q_pool @ K_pool^T
block_index = topk(score_block)
```

主 Triton kernel 同样是一个 program 处理一个 query block。它遍历 `block_index[start_m]` 中的 KV block，每个非零块都是完整 64x64 tile。

这条路径的硬件形态很好：规则 block、连续 K/V、少 gather。但论文消融中 only block-sparse 平均分只有 18.7，说明 block mean pooling 容易把细粒度检索信号抹掉。换句话说，Block-Sparse 的瓶颈不是 CUDA 设计，而是 pattern 表达能力。

## 负载均衡与 GPU 利用率

MInference 的 GPU 利用率取决于四个层面。

### 1. 搜索空间约束

只允许 kernel-friendly pattern，避免随机稀疏 mask。A-shape 是固定连续窗口，Vertical-Slash 可拆成连续 range + 少量 columns，Block-Sparse 是 64x64 tile。

### 2. metadata 预处理

CUDA converter 预先合并 slash range、去重 vertical column，让主 attention kernel 只看 `block_offset/column_index`，不在热路径解释复杂稀疏结构。

### 3. 每个 program 的工作量上界

预算来自 best pattern，例如 vertical/slash 数量固定。这样每个 head 的最大非零数有界，不会无限制增长。

### 4. 仍然存在的负载不均

Vertical-Slash fallback 里，每个 row block 的 `num_blks/num_cols` 仍可能不同。某些 row/head 需要更多 slash range 或 vertical columns，就会让对应 program 跑更久。SGLang/vLLM 的优化 sparse kernel 通常比本地 fallback 更能处理这些调度问题，所以 README 推荐安装它们，且 latency 表里 `MInference w/ SGLang` 明显更快。

## 从实现解释实验结果

MInference 的实验现象和实现链路是对应的：

- **短上下文不占优**：1K/10K 下 index building、逐 head dispatch、kernel launch overhead 比节省的 attention 计算更显著。
- **长上下文加速扩大**：当 `T` 到 100K、300K、1M，dense attention 的 `T^2` 成本成为主项，稀疏计算收益迅速扩大。
- **SGLang 版本更快**：同样的 sparse metadata，优化 kernel 的调度和访存比本地 Triton fallback 更好。
- **only vertical-slash 接近完整方法但仍落后**：Vertical-Slash 捕获了大多数长上下文结构，但有些 head 更适合 A-shape 或 Block-Sparse；per-head 混合是质量来源。
- **only block-sparse 掉分严重**：block pooling 虽硬件友好，但对 KV retrieval 等细粒度任务表达不足。

这也是 MInference 对后续视频/DiT sparse attention 的启示：模式设计必须同时满足“能表达模型真实 attention 结构”和“能被 GPU kernel 高效执行”。

## 工程限制

- MInference 主线是 LLM prefill，不处理视频 DiT 的空间/时间 token 拓扑。
- best pattern 依赖模型和校准数据，跨模型迁移需要重新验证。
- 逐 head Python dispatch 简单但有 overhead，短上下文不划算。
- fallback Triton kernel 对不均匀 sparse list 的负载均衡有限，部署应优先使用 SGLang/vLLM sparse kernel。
- Vertical-Slash 的在线估计只看最后 64 个 query，极端输入中可能漏掉早期 query 特有的全局依赖。

## 附录 A：源码交叉引用

- 配置入口：[minference_configuration.py#L7-L110](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/minference/minference_configuration.py#L7-L110)
- patch 到通用 attention forward：[patch.py#L850-L892](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/minference/patch.py#L850-L892)
- prefill/decoding 分流：[forward.py#L136-L168](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/minference/modules/forward.py#L136-L168)
- prefill registry：[forward.py#L237-L246](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/minference/modules/forward.py#L237-L246)
- per-head search 和在线执行：[minference_forward.py#L129-L276](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/minference/modules/minference_forward.py#L129-L276)、[minference_forward.py#L594-L674](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/minference/modules/minference_forward.py#L594-L674)
- Vertical-Slash sparse attention：[pit_sparse_flash_attention_v2.py#L195-L273](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/minference/ops/pit_sparse_flash_attention_v2.py#L195-L273)
- CUDA index converter：[vertical_slash_index.cu#L27-L167](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/csrc/vertical_slash_index.cu#L27-L167)
- Block-Sparse Triton kernel：[block_sparse_flash_attention.py#L29-L186](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/minference/ops/block_sparse_flash_attention.py#L29-L186)

## 附录 B：Vertical-Slash 执行伪代码

```text
for layer:
    for head:
        ty, vertical_size, slash_size = best_pattern[layer][head]

        if ty == "vertical_and_slash":
            Q_last = Q[-64:]
            P = softmax(Q_last @ K.T / sqrt(D), causal=True)
            vertical_topk = topk(sum_over_queries(P), vertical_size)
            slash_topk = topk(sum_over_diagonals(P), slash_size)

            block_count, block_offset, column_count, column_index =
                convert_vertical_slash_indexes(vertical_topk, slash_topk)

            O_head = sparse_attention(
                Q, K, V,
                block_count, block_offset,
                column_count, column_index
            )
```

## 附录 C：Mixed sparse kernel 伪代码

```text
program(row_block, batch_head):
    q = load Q[row_block]
    m = -inf
    l = 0
    acc = 0

    for each slash block range:
        cols = contiguous 64-token block
        k, v = load contiguous K/V
        qk = q @ k
        qk = apply causal mask
        m, l, acc = online_softmax_update(m, l, acc, qk, v)

    for each vertical column chunk:
        cols = gather column_index
        k, v = gather K/V
        qk = q @ k
        m, l, acc = online_softmax_update(m, l, acc, qk, v)

    store acc / l
```
