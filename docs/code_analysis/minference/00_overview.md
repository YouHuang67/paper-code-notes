# MInference 代码实现：Dynamic Sparse Attention Prefill

对应论文：[MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention](../../paper_reading/minference.md)

官方代码位于 `refs/codes/MInference`，当前解析提交：
`a4eb395f949ea39e871f9bc586d683390692c6be`。MInference 是长上下文 LLM prefill 加速方法，不是视频生成 DiT 方法；它在本仓库中适合作为 training-free sparse attention 的模式发现、标定流程和稀疏 kernel 参考。

## 实现结构

MInference 的工程结构可以分成四层：

1. 配置层：`MInferenceConfig` 解析 `attn_type`、`kv_type`、`config_path`、`is_search`、`starting_layer`，并把模型名映射到预先搜索好的 best pattern JSON。
2. Patch 层：`MInference(...)` / `new_patch(...)` 替换 HuggingFace LLaMA/GLM attention forward，把 prefill 和 decoding 分流。
3. Pattern 层：`minference_prefill_forward` 对每个 layer/head 读取或搜索最佳稀疏模式，支持 `stream_llm`、`vertical_and_slash`、`block_sparse`。
4. Kernel 层：`vertical_slash_sparse_attention` 和 `block_sparse_attention` 把动态索引转成 GPU-friendly 稀疏格式，再调用 SGLang/vLLM sparse kernel 或本地 Triton fallback。

源码交叉引用：

- 配置入口：[minference_configuration.py#L7-L110](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/minference/minference_configuration.py#L7-L110)
- patch 到通用 attention forward：[patch.py#L850-L892](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/minference/patch.py#L850-L892)
- prefill/decoding 分流：[forward.py#L136-L168](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/minference/modules/forward.py#L136-L168)
- prefill registry：[forward.py#L237-L246](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/minference/modules/forward.py#L237-L246)
- per-head search 和在线执行：[minference_forward.py#L129-L276](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/minference/modules/minference_forward.py#L129-L276)、[minference_forward.py#L594-L674](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/minference/modules/minference_forward.py#L594-L674)
- Vertical-Slash sparse attention：[pit_sparse_flash_attention_v2.py#L195-L273](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/minference/ops/pit_sparse_flash_attention_v2.py#L195-L273)
- CUDA index converter：[vertical_slash_index.cu#L27-L167](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/csrc/vertical_slash_index.cu#L27-L167)
- Block-Sparse Triton kernel：[block_sparse_flash_attention.py#L29-L186](https://github.com/microsoft/MInference/blob/a4eb395f949ea39e871f9bc586d683390692c6be/minference/ops/block_sparse_flash_attention.py#L29-L186)

## Patch 与 prefill 执行链路

`new_patch` 从模型中取 `Attention`、`DecoderLayer` 类型，把每个 attention module 的 `forward` 替换成 `attn_forward` 的 partial。`prefill_forward` 由 `prefill_forwards[config.attn_type]` 决定；当 `attn_type="minference"` 时，就是 `minference_prefill_forward`。

`attn_forward` 内部先做标准 Q/K/V 投影、RoPE、KV cache 更新和 GQA repeat。随后用一个条件判断 prefill 或 decode：

```python
if not use_cache or q_len == past_key_value.get_seq_length(layer_idx):
    prefill_forward(...)
else:
    decoding_forward(...)
```

因此 MInference 的核心加速目标是 prefill，即长 prompt 一次性进入模型时的 `QK^T` 计算。decode 阶段可以叠加 SnapKV、Quest、RetrAttn、KIVI 等 KV cache 方法，但这不是 MInference 论文主线。

## 离线 kernel-aware pattern search

MInference 的标定不是训练权重，而是为每个 layer/head 选择最适合的稀疏模式和预算。代码有两套搜索函数：

- `search_pattern`: 用注意力权重召回作为 proxy；
- `search_pattern_v2`: 直接比较 sparse output 与 dense FlashAttention output 的差异。

候选模式包括：

- A-shape / StreamingLLM：固定初始全局 token + 局部窗口；
- Vertical-Slash：保留若干 vertical lines 和 diagonal/slash lines；
- Block-Sparse：按 64 或 32 token block 做 top-k。

论文中的 kernel-aware search space 不是单纯指定“稀疏率”，而是指定真实 kernel 能高效执行的形状。例如 A-shape 是 `(1024,4096)`，Vertical-Slash 是若干 `(vertical_count, slash_count)`，Block-Sparse 是 top-100 block。代码当前候选略有版本差异，例如 `search_pattern_v2` 中 Vertical-Slash 包含 `(30,800),(100,800),(100,750),(500,700),(3500,100),(1000,4096)`。

搜索结果写成 JSON，结构近似：

```json
[
  {
    "0": ["vertical_and_slash", 100, 750, 0.98],
    "1": ["block_sparse", 10, 1, 0.95]
  }
]
```

线上执行时，`minference_prefill_kernel` 用 `config["best_pattern"][layer_idx][head_id]` 取出模式和预算；如果没有找到，则默认回退到 `("vertical_and_slash", 1000, 6096, 1)`。

## Vertical-Slash 在线索引构造

Vertical-Slash 是 MInference 最重要的模式。它的假设是长上下文 LLM attention 里常见两类结构：

- vertical line：很多 query 都关注的全局 key，例如文档开头、特殊分隔符、重要事实 token；
- slash line：沿对角线的局部/近邻依赖，表示 query 主要看前面固定偏移范围。

在线执行并不直接复用离线 token 索引，而是只复用“这个 head 适合 Vertical-Slash，以及保留多少条 line”。实际索引由当前输入动态生成：

1. 取最后 `last_q=min(64,q_len)` 个 query；
2. 计算 `Q_last K^T / sqrt(d)`；
3. 对最后 64 个 query 内部加 causal mask；
4. softmax 后沿 query 维求和得到 vertical 重要性；
5. 用对角线求和 `sum_all_diagonal_matrix` 得到 slash/diagonal 重要性；
6. top-k 得到 `vertical_topk` 和 `slash`，传给 sparse attention kernel。

这一步的计算量是 `O(64 * T * D)`，相对完整 prefill 的 `O(T^2 * D)` 很小；但在短上下文下 overhead 会明显，所以 README 的 latency 表里 1K/10K 时 MInference 不一定比 FlashAttention-2 快。

## CUDA index converter：把 line pattern 变成 kernel 可执行格式

`vertical_slash_sparse_attention` 先对 `v_idx` 升序、`s_idx` 降序排序，然后调用 `convert_vertical_slash_indexes`。本地 CUDA converter 输出四个张量：

- `block_count[B,H,N_ROWS]`: 每个 query row block 有多少个连续 slash block；
- `block_offset[B,H,N_ROWS,NNZ_S]`: slash range 拆成 64-token KV block 后的起点；
- `column_count[B,H,N_ROWS]`: 每个 query row block 有多少个 vertical column；
- `column_index[B,H,N_ROWS,NNZ_V]`: 不在 slash range 内的 vertical token id。

CUDA grid 是：

```cpp
dimBlock(64)
dimGrid(N_HEADS, BATCH_SIZE, ceil(N_ROWS / 64))
```

一个 CUDA thread 负责一个 64-row query block。它把 diagonal/slash 范围转成连续 block range，同时扫描 vertical indices；如果 vertical column 已经落在 slash range 内，就不写入 `column_index`，避免后续 sparse attention 重复计算。

这个 converter 的计算很轻，主要是整数索引整理。负载均衡上它不是重计算 kernel：每个 thread 的循环次数受 `NNZ_S + NNZ_V` 上界约束，且候选预算由离线 search 固定。真正影响吞吐的是后面的 sparse attention 对每个 row block 的非零 block/column 数是否均衡。

## Mixed sparse Triton kernel

如果安装了 SGLang 或 vLLM，代码优先调用它们的 `sparse_attn_func`；否则使用本地 Triton fallback `_triton_mixed_sparse_attn_fwd_kernel`。

fallback grid：

```python
grid = (ceil(N_CTX / BLOCK_M), B * H, 1)
```

一个 Triton program 处理一个 `(batch-head, query row block)`，默认 `BLOCK_M=64`、`BLOCK_N=64`。内部保持 `q`、`m_i`、`l_i`、`acc`，用 online softmax 顺序累积两类稀疏项：

1. slash ranges：`for block_index in range(num_blks)`，从 `block_offset` 取连续 64-token KV block，K/V load 连续、coalescing 好，并应用 causal mask；
2. vertical columns：`for start_n in range(0, num_cols, BLOCK_N)`，从 `column_index` gather 任意 key column，适合表达全局关键 token，但内存访问比 slash range 更离散。

这种“连续 range + 离散 column”的格式是 Vertical-Slash 的硬件核心。Slash line 被转成 64-token block range 后，可以像 block sparse FlashAttention 一样做规则 tile GEMM；vertical line 保留为列 gather，虽然访存不完全连续，但数量受 `vertical_size` 限制。

负载均衡上，Triton fallback 每个 row block 的 `num_blks/num_cols` 可变，尾部 CTA 可能因为某些 head/row 有更多 slash range 或 vertical columns 而拖慢。MInference 通过两点缓解：

- 离线 search 选择的是 kernel-aware 预算，不让稀疏形状只在数学稀疏率上漂亮却在 kernel 中碎片化；
- 优先使用 SGLang/vLLM 的稀疏内核，它们对 block metadata 和调度有更成熟的优化；README 也显示 `MInference w/ SGLang` 在长上下文下明显更快。

## Block-Sparse kernel

Block-Sparse 路径先在 Python 侧构造 block index：

1. 把 Q/K padding 到 64 的倍数；
2. 对每个 64-token block 做 mean pooling；
3. 计算 block-level `Q_pool K_pool^T`；
4. causal mask 后 top-k，得到每个 query block 要看的 KV block id。

Triton kernel 同样使用：

```python
grid = (ceil(q_len / 64), B * H, 1)
```

每个 program 对一个 query block 遍历 `block_index[start_m]`，每个非零项是完整 64x64 dense attention tile。由于 `block_count = min((start_m + 1) * 64 / 64, MAX_BLOCKS_PRE_ROW)`，早期行天然只能看更少历史块，后期行达到 `MAX_BLOCKS_PRE_ROW` 上限。相较 Vertical-Slash，Block-Sparse 访存更规则，但如果 top-k block 捕获不到长程检索 token，质量会掉得更明显；论文 ablation 里 only block-sparse 的 InfiniteBench 平均分远低于完整 MInference。

## GPU 利用率设计要点

MInference 的 GPU 利用率取决于三个工程选择：

1. 稀疏形状和 kernel 绑定：离线搜索的不是抽象 mask，而是 A-shape、Vertical-Slash、Block-Sparse 这些 kernel 能高效表示的模式。
2. 动态索引构造轻量化：只用最后 64 个 query 估计全局/对角线结构，避免在线构造 mask 本身接近 `T^2`。
3. 稀疏 metadata 分离：把 slash 转成 `block_offset`，把 vertical 转成 `column_index`，让主 attention kernel 不需要在循环里解释复杂 pattern。

剩余风险也很明确：如果某些 head 的 sparse list 长度差异很大，row-block program 的执行时间会不一致；如果 vertical columns 太多，gather 访存会降低带宽效率；如果上下文较短，index building 的 5%-20% overhead 和 per-head Python 循环会吃掉收益。

## 实验脚本与复现路径

仓库给出的离线搜索入口在 `experiments/infinite_bench/run_infinitebench.py`，典型参数包括：

```bash
python run_infinitebench.py \
  --task kv_retrieval \
  --model_name_or_path gradientai/Llama-3-8B-Instruct-262k \
  --max_seq_length 30000 \
  --is_search \
  --topk_dims_file_path Llama_3_8B_Instruct_262k_kv_out_v32_fit_o_best_pattern.json \
  --num_eval_examples 20 \
  --starting_layer 0 \
  --attn_type minference
```

端到端 latency 脚本是 `experiments/benchmarks/benchmark_e2e.py`，README 建议长于 700K token 时启用 `--kv_cache_cpu`。对于真实使用，应优先安装 SGLang 或 vLLM sparse kernel；本地 Triton fallback 更适合理解算法和做可用性兜底。
