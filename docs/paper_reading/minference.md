---
tags:
  - Sparse Attention
  - LLM Inference
  - CUDA
  - Triton
---

# MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention

[arXiv 2407.02490](https://arxiv.org/abs/2407.02490) | [代码解析](../code_analysis/minference/00_overview.md) | Microsoft Research

## 定位

MInference 是 **training-free long-context LLM prefill sparse attention** 方法，不是视频生成 DiT 方法。它仍值得放在本仓库的 sparse attention 线索下，因为后续很多视频稀疏注意力会借鉴它的模式：vertical line、slash/diagonal line、block sparse，以及“离线标定每个 head 的模式，在线按输入动态构造索引”的流程。

它解决的是长 prompt prefill 的 `O(T^2D)` attention 成本。论文报告在单 A100 上最高 10x prefill speedup，把 1M token prefill 从约 30 分钟降到约 3 分钟；结合 8x A100 tensor/context parallel 可到约 22 秒。

## 核心思路

MInference 观察到长上下文 LLM 的注意力图并不是任意稀疏，而是不同 head 呈现可归类的结构：

1. **A-shape**：开头全局 token + 近邻局部窗口；
2. **Vertical-Slash**：若干全局 key 列 + 若干对角线/斜线；
3. **Block-Sparse**：以 token block 为单位保留重要块。

方法分三步：

1. 离线搜索每个 layer/head 的最佳稀疏模式和预算；
2. 在线对当前输入动态构造 sparse index；
3. 用优化 sparse attention kernel 执行 prefill。

这和“固定 mask”不同：离线阶段只决定 head 类型和预算，具体 token index 仍由当前 prompt 的 Q/K 动态生成，因此能适应不同输入。

## 数学形式

对某个 head，dense attention 是：

$$
O = \operatorname{Softmax}\left(\frac{QK^\top}{\sqrt{d}} + M_{causal}\right)V
$$

MInference 构造一个稀疏 mask `A_h(x)`，它依赖当前输入 `x`，但模式族和预算由离线搜索确定：

$$
\tilde{O}_h =
\operatorname{Softmax}
\left(
\frac{Q_h K_h^\top}{\sqrt{d}} + M_{causal} + \log A_h(x)
\right)V_h
$$

其中 `A_h(x) in {0,1}^{T x T}` 只允许某些 key 被 query 访问。目标不是最小化抽象稀疏率，而是在真实 kernel 延迟约束下最大化 dense attention 的近似质量：

$$
\max_{p_h \in \mathcal{P}}
\operatorname{Quality}
\left(
\tilde{O}_h(p_h), O_h
\right)
\quad
\text{s.t.}
\quad
t_{index}(p_h) + t_{sparse}(p_h) \le \tau
$$

这里 `p_h` 是 head 的模式与预算，例如 `(vertical_and_slash, vertical_size=100, slash_size=750)`。论文强调 kernel-aware：候选空间要对应实际 kernel 能高效执行的 shape，而不是只看 mask 中 1 的比例。

## 三类模式

### A-shape

A-shape 等价于 StreamingLLM 风格：保留前 `n_init` 个 sink/global token，并保留最近 `n_local` 个 token：

$$
A_{i,j}=1
\quad \text{if} \quad
j < n_{init}
\ \text{or}\
i-n_{local} \le j \le i
$$

论文 search space 中 A-shape 使用 `(1024,4096)`。它访存和 kernel 形态规整，但只能表达“开头 + 局部”，对分散检索任务不够灵活。

### Vertical-Slash

Vertical-Slash 是 MInference 最重要的模式。给定最后 `r=64` 个 query，先计算：

$$
P = \operatorname{Softmax}
\left(
\frac{Q_{T-r:T}K^\top}{\sqrt{d}} + M_{causal}
\right)
$$

Vertical score 是最近 query 对每个 key column 的总注意力：

$$
s^{vert}_j = \sum_{i=T-r}^{T} P_{i,j}
$$

Slash score 是沿对角线/相对位移求和：

$$
s^{slash}_{\delta}
=
\sum_i P_{i, i-\delta}
$$

线上 top-k 后得到 `vertical_topk` 和 `slash`。Vertical 捕获全局重要 token，slash 捕获局部或固定相对位置依赖。只看最后 64 个 query 是一个折中：它用 `O(64TD)` 的开销估计当前 prompt 的稀疏结构，避免在线构造索引本身退化成 `O(T^2D)`。

### Block-Sparse

Block-Sparse 把 Q/K 按 64-token block mean pooling：

$$
\bar{Q}_a = \frac{1}{B}\sum_{i \in a} Q_i,
\quad
\bar{K}_b = \frac{1}{B}\sum_{j \in b} K_j
$$

然后在 block-level attention 上为每个 query block 选择 top-k key block。它的 kernel 最规整，每个非零项都是 64x64 dense tile；但表达能力取决于 block pooling 是否能保留检索信号，论文消融显示 only block-sparse 明显掉分。

## 离线标定流程

MInference 的标定流程是方法的重要组成部分。

官方实验命令示例：

```bash
cd experiments/infinite_bench
python run_infinitebench.py \
  --task kv_retrieval \
  --model_name_or_path gradientai/Llama-3-8B-Instruct-262k \
  --data_dir ./data \
  --output_dir ./results \
  --max_seq_length 30000 \
  --rewrite \
  --is_search \
  --start_example_id 3 \
  --topk_dims_file_path Llama_3_8B_Instruct_262k_kv_out_v32_fit_o_best_pattern.json \
  --num_eval_examples 20 \
  --topk 1 \
  --starting_layer 0 \
  --attn_type minference
```

代码在 `is_search=True` 时逐 layer/head 运行候选模式，评估 attention recall 或 sparse output 与 dense FlashAttention output 的差异，把最佳 pattern 写入 JSON。推理时 `MInferenceConfig` 根据 `model_name` 加载对应 JSON，`minference_prefill_kernel` 按 layer/head 读取模式并执行。

需要注意：标定不是训练，模型权重不变；但它依赖目标模型、上下文长度、任务样本和 kernel 实现。换模型或换极端上下文分布时，最好重新搜索或至少验证预置 pattern。

## 代码实现要点

详细实现见 [MInference 代码实现](../code_analysis/minference/00_overview.md)。

### Patch 链路

`MInferenceConfig` 记录 `attn_type="minference"`、`config_path`、`starting_layer` 和 best pattern。`new_patch` 把 HuggingFace attention forward 替换成通用 `attn_forward`。在 prefill 阶段，`prefill_forwards["minference"]` 指向 `minference_prefill_forward`；decode 阶段可以保持 dense 或叠加 KV cache 压缩方法。

当前实现是逐 head 循环：

```python
for head in range(query_states.size(1)):
    q = query_states[:, head, :, :].unsqueeze(1)
    attn_output = minference_prefill_kernel(...)
```

这让不同 head 可以使用不同 pattern，但 Python/head 粒度调度也会带来 overhead。长上下文下 attention 计算占主导，overhead 可摊薄；短上下文下则不一定划算。

### Vertical-Slash CUDA/Triton 路径

`vertical_slash_sparse_attention` 先把 line indices 转为 kernel metadata：

- slash/diagonal line -> 连续 64-token block ranges；
- vertical line -> 离散 key column gather；
- 如果 vertical column 已落入 slash range，则去重，避免重复 attention。

本地 CUDA converter 使用：

```cpp
dimBlock(64)
dimGrid(N_HEADS, BATCH_SIZE, ceil(N_ROWS / 64))
```

每个 CUDA thread 处理一个 query row block，把可解释的 line pattern 转成 `block_offset` 和 `column_index`。主 Triton fallback kernel 的 grid 是：

```python
grid = (ceil(N_CTX / 64), B * H, 1)
```

每个 program 负责一个 `(batch-head, 64-row query block)`，先遍历连续 slash block，再遍历 vertical column chunk，并用 online softmax 维护 `m_i/l_i/acc`。这种表示的关键是把大部分局部/斜线访问变成连续 block load，只有 vertical 部分需要 gather。

为了高 GPU 利用率，MInference 把稀疏模式约束在 kernel 友好的有限候选中；预算固定也限制了每个 row block 的最大循环次数。不过 fallback Triton kernel 仍可能因为不同 row/head 的 `num_blks/num_cols` 不同产生负载不均，所以代码优先使用 SGLang/vLLM 的 `sparse_attn_func`。

### Block-Sparse 路径

Block-Sparse 先 mean-pool Q/K，构造 `block_index`，再用 Triton kernel 对每个 query block 遍历 selected KV block。每个非零块是 64x64 tile，访存和计算都更接近标准 block sparse FlashAttention。它的弱点不是 kernel，而是模式表达能力：长上下文检索 token 可能被 block pooling 稀释。

## 实验结果

### Latency

README 给出的单 A100、LLaMA-3-8B-Instruct-1M prefill latency 示例：

| Context | FA2 | A-Shape | InfLLM | MInference | MInference + SGLang |
|---:|---:|---:|---:|---:|---:|
| 1K | 0.55 | 1.07 | 2.94 | 2.96 | 1.25 |
| 10K | 0.98 | 1.18 | 2.21 | 2.78 | 2.44 |
| 50K | 8.53 | 5.48 | 14.64 | 7.55 | 6.20 |
| 100K | 24.88 | 10.86 | 27.67 | 13.99 | 10.82 |
| 300K | 169.62 | 32.45 | 80.74 | 41.09 | 31.02 |
| 1M | 1765.56 | 107.86 | 328.59 | 179.12 | 112.38 |

短上下文下 MInference 有 index building 和 Python dispatch overhead；长上下文下 `T^2` attention 成本被削掉，收益快速扩大。

论文还报告 index building overhead 约 5%-20%，其余时间主要是 sparse attention 计算。100K/300K/500K/1M token 上相对 FA2 speedup 约为 1.8x/4.1x/6.8x/10x。

### Downstream

InfiniteBench 上，Full Attention 平均 38.2，MInference 平均 38.8。RULER 上，Full Attention 平均 84.4，MInference 平均 87.0。分数略高不代表 sparse attention 本身提升能力，更可能来自实验噪声、实现差异或某些稀疏模式的正则化效应；核心结论是它在长上下文任务上基本保持质量。

### Ablation

InfiniteBench 消融：

| 方法 | Avg. | 关键现象 |
|---|---:|---|
| Ours | 38.8 | 三类模式混合最好 |
| only block-sparse | 18.7 | KV retrieval 接近 0，说明 block pooling 难捕捉细粒度检索 |
| only vertical-slash | 37.1 | 保留大部分能力，但仍弱于按 head 混合 |

论文还指出 static indices 明显退化，尤其 KV retrieval 任务；这支持“模式可离线确定，但具体 token index 必须在线动态生成”的设计。

## 和视频稀疏注意力的关系

MInference 本身不是视频生成论文，但它提供了三个可复用思想：

- 把 attention map 中的结构模式显式参数化，而不是只做无结构 top-k；
- 离线标定 head 的模式类型，在线根据输入动态生成 sparse index；
- kernel-aware 搜索：稀疏模式必须和实际 CUDA/Triton kernel 的块布局、访存连续性、负载均衡匹配。

视频 DiT 方法如 SVG/SVG2 会把 spatial/temporal token 结构纳入模式设计；PISA/PASA 则进一步把非选中块做近似而非丢弃。MInference 更像 long-context sparse attention 的模式库和工程 baseline。

## 局限

- 面向 LLM prefill，不直接处理视频 DiT 的空间/时间 token 拓扑；
- 离线 pattern 依赖模型和任务，跨模型迁移需要验证；
- 逐 head Python 调度和在线 index building 让短上下文收益不稳定；
- fallback Triton sparse kernel 存在 row/head 非零数不均的负载风险，实际部署应优先用 SGLang/vLLM 优化 kernel；
- 稀疏 attention 对需要全局细粒度比较的任务仍可能漏掉低频关键 token。
