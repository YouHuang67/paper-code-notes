---
tags:
  - LLM Inference
  - KV Cache
  - Sparse Attention
  - CUDA
---
# DeepSeek V4：模型架构

本文只解释 `inference/model.py` 的模型设计，不讨论运行入口和权重转换。核心目标是把论文中的四条主线和源码对齐：

- mHC 如何改写 residual
- Hybrid Attention 如何改写 memory
- MoE 如何改写 FFN
- 低精度推理如何改写线性层

源码主文件：

- [model.py](src/model_py.md)
- [config.json](src/config_json.md)

必要前置：

- [TileLang 编程模型](../cuda_foundations/07_tilelang_programming_model.md)
- [块量化与低精度 GEMM](../cuda_foundations/08_blockwise_quantization_and_low_precision_gemm.md)

## 1. 配置如何定义这台模型

`ModelArgs` 是整份实现的总开关，对应源码 [model.py:L34-L80](src/model_py.md#__codelineno-0-34)。

最关键的字段不是“超参数名字”，而是它们如何分组决定结构：

- 基本模型规模
  - `dim`
  - `n_layers`
  - `n_heads`
  - `head_dim`
- 长上下文 / attention
  - `window_size`
  - `compress_ratios`
  - `rope_head_dim`
  - `compress_rope_theta`
  - `rope_theta`
- MoE
  - `n_routed_experts`
  - `n_shared_experts`
  - `n_activated_experts`
  - `score_func`
  - `route_scale`
- mHC
  - `hc_mult`
  - `hc_sinkhorn_iters`
  - `hc_eps`
- 低精度
  - `dtype`
  - `expert_dtype`
  - `scale_fmt`
  - `scale_dtype`

这里有两个很重要的结构信号：

- `compress_ratios` 是 **逐层** 配置，不是全模型统一比率
- `hc_mult` 是全模型的“并行残差流数量”，这意味着 mHC 不是某一层的局部 trick，而是全局状态表示方式

## 2. 最底层积木：并行 Embedding、量化 Linear、RMSNorm

### 2.1 Embedding 与 TP 切分

`ParallelEmbedding` 在词表维度上切分，对应源码 [model.py:L83-L105](src/model_py.md#__codelineno-0-83)。

```python
class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        ...
        self.part_vocab_size = (vocab_size // world_size)          # 每个 rank 持有局部词表
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx                           # 全局 token id -> 局部 id
            x[mask] = 0
        y = F.embedding(x, self.weight)                            # 先查局部词表
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)                                     # 汇总各 rank 局部命中的结果
        return y
```

这个实现很标准，但重要的是它和后面的并行线性层形成统一模式：

- Column Parallel：输出维切分
- Row Parallel：输入维切分
- Embedding：词表维切分

### 2.2 `linear()` 不是普通 `F.linear`

`linear()` 是整个低精度推理分发入口，对应 [model.py:L108-L120](src/model_py.md#__codelineno-0-108)。

```python
def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert bias is None

    '''
    关键分发逻辑：
    1. FP4 权重 -> 先把输入激活按 block 做 FP8 量化，再走 fp4_gemm
    2. FP8 权重 -> 先把输入激活按 block 做 FP8 量化，再走 fp8_gemm
    3. 其他情况 -> 退回普通 F.linear
    '''
    if weight.dtype == torch.float4_e2m1fn_x2:
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp4_gemm(x, s, weight, weight.scale, scale_dtype)
    elif weight.dtype == torch.float8_e4m3fn:
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp8_gemm(x, s, weight, weight.scale, scale_dtype)
    else:
        return F.linear(x, weight)
```

也就是说，这份模型代码根本没有把“量化权重”和“普通线性层”当成同一实现路径；它们从 `linear()` 开始就走不同后端。

### 2.3 `Linear` 显式持有 scale 张量

对应 [model.py:L123-L152](src/model_py.md#__codelineno-0-123)。

这段实现最关键的不是构造 `weight`，而是它把“scale 是参数的一部分”写死在类型结构里：

- FP4 权重：
  - `weight.shape = [out, in // 2]`
  - `scale.shape = [out, in // 32]`
- FP8 权重：
  - `weight.shape = [out, in]`
  - `scale.shape = [ceil(out / 128), ceil(in / 128)]`

这说明：

- 量化不是运行时临时压缩，而是 checkpoint 原生格式的一部分
- `model.py` 不是“加载 BF16 权重再临时量化”，而是原生消费低精度权重

### 2.4 并行线性层的职责

对应 [model.py:L155-L180](src/model_py.md#__codelineno-0-155)。

- `ColumnParallelLinear`
  - 切输出维
  - 每个 rank 算自己的输出块
  - 不需要 all-reduce
- `RowParallelLinear`
  - 切输入维
  - 每个 rank 算部分和
  - 最后 `all_reduce`

这对 Attention 和 MoE 都很重要，因为后面所有 LoRA-style 低秩投影都建立在这组并行线性层之上。

## 3. 位置编码与长上下文准备

RoPE 相关逻辑在 [model.py:L199-L244](src/model_py.md#__codelineno-0-199)。

这里有两个要点：

- `precompute_freqs_cis` 支持 YaRN 风格缩放
- `apply_rotary_emb` 支持 `inverse=True`

第二点非常关键，因为 Attention 输出在 `sparse_attn` 之后还会执行一次“反旋转”：

- query / kv 前向时对 RoPE 维做旋转
- 输出 `o` 在回到 O 投影前，用 `apply_rotary_emb(..., inverse=True)` 去旋回

这意味着实现里把 RoPE 维度看成一种“中间计算坐标系”，而不是始终裸露在残差流中。

## 4. Hybrid Attention 的记忆结构

这是整份代码最重要的部分。

### 4.1 不是三种 attention，而是三种 memory

相关辅助函数：

- [model.py:L254-L276](src/model_py.md#__codelineno-0-254)

这两个函数本质上在生成两类索引：

- `get_window_topk_idxs`
  - 局部滑动窗口索引
- `get_compress_topk_idxs`
  - 压缩缓存索引

所以从实现角度看，论文里的 SWA / CSA / HCA 在这里已经统一成：

```text
局部窗口记忆 + 压缩远程记忆 + 稀疏 top-k 访问
```

真正的注意力 kernel `sparse_attn` 只认一份 `topk_idxs`，并不关心某个索引来自窗口还是压缩缓存。

### 4.2 `Compressor`：把远程 token 变成压缩记忆

核心代码在 [model.py:L279-L377](src/model_py.md#__codelineno-0-279)。

`Compressor` 可以用一句话概括：

```text
对连续 ratio 个 token 的 KV 做带门控权重的聚合，再把结果写入 compressed cache
```

对应数学形式可写成：

$$
\tilde{k}_{g} = \sum_{i \in \mathcal{B}_g} \alpha_i k_i,\qquad
\alpha_i = \mathrm{softmax}(s_i)
$$

其中：

- `\mathcal{B}_g` 是一个压缩块
- `s_i` 来自 `wgate`
- `k_i` 来自 `wkv`

源码里最关键的一段如下：

```python
kv = self.wkv(x)                                             # 待压缩的值向量
score = self.wgate(x)                                        # 每个 token 的压缩打分

'''
prefill 阶段:
    1. 把序列按 ratio 分块
    2. score + ape 后做 softmax
    3. 在块维度做加权求和，得到 compressed kv

decode 阶段:
    1. 用 kv_state / score_state 累积当前未满块
    2. 当 (start_pos + 1) % ratio == 0 时触发一次压缩
'''
if start_pos == 0:
    ...
    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio)) + self.ape
    if overlap:
        kv = self.overlap_transform(kv, 0)
        score = self.overlap_transform(score, float("-inf"))
    kv = (kv * score.softmax(dim=2)).sum(dim=2)
else:
    ...
```

这里有两个工程细节非常重要：

- `compress_ratio == 4` 时启用 overlap 机制
  - 说明短比率压缩更担心块边界伪影
- 压缩在 FP32 中完成，压缩后才回到低精度模拟
  - 说明作者不想让聚合权重本身受太强量化噪声影响

压缩结束后，代码还会：

- 对压缩后的 KV 重新做 `RMSNorm`
- 只对非 RoPE 维做 FP8 / FP4 模拟
- 写入 `self.kv_cache`

### 4.3 `Indexer`：决定远程记忆到底读哪些块

核心代码在 [model.py:L380-L433](src/model_py.md#__codelineno-0-380)。

`Indexer` 的目标不是做真正注意力，而是生成 **压缩缓存的 top-k 检索索引**。

它的打分过程可以概括成：

$$
\mathrm{score}(t, j) =
\sum_{h}
w_{t,h} \cdot \mathrm{ReLU}(q_{t,h}^{\top}\tilde{k}_{j})
$$

对应实现：

```python
q = self.wq_b(qr)                                            # 从 Q 的低秩表示恢复 index query
q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
apply_rotary_emb(q[..., -rd:], freqs_cis)
q = rotate_activation(q)                                     # Hadamard rotate
fp4_act_quant(q, fp4_block_size, True)                       # 更激进的低精度模拟

self.compressor(x, start_pos)                                # 先更新 compressed kv cache
weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
```

从设计上看，`Indexer` 扮演的是论文里“高压缩记忆的检索器”角色：

- 它不返回 attention output
- 它只返回“值得访问的压缩位置”
- 真正做 softmax attention 的是后面的 `sparse_attn`

### 4.4 `Attention`：把窗口缓存与压缩缓存合并成一份检索空间

核心实现见 [model.py:L436-L543](src/model_py.md#__codelineno-0-436)。

这段代码最值得精读，因为它把论文里的所有 attention 改造压缩成了一个 forward。

```python
def forward(self, x: torch.Tensor, start_pos: int):
    bsz, seqlen, _ = x.size()
    freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
    win = self.window_size
    ratio = self.compress_ratio
    rd = self.rope_head_dim

    '''
    第一次进入该层时，把压缩器和索引器绑定到本层 cache。
    注意 compressed cache 直接放在 kv_cache 的窗口区之后。
    '''
    if self.compress_ratio and self.compressor.kv_cache is None:
        self.compressor.kv_cache = self.kv_cache[:, win:]
        self.compressor.freqs_cis = self.freqs_cis
        if self.indexer is not None:
            self.indexer.freqs_cis = self.freqs_cis

    # 1. 构造 Q
    qr = q = self.q_norm(self.wq_a(x))
    q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
    q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
    apply_rotary_emb(q[..., -rd:], freqs_cis)

    # 2. 构造窗口 KV
    kv = self.wkv(x)
    kv = self.kv_norm(kv)
    apply_rotary_emb(kv[..., -rd:], freqs_cis)
    act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)      # 只量化非 RoPE 维

    # 3. 先拿到窗口索引，再拼上压缩索引
    topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
    if self.compress_ratio:
        offset = kv.size(1) if start_pos == 0 else win
        if self.indexer is not None:
            compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
        else:
            compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
        topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)

    # 4. 写 cache + 做 sparse attention
    if start_pos == 0:
        ...
        if self.compress_ratio:
            if (kv_compress := self.compressor(x, start_pos)) is not None:
                kv = torch.cat([kv, kv_compress], dim=1)            # prefill 阶段把窗口 KV 和 compressed KV 拼成统一记忆
        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
    else:
        self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)        # decode 阶段只写一个窗口位置
        if self.compress_ratio:
            self.compressor(x, start_pos)                           # 视需要更新 compressed cache
        o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
```

从这段实现可以抽出三个关键结论：

- **SWA** 对应 `window_size` + `get_window_topk_idxs`
- **CSA / HCA** 对应 `Compressor` + `Indexer` + `compressed kv cache`
- 最终并不存在三套 attention kernel，只有一套 `sparse_attn` 在“混合索引空间”上运行

这就是论文“Hybrid Attention”在代码里的真实形态。

## 5. MoE：共享专家 + 路由专家

相关实现：

- [model.py:L546-L644](src/model_py.md#__codelineno-0-546)

### 5.1 Gate

`Gate` 同时支持：

- hash routing
- score-based routing

核心逻辑：

```python
scores = linear(x.float(), self.weight.float())
if self.score_func == "softmax":
    scores = scores.softmax(dim=-1)
elif self.score_func == "sigmoid":
    scores = scores.sigmoid()
else:
    scores = F.softplus(scores).sqrt()

if self.bias is not None:
    scores = scores + self.bias                              # 只影响选专家，不影响最终权重归一化

if self.hash:
    indices = self.tid2eid[input_ids]
else:
    indices = scores.topk(self.topk, dim=-1)[1]

weights = original_scores.gather(1, indices)
if self.score_func != "softmax":
    weights /= weights.sum(dim=-1, keepdim=True)
weights *= self.route_scale
```

实现上的重点有两个：

- 对非 softmax routing，代码会显式重新归一化
- `bias` 只参与 top-k 选择，不参与最终 routing weight

### 5.2 Expert 与 MoE

`Expert` 是标准 SwiGLU FFN，但 `MoE` 的组织方式很有代表性：

- 每个 rank 只持有局部 routed experts
- 共享专家始终全量存在
- routed experts 的输出在 rank 间 `all_reduce`

因此它的整体输出是：

$$
y = \sum_{e \in \mathrm{topk}(x)} w_e \, E_e(x) + E_{\mathrm{shared}}(x)
$$

其中 routed experts 可能是 FP4 权重，而 shared expert 仍走常规路径。

## 6. mHC：真正改写残差流的地方

这是最值得对齐论文的部分。相关代码：

- [model.py:L647-L700](src/model_py.md#__codelineno-0-647)

### 6.1 `hc_pre`

`hc_pre` 的输入不是 `[b, s, d]`，而是 `[b, s, hc, d]`。

它先把 `hc` 条状态流 flatten 成一条大向量，再通过线性混合产生三组系数：

- `pre`
- `post`
- `comb`

源码：

```python
def hc_pre(self, x, hc_fn, hc_scale, hc_base):
    shape, dtype = x.size(), x.dtype
    x = x.flatten(2).float()                                  # [b, s, hc*d]
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt                        # 先生成 mixing logits
    pre, post, comb = hc_split_sinkhorn(
        mixes, hc_scale, hc_base,
        self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
    )
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)   # hc 条流收缩成一条主路径
    return y.to(dtype), post, comb
```

数学上可以写成：

$$
\tilde{h}_t = \sum_{r=1}^{H_c} \alpha_{t,r} h_{t,r}
$$

其中 `\alpha` 由 `hc_split_sinkhorn` 产生。

### 6.2 `hc_post`

`hc_post` 再把单条主路径展开回多条状态流：

```python
def hc_post(self, x, residual, post, comb):
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + \
        torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y.type_as(x)
```

对应公式：

$$
h'_{t,r} = \beta_{t,r} \tilde{h}_t + \sum_{r'} \Gamma_{t,r',r} h_{t,r'}
$$

这里最关键的不是公式形式，而是实现含义：

- 主分支输出不会直接覆盖 residual
- 它只是作为一部分，重新注入 `hc` 条状态流
- 原来的 residual 多分支结构继续通过 `comb` 保持传播

这正是 mHC 相比普通 residual 的本质区别。

### 6.3 一个 block 的真实顺序

完整 block forward：

```python
residual = x
x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
x = self.attn_norm(x)
x = self.attn(x, start_pos)
x = self.hc_post(x, residual, post, comb)

residual = x
x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
x = self.ffn_norm(x)
x = self.ffn(x, input_ids)
x = self.hc_post(x, residual, post, comb)
```

可以看到，Attention 与 MoE 两个子层都被包在：

```text
hc_pre -> 子层计算 -> hc_post
```

这说明 mHC 不是“只在 block 外包一层”，而是对子层级残差传播都做了重定义。

## 7. `Transformer`：整网如何拼装

相关代码：

- [model.py:L769-L809](src/model_py.md#__codelineno-0-769)

最重要的一点是模型一开始就把 embedding 扩成 `hc_mult` 条状态流：

```python
h = self.embed(input_ids)
h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)             # 从一开始就进入多流表示
for layer in self.layers:
    h = layer(h, start_pos, input_ids)
logits = self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
```

也就是说：

- 多流表示不是 block 内临时结构
- 它贯穿整个主干
- `ParallelHead` 在末端还会再做一次 hc 聚合

## 8. 从代码角度重新理解论文里的“精度重点”

如果只从 `model.py` 出发，DeepSeek V4 的“重点精度设计”其实就是以下几条约束一起成立：

- RoPE 维度在关键路径上尽量保持高精度
- 压缩聚合在 FP32 中完成
- MoE 内部关键激活在 `float()` 路径中计算
- mHC mixing 明确在 FP32 参数空间中进行
- 低精度主要压在线性层、检索器子路径和 kernel 密集区

这意味着它不是“到处都量化”，而是在非常具体的位置保留数值稳定带。

## 小结

`model.py` 里真正值得记住的不是类名，而是三次重写：

1. **残差重写**：`mHC` 让隐藏状态从单流变成多流
2. **记忆重写**：`Compressor + Indexer + sparse_attn` 让 KV cache 从稠密序列变成混合记忆
3. **线性层重写**：`linear()` 把常规 `F.linear` 改成低精度 kernel 分发入口

下一篇看 [推理链路](02_inference_pipeline.md)，会把这些模块放回 prefill / decode 两阶段里串起来。
