---
tags:
  - LLM Inference
  - KV Cache
  - Sparse Attention
  - CUDA
---
# DeepSeek V4：推理链路

本文关注“推理时到底怎么走”，对应两个入口文件：

- [generate.py](src/generate_py.md)
- [model.py](src/model_py.md)

目标是把这四件事讲清楚：

- prompt 如何进入模型
- prefill 和 decode 怎样共用同一个 `model.forward`
- KV cache / compressed cache 怎样在两阶段里更新
- `start_pos` 在整条推理链中的控制作用

## 1. 入口：从消息到 token

最外层输入协议在 `encoding_dsv4.py`，生成脚本在 [generate.py:L1-L155](src/generate_py.md#__codelineno-0-1)。

交互模式下的链路非常简单：

```text
messages
  -> encode_messages(..., thinking_mode="chat")
  -> tokenizer.encode(...)
  -> generate(...)
  -> tokenizer.decode(...)
  -> parse_message_from_completion_text(...)
```

所以 `encoding_dsv4.py` 的作用是：

- 决定 prompt 字符串格式
- 决定 `<think>` / DSML / system/user/assistant token 如何组织

从模型角度看，进入 `Transformer.forward` 的仍然只是 `input_ids`。

## 2. `generate()` 的增量循环

核心代码见 [generate.py:L22-L63](src/generate_py.md#__codelineno-0-22)。

```python
def generate(model, prompt_tokens, max_new_tokens, eos_id, temperature=1.0):
    prompt_lens = [len(t) for t in prompt_tokens]
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))

    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long)

    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens))
    prompt_mask = tokens != -1

    '''
    cur_pos 从 min(prompt_lens) 开始：
    - 第一次 forward 处理所有样本都具备的公共前缀
    - 后续每轮只把 [prev_pos:cur_pos] 这段新 token 送进模型
    - 仍处在 prompt 范围内的位置，用 ground-truth prompt token 覆盖采样结果
    '''
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        ...
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        prev_pos = cur_pos
```

这段循环的含义是：

- `model.forward(delta_tokens, start_pos)` 从设计上就是 **增量接口**
- 第一次可能吃一段长序列（prefill）
- 后面每次也可能继续补 prompt 的尾巴，而不是立即进入纯生成

`generate()` 用 `prev_pos` / `cur_pos` 把“补 prompt”和“生成新 token”统一成一条增量 loop。

## 3. `start_pos` 的作用

整份 DeepSeek V4 推理代码里，`start_pos` 扮演三重角色：

- 表示本轮输入片段在全序列中的起始绝对位置
- 决定当前是 prefill 分支还是 decode 分支
- 决定窗口缓存和压缩缓存写入哪个槽位

在 `Transformer.forward(input_ids, start_pos)` 内，`start_pos` 会一路传到每个 block 的 attention：

- [model.py:L801-L809](src/model_py.md#__codelineno-0-801)
- [model.py:L688-L700](src/model_py.md#__codelineno-0-688)
- [model.py:L484-L543](src/model_py.md#__codelineno-0-484)

模型本身没有单独的 `prefill()` 和 `decode()` API，而是由 `start_pos == 0` 或 `> 0` 切换行为。

## 4. `Transformer.forward`：网络主干本身很薄

`Transformer.forward` 对应 [model.py:L801-L809](src/model_py.md#__codelineno-0-801)。

```python
@torch.inference_mode()
def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
    h = self.embed(input_ids)                                   # token -> embedding
    h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)           # 扩成 hc_mult 条状态流
    for layer in self.layers:
        h = layer(h, start_pos, input_ids)                     # 每层自己处理 cache / sparse attention
    logits = self.head(h, self.hc_head_fn, self.hc_head_scale,
                       self.hc_head_base, self.norm)
    return logits
```

推理阶段的复杂性不在 `Transformer.forward` 外壳，而在：

- 每层 `Attention.forward`
- 每层 `Compressor.forward`
- kernel 层 `sparse_attn`

## 5. `Attention.forward` 中的 prefill / decode 分支

`Attention.forward` 的核心逻辑见 [model.py:L484-L543](src/model_py.md#__codelineno-0-484)。这一段同时覆盖：

- Q / KV 构造
- 窗口索引与压缩索引拼接
- prefill 分支
- decode 分支

```python
def forward(self, x: torch.Tensor, start_pos: int):
    bsz, seqlen, _ = x.size()
    freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
    win = self.window_size
    ratio = self.compress_ratio
    rd = self.rope_head_dim

    if self.compress_ratio and self.compressor.kv_cache is None:
        self.compressor.kv_cache = self.kv_cache[:, win:]          # compressed cache 放在窗口区之后
        self.compressor.freqs_cis = self.freqs_cis
        if self.indexer is not None:
            self.indexer.freqs_cis = self.freqs_cis

    # Q
    qr = q = self.q_norm(self.wq_a(x))
    q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
    q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
    apply_rotary_emb(q[..., -rd:], freqs_cis)

    # 窗口 KV
    kv = self.wkv(x)
    kv = self.kv_norm(kv)
    apply_rotary_emb(kv[..., -rd:], freqs_cis)
    act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)    # 只对非 RoPE 维做量化模拟

    # 统一索引空间：窗口索引 + 压缩索引
    topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
    if self.compress_ratio:
        offset = kv.size(1) if start_pos == 0 else win
        if self.indexer is not None:
            compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
        else:
            compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
        topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
    topk_idxs = topk_idxs.int()

    # prefill
    if start_pos == 0:
        if seqlen <= win:
            self.kv_cache[:bsz, :seqlen] = kv                     # 短 prompt 直接写入窗口区
        else:
            cutoff = seqlen % win
            self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = \
                kv[:, -win:].split([win - cutoff, cutoff], dim=1) # 长 prompt 写成窗口环形布局
        if self.compress_ratio:
            if (kv_compress := self.compressor(x, start_pos)) is not None:
                kv = torch.cat([kv, kv_compress], dim=1)          # prefill: 临时拼接窗口 KV 和 compressed KV
        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)

    # decode
    else:
        self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)      # decode: 只写当前 token 的窗口槽位
        if self.compress_ratio:
            self.compressor(x, start_pos)                         # decode: 满 ratio 时更新 compressed cache
        o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink,
                        topk_idxs, self.softmax_scale)            # decode: 直接在统一 cache 地址空间内检索

    apply_rotary_emb(o[..., -rd:], freqs_cis, True)
    ...
```

这段代码里能直接看出推理态的缓存组织：

- `kv_cache[: , :win]` 是窗口区
- `kv_cache[: , win:]` 是 compressed cache 区
- prefill 阶段把窗口 KV 和新生成的 compressed KV 临时拼接后送进 `sparse_attn`
- decode 阶段直接在统一的 `kv_cache` 地址空间中检索

`Compressor.forward` 在 decode 态的更新条件见 [model.py:L343-L376](src/model_py.md#__codelineno-0-343)。压缩缓存不是每 token 更新一次，而是每 `ratio` 个 token 更新一次。

## 7. `generate()` 如何把 prefill 和 decode 串成一条 loop

这一点最容易被忽略。

设：

- 第一个样本 prompt 长度为 100
- 第二个样本 prompt 长度为 200

则 `generate()` 的行为是：

1. `cur_pos = 100`
   - 调 `model.forward(tokens[:, 0:100], start_pos=0)`
   - 这是真正的 prefill
2. `cur_pos = 101, 102, ..., 199`
   - 调 `model.forward(tokens[:, prev_pos:cur_pos], start_pos=prev_pos)`
   - 对第一个样本，这时已经是 decode
   - 对第二个样本，仍然是在补 prompt
3. `cur_pos >= 200`
   - 两个样本都进入纯生成

因此实现里必须写成：

```python
next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
```

因为同一个 batch 内，不同样本可能在不同时间从“prompt 填充”切换到“生成”。

## 8. 长上下文推理的控制点

如果从实现角度总结 DeepSeek V4 的长上下文推理设计，核心是三条：

- **近邻 history**
  - 放进固定长度窗口缓存
  - 保留 token 级精度

- **远程 history**
  - 压缩写入 compressed cache
  - 只在需要时被 top-k 检索到

- **计算入口统一**
  - prefill 和 decode 都走同一套 `sparse_attn`
  - 不维护一套 dense prefill 和一套 sparse decode 的双系统

于是它把推理复杂度的控制点，集中到：

- `window_size`
- `compress_ratio`
- `index_topk`

这几个量就是代码里直接控制长上下文推理成本的拨杆。

## 9. 代码层的一个重要判断

DeepSeek V4 的推理代码并没有尝试做“最通用”的缓存抽象。它的实现是明显为这套架构定制的：

- 窗口缓存是环形的
- 压缩缓存是分层的
- 索引空间是手工拼接的
- mHC 状态流从输入到输出贯穿全程

后续看代码时，需要把缓存组织本身视为架构的一部分，而不是附加技巧。

## 小结

DeepSeek V4 的推理链路可以概括成：

```text
generate() 用 prev_pos / cur_pos 把 prompt 补全和生成统一起来，
model.forward() 用 start_pos 把 prefill 和 decode 统一起来，
Attention.forward() 再把窗口缓存和压缩缓存统一成一套稀疏记忆访问。
```

下一篇 [Kernels 与量化](03_kernels_and_quantization.md) 会继续把这条链路下沉到 `kernel.py`。
