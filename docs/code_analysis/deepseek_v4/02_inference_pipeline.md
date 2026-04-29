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
- 为什么 `start_pos` 是整个实现的核心控制量

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

但从模型角度看，真正进入 `Transformer.forward` 的仍然只是 `input_ids`。

## 2. `generate()` 的总体思路

核心代码见 [generate.py:L22-L63](src/generate_py.md#__codelineno-0-22)。

这段实现最容易被误读的地方是：它不是“先把整段 prompt prefill 一次，再逐 token decode”，而是更细地兼容了 **left-padded 变长 batch prompt**。

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
    - 第一次 forward 处理所有样本都已知的公共前缀
    - 后续每一轮只新增 [prev_pos:cur_pos] 这段 token
    - 对于还处在 prompt 范围内的位置，用 ground-truth prompt token 覆盖采样结果
    '''
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        ...
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        prev_pos = cur_pos
```

这个设计的真实含义是：

- `model.forward(delta_tokens, start_pos)` 从设计上就是 **增量接口**
- 第一次可能吃一段长序列（prefill）
- 后面每次也可能继续补 prompt 的尾巴，而不是立即进入纯生成

因此，`generate()` 通过 `prev_pos` / `cur_pos` 把“补 prompt”和“生成新 token”统一成一条 loop。

## 3. 为什么 `start_pos` 这么关键

整份 DeepSeek V4 推理代码里，`start_pos` 扮演三重角色：

- 表示本轮输入片段在全序列中的起始绝对位置
- 决定当前是 prefill 分支还是 decode 分支
- 决定窗口缓存和压缩缓存写入哪个槽位

在 `Transformer.forward(input_ids, start_pos)` 内，`start_pos` 会一路传到每个 block 的 attention：

- [model.py:L801-L809](src/model_py.md#__codelineno-0-801)
- [model.py:L688-L700](src/model_py.md#__codelineno-0-688)
- [model.py:L484-L543](src/model_py.md#__codelineno-0-484)

这意味着：模型本身没有单独的 `prefill()` 和 `decode()` API，而是由 `start_pos == 0` 还是 `> 0` 来切换行为。

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

所以推理阶段真正的复杂性不在 `Transformer.forward` 外壳，而在：

- 每层 `Attention.forward`
- 每层 `Compressor.forward`
- kernel 层 `sparse_attn`

## 5. Prefill：第一段长输入时发生了什么

当 `start_pos == 0` 时，`Attention.forward` 进入 prefill 路径，相关代码见 [model.py:L484-L543](src/model_py.md#__codelineno-0-484)。

### 5.1 先构造 Q / KV / top-k 索引

这一部分在 prefill / decode 都会发生：

- 先生成 query
- 生成窗口 KV
- 对窗口 KV 的非 RoPE 维做量化模拟
- 先构造窗口索引
- 如果当前层有压缩记忆，再拼接压缩索引

### 5.2 写窗口缓存

prefill 路径里，窗口缓存更新逻辑是：

```python
if seqlen <= win:
    self.kv_cache[:bsz, :seqlen] = kv
else:
    cutoff = seqlen % win
    self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = \
        kv[:, -win:].split([win - cutoff, cutoff], dim=1)
```

这说明窗口缓存不是“保留整段历史”，而是一个长度为 `win` 的环形缓冲区。

从缓存语义看：

- 它只服务近邻精确注意力
- 长距离历史不应该继续占用同样精细的存储预算

### 5.3 生成 compressed KV

如果本层 `compress_ratio > 0`，prefill 阶段会尝试整段压缩：

```python
if self.compress_ratio:
    if (kv_compress := self.compressor(x, start_pos)) is not None:
        kv = torch.cat([kv, kv_compress], dim=1)
```

这里非常关键：

- `kv` 在 prefill 阶段被扩展成“窗口 KV + compressed KV”的临时拼接记忆
- 传给 `sparse_attn` 的 `topk_idxs` 已经按这个拼接布局计算过 offset

所以 prefill 阶段的 `sparse_attn` 实际上是在一份 **逻辑拼接后的混合记忆矩阵** 上工作。

### 5.4 Prefill 注意力不是 dense attention

最终计算：

```python
o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
```

也就是说，即使是 prefill，也没有退回到完整 dense attention。

这和很多实现不一样：

- 很多模型 prefill 走 dense，decode 走 cache
- DeepSeek V4 从 prefill 开始就按 hybrid sparse memory 组织

这正是论文强调“长上下文效率”能落地到推理实现的关键。

## 6. Decode：单 token 增量时发生了什么

当 `start_pos > 0` 时，进入 decode 路径。

### 6.1 当前 token 写入窗口环形缓存

```python
self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
```

这说明窗口缓存的写法是：

- 只保留最近 `win` 个 token
- 用绝对位置模 `win` 定位槽位

因此 decode 阶段根本不会复制整段历史，只会覆盖环形槽位。

### 6.2 压缩缓存按比率增量更新

```python
if self.compress_ratio:
    self.compressor(x, start_pos)
```

`Compressor.forward` 在 decode 阶段的关键分支是 [model.py:L343-L376](src/model_py.md#__codelineno-0-343)：

```python
should_compress = (start_pos + 1) % self.compress_ratio == 0

'''
如果一个压缩块还没凑满:
    只把当前 token 放入 kv_state / score_state

如果压缩块凑满:
    再做一次 gated pooling
    然后写入 compressed cache 的 start_pos // ratio 槽位
'''
...
if not should_compress:
    return
...
self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
```

因此 decode 阶段的 compressed cache 更新不是“每 token 更新一次”，而是 **每 `ratio` 个 token 更新一次**。

### 6.3 Decode 阶段的注意力记忆空间

decode 阶段传给 kernel 的记忆不再是临时拼接的 `kv`，而是整个 `self.kv_cache[:bsz]`：

```python
o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
```

这里 `self.kv_cache` 的布局是：

```text
[ 0 : window_size )                   -> 窗口环形缓存
[ window_size : window_size + ... )   -> compressed cache
```

所以 decode 阶段的 `topk_idxs` 本质上是在这块统一缓存地址空间上检索。

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

这就是为什么它必须写成：

```python
next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
```

因为同一个 batch 内，不同样本可能在不同时间从“prompt 填充”切换到“生成”。

## 8. 为什么这种设计适合长上下文

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

这几个量正好就是论文中长上下文设计的核心拨杆。

## 9. 代码层的一个重要判断

DeepSeek V4 的推理代码并没有尝试做“最通用”的缓存抽象。它的实现是明显为这套架构定制的：

- 窗口缓存是环形的
- 压缩缓存是分层的
- 索引空间是手工拼接的
- mHC 状态流从输入到输出贯穿全程

这也是为什么后续做代码解析时，不能把它当成“普通 Transformer + 一些技巧”，而要把缓存组织本身视为架构的一部分。

## 小结

DeepSeek V4 的推理链路可以概括成：

```text
generate() 用 prev_pos / cur_pos 把 prompt 补全和生成统一起来，
model.forward() 用 start_pos 把 prefill 和 decode 统一起来，
Attention.forward() 再把窗口缓存和压缩缓存统一成一套稀疏记忆访问。
```

下一篇 [Kernels 与量化](03_kernels_and_quantization.md) 会继续把这条链路下沉到 `kernel.py`。
