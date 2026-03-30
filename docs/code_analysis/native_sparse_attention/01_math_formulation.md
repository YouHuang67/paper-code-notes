---
tags:
  - Online Softmax
---
# 数学公式推导

本节介绍 NSA 的数学基础，包括标准注意力、Online Softmax 以及 NSA 的三路融合公式。

## 1. 标准注意力

给定 Query $\mathbf{Q} \in \mathbb{R}^{T \times d}$，Key $\mathbf{K} \in \mathbb{R}^{T \times d}$，Value $\mathbf{V} \in \mathbb{R}^{T \times d}$：

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}
$$

对于因果注意力，需要 mask 掉未来位置：

$$
\mathbf{o}_t = \sum_{i=1}^{t} \frac{\exp(s_{ti})}{\sum_{j=1}^{t}\exp(s_{tj})} \mathbf{v}_i, \quad s_{ti} = \frac{\mathbf{q}_t \cdot \mathbf{k}_i}{\sqrt{d}}
$$

## 2. Online Softmax

直接计算 softmax 需要两次遍历（求 max、求 sum），FlashAttention 使用 online softmax 实现单次遍历：

$$
\begin{aligned}
m_t^{(j)} &= \max(m_t^{(j-1)}, \max_i(s_{ti}^{(j)})) \\
\ell_t^{(j)} &= e^{m_t^{(j-1)} - m_t^{(j)}} \ell_t^{(j-1)} + \sum_i e^{s_{ti}^{(j)} - m_t^{(j)}} \\
\mathbf{o}_t^{(j)} &= e^{m_t^{(j-1)} - m_t^{(j)}} \mathbf{o}_t^{(j-1)} + \sum_i e^{s_{ti}^{(j)} - m_t^{(j)}} \mathbf{v}_i
\end{aligned}
$$

最终输出：

$$
\mathbf{o}_t = \frac{\mathbf{o}_t^{(N)}}{\ell_t^{(N)}}, \quad \text{LSE}_t = m_t^{(N)} + \log(\ell_t^{(N)})
$$

**源码对应** ([parallel.py#L87-L122](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L87-L122))：

```python
'''
Online softmax 累积更新
- b_m: 当前最大值
- b_acc: 累积 exp sum
- b_o: 累积加权输出
'''
b_o = tl.zeros([G, BV], dtype=tl.float32)
b_m = tl.full([G], float('-inf'), dtype=tl.float32)
b_acc = tl.zeros([G], dtype=tl.float32)

for i_c in range(0, NC, BC):
    # ... 加载 K, V block ...

    # [G, BC] attention scores
    b_s = tl.dot(b_q, b_k)
    b_s = tl.where((o_c < NC)[None, :], b_s, float('-inf'))

    # 更新 max 并计算 rescale 系数
    b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m  # [G]
    b_r = tl.exp(b_mp - b_m)                          # [G] rescale factor

    # softmax 概率
    b_p = tl.exp(b_s - b_m[:, None])                  # [G, BC]

    # 累积更新
    b_acc = b_acc * b_r + tl.sum(b_p, 1)              # [G]
    b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)  # [G, BV]

# 最终归一化
b_o = b_o / b_acc[:, None]
b_lse = b_m + tl.log(b_acc)
```

## 3. NSA 三路融合

NSA 将注意力分解为三个分支，通过门控机制融合：

$$
\mathbf{o} = g_{\text{cmp}} \odot \mathbf{o}_{\text{cmp}} + g_{\text{slc}} \odot \mathbf{o}_{\text{slc}} + g_{\text{swa}} \odot \mathbf{o}_{\text{swa}}
$$

### 3.1 压缩注意力 (Compression)

将 KV 序列按 block 压缩（mean pooling），降低序列长度：

$$
\bar{\mathbf{k}}_c = \frac{1}{B} \sum_{i=(c-1)B+1}^{cB} \mathbf{k}_i, \quad \bar{\mathbf{v}}_c = \frac{1}{B} \sum_{i=(c-1)B+1}^{cB} \mathbf{v}_i
$$

压缩后的注意力：

$$
\mathbf{o}_{\text{cmp},t} = \sum_{c=1}^{\lfloor t/B \rfloor} \text{softmax}\left(\frac{\mathbf{q}_t \cdot \bar{\mathbf{K}}^\top}{\sqrt{d}}\right)_c \bar{\mathbf{v}}_c
$$

### 3.2 选择注意力 (Selection)

基于压缩注意力的分数，选择 top-k 个最重要的 block：

$$
\mathcal{I}_t = \text{top-}k\left(\left\{\sum_{g=1}^{G} p_{t,c}^{(g)} : c = 1, \ldots, \lfloor t/B \rfloor\right\}\right)
$$

其中 $G$ 为 GQA 的 group 数量。选择后对原始 KV 做精细注意力：

$$
\mathbf{o}_{\text{slc},t} = \sum_{c \in \mathcal{I}_t} \sum_{i \in \text{block}_c} \text{softmax}(\mathbf{q}_t \cdot \mathbf{K}_{\mathcal{I}_t}^\top / \sqrt{d})_i \mathbf{v}_i
$$

### 3.3 滑动窗口注意力 (Sliding Window)

固定窗口大小 $W$ 的局部注意力：

$$
\mathbf{o}_{\text{swa},t} = \sum_{i=\max(1, t-W+1)}^{t} \text{softmax}(\mathbf{q}_t \cdot \mathbf{K}_{[t-W+1:t]}^\top / \sqrt{d})_i \mathbf{v}_i
$$

## 4. 反向传播

对于 softmax attention 的反向传播，需要计算：

$$
\begin{aligned}
\delta_t &= \sum_j o_{tj} \cdot \frac{\partial L}{\partial o_{tj}} \\
\frac{\partial L}{\partial s_{ti}} &= p_{ti} \left(\frac{\partial L}{\partial o_{ti}} \cdot v_i - \delta_t\right) \\
\frac{\partial L}{\partial \mathbf{q}_t} &= \sum_i \frac{\partial L}{\partial s_{ti}} \cdot \mathbf{k}_i
\end{aligned}
$$

**源码对应** ([parallel.py#L586-L602](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L586-L602))：

```python
'''
预处理：计算 delta = sum(o * do)
用于反向传播中 ds = p * (dp - delta) 的计算
'''
@triton.jit
def parallel_nsa_bwd_kernel_preprocess(
    o, do, delta,
    B: tl.constexpr,
    V: tl.constexpr
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < V

    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)  # delta = o · do

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))
```
