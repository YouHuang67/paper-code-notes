---
tags:
  - LLM Inference
  - KV Cache
  - Sparse Attention
  - CUDA
---
# DeepSeek V4 代码分析：总览

**源码仓库**: `refs/codes/deepseek_v4_pro`

**分析范围**: `config.json`、`encoding/encoding_dsv4.py`、`inference/generate.py`、`inference/model.py`、`inference/kernel.py`

**关联论文笔记**: [DeepSeek-V4](../../paper_reading/deepseek_v4.md)

本文不做运行教程，也不讨论权重转换细节，重点只放在两件事：

- 模型架构如何把论文里的 mHC、Hybrid Attention、MoE 和长上下文设计落到代码里
- 推理期如何通过 KV 压缩、滑动窗口、稀疏检索和低精度 kernel 把 1M context 做到可实现

## 1. 本仓库真正值得看的文件

按重要性排序：

- [model.py](src/model_py.md)：主体实现，包含 `Compressor`、`Indexer`、`Attention`、`MoE`、`Block(mHC)`、`Transformer`
- [kernel.py](src/kernel_py.md)：TileLang kernel，包含量化、GEMM、稀疏注意力、Sinkhorn
- [generate.py](src/generate_py.md)：prefill / decode 推理主循环
- [config.json](src/config_json.md)：真实模型配置，能看出论文参数如何落地
- [encoding_dsv4.py](src/encoding_dsv4_py.md)：对话协议与 DSML tool 格式

这套代码的结构非常收敛，几乎可以概括成：

```text
generate.py 负责推理驱动
model.py    负责网络结构与缓存组织
kernel.py   负责最重的低精度和稀疏计算
```

## 2. 论文概念到代码模块的映射

如果先按论文去找代码，对应关系基本如下：

- **mHC**
  - `Block.hc_pre / hc_post`
  - `hc_split_sinkhorn`
- **Hybrid Attention**
  - `Attention`
  - `Compressor`
  - `Indexer`
  - `sparse_attn`
- **MoE**
  - `Gate`
  - `Expert`
  - `MoE`
- **低精度推理**
  - `linear`
  - `act_quant`
  - `fp8_gemm`
  - `fp4_gemm`
- **长上下文推理**
  - `get_window_topk_idxs`
  - `get_compress_topk_idxs`
  - `Attention.kv_cache`
  - `Compressor.kv_cache`

这里最重要的判断是：这份代码并没有把 CSA / HCA / SWA 分成论文里的三套独立类，而是把它们统一折叠成“**窗口缓存 + 压缩缓存 + 稀疏索引**”三部分。

## 3. 整体数据流

可以先忽略所有细节，直接把一层推理看成下面这个过程：

```text
token ids
  -> embedding
  -> 扩成 hc_mult 条隐藏状态支路
  -> hc_pre: 把 hc 支路收缩成单条主路径
  -> attention:
       1. 生成 Q
       2. 生成窗口 KV
       3. 生成 compressed KV / topk 索引
       4. sparse_attn 在“窗口 + 压缩记忆”上做注意力
  -> hc_post: 把主路径重新散回 hc 支路
  -> hc_pre
  -> MoE
  -> hc_post
  -> 下一层
  -> hc_head + lm_head
  -> logits
```

理解 DeepSeek V4 代码时，最关键的是把它当成一个 **“残差状态被 mHC 改写、注意力记忆被 hybrid memory 改写、线性层被低精度 kernel 改写”** 的 Transformer。

## 4. 为什么这份实现和论文风格一致

从代码结构上看，这份实现很明确地在追三件事：

- **长上下文内存重写**
  - 不再把全部历史 token 当成同质 KV cache
  - 近邻 token 留在窗口缓存
  - 远程 token 压缩后进入 compressed cache
  - 检索器只把最值得访问的压缩块送进 attention

- **残差传播重写**
  - 不用单一 residual stream
  - 用 `hc_mult` 份状态并行传播
  - 每个 block 前后通过 Sinkhorn 归一化的 mixing 进行聚合和展开

- **推理算子重写**
  - 激活走 block-wise FP8
  - 专家权重走 FP4
  - 稀疏注意力不再是“先 gather 完整矩阵再用 PyTorch softmax”，而是直接走 TileLang kernel

## 5. 阅读顺序

建议按下面顺序读：

1. [模型架构](01_model_architecture.md)
2. [推理链路](02_inference_pipeline.md)
3. [Kernels 与量化](03_kernels_and_quantization.md)

前置公共基础只需要补这两页：

- [TileLang 编程模型](../cuda_foundations/07_tilelang_programming_model.md)
- [块量化与低精度 GEMM](../cuda_foundations/08_blockwise_quantization_and_low_precision_gemm.md)

## 6. 本系列文档范围

这一轮只保留 4 篇主文档，不扩散成很细的章节：

- [总览](00_overview.md)
- [模型架构](01_model_architecture.md)
- [推理链路](02_inference_pipeline.md)
- [Kernels 与量化](03_kernels_and_quantization.md)

源码浏览页集中在：

- [源码索引](src/index.md)
- [model.py](src/model_py.md)
- [kernel.py](src/kernel_py.md)
- [generate.py](src/generate_py.md)
- [config.json](src/config_json.md)
- [encoding_dsv4.py](src/encoding_dsv4_py.md)

## 小结

这份代码最值得看的地方，不是“它怎么把标准 Transformer 重新写一遍”，而是它如何同时重写三件事：

- 状态传播：`mHC`
- 记忆组织：`window cache + compressed cache + sparse topk`
- 算子实现：`TileLang low-precision kernels`

后面三篇文档会分别把这三条线彻底展开。
