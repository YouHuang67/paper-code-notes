---
tags:
  - CUDA
  - CUTLASS
  - Sparse Attention
  - Flash Attention
---

# FlashInfer Variable Block Sparse Attention：总览

**源码仓库**: [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)

**分析范围**: `VariableBlockSparseAttentionWrapper` 的完整调用链——从 Python wrapper 到 Hopper CUDA kernel

## 解决什么问题

标准 block sparse attention（如 FlashAttention 的 block_mask）要求所有 block 使用统一的固定大小（如 128×128）。当实际稀疏模式的 block 大小不均匀时（如 SVG2 中 k-means 聚类产生的簇大小为 `[96, 128, 64, ...]`），固定 block 要么浪费计算（pad 小 block），要么损失精度（截断大 block）。

`VariableBlockSparseAttentionWrapper` 支持每个 block 行/列有独立的大小，同时每个 KV head 可以有不同的稀疏模式。

## 核心设计

### plan/run 两阶段

- **plan()**: 接收高层描述（block_mask_map, block_row_sz, block_col_sz），编译成底层 paged attention 可消费的索引结构
- **run()**: reshape Q/K/V，调用 `paged_run` 执行计算

### 关键映射：variable block → paged attention

wrapper 并不实现专门的"variable block kernel"。它的核心技巧是把 variable block sparse 问题**映射**成 paged attention 问题：

1. 设置 `page_size = 1`，让每个 token 成为独立的"page"
2. 将 block 级稀疏图展开为 token 级索引（`kv_indptr` + `kv_indices`）
3. 复用 FlashInfer 已有的 batch prefill / paged attention 基础设施

这样就把"支持任意 block 大小"转化为"支持任意长度的 KV 序列"——后者是 paged attention 天然支持的。

## 代码架构

```
Python Layer
├── VariableBlockSparseAttentionWrapper (sparse.py)
│   ├── plan()
│   │   ├── block_row_sz → qo_indptr (Q 侧分段)
│   │   ├── _block_mask_map_to_expanded_indices() → kv_indptr, kv_indices (KV 侧稀疏索引)
│   │   └── get_batch_prefill_module() → JIT 编译/加载 kernel module
│   └── run()
│       ├── Q: [num_qo_heads, qo_len, D] → [num_kv_heads * qo_len, gqa_group_size, D]
│       ├── K/V: [num_kv_heads, kv_len, D] → [num_kv_heads * kv_len, 1, 1, D]
│       └── cached_module.paged_run(q, k, v, qo_indptr, kv_indptr, kv_indices, ...)
│
CUDA Layer (Hopper / FA3 path)
├── SparseCollectiveMainloop (sparse_mainloop.cuh)
│   ├── Q: TMA 连续加载 (SM90_TMA_LOAD)
│   ├── K/V: cp.async sparse gather (SM80_CP_ASYNC_CACHEGLOBAL_ZFILL)
│   │   ├── prefetch_kv_offset: page table → 预计算全局偏移
│   │   └── load_kv_with_gather: __shfl_sync 广播 + cp_async_zfill 加载
│   └── dense MMA 计算 (与标准 FA3 相同)
│
CUDA Layer (SM80 / FA2 path)
└── 通用 prefill kernel (prefill.cuh) + paged KV 索引
```

## 关键文件

| 文件 | 功能 | 行数 |
|---|---|---|
| [sparse.py](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/sparse.py) | Python wrapper，plan/run 两阶段 | ~520 行（L649-L1168） |
| [sparse_mainloop.cuh](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/hopper/sparse_mainloop.cuh) | Hopper sparse mainloop | 448 行 |
| [mainloop_sparse_load.cuh](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/hopper/quantization/mainloop_sparse_load.cuh) | FP8 Hopper sparse mainloop | 467 行 |
| [prefill.py](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/prefill.py) | `get_batch_prefill_module` 入口 | ~4275 行 |
| [test_block_sparse.py](https://github.com/flashinfer-ai/flashinfer/blob/main/tests/attention/test_block_sparse.py) | correctness 测试 | 304 行 |

## 后续章节

- [Python Wrapper 实现](01_python_wrapper.md)：plan/run 的完整数据流，索引展开的数值例子
- [Hopper Sparse Mainloop](02_sparse_mainloop.md)：CUDA kernel 层面的 sparse loading + dense computation
