---
tags:
  - CUDA
  - Flash Attention
  - Sparse Attention
---

# FlashInfer Variable Block Sparse：总览

**源码仓库**: `refs/codes/flashinfer_variable_block_sparse/variable_block_attn`

**分析范围**: [`wrapper.py:L41-L397`](src/wrapper_py.md#__codelineno-0-41)、[`metadata.py:L22-L88`](src/metadata_py.md#__codelineno-0-22)、[`prefill_runtime.py:L32-L108`](src/prefill_runtime_py.md#__codelineno-0-32)、[`jit/attention/modules.py:L372-L1580`](src/jit_attention_modules_py.md#__codelineno-0-372)、[`data/csrc/batch_prefill_jit_binding.cu:L22-L50`](src/batch_prefill_jit_binding_cu.md#__codelineno-0-22)、[`data/csrc/batch_prefill.cu:L47-L334`](src/batch_prefill_cu.md#__codelineno-0-47)

这组代码不是完整 FlashInfer，也不是一套重新实现的 variable-block attention kernel。它是从上游剥离出的一个最小闭环，目标只有一个：让 `VariableBlockSparseAttentionWrapper` 能把 variable block 稀疏描述翻译成 FlashInfer 已有的 FA2 paged prefill 执行路径。

## 核心判断

- 对外接口仍保留 `backend="auto" | "fa2" | "fa3"`，但当前本地实现内部只真正接通了 `fa2`
- 这套代码最重要的部分不是 CUDA kernel，而是 Python 侧的 metadata 翻译层
- variable block 语义在进入 C++/CUDA 前就已经被压平成 token-level paged KV 索引
- 底层执行的仍是 FlashInfer 原生的 batch prefill plan + paged_run + kernel dispatch 体系

如果只记一句话，可以记成：

> `variable_block_attn` 的价值不在“写了新 kernel”，而在“把 variable block 稀疏图稳定翻译到 FA2 paged prefill ABI 上”。

## 核心文件

- [wrapper.py:L41-L397](src/wrapper_py.md#__codelineno-0-41)：公开入口 `VariableBlockSparseAttentionWrapper`，负责 `plan()` / `run()` 生命周期
- [metadata.py:L22-L88](src/metadata_py.md#__codelineno-0-22)：把 block 级稀疏图展开成 token 级 `qo_indptr / kv_indptr / kv_indices`
- [prefill_runtime.py:L32-L108](src/prefill_runtime_py.md#__codelineno-0-32)：当前最薄的 runtime 封装，只保留 `plan` 和 `paged_run`
- [backend.py:L28-L50](src/backend_py.md#__codelineno-0-28)：`auto/fa2/fa3` 边界，明确拒绝未接通的 `fa3`
- [jit/attention/modules.py:L956-L1580](src/jit_attention_modules_py.md#__codelineno-0-956)：batch prefill 模块规格编码、模板渲染与实例生成
- [data/csrc/batch_prefill_jit_binding.cu:L22-L50](src/batch_prefill_jit_binding_cu.md#__codelineno-0-22)：TVM FFI 导出 `plan/ragged_run/paged_run`
- [data/csrc/batch_prefill.cu:L47-L334](src/batch_prefill_cu.md#__codelineno-0-47)：`PrefillPlanInfo`、`PagedParams`、workspace 偏移恢复与最终 dispatch

## 执行链路

```text
block_mask_map / block_row_sz / block_col_sz
  -> build_qo_indptr()
  -> build_last_page_len()
  -> block_mask_map_to_expanded_indices()
  -> resolve_attention_backend()
  -> get_batch_prefill_module()
  -> module.plan(...)
  -> run() 中重排 q / k / v
  -> module.paged_run(...)
  -> BatchPrefillWithPagedKVCacheRun(...)
  -> FlashInfer FA2 paged prefill kernel
```

这里真正“新”的部分只到 `block_mask_map_to_expanded_indices()` 为止。后面的 plan、JIT、binding、kernel 都是在复用 FlashInfer 原有的 paged prefill 基础设施。

## 代码结构

```text
variable_block_attn/
├── __init__.py / api.py
├── wrapper.py
├── metadata.py
├── backend.py
├── prefill_runtime.py
├── common.py
├── jit/
│   ├── core.py / cpp_ext.py / env.py / utils.py
│   └── attention/
│       ├── modules.py
│       ├── utils.py
│       └── variants.py
├── data/csrc/
│   ├── batch_prefill_jit_binding.cu
│   ├── batch_prefill.cu
│   ├── batch_prefill_customize_config.jinja
│   ├── batch_prefill_paged_kernel_inst.jinja
│   └── batch_prefill_ragged_kernel_inst.jinja
└── data/include/flashinfer/**
```

可以把它压成三层：

- Python 语义层：variable block 元数据翻译
- runtime/JIT 层：把规格编码成可缓存、可编译的 batch prefill 模块
- C++/CUDA 层：恢复 `plan_info`、拼装 `PagedParams`、进入标准 paged prefill kernel

## 文档导航

| 文档 | 内容 |
|------|------|
| [01 Python Wrapper 与 Metadata](01_python_wrapper_and_metadata.md) | `wrapper.py` / `metadata.py`：`plan()`、`run()`、`page_size=1` 与 token 级 CSR 展开 |
| [02 Runtime 与 JIT](02_runtime_and_jit.md) | `prefill_runtime.py`、backend 选择、URI、batch prefill 模块生成 |
| [03 C++ Binding 与 Kernel](03_cpp_binding_and_kernel.md) | `plan_info` 序列化/恢复、`PagedParams` 装配、workspace 偏移与最终 dispatch |
| [源码浏览：wrapper.py / metadata.py / batch_prefill.cu](src/index.md) | `variable_block_attn` 关键 Python、JIT、C++/CUDA 文件的源码页索引 |

## 源码浏览

| 文件 | 行数 | 源码页 |
|------|------|--------|
| wrapper.py | 397 | [浏览](src/wrapper_py.md) |
| metadata.py | 88 | [浏览](src/metadata_py.md) |
| backend.py | 51 | [浏览](src/backend_py.md) |
| prefill_runtime.py | 108 | [浏览](src/prefill_runtime_py.md) |
| common.py | 1236 | [浏览](src/common_py.md) |
| jit/attention/modules.py | 1888 | [浏览](src/jit_attention_modules_py.md) |
| data/csrc/batch_prefill_jit_binding.cu | 50 | [浏览](src/batch_prefill_jit_binding_cu.md) |
| data/csrc/batch_prefill.cu | 334 | [浏览](src/batch_prefill_cu.md) |

完整索引见 [src/index.md](src/index.md)。

## 前置知识与关联笔记

阅读这组代码时，最常需要回看的不是 Hopper 新指令，而是下面这些基础页：

- [CUDA 基础：执行模型与内存访问](../cuda_foundations/01_cuda_execution_model_and_memory.md)
- [CUDA 基础：CUTLASS/CuTe 编程模型](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)
- [CUDA 基础：分块、数据搬运与局部性](../cuda_foundations/04_cuda_tiling_data_movement_and_locality.md)
- [CUDA 基础：计算、带宽与算存权衡](../cuda_foundations/06_cuda_compute_memory_tradeoffs.md)
- [Flash Attention V2：调度与实例化](../flash_attention_v2/05_dispatch_and_instantiation.md)

最后一篇尤其重要，因为 `variable_block_attn` 虽然不是 FA2 原仓源码，但它在“模板规格编码 -> JIT 生成 -> 编译期实例 + 运行期 dispatch”这条链路上和 FA2 是同类问题。

## 阅读抓手

读后面三篇时，可以始终盯住三个问题：

1. variable block 稀疏图是怎样被压平成 token 级 CSR / paged KV 索引的？
2. 为什么当前实现能说“接口保留 auto/fa2/fa3，内部只保留 fa2 最小闭环”？
3. C++/CUDA 层到底有没有 variable block 专属 kernel，还是只是在复用 paged prefill kernel？

后面的拆解基本都在回答这三个问题。
