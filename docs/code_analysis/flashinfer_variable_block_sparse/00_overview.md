---
tags:
  - CUDA
  - Flash Attention
  - Sparse Attention
---

# FlashInfer Variable Block Sparse：总览

**源码位置**: `refs/codes/flashinfer_variable_block_sparse/variable_block_attn/` · [源码索引](src/index.md)

**分析对象**: 本库同步的 `variable_block_attn/`，它不是完整 FlashInfer，而是从上游定制化剥离出来、专门服务 `VariableBlockSparseAttentionWrapper` 的最小闭环。

## 先看清边界

这一组代码的定位，和上游 `flashinfer` 仓库里那种“完整注意力内核集合”不同。

- 对外 API 仍保留 `backend="auto" | "fa2" | "fa3"`
- 内部真实可执行路径已经收敛为 `fa2`
- `fa3` 目前只保留接口边界和错误提示
- 真正的计算内核并不是“新写了一套 variable block attention kernel”
- 核心定制点是把 variable block 元数据翻译成 FA2 paged prefill 能直接消费的 ABI

这也是阅读这套代码时最容易走偏的地方。它看上去带了很多 FlashInfer JIT/runtime 文件，但真正和 variable block 语义直接绑定的部分非常少，主要集中在 [`wrapper.py`](src/wrapper_py.md) 和 [`metadata.py`](src/metadata_py.md)。

## 这套实现到底在解决什么问题

普通 block sparse attention 往往默认所有 block 的行高、列宽固定，例如 `128 x 128`。但在一些聚类、压缩或动态路由场景里，Q/KV block 实际是变长的：

- 第 0 个 query block 可能长 64
- 第 1 个 query block 可能长 192
- 第 2 个 key block 可能长 96
- 不同 KV head 还可能拥有不同稀疏图

如果沿用固定 block 方案，就只能：

- pad 到统一大小，浪费计算和带宽
- 或者截断/拆块，引入额外复杂度

`VariableBlockSparseAttentionWrapper` 的办法不是为“任意 block 大小”重新发明 kernel，而是做一个映射：

```text
variable block sparse attention
-> token-level paged KV representation
-> FlashInfer FA2 batch prefill runtime
```

换句话说，它把“变长块稀疏”问题翻译成了“page size = 1 的 paged prefill”问题。

## 一句话版执行链路

```text
用户传入 block_mask_map / block_row_sz / block_col_sz
-> Python 侧展开出 qo_indptr / kv_indptr / kv_indices
-> 选择 backend，拿到 batch prefill JIT 模块
-> plan() 产出 split-k / tile / merge 的 plan_info
-> run() 重排 q/k/v 为 paged prefill 期望布局
-> paged_run() 进入 TVM FFI binding
-> C++ 侧把 plan_info 和张量指针拼成 PagedParams
-> 复用 FlashInfer 原生 FA2 paged prefill kernel
```

这个链路里，真正和 variable block 语义绑定的，只有前半段的“元数据翻译层”。一旦 `qo_indptr / kv_indptr / kv_indices / last_page_len` 准备好，后面就已经是标准 paged prefill 路径。

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
│   ├── __init__.py
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

## 代码阅读顺序

### 1. 先读 Python 语义层

- [01 Python Wrapper 与 Metadata](01_python_wrapper_and_metadata.md)

重点是：

- wrapper 对象缓存了什么
- `plan()` 如何把 block 级描述翻译成 token 级 CSR
- 为什么 `page_size=1`
- `run()` 如何把外部张量变成 runtime ABI

### 2. 再读 runtime / JIT 层

- [02 Runtime 与 JIT](02_runtime_and_jit.md)

重点是：

- `backend="auto"` 为什么在当前环境会落到 `fa2`
- `prefill_runtime.py` 为什么说是“更薄的一层”
- `jit/attention/modules.py` 如何生成 FA2 的 batch prefill 模块
- 为什么当前实现可以说是 `fa2` only 最小闭环

### 3. 最后读 C++ binding / kernel 层

- [03 C++ Binding 与 Kernel](03_cpp_binding_and_kernel.md)

重点是：

- `plan` / `paged_run` 符号如何导出
- `plan_info` 如何在 C++ 侧恢复成 `PrefillPlanInfo`
- `PagedParams` 到底装了哪些东西
- 为什么这里没有“variable block kernel”，只有“paged prefill kernel”

### 4. 源码浏览

- [源码索引](src/index.md)

## 阅读前置

这组代码不要求你先懂 Hopper sparse mainloop，但最好先有下面几块基础：

- [CUDA 基础：执行模型与内存访问](../cuda_foundations/01_cuda_execution_model_and_memory.md)
- [CUDA 基础：CUTLASS/CuTe 编程模型](../cuda_foundations/02_cuda_cutlass_cute_programming_model.md)
- [CUDA 基础：分块、数据搬运与局部性](../cuda_foundations/04_cuda_tiling_data_movement_and_locality.md)
- [CUDA 基础：计算、带宽与算存权衡](../cuda_foundations/06_cuda_compute_memory_tradeoffs.md)
- [Flash Attention V2：Dispatch 与实例化](../flash_attention_v2/05_dispatch_and_instantiation.md)

其中最后一篇尤其有帮助，因为这套代码虽然不是 FA2 原仓代码，但它在 JIT 模板生成、编译期分发、实例化策略上的思路是同类问题。

## 核心结论

如果要把整套实现压缩成一句工程判断，那就是：

> `variable_block_attn` 的价值不在于“另起炉灶写了新 attention kernel”，而在于“把 variable block 语义稳定地翻译到了 FlashInfer FA2 paged prefill 的成熟执行路径上”。

所以阅读它时，最该关注的是：

1. block 级稀疏图如何被压平为 token 级索引
2. plan/runtime/JIT 边界如何最小化复用上游
3. 这套迁移删掉了什么、保留了什么、故意没有接什么
