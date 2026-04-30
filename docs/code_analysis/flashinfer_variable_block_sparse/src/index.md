---
tags:
  - CUDA
  - Flash Attention
  - Sparse Attention
---

# FlashInfer Variable Block Sparse 源码

**本库路径**: `refs/codes/flashinfer_variable_block_sparse/variable_block_attn/` · **解读**: [代码分析](../00_overview.md)

## 包入口与迁移说明

| 文件 | 行数 | 说明 |
|------|------|------|
| [README.md](readme_md.md) | 233 | 当前迁移范围、调用方式、边界说明 |
| [DEVIATIONS.md](deviations_md.md) | 25 | 相对上游 FlashInfer 的主动偏离项 |
| [UPSTREAM_FILE_MAP.md](upstream_file_map_md.md) | 39 | 上游文件到本地迁移路径映射 |
| [__init__.py](__init___py.md) | 3 | 顶层导出 |
| [api.py](api_py.md) | 3 | 同样只导出 wrapper |

## Python 主链路

| 文件 | 行数 | 说明 |
|------|------|------|
| [wrapper.py](wrapper_py.md) | 397 | `VariableBlockSparseAttentionWrapper` 主入口，含 `plan()` / `run()` |
| [metadata.py](metadata_py.md) | 88 | variable block 到 token 级 paged prefill 索引展开 |
| [backend.py](backend_py.md) | 51 | `auto/fa2/fa3` 边界解析 |
| [prefill_runtime.py](prefill_runtime_py.md) | 108 | 最小 batch prefill runtime 封装 |
| [common.py](common_py.md) | 1236 | dtype/layout/backend/tooling 共用基础设施 |

## JIT 基础设施

| 文件 | 行数 | 说明 |
|------|------|------|
| [jit/__init__.py](jit___init___py.md) | 30 | JIT 入口导出 |
| [jit/core.py](jit_core_py.md) | 518 | JIT spec、构建与加载主逻辑 |
| [jit/cpp_ext.py](jit_cpp_ext_py.md) | 344 | C++ 扩展构建与编译工具 |
| [jit/env.py](jit_env_py.md) | 176 | JIT 目录、缓存与环境配置 |
| [jit/utils.py](jit_utils_py.md) | 83 | 通用 JIT 辅助函数 |

## Attention 模块生成

| 文件 | 行数 | 说明 |
|------|------|------|
| [jit/attention/modules.py](jit_attention_modules_py.md) | 1888 | URI 生成、模板参数装配、batch prefill 模块生成 |
| [jit/attention/utils.py](jit_attention_utils_py.md) | 90 | attention JIT 辅助逻辑 |
| [jit/attention/variants.py](jit_attention_variants_py.md) | 171 | attention 变体定义 |

## C++ / CUDA / 模板

| 文件 | 行数 | 说明 |
|------|------|------|
| [data/csrc/batch_prefill_jit_binding.cu](batch_prefill_jit_binding_cu.md) | 50 | TVM FFI 导出 `plan/ragged_run/paged_run` |
| [data/csrc/batch_prefill.cu](batch_prefill_cu.md) | 334 | `PrefillPlanInfo` / `PagedParams` / dispatch 主体 |
| [data/csrc/batch_prefill_customize_config.jinja](batch_prefill_customize_config_jinja.md) | 121 | JIT 配置模板 |
| [data/csrc/batch_prefill_paged_kernel_inst.jinja](batch_prefill_paged_kernel_inst_jinja.md) | 14 | paged kernel 实例化模板 |
| [data/csrc/batch_prefill_ragged_kernel_inst.jinja](batch_prefill_ragged_kernel_inst_jinja.md) | 14 | ragged kernel 实例化模板 |
| [data/csrc/tvm_ffi_utils.h](tvm_ffi_utils_h.md) | 318 | TVM FFI 张量视图与导出辅助 |

