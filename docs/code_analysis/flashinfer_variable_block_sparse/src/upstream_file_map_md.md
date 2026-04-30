---
tags:
  - CUDA
  - Flash Attention
  - Sparse Attention
---
# UPSTREAM_FILE_MAP.md

**原始文件**: `refs/codes/flashinfer_variable_block_sparse/variable_block_attn/UPSTREAM_FILE_MAP.md`

**解读**: [代码分析](../00_overview.md)

```markdown linenums="1"
# Upstream File Map

Primary upstream source root:

- `projects/flashinfer/flashinfer`
- `projects/flashinfer/csrc`
- `projects/flashinfer/include/flashinfer`

Current local mapping:

- `projects/flashinfer/flashinfer/sparse.py` -> `variable_block_attn/wrapper.py`
- `projects/flashinfer/flashinfer/prefill.py` -> `variable_block_attn/prefill_runtime.py`
- `projects/flashinfer/flashinfer/utils.py` -> `variable_block_attn/common.py`
- `projects/flashinfer/flashinfer/api_logging.py` -> `variable_block_attn/api_logging.py`
- `projects/flashinfer/flashinfer/version.py` -> `variable_block_attn/version.py`
- `projects/flashinfer/flashinfer/compilation_context.py` -> `variable_block_attn/compilation_context.py`
- `projects/flashinfer/flashinfer/jit/env.py` -> `variable_block_attn/jit/env.py`
- `projects/flashinfer/flashinfer/jit/core.py` -> `variable_block_attn/jit/core.py`
- `projects/flashinfer/flashinfer/jit/cpp_ext.py` -> `variable_block_attn/jit/cpp_ext.py`
- `projects/flashinfer/flashinfer/jit/cubin_loader.py` -> `variable_block_attn/jit/cubin_loader.py`
- `projects/flashinfer/flashinfer/jit/utils.py` -> `variable_block_attn/jit/utils.py`
- `projects/flashinfer/flashinfer/jit/spdlog.py` -> `variable_block_attn/jit/spdlog.py`
- `projects/flashinfer/flashinfer/jit/attention/modules.py` -> `variable_block_attn/jit/attention/modules.py`
- `projects/flashinfer/flashinfer/jit/attention/utils.py` -> `variable_block_attn/jit/attention/utils.py`
- `projects/flashinfer/flashinfer/jit/attention/variants.py` -> `variable_block_attn/jit/attention/variants.py`
- `projects/flashinfer/flashinfer/jit/attention/fmha_v2/*` -> `variable_block_attn/jit/attention/fmha_v2/*`
- `projects/flashinfer/csrc/batch_prefill*` -> `variable_block_attn/data/csrc/batch_prefill*`
- `projects/flashinfer/csrc/logging.cc` -> `variable_block_attn/data/csrc/logging.cc`
- `projects/flashinfer/csrc/tvm_ffi_utils.h` -> `variable_block_attn/data/csrc/tvm_ffi_utils.h`
- `projects/flashinfer/include/flashinfer/**` -> `variable_block_attn/data/include/flashinfer/**`

Local-only files added during migration:

- `variable_block_attn/backend.py`
- `variable_block_attn/metadata.py`
- `variable_block_attn/api.py`
- `variable_block_attn/README.md`
- `variable_block_attn/UPSTREAM_FILE_MAP.md`
- `variable_block_attn/DEVIATIONS.md`

```
