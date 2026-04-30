---
tags:
  - CUDA
  - Flash Attention
  - Sparse Attention
---
# DEVIATIONS.md

**原始文件**: `refs/codes/flashinfer_variable_block_sparse/variable_block_attn/DEVIATIONS.md`

**解读**: [代码分析](../00_overview.md)

```markdown linenums="1"
# Deviations

Current intentional deviations from upstream FlashInfer:

1. The code is extracted into a repository-local top-level directory:
   - `variable_block_attn/`

2. The package only keeps the variable-block wrapper closure:
   - `VariableBlockSparseAttentionWrapper`
   - paged prefill planning/runtime
   - JIT pieces needed for batch prefill

3. Python structure is partially refactored for readability:
   - metadata helpers moved to `metadata.py`
   - backend resolution moved to `backend.py`
   - top-level API exposed from `__init__.py` / `api.py`

4. Import-time side effects were reduced:
   - JIT-heavy modules are imported lazily where possible
   - this avoids pulling `tvm_ffi` during a plain package import

5. `fa3` is not wired up yet:
   - `backend="auto"` still uses device-aware detection semantics
   - if detection or explicit request resolves to `fa3`, the current code raises
     a clear `NotImplementedError`

```
