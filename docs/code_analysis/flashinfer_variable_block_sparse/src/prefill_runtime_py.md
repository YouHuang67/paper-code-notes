---
tags:
  - CUDA
  - Flash Attention
  - Sparse Attention
---
# prefill_runtime.py

**原始文件**: `refs/codes/flashinfer_variable_block_sparse/variable_block_attn/prefill_runtime.py`

**解读**: [代码分析](../00_overview.md)

```python linenums="1"
"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
from types import SimpleNamespace
from typing import List, Optional

import torch

from .common import is_float8


_FA3_NOT_SUPPORTED_MSG = (
    "The fa3 backend is detected or requested, but this variable_block_attn "
    "migration has not wired up the Hopper runtime yet."
)


@functools.cache
def get_batch_prefill_module(backend, *args):
    from .jit import gen_batch_prefill_module

    if backend == "fa3":
        raise NotImplementedError(_FA3_NOT_SUPPORTED_MSG)

    module = gen_batch_prefill_module(backend, *args).build_and_load()
    paged_run_func = module.paged_run

    def paged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        paged_k_cache: torch.Tensor,
        paged_v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        enable_pdl: bool,
        maybe_custom_mask: Optional[torch.Tensor],
        maybe_mask_indptr: Optional[torch.Tensor],
        maybe_alibi_slopes: Optional[torch.Tensor],
        maybe_prefix_len_ptr: Optional[torch.Tensor],
        maybe_token_pos_in_items_ptr: Optional[torch.Tensor],
        maybe_max_item_len_ptr: Optional[torch.Tensor],
        logits_soft_cap: float,
        sm_scale: float,
        scale_q: Optional[torch.Tensor],
        scale_k: Optional[torch.Tensor],
        scale_v: Optional[torch.Tensor],
        rope_scale: float,
        rope_theta: float,
        token_pos_in_items_len: int,
        workspace_size: int,
    ) -> None:
        assert backend == "fa2"
        assert not is_float8(q)
        paged_run_func(
            float_workspace_buffer,
            int_workspace_buffer,
            plan_info_vec,
            q,
            paged_k_cache,
            paged_v_cache,
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            o,
            maybe_lse,
            mask_mode,
            layout,
            window_left,
            enable_pdl,
            maybe_custom_mask,
            maybe_mask_indptr,
            maybe_alibi_slopes,
            maybe_prefix_len_ptr,
            maybe_token_pos_in_items_ptr,
            maybe_max_item_len_ptr,
            logits_soft_cap,
            sm_scale,
            1.0 / rope_scale,
            1.0 / rope_theta,
            token_pos_in_items_len,
        )
        return o

    return SimpleNamespace(plan=module.plan, paged_run=paged_run)

```
