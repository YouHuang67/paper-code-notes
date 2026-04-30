---
tags:
  - CUDA
  - Flash Attention
  - Sparse Attention
---
# wrapper.py

**原始文件**: `refs/codes/flashinfer_variable_block_sparse/variable_block_attn/wrapper.py`

**解读**: [代码分析](../00_overview.md)

```python linenums="1"
"""
Copyright (c) 2024 by FlashInfer team.

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

import math
from typing import Optional, Tuple, Union

import torch

from .api_logging import flashinfer_api
from .backend import resolve_attention_backend
from .common import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_pos_encoding_mode,
    canonicalize_torch_dtype,
    check_shape_dtype_device,
    device_support_pdl,
)
from .metadata import (
    block_mask_map_to_expanded_indices,
    build_last_page_len,
    build_qo_indptr,
)
from .prefill_runtime import get_batch_prefill_module


class VariableBlockSparseAttentionWrapper:
    r"""Wrapper class for attention computation with a block-sparse matrix as attention mask.
    This API supports variable block sizes provided by ``block_row_sz`` and ``block_col_sz``.
    Besides, each ``kv_head_idx`` can specify its own sparse patterns without using the same mask.

    Example
    -------
    >>> import torch
    >>> from variable_block_attn import VariableBlockSparseAttentionWrapper
    >>> num_qo_heads = 1
    >>> num_kv_heads = 1
    >>> head_dim = 128
    >>> seq_len = 6
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> wrapper = VariableBlockSparseAttentionWrapper(workspace_buffer)
    >>> block_mask_map = torch.tensor([[[0, 0, 1], [1, 0, 1], [0, 1, 1]]], dtype=torch.bool, device="cuda:0")
    >>> block_row_sz = torch.tensor([[1, 2, 3]], dtype=torch.int32, device="cuda:0")
    >>> block_col_sz = torch.tensor([[3, 1, 2]], dtype=torch.int32, device="cuda:0")
    >>> wrapper.plan(
    ...     block_mask_map,
    ...     block_row_sz,
    ...     block_col_sz,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ... )
    >>> q = torch.randn((num_qo_heads, seq_len, head_dim), dtype=torch.float16, device="cuda:0")
    >>> k = torch.randn((num_kv_heads, seq_len, head_dim), dtype=torch.float16, device="cuda:0")
    >>> v = torch.randn((num_kv_heads, seq_len, head_dim), dtype=torch.float16, device="cuda:0")
    >>> o = wrapper.run(q, k, v)
    """

    @flashinfer_api
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        backend: str = "auto",
    ) -> None:
        r"""Constructs of :class:`VariableBlockSparseAttentionWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.
        backend : str
            The implementation backend, could be ``auto``/``fa2`` or ``fa3``. Defaults to ``auto``.
            If set to ``auto``, the function will automatically choose the backend based on the
            device architecture and kernel availability.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._workspace_size = (
            float_workspace_buffer.numel() * float_workspace_buffer.element_size()
        )
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
        )
        self._kv_layout = "NHD"
        self._qo_indptr: Optional[torch.Tensor] = None
        self._paged_kv_indptr_buf: Optional[torch.Tensor] = None
        self._paged_kv_indices_buf: Optional[torch.Tensor] = None
        self._paged_kv_last_page_len: Optional[torch.Tensor] = None
        self._backend = backend

    def reset_workspace_buffer(
        self,
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
    ) -> None:
        r"""Reset the workspace buffer."""
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._workspace_size = (
            float_workspace_buffer.numel() * float_workspace_buffer.element_size()
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            pin_memory=True,
        )

    @flashinfer_api
    def plan(
        self,
        block_mask_map: torch.Tensor,
        block_row_sz: torch.Tensor,
        block_col_sz: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        non_blocking: bool = True,
        q_data_type: Union[str, torch.dtype] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        r"""Create auxiliary data structures for variable block sparse attention."""
        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)
        self._o_dtype = q_data_type

        if logits_soft_cap is None:
            logits_soft_cap = 0.0

        num_blocks_row = block_row_sz.shape[-1]
        num_blocks_col = block_col_sz.shape[-1]

        qo_indptr = build_qo_indptr(block_row_sz)
        qo_indptr_host = qo_indptr.to("cpu", non_blocking=non_blocking)
        last_block_len = build_last_page_len(
            block_mask_map,
            num_blocks_row,
            num_kv_heads,
        )

        kv_indptr, kv_indices = block_mask_map_to_expanded_indices(
            block_mask_map,
            block_col_sz,
        )
        kv_indptr_host = kv_indptr.to("cpu", non_blocking=non_blocking)
        kv_indices_host = kv_indices.to("cpu", non_blocking=non_blocking)

        self._qo_indptr = qo_indptr.to(self.device, non_blocking=non_blocking)
        self._paged_kv_indptr_buf = kv_indptr.to(self.device, non_blocking=non_blocking)
        self._paged_kv_indices_buf = kv_indices.to(
            self.device,
            non_blocking=non_blocking,
        )
        self._paged_kv_last_page_len = last_block_len.to(
            self.device,
            non_blocking=non_blocking,
        )
        torch.cuda.synchronize()
        self._mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value

        assert num_qo_heads % num_kv_heads == 0, (
            "num_qo_heads must be a multiple of num_kv_heads"
        )
        assert num_blocks_row * num_kv_heads + 1 == kv_indptr_host.shape[0]
        assert kv_indptr_host[-1].item() == kv_indices_host.shape[0], (
            f"{kv_indptr_host[-1].item()} != {kv_indices_host.shape[0]}"
        )
        assert num_kv_heads == block_mask_map.shape[0]
        assert num_kv_heads == block_row_sz.shape[0]
        assert num_kv_heads == block_col_sz.shape[0]
        assert num_blocks_row == block_mask_map.shape[1]
        assert num_blocks_col == block_mask_map.shape[2]

        self._backend = resolve_attention_backend(
            self._backend,
            self.device,
            PosEncodingMode[pos_encoding_mode].value,
            use_fp16_qk_reduction,
            self._mask_mode == MaskMode.CUSTOM.value,
            q_data_type,
            kv_data_type,
        )

        get_module_args = (
            q_data_type,
            kv_data_type,
            self._o_dtype,
            kv_indptr_host.dtype,
            head_dim,
            head_dim,
            PosEncodingMode[pos_encoding_mode].value,
            False,
            logits_soft_cap > 0,
            use_fp16_qk_reduction,
        )
        self._cached_module = get_batch_prefill_module(self._backend, *get_module_args)

        kv_lens_arr_host = kv_indptr_host[1:] - kv_indptr_host[:-1]

        args = [
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_host,
            kv_indptr_host,
            kv_lens_arr_host,
            qo_indptr_host[-1].item(),
            num_blocks_row * num_kv_heads,
            num_qo_heads // num_kv_heads,
            1,
            1,
            False,
            head_dim,
            head_dim,
            causal,
            -1,
        ]
        if self._backend == "fa2":
            args.append(-1)
            args.append(False)
            args.append(0)
        self._plan_info = self._cached_module.plan(*args)

        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        self._num_kv_heads = num_kv_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Warning: This method is deprecated, please use :meth:`run` instead."""
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(q, k, v)

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute variable block sparse attention between Q/K/V tensors."""
        import einops

        if enable_pdl is None:
            enable_pdl = device_support_pdl(q.device)

        pos_encoding_mode = self._pos_encoding_mode
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        _check_pos_encoding_mode(pos_encoding_mode)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4

        q = einops.rearrange(
            q,
            "(num_kv_heads gqa_group_size) qo_len head_dim -> (num_kv_heads qo_len) gqa_group_size head_dim",
            num_kv_heads=self._num_kv_heads,
        ).contiguous()
        k = einops.rearrange(
            k,
            "num_kv_heads kv_len head_dim -> (num_kv_heads kv_len) 1 1 head_dim",
        ).contiguous()
        v = einops.rearrange(
            v,
            "num_kv_heads kv_len head_dim -> (num_kv_heads kv_len) 1 1 head_dim",
        ).contiguous()

        if return_lse:
            if lse is None:
                lse = torch.empty(
                    (q.size(0), q.size(1)),
                    dtype=torch.float32,
                    device=q.device,
                )
            else:
                check_shape_dtype_device(
                    lse,
                    (q.size(0), q.size(1)),
                    torch.float32,
                    q.device,
                    "lse",
                )

        if out is None:
            out = torch.empty_like(q, dtype=self._o_dtype)
        else:
            check_shape_dtype_device(out, q.shape, self._o_dtype, q.device, "out")

        self._cached_module.paged_run(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q,
            k,
            v,
            self._qo_indptr,
            self._paged_kv_indptr_buf,
            self._paged_kv_indices_buf,
            self._paged_kv_last_page_len,
            out,
            lse,
            self._mask_mode,
            TensorLayout[self._kv_layout].value,
            -1,
            enable_pdl,
            None,
            None,
            None,
            None,
            None,
            None,
            logits_soft_cap,
            sm_scale,
            None,
            None,
            None,
            rope_scale,
            rope_theta,
            0,
            self._workspace_size,
        )

        out = einops.rearrange(
            out,
            "(num_kv_heads qo_len) gqa_group_size head_dim -> (num_kv_heads gqa_group_size) qo_len head_dim",
            num_kv_heads=self._num_kv_heads,
        ).contiguous()

        if return_lse:
            lse = einops.rearrange(
                lse,
                "(num_kv_heads qo_len) gqa_group_size -> (num_kv_heads gqa_group_size) qo_len",
                num_kv_heads=self._num_kv_heads,
            ).contiguous()

        return (out, lse) if return_lse else out

```
