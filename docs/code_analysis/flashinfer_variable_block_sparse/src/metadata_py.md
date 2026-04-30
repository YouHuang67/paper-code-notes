---
tags:
  - CUDA
  - Flash Attention
  - Sparse Attention
---
# metadata.py

**原始文件**: `refs/codes/flashinfer_variable_block_sparse/variable_block_attn/metadata.py`

**解读**: [代码分析](../00_overview.md)

```python linenums="1"
"""
Copyright (c) 2026 by variable_block_attn authors.

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

from typing import Tuple

import torch


def build_qo_indptr(block_row_sz: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=block_row_sz.device),
            torch.cumsum(block_row_sz.flatten(), dim=0, dtype=torch.int32),
        ],
        dim=0,
    )


def build_last_page_len(
    block_mask_map: torch.Tensor,
    num_blocks_row: int,
    num_kv_heads: int,
) -> torch.Tensor:
    return torch.full(
        (num_blocks_row * num_kv_heads,),
        1,
        dtype=torch.int32,
        device=block_mask_map.device,
    )


def block_mask_map_to_expanded_indices(
    block_mask_map: torch.Tensor,  # [H, R, C] bool / {0,1}
    block_col_sz: torch.Tensor,  # [H, C]     int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        block_mask_map:  bool/int  [num_kv_heads, num_blocks_row, num_blocks_col]
        block_col_sz:    int32/64  [num_kv_heads, num_blocks_col]
    Returns:
        kv_indptr:  [H*R + 1]  int32  —  CSR indptr
        kv_indices: [nnz]      int32  —  token indices per (head, row)
    """
    device = block_mask_map.device
    dtype_i = torch.int32

    # 1) Calculate the total length of each row (head, row)
    row_lengths = (block_mask_map * block_col_sz[:, None, :]).sum(-1)  # [H,R]
    kv_indptr = torch.cat(
        [
            torch.zeros(1, dtype=dtype_i, device=device),
            torch.cumsum(row_lengths.flatten(), 0),
        ],
        dim=0,
    )

    # 2) Calculate the offset of each column block within its head
    col_offset = torch.cumsum(block_col_sz.to(dtype_i), 1) - block_col_sz  # [H,C]
    head_len = block_col_sz.sum(1, dtype=dtype_i)
    head_offset = torch.cumsum(head_len, 0) - head_len

    # 3) Find all selected (h,r,c)
    h_idx, _, c_idx = block_mask_map.nonzero(as_tuple=True)
    lengths = block_col_sz[h_idx, c_idx].to(dtype_i)  # [N]
    base = head_offset[h_idx] + col_offset[h_idx, c_idx]  # [N]

    # 4) Expand variable-length column blocks into token-level indices
    cum = torch.cumsum(lengths, 0)
    starts = torch.repeat_interleave(cum - lengths, lengths)  # [total]
    offsets_within = torch.arange(cum[-1], device=device) - starts
    kv_indices = torch.repeat_interleave(base, lengths) + offsets_within

    return kv_indptr.to(dtype=dtype_i, device=device), kv_indices.to(
        dtype=dtype_i, device=device
    )

```
