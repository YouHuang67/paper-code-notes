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

import torch

from .common import determine_attention_backend


_FA3_NOT_SUPPORTED_MSG = (
    "The fa3 backend is detected or requested, but this variable_block_attn "
    "migration has not wired up the Hopper runtime yet."
)


def resolve_attention_backend(
    backend: str,
    device: torch.device,
    pos_encoding_mode: int,
    use_fp16_qk_reduction: bool,
    use_custom_mask: bool,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
) -> str:
    if backend == "auto":
        backend = determine_attention_backend(
            device,
            pos_encoding_mode,
            use_fp16_qk_reduction,
            use_custom_mask,
            q_data_type,
            kv_data_type,
        )

    if backend == "fa3":
        raise NotImplementedError(_FA3_NOT_SUPPORTED_MSG)
    if backend != "fa2":
        raise ValueError(f"Unsupported backend: {backend}")
    return backend
