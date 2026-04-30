---
tags:
  - CUDA
  - Flash Attention
  - Sparse Attention
---
# README.md

**原始文件**: `refs/codes/flashinfer_variable_block_sparse/variable_block_attn/README.md`

**解读**: [代码分析](../00_overview.md)

```markdown linenums="1"
# variable_block_attn

`variable_block_attn` 是从本仓 `projects/flashinfer` 中摘录并移植出来的
variable-block sparse attention 闭包，实现目标不是复刻完整 FlashInfer，
而是把 `VariableBlockSparseAttentionWrapper` 这条链路独立出来，并尽量保持
调用方式、元数据展开语义和 FA2 运行路径与上游一致。

当前范围：

- 保留 `VariableBlockSparseAttentionWrapper` 的 `plan` / `run` 接口
- 保留 variable-block 到 paged prefill 元数据的展开语义
- 保留 FA2 paged prefill 的 JIT / CUDA 执行路径
- 保留 `backend="auto"` 的自动检测语义
- 预留 `fa3` 接口边界，但当前未接入 Hopper runtime

当前不包含：

- decode / MLA / POD / cuDNN / TRT-LLM
- 与 variable block 无关的其它 FlashInfer wrapper
- 完整 FlashInfer 包结构

## 公开接口

```python
from variable_block_attn import VariableBlockSparseAttentionWrapper
```

构造函数：

```python
VariableBlockSparseAttentionWrapper(
    float_workspace_buffer: torch.Tensor,
    backend: str = "auto",
)
```

其中 `float_workspace_buffer` 需要是 CUDA 上的 `uint8` workspace，推荐先给
`128 * 1024 * 1024` 字节。

## 调用流程

使用方式与上游 wrapper 一样，分两步：

1. `plan(...)`
2. `run(q, k, v)`

示例：

```python
import torch

from variable_block_attn import VariableBlockSparseAttentionWrapper

device = torch.device("cuda")
num_qo_heads = 8
num_kv_heads = 2
head_dim = 128

block_mask_map = torch.tensor(
    [
        [[1, 0, 1], [1, 1, 0], [0, 1, 1]],
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
    ],
    dtype=torch.bool,
    device=device,
)
block_row_sz = torch.tensor(
    [
        [64, 128, 64],
        [64, 128, 64],
    ],
    dtype=torch.int32,
    device=device,
)
block_col_sz = torch.tensor(
    [
        [128, 64, 64],
        [128, 64, 64],
    ],
    dtype=torch.int32,
    device=device,
)

qo_len = int(block_row_sz[0].sum().item())
kv_len = int(block_col_sz[0].sum().item())

workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
wrapper = VariableBlockSparseAttentionWrapper(workspace, backend="auto")

wrapper.plan(
    block_mask_map=block_mask_map,
    block_row_sz=block_row_sz,
    block_col_sz=block_col_sz,
    num_qo_heads=num_qo_heads,
    num_kv_heads=num_kv_heads,
    head_dim=head_dim,
    causal=False,
    q_data_type=torch.float16,
    kv_data_type=torch.float16,
)

q = torch.randn(num_qo_heads, qo_len, head_dim, dtype=torch.float16, device=device)
k = torch.randn(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device=device)
v = torch.randn(num_kv_heads, kv_len, head_dim, dtype=torch.float16, device=device)

out = wrapper.run(q, k, v)
out2, lse = wrapper.run(q, k, v, return_lse=True)
```

## 输入张量语义

`plan(...)` 需要三组 variable-block 元数据：

### `block_mask_map`

形状：

```python
[num_kv_heads, num_blocks_row, num_blocks_col]
```

含义：

- 每个 `kv_head` 都可以有独立的 block-sparse mask
- `block_mask_map[h, r, c] == 1` 表示第 `h` 个 kv head 的第 `r` 个 query block
  会访问第 `c` 个 key/value block

### `block_row_sz`

形状：

```python
[num_kv_heads, num_blocks_row]
```

含义：

- 描述每个 query block 的真实 token 长度
- 当前实现沿用上游 variable-block wrapper 的行为，按 head 分组输入
- 常见使用方式是不同 head 传相同的 row sizes

### `block_col_sz`

形状：

```python
[num_kv_heads, num_blocks_col]
```

含义：

- 描述每个 key/value block 的真实 token 长度
- `block_mask_map` 中被选中的列块，会在 `plan(...)` 中展开成 token 级
  `paged_kv_indices`

## Q / K / V 形状

`run(...)` 期望：

```python
q: [num_qo_heads, qo_len, head_dim]
k: [num_kv_heads, kv_len, head_dim]
v: [num_kv_heads, kv_len, head_dim]
```

约束：

- `num_qo_heads % num_kv_heads == 0`
- `qo_len` 需要与 `block_row_sz` 的每个 head 总和一致
- `kv_len` 需要与 `block_col_sz` 的每个 head 总和一致
- 当前内部 layout 与上游一样走 `NHD`

## `backend` 语义

支持：

- `backend="auto"`
- `backend="fa2"`
- `backend="fa3"`

行为：

- `auto` 会先按设备能力和 kernel 可用性自动检测
- 在 RTX 3090 / `sm_86` 上，当前会自动解析到 `fa2`
- 如果自动检测结果或显式指定结果为 `fa3`，当前不会静默回退，而是直接报错，
  提示该移植版本尚未接通 Hopper runtime

这保证了：

- `auto` 的行为与上游判定语义一致
- 将来接入 `fa3` 时，不需要改调用接口
- 当前不会把 `fa3` 请求偷偷落回 `fa2`

## 运行要求

- Python 环境需要能导入 `torch`、`triton`、`einops`
- 需要可用 CUDA 和 NVCC，用于首次 JIT 编译
- 建议显式设置可写的 `TORCH_EXTENSIONS_DIR`

本仓常用启动方式示例：

```bash
conda run -n fta env \
  PYTHONPATH=/home/hy/triton/flash_topk_attention \
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_variable_block_attn \
  CUDA_VISIBLE_DEVICES=1 \
  python your_script.py
```

如果当前环境里 `tvm_ffi -> torch_c_dlpack_ext` 有 ABI 冲突，可额外加：

```bash
TVM_FFI_DISABLE_TORCH_C_DLPACK=1
```

## 目录说明

- `wrapper.py`: `VariableBlockSparseAttentionWrapper` 主入口
- `metadata.py`: variable-block 到 paged prefill 元数据展开
- `prefill_runtime.py`: 最小 batch prefill runtime 封装
- `jit/`: JIT 编译与模块加载闭包
- `data/csrc` 与 `data/include`: 上游 CUDA/C++/头文件摘录

## 设计边界

这套实现追求的是：

- variable-block wrapper 闭包可独立运行
- 尽量对齐上游调用方式
- 避免把与 variable-block 无关的 FlashInfer 子系统一并搬进来

如果后续要接 `fa3`，建议直接在当前接口边界上补 Hopper runtime，而不是再回退成
“重新包一层上游 flashinfer 调用”。

```
