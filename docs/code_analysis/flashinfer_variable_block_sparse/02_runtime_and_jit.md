---
tags:
  - CUDA
  - Flash Attention
  - Sparse Attention
---

# plan 阶段：模块准备与调度生成

**源码**:

- [prefill_runtime.py:L32-L108](src/prefill_runtime_py.md#__codelineno-0-32)
- [backend.py:L28-L50](src/backend_py.md#__codelineno-0-28)
- [common.py:L459-L501](src/common_py.md#__codelineno-0-459)
- [jit/attention/modules.py:L372-L396](src/jit_attention_modules_py.md#__codelineno-0-372)
- [jit/attention/modules.py:L956-L1042](src/jit_attention_modules_py.md#__codelineno-0-956)
- [jit/attention/modules.py:L1473-L1580](src/jit_attention_modules_py.md#__codelineno-0-1473)

这一章只讲 `plan()` 后半段真正发生的事情，也就是：

1. Python 侧怎样拿到 batch prefill 模块
2. JIT 怎样把规格编码成可编译模块
3. `module.plan(...)` 最终怎样落到 C++ `BatchPrefillWithKVCachePlan(...)`

这正好对应代码执行顺序里“metadata 已经准备好，接下来开始生成执行模块和调度信息”的阶段。

## `get_batch_prefill_module()`：先拿到可执行模块

当前 runtime 主入口只有 [`prefill_runtime.py:L32-L108`](src/prefill_runtime_py.md#__codelineno-0-32)：

```python
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
            ...,
            logits_soft_cap,
            sm_scale,
            1.0 / rope_scale,
            1.0 / rope_theta,
            token_pos_in_items_len,
        )
        return o

    return SimpleNamespace(plan=module.plan, paged_run=paged_run)
```

这段代码很能说明当前迁移的定位：

- runtime 不再维护一大组 prefill 入口矩阵
- `fa3` 明确拒绝，不做半接线状态的伪支持
- 对 wrapper 来说，导出面只剩下 `plan` 和 `paged_run`

所以它确实是“面向 variable-block wrapper 的 batch prefill runtime 薄封装”。

## 这一阶段为什么只会落到 `fa2`

外层边界在 [`backend.py:L28-L50`](src/backend_py.md#__codelineno-0-28)：

```python
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
```

而 [`common.py:L459-L501`](src/common_py.md#__codelineno-0-459) 里的检测逻辑仍然保留：

```python
def determine_attention_backend(
    device: torch.device,
    pos_encoding_mode: int,
    use_fp16_qk_reductions: bool,
    use_custom_mask: bool,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
) -> str:
    if is_sm90a_supported(device) and is_fa3_backend_supported(
        pos_encoding_mode,
        use_fp16_qk_reductions,
        use_custom_mask,
        dtype_q,
        dtype_kv,
    ):
        return "fa3"
    else:
        return "fa2"
```

这说明当前实现刻意保留了两层语义：

- API 层仍与上游对齐，可以说 `auto/fa2/fa3`
- 执行层则只允许真正打通的 `fa2`

这么做的好处很工程化：

- 调用端接口不用因为迁移而改写
- 将来若要接回 `fa3`，边界仍然清楚
- 当前不会把没打通的能力伪装成“已支持”

## URI：把这次 `plan()` 所需规格编码成模块身份

JIT 的第一步是把“规格”编码进 URI。见 [`jit/attention/modules.py:L372-L396`](src/jit_attention_modules_py.md#__codelineno-0-372)：

```python
def get_batch_prefill_uri(
    backend: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
) -> str:
    return (
        f"batch_prefill_with_kv_cache_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"dtype_idx_{filename_safe_dtype_map[dtype_idx]}_"
        f"head_dim_qk_{head_dim_qk}_"
        f"head_dim_vo_{head_dim_vo}_"
        f"posenc_{pos_encoding_mode}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}_"
        f"f16qk_{use_fp16_qk_reduction}" + ("_sm90" if backend == "fa3" else "")
    )
```

它本质上是在做 JIT 缓存 key：

- dtype 组合不同，URI 不同
- head dim 不同，URI 不同
- pos encoding / soft cap / fp16-qk reduction 不同，URI 不同

因此 URI 不是装饰，它是在把“模板实例化规格”编码成可缓存、可编译、可复用的模块身份。

## `gen_batch_prefill_module()`：按规格选择 batch prefill 模块形态

wrapper 最终调用的是 [`jit/attention/modules.py:L956-L1042`](src/jit_attention_modules_py.md#__codelineno-0-956)：

```python
def gen_batch_prefill_module(
    backend: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
) -> JitSpec:
    uri = get_batch_prefill_uri(...)
    fp8_enabled = dtype_q in [torch.float8_e4m3fn, torch.float8_e5m2]

    assert backend in ["fa2", "fa3"]
    assert dtype_o not in [torch.float8_e4m3fn, torch.float8_e5m2]

    if backend == "fa2":
        assert not fp8_enabled, "fp8 tensor core is not supported in fa2 backend"
        additional_tensor_names = [
            "maybe_custom_mask",
            "maybe_mask_indptr",
            "maybe_alibi_slopes",
            "maybe_prefix_len_ptr",
            "maybe_token_pos_in_items_ptr",
            "maybe_max_item_len_ptr",
        ]
        additional_tensor_dtypes = [
            "uint8_t",
            "int32_t",
            "float",
            "uint32_t",
            "uint16_t",
            "uint16_t",
        ]
        additional_scalar_names = [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
            "token_pos_in_items_len",
        ]
        additional_scalar_dtypes = ["double", "double", "double", "double", "int64_t"]
        variant_name = (
            f"DefaultAttention<use_custom_mask, {str(use_sliding_window).lower()}, "
            f"{str(use_logits_soft_cap).lower()}, {str(pos_encoding_mode == 2).lower()}>"
        )
        variant_decl = "#include<flashinfer/attention/variants.cuh>"
    else:
        ...

    return gen_customize_batch_prefill_module(...)
```

虽然这里还保留了 `fa3` 分支的壳，但对当前 `variable_block_attn` 闭环来说，真正重要的是：

- wrapper 经过 `resolve_attention_backend()` 之后只会允许 `fa2`
- `fa2` 下直接排除了 fp8 tensor core 路径
- additional tensor/scalar 参数仍保持与上游 batch prefill ABI 一致

这意味着本地版本不是重新发明 JIT 逻辑，而是在复用上游 batch prefill 的参数化模板体系。

## `gen_customize_batch_prefill_module()`：把规格落成可编译源码

真正落到磁盘的是 [`jit/attention/modules.py:L1473-L1580`](src/jit_attention_modules_py.md#__codelineno-0-1473)：

```python
def gen_customize_batch_prefill_module(
    backend: str,
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    idtype: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    additional_tensor_names: List[str],
    additional_tensor_dtypes: List[str],
    additional_scalar_names: List[str],
    additional_scalar_dtypes: List[str],
    variant_name: str,
    variant_decl: str,
    ...
) -> JitSpec:
    if backend != "fa2":
        raise NotImplementedError(
            "variable_block_attn only keeps the fa2 batch prefill codegen path; "
            f"got backend={backend!r}"
        )

    kwargs = {
        "variant_decl": variant_decl,
        "variant_name": variant_name,
        "dtype_q": dtype_map[dtype_q],
        "dtype_kv": dtype_map[dtype_kv],
        "dtype_o": dtype_map[dtype_o],
        "idtype": dtype_map[idtype],
        "head_dim_qk": head_dim_qk,
        "head_dim_vo": head_dim_vo,
        "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
        "use_sliding_window": str(use_sliding_window).lower(),
        "use_logits_soft_cap": str(use_logits_soft_cap).lower(),
        "use_fp16_qk_reduction": str(use_fp16_qk_reduction).lower(),
    }

    with open(... / "batch_prefill_customize_config.jinja") as f:
        config_templ = jinja2.Template(f.read())
    with open(... / "batch_prefill_paged_kernel_inst.jinja") as f:
        paged_kernel_inst_templ = jinja2.Template(f.read())
    with open(... / "batch_prefill_ragged_kernel_inst.jinja") as f:
        ragged_kernel_inst_templ = jinja2.Template(f.read())

    generated_inc_str = config_templ.render(**kwargs)
    os.makedirs(gen_directory, exist_ok=True)

    for mask_mode in [0, 1, 2, 3]:
        ...
        source = paged_kernel_inst_templ.render(mask_mode=mask_mode_literal[mask_mode], **kwargs)
        ...
        source = ragged_kernel_inst_templ.render(mask_mode=mask_mode_literal[mask_mode], **kwargs)
        ...

    for filename in [
        "batch_prefill.cu",
        "batch_prefill_jit_binding.cu",
    ]:
        ...

    generated_config_path = gen_directory / "batch_prefill_config.inc"
    write_if_different(generated_config_path, generated_inc_str)
    return gen_jit_spec(uri, source_paths)
```

这里可以直接看出当前 JIT 输出物的形态：

- 一个 `batch_prefill_config.inc`，把 dtype / head dim / variant / mask mode 等规格写成配置
- 4 个 paged kernel 实例化单元
- 4 个 ragged kernel 实例化单元
- 两个公共 C++ 源文件：`batch_prefill.cu` 和 `batch_prefill_jit_binding.cu`

也就是说，JIT 这一层做的事是：

1. 把规格展开进模板参数
2. 生成当前规格需要的最小源码集合
3. 交给 `jit/core.py`、`jit/cpp_ext.py`、`jit/env.py` 的通用机制完成编译与加载

如果你在理解“为什么要写 URI，为什么要有 config + inst 模板”时感觉熟悉，可以对照 [Flash Attention V2：调度与实例化](../flash_attention_v2/05_dispatch_and_instantiation.md) 一起看。这两者都属于“编译期模板实例化 + 运行期选择”的同一类工程问题。

## `module.plan(...)`：Python plan 最终落到哪

在 [`wrapper.py:L214-L252`](src/wrapper_py.md#__codelineno-0-214) 里，Python 侧最终会调用：

```python
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
```

这一步是整个 `plan()` 的真正终点：

- 前一章准备好的 metadata 被送进 runtime 模块
- JIT 产出的 `.plan` 符号被真正调用
- 返回值 `self._plan_info` 会成为后续 `run()` 阶段的核心输入

它在 C++ 侧对应的是 [`batch_prefill.cu:L47-L75`](src/batch_prefill_cu.md#__codelineno-0-47)：

```cpp
Array<int64_t> BatchPrefillWithKVCachePlan(
    TensorView float_workspace_buffer, TensorView int_workspace_buffer,
    TensorView page_locked_int_workspace_buffer, TensorView qo_indptr, TensorView kv_indptr,
    TensorView kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal, int64_t window_left, int64_t fixed_split_size,
    bool disable_split_kv, int64_t num_colocated_ctas = 0) {
  ...
  PrefillPlanInfo plan_info;
  ...
  cudaError_t status = PrefillPlan<IdType>(..., plan_info, ...);
  ...
  return Array(plan_info.ToVector());
}
```

这里最重要的语义是：

- C++ `plan` 会把调度结果写入 `PrefillPlanInfo`
- 返回 Python 的不是结构体，而是序列化后的 `Array<int64_t>`
- 后续 `run()` 再把这串 `plan_info` 恢复回来

所以如果按执行顺序理解，这一章的落点其实很明确：

> `plan()` 的后半段并不负责执行 attention，它负责准备“之后怎么执行”。

## 这一层的结论

按执行顺序看，`plan()` 的后半段做了三件事：

1. 根据 dtype/head_dim/backend 选择并生成 batch prefill 模块。
2. 通过 JIT 把这组规格变成可编译的 C++/CUDA 源码。
3. 调用 C++ `BatchPrefillWithKVCachePlan(...)` 生成序列化后的 `plan_info`。

到了这一章结束时，真正执行 attention 所需的调度信息已经生成完毕；下一章才开始进入 `run()` 和底层 kernel。
