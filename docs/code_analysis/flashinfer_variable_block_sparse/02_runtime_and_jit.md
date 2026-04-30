---
tags:
  - CUDA
  - Flash Attention
  - Sparse Attention
---

# Runtime 与 JIT

**核心文件**:

- [`prefill_runtime.py`](src/prefill_runtime_py.md)
- [`backend.py`](src/backend_py.md)
- [`common.py`](src/common_py.md)
- [`jit/attention/modules.py`](src/jit_attention_modules_py.md)
- [`jit/core.py`](src/jit_core_py.md)
- [`jit/cpp_ext.py`](src/jit_cpp_ext_py.md)
- [`jit/env.py`](src/jit_env_py.md)

## runtime 层为什么说是“更薄的一层”

当前的 `prefill_runtime.py` 只有一个关键入口：`get_batch_prefill_module`。

完整主逻辑如下：

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

这份实现和完整 FlashInfer 相比，最大的变化不是“多了什么”，而是“删掉了很多没必要的东西”：

- 不再保留庞杂的 prefill 入口矩阵
- 不再暴露与 variable block 无关的其它 runtime 路径
- 只留下 `plan` 和 `paged_run` 这一条最小闭环
- 对 `fa3` 直接报错，而不是继续分叉复杂 runtime

这就解释了为什么它可以说是：

> 面向 variable-block wrapper 的 batch prefill runtime 薄封装。

## backend 选择逻辑：接口保留，能力收缩

backend 的边界写在 [`backend.py`](src/backend_py.md)：

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

这里最容易误解的是：

- `auto` 不是写死成 `fa2`
- 而是仍然保留了设备感知检测
- 只是检测结果如果是 `fa3`，当前版本不会继续执行

真正的检测逻辑在 [`common.py`](src/common_py.md)：

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

这代表当前设计刻意保留了两层语义：

- 接口层仍与上游一致
- 实现层则明确收缩到当前真正打通的 `fa2`

这么做的好处是：

- 调用方式不需要因为迁移而重写
- 将来若要接回 `fa3`，边界还在
- 当前不会把“未接通”伪装成“已支持”

## JIT 入口：规格先编码，再生成源码，再编译

JIT 层的第一步是把规格编码成 URI。

[`jit/attention/modules.py`](src/jit_attention_modules_py.md) 中的 `get_batch_prefill_uri`：

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

这一步不是装饰，而是整个 JIT 缓存的 key：

- dtype 组合不同，URI 不同
- head dim 不同，URI 不同
- pos encoding / soft cap / fp16 qk reduction 不同，URI 不同

所以 URI 本质上是在做一件事：

> 把“模板实例化规格”编码成可缓存、可编译、可复用的模块身份。

## `gen_batch_prefill_module`：当前 variable-block 真正会走到哪一支

现在的 variable-block 路径最终会走进：

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
        variant_name = f"DefaultAttention<use_custom_mask, {str(use_sliding_window).lower()}, {str(use_logits_soft_cap).lower()}, {str(pos_encoding_mode == 2).lower()}>"
        variant_decl = "#include<flashinfer/attention/variants.cuh>"
    else:
        ...

    return gen_customize_batch_prefill_module(...)
```

虽然这个函数仍然保留了 `fa3` 分支结构，但对当前 variable-block 闭环来说，真正重要的是：

- 它使用的是 `fa2` 的 batch prefill 模板
- `fp8_enabled` 在当前闭环里直接被排除
- 额外参数位虽然保留，但 `run()` 大多传 `None`

因此，从“当前实际执行路径”的角度，这段代码应当读成：

> 用上游 batch prefill 的 `fa2` 模板体系，实例化一个最适合当前 `dtype/head_dim/pos_encoding` 组合的模块。

## `gen_customize_batch_prefill_module`：真正生成哪些源码

JIT 的主体在 `gen_customize_batch_prefill_module`：

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
    pos_encoding_mode: int = 0,
    use_sliding_window: bool = False,
    use_logits_soft_cap: bool = False,
    use_fp16_qk_reduction: bool = False,
    fp8_enabled: bool = False,
) -> JitSpec:
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
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    (additional_params_decl, additional_func_params, additional_params_setter) = (
        generate_additional_params(...)
    )

    with open(jit_env.FLASHINFER_CSRC_DIR / "batch_prefill_customize_config.jinja") as f:
        config_templ = jinja2.Template(f.read())

    with open(jit_env.FLASHINFER_CSRC_DIR / "batch_prefill_paged_kernel_inst.jinja") as f:
        paged_kernel_inst_templ = jinja2.Template(f.read())

    with open(jit_env.FLASHINFER_CSRC_DIR / "batch_prefill_ragged_kernel_inst.jinja") as f:
        ragged_kernel_inst_templ = jinja2.Template(f.read())

    ...

    for mask_mode in [0, 1, 2, 3]:
        dest_path = gen_directory / f"batch_prefill_paged_kernel_mask_{mask_mode}.cu"
        source = paged_kernel_inst_templ.render(
            mask_mode=mask_mode_literal[mask_mode],
            **kwargs,
        )
        write_if_different(dest_path, source)

        dest_path = gen_directory / f"batch_prefill_ragged_kernel_mask_{mask_mode}.cu"
        source = ragged_kernel_inst_templ.render(
            mask_mode=mask_mode_literal[mask_mode],
            **kwargs,
        )
        write_if_different(dest_path, source)

    for filename in [
        "batch_prefill.cu",
        "batch_prefill_jit_binding.cu",
    ]:
        src_path = jit_env.FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        ...

    generated_config_path = gen_directory / "batch_prefill_config.inc"
    write_if_different(generated_config_path, generated_inc_str)
    return gen_jit_spec(uri, source_paths)
```

这段代码说明当前 JIT 策略不是“直接拿现成 `.so`”，而是：

1. 根据 dtype/head_dim/mask_mode 等规格生成配置
2. 用 Jinja 模板生成实例化 `.cu`
3. 拷贝通用的 `batch_prefill.cu` 与 `batch_prefill_jit_binding.cu`
4. 形成一组源码后再交给编译器构建扩展

所以这层 JIT 的真正含义是：

- Python 侧并不自己维护一堆静态编译产物
- 它在运行时按需求拼出一份“刚好够用”的源码集合
- 然后由 `jit/core.py`、`jit/cpp_ext.py` 等基础设施负责编译和加载

## 为什么说当前已经是 `fa2 only` 最小闭环

从代码组织上看，你仍能在 [`modules.py`](src/jit_attention_modules_py.md) 里看到 `fa3` 甚至更多更大的上游结构。但结合整个目录真实保留下来的东西看，当前这套闭环已经是明显收缩后的版本：

- [`prefill_runtime.py`](src/prefill_runtime_py.md) 里显式拒绝 `fa3`
- 当前目录的 `data/csrc/` 只保留了 `fa2` 相关 `batch_prefill*` 文件
- 文档链路和调用链都只围绕 `paged prefill -> plan -> paged_run`
- `VariableBlockSparseAttentionWrapper` 也只依赖这条路径

所以工程上最准确的理解不是“这还是完整的多后端 runtime”，而是：

> 接口层保留了上游兼容外壳，执行层已经收敛为 variable-block 所需的 `fa2` 最小子集。

## 这一层应该怎么读

读 runtime/JIT 时，建议不要把注意力放在“所有可能性”上，而要聚焦“当前 variable-block 主链路会实际命中的部分”：

1. backend 解析结果如何落到 `fa2`
2. `get_batch_prefill_module` 如何把 plan/run 句柄拿出来
3. JIT 如何从规格生成 URI、配置、模板实例化源码
4. 这些产物最终怎样组成可加载模块

这一层的价值不在数学，而在工程边界：

- 上层 wrapper 提供什么规格
- 下层 C++/CUDA 需要什么实例
- 中间如何以最薄的方式接起来

