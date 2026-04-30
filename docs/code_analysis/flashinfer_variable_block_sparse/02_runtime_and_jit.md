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
- [batch_prefill.cu:L47-L75](src/batch_prefill_cu.md#__codelineno-0-47)

这一章只讲 `plan()` 后半段真正发生的事情，也就是：

1. Python 侧怎样拿到 batch prefill 模块
2. JIT 怎样把规格编码成可编译模块
3. `module.plan(...)` 最终怎样落到 C++ `BatchPrefillWithKVCachePlan(...)`

这正好对应代码执行顺序里“metadata 已经准备好，接下来开始生成执行模块和调度信息”的阶段。

## 1. 这一章的输入和输出

从执行顺序看，这一章接收的是前一章已经准备好的状态：

- `self._backend`
- `self._qo_indptr`
- `self._paged_kv_indptr_buf`
- `self._paged_kv_indices_buf`
- `self._paged_kv_last_page_len`

这一章结束时，最关键的新产物有两个：

- `self._cached_module`：JIT 生成并加载好的 batch prefill 模块
- `self._plan_info`：C++ `plan` 返回的序列化调度结果

这两个对象的分工要先记住：

- `self._cached_module` 决定“以后调用哪个 `.plan` / `.paged_run` 符号”
- `self._plan_info` 决定“以后真正执行时该怎样切 tile、怎样做 split-k、怎样在 workspace 里找调度数组”

## 2. `get_batch_prefill_module()`：先拿到可执行模块

当前 runtime 主入口只有 [`prefill_runtime.py:L32-L108`](src/prefill_runtime_py.md#__codelineno-0-32)：

```python
@functools.cache
def get_batch_prefill_module(backend, *args):
    from .jit import gen_batch_prefill_module

    if backend == "fa3":
        raise NotImplementedError(_FA3_NOT_SUPPORTED_MSG)

    module = gen_batch_prefill_module(backend, *args).build_and_load()   # JIT 生成并 dlopen
    paged_run_func = module.paged_run                                     # 取出底层 paged_run 符号

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
            float_workspace_buffer,   # [workspace_bytes]
            int_workspace_buffer,     # [int_workspace_bytes]
            plan_info_vec,            # [plan_info_words]
            q,                        # [H_kv * qo_len, G, D]
            paged_k_cache,            # [H_kv * kv_len, 1, 1, D]
            paged_v_cache,            # [H_kv * kv_len, 1, 1, D]
            qo_indptr,                # [H_kv * R + 1]
            paged_kv_indptr,          # [H_kv * R + 1]
            paged_kv_indices,         # [nnz_tokens]
            paged_kv_last_page_len,   # [H_kv * R]
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
            logits_soft_cap,          # scalar
            sm_scale,                 # scalar
            1.0 / rope_scale,         # runtime 侧要求传倒数
            1.0 / rope_theta,         # runtime 侧要求传倒数
            token_pos_in_items_len,   # 当前 variable block 路径固定传 0
        )
        return o

    return SimpleNamespace(plan=module.plan, paged_run=paged_run)
```

这段代码说明了两个事实：

1. runtime 这一层已经被裁得很薄。
   对 wrapper 来说，它只保留了两个真正需要的能力：
   - `.plan`
   - `.paged_run`

2. `paged_run()` 虽然还保留了很多扩展参数位，但当前 variable block 路径里，大部分都会传 `None` 或固定标量。
   这也是“本地最小闭环”这个说法的直接体现。

## 3. 为什么这一阶段只会落到 `fa2`

上一章已经把 `backend` 收敛清楚了，这里再从 runtime 视角看一次边界。见 [`backend.py:L28-L50`](src/backend_py.md#__codelineno-0-28)：

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

所以这一章里所有“模块准备”和“调度生成”的真实执行路径，都应该按 `fa2` 去理解。

## 4. wrapper 里 `module.plan(...)` 的实参到底是什么

这一章最容易读糊涂的地方，其实不是 JIT，而是 `wrapper.py` 里这一段位置参数调用。见 [`wrapper.py:L214-L252`](src/wrapper_py.md#__codelineno-0-214)：

```python
get_module_args = (
    q_data_type,                                                # q dtype
    kv_data_type,                                               # kv dtype
    self._o_dtype,                                              # output dtype
    kv_indptr_host.dtype,                                       # index dtype, 当前是 int32
    head_dim,                                                   # head_dim_qk = D
    head_dim,                                                   # head_dim_vo = D
    PosEncodingMode[pos_encoding_mode].value,                   # pos encoding mode
    False,                                                      # use_sliding_window
    logits_soft_cap > 0,                                        # use_logits_soft_cap
    use_fp16_qk_reduction,                                      # use_fp16_qk_reduction
)
self._cached_module = get_batch_prefill_module(self._backend, *get_module_args)

kv_lens_arr_host = kv_indptr_host[1:] - kv_indptr_host[:-1]     # [H_kv * R]

args = [
    self._float_workspace_buffer,                                # [workspace_bytes]
    self._int_workspace_buffer,                                  # [8MiB]
    self._pin_memory_int_workspace_buffer,                       # [8MiB]
    qo_indptr_host,                                              # [H_kv * R + 1]
    kv_indptr_host,                                              # [H_kv * R + 1]
    kv_lens_arr_host,                                            # [H_kv * R]
    qo_indptr_host[-1].item(),                                   # scalar，所有逻辑 request 的 query token 总数
    num_blocks_row * num_kv_heads,                               # scalar，逻辑 batch size = H_kv * R
    num_qo_heads // num_kv_heads,                                # scalar，单个逻辑 request 内的 query 头数 = G
    1,                                                           # scalar，单个逻辑 request 内的 KV 头数
    1,                                                           # scalar，page_size = 1
    False,                                                       # scalar，不启用 cuda graph
    head_dim,                                                    # scalar，QK 头维度 = D
    head_dim,                                                    # scalar，VO 头维度 = D
    causal,                                                      # scalar，mask mode
    -1,                                                          # scalar，window_left = -1
]
if self._backend == "fa2":
    args.append(-1)                                              # fixed_split_size，交给 planner 自己决定
    args.append(False)                                           # disable_split_kv
    args.append(0)                                               # num_colocated_ctas
self._plan_info = self._cached_module.plan(*args)
```

这段代码最重要的不是“参数很多”，而是它暴露了底层 runtime 看到的已经不是原始输入，而是 **逻辑 request 视角**：

- `num_blocks_row * num_kv_heads`
  - 这不是原始 batch size
  - 它是逻辑 request 数量，也就是 `(kv_head, row_block)` 对的个数

- `num_qo_heads // num_kv_heads`
  - 这不是原始 `num_qo_heads`
  - 它是单个逻辑 request 内部的 query 头数，也就是 GQA group size `G`

- `1`（逻辑 request 内的 KV 头数）
  - 因为 Python 侧已经按 `kv_head` 把 request 拆开了
  - 所以从 planner 视角，每个 request 只面对 1 个 KV head

- `1`（page_size）
  - 因为当前实现把每个 token 都当成了一个 page

所以这一步本质上是在告诉 planner：

> “我已经把 variable block 语义压平成 `H_kv * R` 个独立 request 了，每个 request 自己只有 1 个 KV head、page size 也是 1，请你按普通 paged prefill 的方式给我做调度规划。”

## 5. URI：JIT 怎样唯一标识这组规格

JIT 的第一步是把规格编码进 URI。见 [`jit/attention/modules.py:L372-L396`](src/jit_attention_modules_py.md#__codelineno-0-372)：

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

它本质上就是 JIT 缓存 key：

- dtype 不同，URI 不同
- head dim 不同，URI 不同
- pos encoding / soft cap / fp16-qk reduction 不同，URI 不同

因此 URI 不是装饰，而是在把这次 `plan()` 所需的模板实例化规格编码成模块身份。

## 6. `gen_batch_prefill_module()`：按规格选择 batch prefill 模块形态

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

虽然这里还保留了 `fa3` 分支的壳，但对当前闭环来说，真正重要的是：

- `resolve_attention_backend()` 之后只会允许 `fa2`
- `fa2` 下直接排除了 fp8 tensor core 路径
- additional tensor/scalar 参数位仍然保持与上游 batch prefill ABI 一致

也就是说，本地版本不是重新发明 JIT 逻辑，而是在复用上游 batch prefill 的参数化模板体系。

## 7. `gen_customize_batch_prefill_module()`：怎样把规格落成源码

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

- 一个 `batch_prefill_config.inc`
- 4 个 paged kernel 实例化单元
- 4 个 ragged kernel 实例化单元
- 两个公共 C++ 源文件：
  - `batch_prefill.cu`
  - `batch_prefill_jit_binding.cu`

所以 JIT 这一层做的事情并不神秘，就是：

1. 把 dtype/head_dim/variant/mask_mode 写进模板参数
2. 生成当前规格真正需要的最小源码集合
3. 交给 `jit/core.py`、`jit/cpp_ext.py`、`jit/env.py` 的通用机制完成编译和加载

## 8. `module.plan(...)` 最终落到哪

前面 Python 侧调用的：

```python
self._plan_info = self._cached_module.plan(*args)
```

在 C++ 侧最终对应的是 [`batch_prefill.cu:L47-L75`](src/batch_prefill_cu.md#__codelineno-0-47)：

```cpp
Array<int64_t> BatchPrefillWithKVCachePlan(
    TensorView float_workspace_buffer, TensorView int_workspace_buffer,
    TensorView page_locked_int_workspace_buffer, TensorView qo_indptr, TensorView kv_indptr,
    TensorView kv_len_arr, int64_t total_num_rows, int64_t batch_size, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph, int64_t head_dim_qk,
    int64_t head_dim_vo, bool causal, int64_t window_left, int64_t fixed_split_size,
    bool disable_split_kv, int64_t num_colocated_ctas = 0) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * get_element_size(float_workspace_buffer);
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * get_element_size(int_workspace_buffer);

  PrefillPlanInfo plan_info;

  ffi::CUDADeviceGuard device_guard(float_workspace_buffer.device().device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer.device());
  cudaError_t status = PrefillPlan<IdType>(
      float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
      int_workspace_buffer.data_ptr(), page_locked_int_workspace_buffer.data_ptr(),
      int_workspace_size_in_bytes, plan_info, static_cast<IdType*>(qo_indptr.data_ptr()),
      static_cast<IdType*>(kv_indptr.data_ptr()), total_num_rows, batch_size, num_qo_heads,
      num_kv_heads, head_dim_qk, head_dim_vo, page_size, enable_cuda_graph,
      /*sizeof_dtype_o=*/2, window_left, fixed_split_size, disable_split_kv, num_colocated_ctas,
      stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Failed to plan prefill with error: " << cudaGetErrorString(status);

  return Array(plan_info.ToVector());
}
```

这一段最重要的语义不是模板，而是输入输出：

- 输入：
  - workspace
  - `qo_indptr`
  - `kv_indptr`
  - `kv_len_arr`
  - 一组描述逻辑 request 形态的标量

- 输出：
  - `PrefillPlanInfo plan_info`
  - 但返回给 Python 的不是结构体，而是 `plan_info.ToVector()` 之后的 `Array<int64_t>`

所以 `self._plan_info` 的本质是：

> **序列化后的调度结果**，不是高层 Python 对象。

后面 `run()` 阶段会再把它恢复回来。

## 小结

按执行顺序看，这一章做了三件事：

1. 根据 dtype/head_dim/backend 选择并生成 batch prefill 模块。
2. 通过 JIT 把这组规格变成可编译的 C++/CUDA 源码。
3. 调用 C++ `BatchPrefillWithKVCachePlan(...)` 生成序列化后的 `self._plan_info`。

到了这一章结束时，真正执行 attention 所需的“执行模块”和“调度信息”都已经准备完毕；下一章才开始进入 `run()` 和底层 kernel。
