---
tags:
    - Diffusion Model
    - Flow Matching
---

# ELF 代码分析：总览

!!! abstract "TL;DR"
    ELF 在冻结 T5 encoder 的连续嵌入空间上做 Flow Matching，整个去噪轨迹保持在连续空间，仅最后一步用共享权重的网络 decode 到离散 token。JAX/Flax 实现，16 个 Python 文件，核心是双分支训练（80% MSE denoising + 20% CE decoding）+ 训练时 CFG。

**源码仓库**: [lillian039/ELF](https://github.com/lillian039/ELF)

**论文**: [ELF: Embedded Language Flows](https://arxiv.org/abs/2605.10938)

**前置知识**: Flow Matching 基础（rectified flow 线性插值 + x-prediction 参数化）见[论文笔记](../../paper_reading/elf.md)。

## 核心思想

标准 Flow Matching 在图像生成中定义 $z_t = t x + (1-t) \epsilon$，网络学习 velocity field $v = x - \epsilon$。ELF 将这一框架搬到语言建模，核心改变：

$$
\underbrace{\text{tokens } s}_{\text{离散}} \xrightarrow{\text{T5 encoder}} \underbrace{x \in \mathbb{R}^{L \times 512}}_{\text{连续嵌入}} \xrightarrow{\text{Flow Matching}} \underbrace{\hat{x}}_{\text{去噪嵌入}} \xrightarrow[t=1]{\text{shared-weight decode}} \underbrace{\hat{s}}_{\text{离散 tokens}}
$$

与传统连续 DLM 的关键区别：

| 设计维度 | 传统连续 DLM | ELF |
|---------|-------------|-----|
| 中间步离散化 | 施加 CE loss 监督 | **不施加**，纯连续去噪 |
| 最终步解码 | 独立训练的 decoder | **共享权重**网络，切换 mode token |
| 时间建模 | DDPM (离散时间) | **连续时间** Flow Matching |
| CFG | 难以适配 | **天然兼容**，训练时 CFG 无推理开销 |

## 代码结构

```
src/
├── configs/
│   └── config.py            # Config + SamplingConfig 定义
├── modules/
│   ├── model.py             # ELF DiT 主模型 (ELFBlock + ELF)
│   ├── layers.py            # Attention / RMSNorm / SwiGLU / RoPE / Bottleneck
│   └── t5_encoder.py        # 冻结 T5 encoder 加载
├── train.py                 # 训练主循环 + 数据加载 + checkpoint
├── train_step.py            # 单步训练 (pmap): 双分支 MSE/CE + training-time CFG
├── generation.py            # 推理采样 + 评估 (Gen. PPL / BLEU / ROUGE)
├── eval.py                  # 独立评估脚本
└── utils/
    ├── sampling_utils.py    # 噪声/时间调度、ODE/SDE step、x↔v 转换、CFG forward
    ├── generation_utils.py  # lax.scan 采样循环、pmap 封装、最终步 decode
    ├── encoder_utils.py     # T5 encoder 前向 + attention mask 构建
    ├── train_utils.py       # TrainState / LR schedule / Muon optimizer
    ├── data_utils.py        # HuggingFace datasets 加载 + tokenize
    ├── checkpoint_utils.py  # Checkpoint 存取 + EMA
    ├── logging_utils.py     # 分布式日志
    └── metrics_utils.py     # GPT-2 Large PPL / BLEU / ROUGE 评估
```

**核心函数一览**：

| 函数 | 功能 | 源码位置 |
|------|------|----------|
| `ELF.__call__` | DiT 前向：self-cond → bottleneck → in-context conditioning → blocks → decode | [model.py#L75-L157](https://github.com/lillian039/ELF/blob/main/src/modules/model.py#L75-L157) |
| `train_step` | 双分支训练：Bernoulli(0.2) 选择 MSE/CE，training-time CFG | [train_step.py#L19-L270](https://github.com/lillian039/ELF/blob/main/src/train_step.py#L19-L270) |
| `_ode_step` / `_sde_step` | ODE Euler / SDE noise re-injection 单步 | [sampling_utils.py#L211-L254](https://github.com/lillian039/ELF/blob/main/src/utils/sampling_utils.py#L211-L254) |
| `_generate_samples_single_batch` | lax.scan 驱动 N 步采样循环 | [generation_utils.py#L108-L141](https://github.com/lillian039/ELF/blob/main/src/utils/generation_utils.py#L108-L141) |
| `_dlm_decode_batch` | 最终步 decode: t=1, mode=decode, unembed → argmax | [generation_utils.py#L144-L159](https://github.com/lillian039/ELF/blob/main/src/utils/generation_utils.py#L144-L159) |
| `_forward_sample` | 单次采样前向：self-cond + input CFG 组合 | [sampling_utils.py#L180-L208](https://github.com/lillian039/ELF/blob/main/src/utils/sampling_utils.py#L180-L208) |
| `add_noise` | Flow Matching 插值加噪 $z_t = t x + (1-t) \epsilon$ | [sampling_utils.py#L12-L18](https://github.com/lillian039/ELF/blob/main/src/utils/sampling_utils.py#L12-L18) |
| `net_out_to_v_x` | x-prediction → velocity: $v = (x - z) / (1-t)$ | [sampling_utils.py#L110-L121](https://github.com/lillian039/ELF/blob/main/src/utils/sampling_utils.py#L110-L121) |

## 文档导航

1. [模型结构](01_model.md) — ELF DiT 前向路径：self-cond → bottleneck → in-context conditioning → DiT blocks → decode head，按执行顺序逐段拆解
2. [训练流程](02_training.md) — 双分支混合训练（80% denoising / 20% decoding）+ training-time CFG + self-conditioning mask 机制的完整实现
3. [采样与生成](03_sampling.md) — ODE / SDE 采样器 → lax.scan 迭代 → self-cond + input CFG 前向 → 最终步 argmax decode

## 模型规模

| Model | Depth | Hidden | Heads | Params |
|-------|-------|--------|-------|--------|
| ELF-B | 12 | 768 | 12 | 105M |
| ELF-M | 24 | 1056 | 16 | 342M |
| ELF-L | 32 | 1280 | 16 | 652M |

工厂函数见 [model.py#L161-L167](https://github.com/lillian039/ELF/blob/main/src/modules/model.py#L161-L167)。
