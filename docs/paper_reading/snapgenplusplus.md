---
tags:
  - Sparse Attention
  - Diffusion Model
---

# SnapGen++: Unleashing Diffusion Transformers for Efficient High-Fidelity Image Generation on Edge Devices

[arXiv 2601.08303](https://arxiv.org/abs/2601.08303) | [项目页](https://snap-research.github.io/snapgenplusplus/) | Snap Inc. / 墨尔本大学

## 概述

现有 DiT 模型（Flux、SDXL 等）需要服务器级 GPU，无法部署到移动/边缘设备。SnapGen++ 提出了一套完整的移动端 DiT 框架，核心三件套：

1. **ASSA（自适应全局-局部稀疏注意力）**：解决 1K 分辨率下 4096 token 的注意力瓶颈
2. **Elastic Training（弹性训练）**：单个 supernetwork 内联合优化多个不同容量的 sub-DiT，一次训练适配不同硬件
3. **K-DMD（知识引导分布匹配蒸馏）**：4 步生成达到接近 28 步的质量

主要结果：
- Ours-small (0.4B) 在 iPhone 16 Pro Max 上 **1.8 s** 生成 1024×1024 图像
- 在 DPG/GenEval/T2I-CompBench 上优于 SD3.5-L (8.1B)、Flux.1-dev (12B)（参数量少 20-30×）

## 方法

### 三段式 DiT 架构

受 HourGlass-DiT 启发，将 PixArt-α 基线扩展为三阶段：

- **Down blocks × 6**：高分辨率（128×128 latent），使用 ASSA
- **Middle blocks × 20**：经 2×2 下采样至 32×32（1024 token），使用标准 SA
- **Up blocks × 8**：上采样回高分辨率，使用 ASSA

通过三段式设计，延迟从 2000ms 降到 550ms。Middle stage 用标准注意力处理小分辨率 token，Down/Up stage 用 ASSA 处理大分辨率 token。

### ASSA：自适应稀疏自注意力

替换 Down/Up stage 中对 4096 token 的全量注意力，并行两条路径：

**全局注意力（Global）——粗粒度 KV 压缩**

用 stride=2 的 2×2 卷积压缩 k, v：
$$k_c = \text{Conv}_{2\times2, s=2}(k), \quad v_c = \text{Conv}_{2\times2, s=2}(v)$$

KV token 数缩减 4 倍（$\frac{H}{2} \times \frac{W}{2}$），每个 query 仍能 attend 到全局上下文。

**局部注意力（Local）——块状邻域注意力（BNA）**

将 token 网格划分为 $B$ 个不重叠的空间块，每个 query block $q_b$ 仅 attend 到邻域半径 $r$ 内的 KV 块：

$$A_b = \text{Softmax}\left(\frac{q_b [k_{N_r(b)}]^\top}{\sqrt{d}}\right)[v_{N_r(b)}]$$

复杂度从 $O(N^2)$ 降为 $O(N^2/B)$。实验设置 $b=16, r=1$，在 1024×1024 分辨率下等价于 9 个空间邻居 token。

BNA 在移动端通过 for-loop 并行执行各块，用 split einsum 实现，导出为 CoreML 计算图。

**自适应融合**

两路输出按 per-head 插值合并，插值权重由输入 hidden state 动态生成（sigmoid 激活）：

$$h \leftarrow \text{Interpolate}(h_{\text{local}}, h_{\text{global}}, w_{\text{head}})$$

ASSA 引入后，延迟从 550ms 降至 293ms，验证损失不变（0.513）。

### Elastic DiT 框架

受 Matformer/Slimmable Networks 启发，沿 hidden dim 切片，从单个 supernetwork 派生多个 sub-network：

- Tiny (0.3B, 0.375× width)：低端 Android 设备
- Small (0.4B, 0.5× width)：高端智能手机
- Full (1.6B, 1× width)：4-bit 量化后可设备端部署或服务端推理

交叉注意力层的 KV 投影不切片（维度与 hidden dim 无关）。

**训练目标**

联合优化 supernetwork $\Theta$ 和采样到的 subnetwork $\Theta_s$，使用 flow matching 损失：

$$\mathcal{L}_{\text{diff}}(\theta) = \mathbb{E}_{\epsilon, t}\left[\|(\epsilon - x_0) - v_\theta(x_t, t)\|_2^2\right]$$

同时加入 subnetwork → supernetwork 的蒸馏损失（stop gradient）：

$$\mathcal{L}_{\text{dist}}(\Theta_s) = \|v_{\Theta_s}(x_t, t) - \overline{\nabla} v_{\Theta}(x_t, t)\|_2^2$$

自适应 gradient scaling 保证各 subnetwork 梯度平衡。结果：弹性训练下 0.4B/1.6B 验证损失与独立训练几乎相同，但节省了独立模型的存储开销。

### K-DMD：知识引导分布匹配蒸馏

标准 DMD 在小模型上超参数敏感且收敛不稳定。K-DMD 在 DMD 目标基础上，额外引入 few-step teacher（Qwen-Image-Lightning，通过 LoRA 激活，无额外内存开销）的输出和特征蒸馏：

$$\mathcal{L}_{\text{K-DMD}}(\theta, \phi) = \mathcal{L}_{\text{DMD}}^\xi + \mathcal{L}_{\text{out}}^{\xi'} + \mathcal{L}_{\text{feat}}^{\xi'}$$

DMD 的分布对齐项：
$$\nabla_\theta \mathcal{L}_{\text{DMD}}^\xi = \left[f_c(F(\hat{x}_0, \tau), \tau) - f_\xi(F(\hat{x}_0, \tau), \tau)\right]\frac{d\hat{x}_0}{d\theta}$$

其中 critic $c$ 用与 student 相同权重初始化，交替用 flow matching 损失训练。

## 实验

### 架构消融（ImageNet 256×256）

| 配置 | 参数量 | 延迟（iPhone 16 PM） | Val Loss |
|------|--------|---------------------|----------|
| 基线 DiT | 424M | 2000ms | 0.5060 |
| + 三段式 | 406M | 550ms | 0.5130 |
| + ASSA | 408M | 293ms | 0.5130 |
| + 增强 | 429M | 360ms | 0.5090 |
| SnapGen (U-Net 对比) | 372M | 274ms | 0.5131 |

最终以 360ms 延迟接近 SnapGen，但 Val Loss 明显更低（0.5090 vs 0.5131）。

### T2I 评测（1024×1024）

| 模型 | 参数 | FPS | 延迟 | DPG↑ | GenEval↑ |
|------|------|-----|------|------|---------|
| SnapGen | 0.4B | 0.51 | 274ms | 81.1 | 0.66 |
| SANA | 1.6B | 0.91 | – | 84.8 | 0.66 |
| SD3.5-L | 8.1B | 0.08 | – | 85.6 | 0.71 |
| Flux.1-dev | 12B | 0.04 | – | 83.8 | 0.66 |
| **Ours-small** | **0.4B** | **0.62** | **360ms** | **85.2** | **0.70** |
| **Ours-full** | **1.6B** | **0.28** | **1580ms** | **87.2** | **0.76** |

0.4B small 模型在 DPG 和 GenEval 上超过所有对比模型（包括 8.1B SD3.5-L），4 步 K-DMD 版本（1.8s）相比 28 步基础版本仅微降（DPG 85.2→82.7）。

### 部署细节

- 模型用 k-means 量化（大多数层 4-bit，敏感层 8-bit，整体 4.3-bit）
- 量化后 fine-tune bias 和 normalization 层（self-distillation，几千步）
- CoreML 导出，split einsum 优化 BNA 的移动端执行

## 关键启示

- **全局-局部稀疏注意力组合**：KV 压缩（4×降）提供全局上下文，BNA 提供精细局部关系，两路自适应融合，优于只用其中一种
- **三段式 DiT 是关键基础**：下采样中间阶段处理小 token 数，高分辨率阶段的 ASSA 才可行，延迟从 2000→293ms
- **弹性训练无代价**：单个 supernetwork 内联合优化多个 sub-DiT，验证损失几乎不变，但节省了独立训练多个模型的开销
- **K-DMD 稳定小模型步骤蒸馏**：用 few-step teacher 的知识蒸馏补充 DMD 的分布对齐，解决标准 DMD 在小模型上的超参敏感和收敛不稳定问题
