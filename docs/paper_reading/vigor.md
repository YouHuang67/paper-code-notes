---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
  - Flow Matching
  - DPO
---

# VIGOR: VIdeo Geometry-Oriented Reward for Temporal Generative Alignment

- 论文：https://arxiv.org/abs/2603.16271
- 代码：https://vigor-geometry-reward.com/
- 团队：南开大学, 北京邮电大学, École Polytechnique (IP Paris)

## 概述

VIGOR 提出一种基于几何一致性的 reward 模型，利用预训练几何基础模型 VGGT 通过逐点跨帧重投影误差来评估视频的多视图一致性。与先前在像素空间计算几何不一致性的方法不同，VIGOR 在点级别计算误差，物理上更合理且鲁棒。此外引入几何感知采样策略，利用 VGGT 浅层全局注意力热图过滤低纹理和非语义区域，仅在几何上有意义的区域采样。基于此 reward，VIGOR 通过两条路径对齐视频扩散模型：(1) 后训练路径——SFT 和 Flow-DPO 对双向模型进行参数更新；(2) 推理时优化路径——针对双向模型的 Best-of-N 采样，以及针对因果自回归视频模型的三种搜索策略（Search on Start / Search on Path / Beam Search），利用 reward 作为路径验证器在推理时提升几何一致性，无需重训练。

## 动机

- 视频扩散模型训练缺乏显式几何监督，导致物体变形、空间漂移、深度违反等几何伪影
- 闭源模型（Sora、Veo 3）通过海量数据隐式学到几何先验，但开源模型缺乏此优势
- 显式几何监督（深度图、相机位姿条件化）受限于几何标注数据的稀缺性
- 现有几何 reward 的局限：
  - Epipolar-DPO 使用 Sampson 距离，仅度量极线约束
  - VideoGPA 使用 VGGT 重建后在像素空间计算 warp MSE，像素强度引入额外噪声
  - 两者均依赖离线偏好数据、仅支持静态场景

## 方法

### 几何感知采样（Geometry-Aware Sampling）

利用 VGGT 浅层全局注意力层自然强调几何有意义区域的特性：

- 对输入视频帧 $I = \{I_i\}_{i=0}^{N}$，提取 VGGT 浅层的 scaled dot-product attention score：

$$A_i = \frac{1}{N-1} \sum_{j \neq i} \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d}}\right)$$

- 对 $A_i$ 做跨 head 均值 + 跨帧求和，上采样到全分辨率得到几何注意力热图 $M_i$
- 将每帧分割为 $p \times p$ 不重叠 patch，按注意力值取 top-$\tau$% patch，取各 patch 中心像素作为采样点
- 效果：自动过滤天空、地面等低纹理区域，聚焦建筑结构、物体轮廓等几何显著区域

### 逐点重投影误差（Pointwise Reprojection Error）

与像素级 warp MSE 不同，VIGOR 在点级别计算误差：

1. **点跟踪**：对每个参考帧 $i$ 的采样点 $P_i$，用 tracker 追踪其在所有其他帧 $j$ 的对应位置 $p_k^{(j)}$ 及置信度 $c_k^{(j)}$
2. **反投影到 3D**：利用 VGGT 预测的深度图 $D_i$ 和相机参数 $(K_i, R_i, t_i)$，将采样点反投影到世界坐标系：$X_k^w = R_i^{-1}(d_k K_i^{-1} [u_k, v_k, 1]^T - t_i)$
3. **重投影到目标帧**：将 3D 点投影到目标帧 $j$，得到几何预测位置 $\hat{p}_k^{(j)}$
4. **误差计算**：对所有有效点-帧对计算平均 L2 距离：

$$E_{\text{reproj}} = \frac{1}{|V|} \sum_{(k,i,j) \in V} \|\hat{p}_k^{(j)} - p_k^{(j)}\|_2$$

逐点方式将几何误差与像素强度解耦，更物理可靠。

### 后训练对齐

**偏好数据集 GB3DV-25k**：
- 2,560 个 prompt（RealEstate10K 室内 + GLDv2 室外，Qwen3-VL 生成详细描述）
- 每个 prompt 用 CausVid 生成 10 个视频，按几何 reward 取最优和最差构成偏好对
- 共 25,600 个视频片段

**SFT**：在 reward 筛选的高质量子集 $D^*$ 上用标准 flow matching objective 微调（LoRA）

**Flow-DPO**：采用 rectified flow 适配的 DPO 目标，训练权重 $\beta_t = \beta(1-t^2)$ 依赖噪声水平。额外添加辅助损失防止模式坍缩：

$$\mathcal{L}_{\text{aux}} = -\mathbb{E}[\text{Var}_t(\hat{x}_0)] + \gamma \cdot \mathbb{E}[\|\Delta_t^2 \hat{x}_0\|^2]$$

第一项鼓励时间维度变化（防止静止），第二项惩罚二阶时间差异（鼓励平滑运动）。

### 推理时优化（Test-Time Scaling）

针对因果自回归视频模型（Causal-Forcing），利用其逐帧生成的结构探索更丰富的搜索空间：

- **Search on Start (SoS)**：沿 seed 轴搜索，$S$ 个 seed 各生成完整视频，选 reward 最高者。复杂度 $O(SN)$
- **Search on Path (SoP)**：沿时间轴搜索，每帧从 $S$ 个 seed 候选中选 reward 最高者，使用滑动窗口 $W$ 评估。复杂度 $O(SN)$
- **Beam Search (BS)**：维护 $K$ 条候选路径，每步评估 $K \times S$ 个子节点保留 top-$K$。复杂度 $O(KSN)$

SoS 和 SoP 是 BS 的特例：SoS = $(K,1)$，SoP = $(1,S)$。

## 实验

### 训练配置

- 后训练基础模型：Wan2.1-T2V-1.3B
- TTS 双向模型：CausVid（蒸馏自 Wan2.1）
- TTS 因果模型：Causal-Forcing
- LoRA：rank 64, $\alpha$ = 128，应用于 DiT 的 q/k/v/o 投影
- 评估：3D 重建指标（PSNR/SSIM/LPIPS）+ 多视图一致性指标（EPI/RPX/RPT）+ VBench

### Best-of-N 采样（双向模型）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | EPI↓ | RPT↓ | VBench Total↑ |
|------|-------|-------|--------|------|------|---------------|
| Baseline (CausVid) | 19.68 | 0.638 | 0.360 | 5.553 | 4.706 | 83.33 |
| Epipolar | 22.45 | 0.756 | 0.243 | - | 2.815 | 84.50 |
| Reproj-Pix | 21.07 | 0.727 | 0.304 | 4.549 | 3.473 | 83.97 |
| **Reproj-Pts (Ours)** | **22.66** | **0.767** | **0.233** | **3.442** | - | **84.52** |

### 推理时搜索（因果模型）

三种搜索策略均随预算增加持续提升：
- Beam Search 在 3D 指标上最强（兼具多样性和细粒度优化）
- SoP 在 VBench 综合分数上最佳（逐帧优化更稳定）
- SoS 在 RPT 上最低但其他指标较弱（仅优化起点）

### 后训练对齐（Wan2.1-1.3B）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | EPI↓ | SC↑ | BC↑ |
|------|-------|-------|--------|------|-----|-----|
| Baseline | 22.45 | 0.755 | 0.224 | 2.832 | 95.98 | 94.43 |
| + SFT | 23.52 | 0.793 | 0.184 | 2.337 | 96.97 | 95.15 |
| + DPO (Epipolar) | 23.57 | 0.797 | 0.182 | 2.187 | 96.98 | 95.16 |
| + DPO (Reproj-Pts) | 23.54 | **0.798** | **0.179** | **2.127** | **97.05** | **95.25** |

SFT 已显著提升几何一致性，DPO 进一步通过对比正负样本带来改善，两种 DPO reward 差异不大但 Reproj-Pts 在 EPI 和感知质量上略优。

## 关键启示

- **逐点重投影优于像素级 warp**：将几何误差与像素强度解耦，避免纹理差异引入噪声，提供物理上更合理的 reward 信号
- **几何基础模型的注意力可直接用于采样策略**：VGGT 浅层注意力自然聚焦几何有意义区域，无需额外训练即可过滤低质采样点
- **因果自回归模型为推理时优化提供更丰富的搜索空间**：逐帧生成的 Markov 结构允许在时间轴和 seed 轴上进行结构化搜索，突破 Best-of-N 的局限
- **SFT + DPO 是互补的后训练路径**：SFT 将分布偏移到高 reward 区域，DPO 通过正负对比进一步精细化
- **辅助损失对防止运动退化至关重要**：temporal variance + smoothness penalty 平衡动态性和几何一致性
