---
tags:
  - Video Generation
  - Diffusion Model
  - Reinforcement Learning
  - GRPO
  - DPO
---

# TeleBoost: A Systematic Alignment Framework for High-Fidelity, Controllable, and Robust Video Generation

- 论文：https://arxiv.org/abs/2602.07595
- 代码：https://github.com/Tele-AI
- 团队：TeleBoost Team（通讯作者 Xuelong Li）

## 概述

TeleBoost 提出一个系统性的视频生成后训练框架，将后训练组织为三阶段顺序优化流水线：Stage I 监督微调（SFT）建立稳定可控的参考策略，Stage II 基于 GRPO 的强化学习驱动可度量目标的改善，Stage III 基于 DPO 的人类偏好对齐捕获难以量化的主观质量。框架围绕视频生成的实际约束设计：高 rollout 成本、时序误差累积、反馈信号异质且弱区分性。在 GRPO 阶段引入三个模块化增强：ViPO（时空信用分配）、BPGO（贝叶斯先验信任分配）、Self-Paced GRPO（自适应 reward 课程），加上多目标联合 reward 平衡机制。基础设施层面提出 Ray 时间复用、NVIDIA MPS 并行 reward 评估、解耦梯度 DPO 三项工程优化。在 Wan2.2 上验证，人类评估在运动质量和文本对齐上以 24%+ margin 超越 Wan2.2-14B。

## 动机

视频后训练本质上比语言和图像后训练更难，体现在四个维度：
- **高 rollout 成本**：单条视频需数十到数百步扩散 + 帧解码，朴素 RL 不可行
- **时序误差累积**：伪影在帧间传播，帧级指标无法捕捉
- **模糊监督**：文本-视频映射是多对多的，同一 prompt 有多种合理实现
- **评估器脆弱性**：CLIP、VLM 评分器对解码参数敏感，随模型改善会饱和或偏置

因此后训练不能简单"加 RL"，需要系统级设计来管理不确定性、结构化学习信号、维持优化稳定性。

## 方法

### Stage I: 监督微调（SFT）

SFT 的角色不是优化输出质量，而是**策略塑形**——定义模型允许做什么、建立稳定的参考策略。

三个子模块：

**指令与控制 SFT**：
- 用指令导向数据（时间结构描述、相机行为、组合约束）训练模型遵循结构化 prompt
- 建立可预测、可组合的基线策略

**空间结构感知 SFT**：
- 针对相机运动下的 3D 结构退化（背景坍缩、物体变形、深度序紊乱）
- 通过结构评估信号（场景稳定性 + 物体几何一致性）进行损失重加权
- 结构畸变样本获得更强梯度，结构稳定样本降权
- 数据来源：真实视频 + 仿真数据 + 模型生成序列

**物理感知 SFT**：
- 针对流体和可变形材料的物理违反
- 联合真实流体视频 + 物理仿真数据训练
- 辅助运动预测分支：预测帧间光流，特征通过零初始化模块融入 RGB 解码器
- 仅更新解码器参数，冻结预训练 backbone

### Stage II: 基于 GRPO 的强化学习

#### GRPO 基线

采用 GRPO 消除对 value critic 的依赖（视频生成中训练 critic 计算量过高）：
- 对每个 prompt $c$，策略 $\pi_\theta$ 采样一组视频 $\{v_1, \ldots, v_G\}$
- 组内标准化得到相对 advantage：$A_i = \frac{r_i - \text{mean}(\{r\})}{\text{std}(\{r\}) + \epsilon}$
- 最大化 advantage 加权的代理目标，KL 约束防止偏离参考策略

#### ViPO: 时空信用分配

解决标量 reward 对视频过于粗粒度的问题（如视频整体好但第 3 秒手部崩坏，标量分数无法定位）：
- 用冻结视觉 backbone（DINOv2、VideoMAE）提取时空特征图
- 通过特征显著性分析构建 Advantage Map $M \in \mathbb{R}^{T \times H \times W}$
- 策略梯度按位置加权：$\mathcal{L}_{\text{ViPO}} = \mathbb{E}\left[\sum_{t,h,w} M_i^{(t,h,w)} \cdot A_i \cdot \log \pi_\theta(v_i)\right]$
- 效果：梯度重定向到视觉关键/失败区域，精确修复局部伪影而不破坏全局结构

#### BPGO: 贝叶斯先验信任分配

解决不同 prompt 组 reward 可靠性不一致的问题：
- **组间信任分配**（RAS）：将当前 rollout 组的 reward 分布与先验分布比较，高方差/大偏差组标记为"不可靠"并降权
- **组内先验锚定重归一化**（CRT）：以先验分数为基线拉伸高置信正样本的 advantage，压缩模糊样本的 advantage 空间
- 效果：防止在噪声 prompt 上的 reward hacking，在高置信信号上激进优化

#### Self-Paced GRPO: 自适应 Reward 课程

解决 reward 饱和导致训练停滞的问题：
- 实时监控生成器性能统计（reward 分布稀疏度、组内区分度）
- Phase 1（视觉保真度）→ Phase 2（时序一致性）→ Phase 3（语义对齐）自动过渡
- 确保模型始终处于"最近发展区"，获得有挑战性但有信息量的反馈

#### 多目标联合 Reward

将多 reward 平衡转化为 advantage 级多目标优化：

$$\min_{\{c_i\}} \left\|\sum_{i=1}^{N} c_i A_i\right\|^2 \quad \text{s.t.} \quad \sum_i c_i = 1, c_i \geq 0$$

相比梯度级优化避免了显式梯度计算的显存开销，保留多目标平衡的收益。

### Stage III: DPO 偏好对齐

以 Stage II 最终 checkpoint 为参考策略（非 SFT 模型），捕获 GRPO 无法量化的主观质量。

**偏好数据构造三策略**：
1. **Policy-on-Policy 硬负例**：从当前 GRPO 策略生成，VLM 评审团 + 启发式过滤器排序，选高语义重叠但质量差异大的对
2. **合成时序负例**（无判别器）：对高质量视频做时序反转/帧混洗/帧冻结构造负例，强制学习因果性和时序连贯
3. **人类专家标注**：小规模高质量数据，聚焦灯光一致性、叙事逻辑、情感基调等"软"标准

**训练细节**：
- 动态 $\beta$ 调度：起始高值维稳定，逐步退火允许探索偏好前沿
- 混合分辨率策略：偏好梯度主要在低分辨率/短时长计算（结构失败最明显处），扩大 batch size

### 基础设施优化

**Ray 时间复用**：同一 GPU 节点池按阶段动态切换（rollout → reward → update），消除空闲时间

**NVIDIA MPS 并行 Reward**：
- 轻量 reward worker 通过 MPS 逻辑分区在同一 GPU 并发运行
- 贪心搜索最优 worker 分组和配额分配，最小化 reward 阶段时延

**解耦梯度 DPO**：
- 标准 DPO 反传因共享参数依赖导致峰值显存 $O(L_w + L_l)$
- 将梯度分解为两个独立反传：$J_w(\theta) = -\eta \cdot \nabla_\theta \log \pi_\theta(y_w|x)$ 和 $J_l(\theta) = \eta \cdot \nabla_\theta \log \pi_\theta(y_l|x)$
- 峰值显存降为 $O(\max(L_w, L_l))$，零计算开销

## 实验

### 人类评估（vs Wan2.2-14B I2V）

| 评估维度 | WinRate↑ | Preference↑ | Margin↑ |
|---------|----------|-------------|---------|
| 视觉质量 | 58.71% | 52.17% | 4.35% |
| 运动质量 | 70.72% | 62.47% | 24.90% |
| 文本对齐 | 77.39% | 62.15% | 24.13% |
| 内容保持 | 63.28% | 54.15% | 8.15% |
| 综合 | **71.18%** | **66.38%** | **32.71%** |

运动质量和文本对齐的改善最显著（margin > 24%），说明 Stage II GRPO 的时序优化和 Stage III DPO 的语义对齐都起到关键作用。

### 消融：物理感知 SFT

在流体效果视频上，物理感知 SFT 在真实视频测试集上 EPE 0.538、>1px 14.7%，在仿真测试集上 EPE 1.541，证实辅助光流监督有效改善时序连贯性。

## 关键启示

- **三阶段后训练是互补而非冗余的**：SFT 塑形行为空间、GRPO 优化可度量目标、DPO 对齐主观偏好，各阶段解决不同层次的问题
- **ViPO 的时空信用分配是视频 GRPO 的关键增强**：标量 reward 对视频过于粗粒度，将 advantage 反投影到像素/时间级别可大幅提升样本效率
- **BPGO 的信任分配机制适用于所有多对多映射任务**：当 reward 在不同 prompt 上可靠性差异大时，贝叶斯先验可防止过拟合噪声
- **自适应 reward 课程是长期训练的必要条件**：静态 reward 模型随模型改善饱和，逐阶段提升难度保持训练动力
- **解耦梯度 DPO 是通用的显存优化**：利用 DPO 梯度的可分解性将峰值显存减半，不影响精度
- **物理感知 SFT 应在最早阶段引入**：物理违反被下游评估器弱惩罚，若不在 SFT 阶段显式解决会持续到最终模型
