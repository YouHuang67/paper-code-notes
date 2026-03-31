# 论文阅读

这里收录各领域的论文阅读笔记。

## Sparse Attention

- [HySparse](hysparse.md) - 混合稀疏注意力架构，full attention 层作为 oracle 选择 token 并共享 KV cache 给后续 sparse 层，80B MoE 仅 5 层 full attn 实现 10x KV cache 减少
- [Kascade](kascade.md) - 跨层 Top-k 复用的 training-free 稀疏注意力，decode 4.1x / prefill 2.2x 加速
- [IndexCache](indexcache.md) - DSA indexer 跨层索引复用，去除 75% indexer 计算，prefill 1.82x / decode 1.48x 加速

## 视频生成

### 视频生成对齐（RL / Human Feedback）

- [VideoAlign](videoalign.md) - 完整的视频生成人类偏好对齐流水线：182k 偏好数据集 + VideoReward（BTT loss + token 位置策略） + Flow-DPO/RWR/NRG 三种对齐算法
- [VideoDPO](videodpo.md) - 基于 OmniScore 多维度评分的视频 DPO，自动构造偏好对无需人工标注
- [DenseDPO](densedpo.md) - 段级密集偏好标注 + 引导采样，细粒度 DPO 对齐
- [DeDPO](dedpo.md) - 半监督 DPO 用双重稳健估计器去偏，解决运动偏差问题
- [SAGE-GRPO](sage_grpo.md) - 精确流形感知 SDE（对数曲率修正）+ 梯度范数均衡器 + 双信赖域 GRPO
- [VGGRPO](vggrpo.md) - Latent Geometry Model 拼接 VAE latent 到 Any4D，几何重投影一致性 reward + GRPO
- [Reward-Forcing](reward_forcing.md) - 自回归视频生成：少量 ODE 轨迹学运动先验 + ImageReward 引导纹理优化
- [PRFL](prfl.md) - 视频生成模型自身作为 latent reward 模型，过程级监督无需 VAE 解码
- [PhysMaster](physmaster.md) - PhysEncoder 物理表征学习 + 三阶段 Flow-DPO（SFT → DPO-DiT → DPO-PhysEncoder）
- [VPO](vpo.md) - 从 prompt 优化角度对齐视频生成，文本级+视频级多反馈 DPO

### 视频 Reward 模型

- [VR-Thinker](vr_thinker.md) - Thinking-with-Image 框架，推理时主动检索帧的视频 reward 模型
- [REACT](react.md) - 帧级结构性失真评估，8 类失真分类 + masked SFT + GRPO

### 自动驾驶视频生成

- [RLGF](rlgf.md) - 层级几何 reward（点-线-面-体素-特征）+ latent 空间滑动窗口 RL 优化

### 视频生成架构

- [SkyReels-V2](skyreels_v2.md) - 无限时长电影级视频生成（Diffusion Forcing + RL 运动优化）
