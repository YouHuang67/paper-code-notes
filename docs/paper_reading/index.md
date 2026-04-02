# 论文阅读

这里收录各领域的论文阅读笔记。

## Vision Language Model

- [Qwen2.5-VL](qwen25_vl.md) - Qwen VLM 旗舰，ViT Window Attention + MRoPE 绝对时间对齐 + 4.1T token 预训练，72B 达 GPT-4o 水平
- [Qwen3-VL](qwen3_vl.md) - Qwen VLM 最强版本，Interleaved MRoPE + DeepStack 多层 ViT 注入 + 256K 原生长上下文，dense/MoE 六规模
- [Qwen3-Omni](qwen3_omni.md) - 首个四模态零退化单一模型，Thinker-Talker MoE + AuT 音频编码器 + 多 Codebook 流式语音，234ms 首包延迟

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
- [VIGOR](vigor.md) - 视频生成 GRPO 框架：多粒度 reward + 帧级/视频级双层 GRPO
- [TeleBoost](teleboost.md) - Telescoping Boosting：多智能体迭代蒸馏框架，每轮 reward 提升 → 重新蒸馏
- [PISCES](pisces.md) - 双鱼对抗系统：verifier + falsifier 对抗训练的视频生成对齐
- [DDRL](ddrl.md) - 数据正则化扩散 RL：前向 KL 锚定数据分布防止 reward hacking，工业级验证
- [Euphonium](euphonium.md) - 过程 Reward 梯度注入 SDE 漂移项 + 双重 Reward GRPO，统一 Flow-GRPO 和 DanceGRPO
- [RealDPO](realdpo.md) - 真实视频锚定 DPO：用真实视频作为偏好锚点，避免合成数据偏差
- [TAGRPO](tagrpo.md) - 记忆库对比 I2V 对齐：高质量记忆库渐进提升生成下界
- [DreaMontage](dreamontage.md) - 多参考图对齐视频生成：CLIP-I2T reward 驱动 GRPO
- [DPP-GRPO](dppgrpo.md) - 行列式点过程 GRPO：DPP 采样替代随机采样增加组内多样性
- [AR-CoPO](arcopo.md) - 流式 AR 模型 chunk-level forking + semi-on-policy 训练，解决 few-step CM 的 SDE 失效
- [What Happens Next](whathappensnext.md) - 统一视频理解+生成的下一场景预测，因果一致性 reward + GRPO
- [McSc](mcsc.md) - 多维 self-critic 推理式 RM + 运动校正 DPO，解决运动偏差 reward hacking
- [MapReduce LoRA](mapreducelora.md) - 迭代式 LoRA 专家合并的多偏好对齐，Pareto 前沿逐步推进
- [VANS](vans.md) - Joint-GRPO 对齐 VLM+VDM，视频作为答案的下一事件预测
- [Identity-GRPO](identitygrpo.md) - 多人身份保持 GRPO：专用身份 RM + BTT + 大组差异化噪声采样
- [AR-Drag](ardrag.md) - Self-Rollout + selective stochasticity 实现 few-step AR 视频的实时运动控制
- [LocalDPO](localdpo.md) - 局部化偏好优化：真实视频 vs 局部腐蚀修复构建偏好对，region-aware DPO
- [CPS](cps.md) - 系数保持采样器消除 Flow-SDE 噪声伪影，更准确的 reward 建模

### 视频生成对齐综述

- [GRPO Survey](grposurvey.md) - Flow-GRPO 及后续 200+ 论文综述：reward 设计、credit assignment、采样效率、多样性保持、reward hacking 缓解

### 视频 Reward 模型

- [VR-Thinker](vr_thinker.md) - Thinking-with-Image 框架，推理时主动检索帧的视频 reward 模型
- [REACT](react.md) - 帧级结构性失真评估，8 类失真分类 + masked SFT + GRPO
- [VideoScore2](videoscore2.md) - CoT 推理式视频评估：三维度评分（VQ/TA/PC）+ SFT → GRPO 两阶段训练
- [RewardDance](rewarddance.md) - 生成式 reward 范式：VLM 预测 "yes" 概率做 reward，1B→26B 系统化扩展

### 自动驾驶视频生成

- [RLGF](rlgf.md) - 层级几何 reward（点-线-面-体素-特征）+ latent 空间滑动窗口 RL 优化

### 视频生成架构

- [SkyReels-V2](skyreels_v2.md) - 无限时长电影级视频生成（Diffusion Forcing + RL 运动优化）
- [Seedance 1.5](seedance15.md) - 音视频联合生成：Native AV-DiT + 多维 RLHF + 10× 推理加速
- [CAMVERSE](camverse.md) - 高质量 I2V + 可控运动：MoRA 运动适配器 + 精细镜头运动控制
- [Kandinsky 5.0](kandinsky5.md) - 统一图像/视频/音频生成基础模型族，多阶段课程训练 + RLHF
