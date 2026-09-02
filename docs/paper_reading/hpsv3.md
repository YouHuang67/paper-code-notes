---
tags:
  - Reward Model
  - Reinforcement Learning
  - VLM
  - Post Training
---

# HPSv3: Towards Wide-Spectrum Human Preference Score

- 论文：https://arxiv.org/abs/2508.03789
- 代码：https://github.com/MizzenAI/HPSv3
- 团队：Mizzen AI, CUHK MMLab, King's College London, Shanghai Jiao Tong University, Shanghai AI Lab
- 发表：ICCV 2025

## 概述

HPSv3 是 CLIP 系人类偏好分（HPS / HPSv2 / PickScore / ImageReward）在 2025 年的主干升级：把训练数据从「中低质量扩散图」扩到包含 DiT、自回归模型和高质量真实图的宽光谱 HPDv3（1.08M 图文对、1.17M 成对标注），骨干从 CLIP/BLIP 换成 Qwen2-VL-7B，损失从确定性 pairwise logistic / Bradley-Terry 换成把标量 reward 建模为一维高斯的 uncertainty-aware ranking。附录证明常见 KL 偏好损失与 BT 损失在优化目标上等价于同一 pairwise logistic，因此 HPSv3 仍属 BT 路线，只是对标注噪声显式建模。在 HPDv3 测试集上偏好准确率 76.9%（HPSv2 65.3%，PickScore 65.6%）。下游既可作 DanceGRPO 等 RL 的 reward，也可驱动无额外训练的 CoHP（模型选择 + 样本迭代精炼）。

## 动机

- 旧人类偏好数据集质量谱过窄，主要是扩散模型输出，难以评价 FLUX、Infinity 等新架构。
- CLIP/BLIP 特征对多模态排序不够，确定性 $P(x_1 \succ x_2)=\mathrm{sigmoid}(r_1-r_2)$ 把所有对当作同样置信，硬例和标注冲突会被同等惩罚。

## 数据：HPDv3

三路来源：

- 扩展 HPDv2：额外 10 个当时 SOTA 生成模型出图
- 互联网高质量照片，VLM 配 caption 后再生成对照图
- Midjourney：每 prompt 四图 + 用户选择

成对比较由 9–19 名标注员完成。另抽 1000 prompt × 11 模型构成 132k 对的 HPDv3 Benchmark。

## 方法：BT + 不确定度

骨干：Qwen2-VL-7B，全参可训。图像 448×448 保比例。最后两层线性头分别预测 $\mu,\sigma$，reward $r\sim\mathcal{N}(\mu,\sigma)$。偏好概率对两高斯积分：

$$P(x_1 \succ x_2|c)=\iint \mathrm{sigmoid}(r_1-r_2)\, \mathcal{N}(r_1|\mu_1,\sigma_1)\,\mathcal{N}(r_2|\mu_2,\sigma_2)\,dr_1 dr_2$$

损失为胜者对的负对数似然。确定性 RankNet/BT 是 $\sigma\to 0$ 的特例。

训练：1.5M 成对样本，2 epoch，48×A800，LR $2\times 10^{-6}$，warmup 0.05，全局 batch 384。

## CoHP

无参推理时缩放：先对候选生成器做 model-wise 选择（HPSv3 打分），再对赢家模型做 sample-wise 加噪重绘，逐步选最高分。实验用 Flux-dev / Playground v2.5 / SD3 / Kolors，两阶段各 4 round。

## 实验

与人类排序相关（HPDv3 benchmark）：Spearman 0.94，Kendall 0.82，高于 HPSv2（0.87 / 0.76）和 PickScore（0.81 / 0.63）。

偏好准确率（%）：

| 模型 | ImageReward | PickScore | HPDv2 | HPDv3 |
|------|:---:|:---:|:---:|:---:|
| ImageReward | 65.1 | 61.1 | 74.0 | 58.6 |
| PickScore | 61.6 | 70.5 | 79.8 | 65.6 |
| HPSv2 | 65.7 | 63.8 | 83.3 | 65.3 |
| HPSv3 | 66.8 | 72.8 | 85.4 | **76.9** |

消融：Qwen2VL-7B + RankNet 已强于 CLIP；换成 uncertainty loss 后 PickScore 测试集再 +2.2 点。DanceGRPO（SD1.4，300 iter）上 HPSv3 比 HPSv2 更少「堆装饰物」类 hacking。

## 关键启示

- 视觉 BT 线在 2025 的有效升级路径是「宽光谱成对数据 + VLM 骨干 + 对标注噪声建模」，不必改成生成式 judge。
- 论文明确写出 BT 与 KL 偏好损失同为 $\log(1+e^{r_l-r_h})$，读后续 VideoReward / HPSv3 引用时可以把它们放在同一优化族。
- HPSv3 仍是点式标量 RM，适合 GRPO 的 $O(n)$ 打分；代价是可解释性弱于 checklist / CoT GRM。
