# 论文阅读

这里收录各领域的论文阅读笔记。

## Sparse Attention

- [HySparse](hysparse.md) - 混合稀疏注意力架构，full attention 层作为 oracle 选择 token 并共享 KV cache 给后续 sparse 层，80B MoE 仅 5 层 full attn 实现 10x KV cache 减少
- [Kascade](kascade.md) - 跨层 Top-k 复用的 training-free 稀疏注意力，decode 4.1x / prefill 2.2x 加速
- [IndexCache](indexcache.md) - DSA indexer 跨层索引复用，去除 75% indexer 计算，prefill 1.82x / decode 1.48x 加速

## 视频生成

- [SkyReels-V2](skyreels_v2.md) - 无限时长电影级视频生成（Diffusion Forcing + RL 运动优化）
