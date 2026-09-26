---
name: post
description: >-
  把一篇论文做成可上线的解读笔记并提交。用户在微信 /paper 里用 $post 触发。
  默认只解读论文本身。只有用户写明必须深入开源代码或算子实现时，才克隆仓库做代码级分析。
---

# $post

在 `paper-code-notes` 里完成一篇论文的解读，并提交到可部署的网页。先读本文件，再读 `.codex/skills/doc/SKILL.md` 和 `.codex/skills/commit/SKILL.md`。长任务同时遵守 `.codex/skills/exec/SKILL.md`。不要用子代理，不要发问卷。

## 深度

按用户原话选一档，不要自行加深。

- 默认：只解读论文已有内容。还原论证，中文写清楚，专名保留英文。补必要背景。不克隆代码，不展开算子。
- 用户写了「必须基于其提供的开源代码深入解析」：在默认之外，克隆官方仓库，对照论文核对模块、训练或推理入口、关键数据流。权重和数据不入库。
- 用户写了「必须深入解析其开源代码的算子设计以及其他所有具体实现部分」：在上一档之外，写清算子的张量形状、算法阶段和与论文公式的对应。仓库未开源或没有该算子时，写明缺了什么，不要编造实现。

## 顺序

1. 定位论文。日报里的短名先查 `refs/scans/daily/index.md`，再打开对应日期文件拿编号。PDF 放 `refs/papers/`，不提交。
2. 笔记写到 `docs/paper_reading/<短名>.md`。开头是论文链接、代码链接、团队，然后是「概述」。正文按论文顺序，公式和关键配置保留。末尾写可迁移的设计点。
3. 同步 `mkdocs.yml` 的 nav 和 `docs/data/tag_index.json`。索引里要有 `category` 和 `tags`。未选分类时出现在全部列表。`category` 用已有分类名。
4. 代码档仅在深度要求写到代码时才写，放 `docs/code_analysis/<项目>/`。克隆的仓库、权重、`refs/` 不提交。
5. 提交这三处：笔记、`mkdocs.yml`、`docs/data/tag_index.json`。说明用简短英文，不要 `Co-authored-by`。
6. 核对 HEAD 里这三处都在。缺一处就继续改，不要说已上线。
7. 执行 `git push origin main`。网页由 `.github/workflows/deploy.yml` 在 `main` 被推送后构建。不要手推 `gh-pages`。推送后用 `git status -sb` 确认没有领先 `origin/main`。
8. 回复笔记路径、深度档、提交号，以及页面地址 `https://YouHuang67.github.io/paper-code-notes/paper_reading/<短名>/`。Actions 未成功时写明还没构建完，不要说已经能看到。

日报不是论文页。不要把 `refs/scans/` 写进 `docs/`。

## 文风

概述能单独读完。正文不省略方法主链。表格只用于实验对比。不用 admonition。中文句子里保留 WanPE、GRPO 这类专名。句首用可指代的短名。
