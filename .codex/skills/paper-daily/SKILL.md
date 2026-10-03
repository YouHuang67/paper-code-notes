---
name: paper-daily
description: >-
  论文日报的主题、落盘位置和微信汇报方式。/paper 里用户提到日报、主题或 PE 时先读本文件。
---

# 论文日报

`/paper` 里凡是日报、主题或 PE，先读本文件，再打开它指出的两个数据文件。用户当前这句话优先于日报。

## 打开哪些文件

1. `scripts/paper-daily.json`
   主题名、`scope`（收）、`reject`（不收）、全天篇数上限 `maxResults`。主题名只认这里的 `name`。
2. `refs/scans/daily/index.md`
   按日期进入 `refs/scans/daily/by-date/<YYYY-MM-DD>.md`，优先今天。

`refs/scans/daily/` 下的时间戳目录是旧发送缓存，不读。那里的主题名已经作废。

## 汇报

用户要看日报、重看某一天、或按主题找论文时，用这个版式：

```text
【主题名】

— 英文短名：一句中文，写清对象和做法。
```

主题名用全角【】，论文用长横线 `—`。一条论文一句。发送层会加 `[论文]`，回复正文里不再写。没有论文的主题合成一行：`无更新：主题甲、主题乙`。

转述 `by-date` 之前，用 `scripts/paper-daily.json` 的 `reject` 再滤一遍。对象落在「不收」里的条目不放进该主题。

## 主题名

PE 指 `VLM调PE`：图像或视频的 prompt enhancer，或用 VLM 改写、评估这类 prompt。

下列旧名不再使用：PE文本蒸馏、SFT拟合、偏好负样本、训练监测。文件里若出现这些字样，不当作主题，回到 `scripts/paper-daily.json` 的 `name`。

当前主题分两组。视频类：视频后训练、视频高效计算、视频强化学习、视频奖励、VLM调PE、信息图后训练。语言模型类：拟合与蒸馏、样本偏好、奖励强化学习、注意力现象、训练信号。每一类收什么、不收什么，以 json 里该主题的 `scope` 和 `reject` 为准。

## 维护配置

新增或改范围时，只改 `scripts/paper-daily.json`。每个主题都要有 `name`、`scope`、`reject`、`enabled`。`scope` 与 `reject` 进入 DeepSeek 的 system prompt。缺任何一项，`scripts/paper_daily.py` 退出。

`reject` 写最容易混进来的邻近工作，例如视频主题里的 LLM 缓存、VLA、唇读，以及拟合与蒸馏里的扩散蒸馏。不要把范围写成一串检索词。

全天所有主题合计不超过 `maxResults`，当前是 30。超过时按主题轮流保留。单主题不再设 8 篇上限。

抓取用 OAI 当天记录，但只保留首次提交日。周二到周五保留当天和前一天，周一保留周五到当天，这样跨月编号（10 月 1 日仍是 `2609.*`）不会被丢掉，旧论文的修订也不会因为当天被更新而混进来。

改完后用一天 `--dry-run` 看归类，确认再发。`refs/scans/`、`refs/papers/`、`refs/research/` 的抓取结果不提交。定时是北京时间工作日 09:30，命令是 `python3 scripts/paper_daily.py run`。
