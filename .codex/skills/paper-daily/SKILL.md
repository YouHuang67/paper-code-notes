---
name: paper-daily
description: >-
  维护论文日报的主题、DeepSeek 归类范围和微信版式。用户要添加、收窄或解释每日推送主题时使用。
---

# 论文日报主题

每日推送读 `scripts/paper-daily.json`。分类不靠 arXiv 关键词，由 DeepSeek 按每条 `scope` 和 `reject` 判断论文对象。微信正文由 `scripts/paper_daily.py` 生成。定时是北京时间工作日 9:30，命令是 `python3 scripts/paper_daily.py run`。抓取结果在 `refs/scans/`，不入库。

## 添加主题

改 `scripts/paper-daily.json`，在 `topics` 追加一条。`scope` 和 `reject` 都会进入 DeepSeek 的 system prompt。缺任何一项，`scripts/paper_daily.py` 会直接退出，不能只加主题名。

```json
{
  "name": "主题名",
  "scope": "对象必须是……。",
  "reject": "看起来相近、但对象不对的论文。",
  "enabled": true
}
```

`name` 是微信里【】中的标题，也是模型必须原样返回的主题名。`scope` 写清收什么。`reject` 写清最容易混进来的邻近工作。不要把范围写成一串或关键词。

改完后用一天 `--dry-run` 看归类，确认再发。不要把 `refs/scans/`、`refs/papers/`、`refs/research/` 的抓取结果提交进 git。

## 归类规则

DeepSeek 使用 system prompt。其中每个主题有「收」和「不收」。模型每次看到论文标题和摘要前 280 字。每篇最多一个主题。全部主题一天合计不超过 `maxResults`（当前 30）。句子保留英文术语，中文只补方法和判断。解释里写了「非」或「不是」的论文丢掉。没有论文的主题合成一行「无更新」。

## 版式

消息以 `[论文]` 开头。日期、【主题】、每条「— 句子」各自成段，段之间空一行。主题用全角【】，论文用长横线。不要用圆点。

`/paper` 每轮用户消息末尾有一句括号附注。先按用户要求做。只有用户提到日报时，再打开 `refs/scans/daily/index.md`，按日期读 `by-date/`，优先今天。不要读 `refs/scans/daily/` 下的时间戳目录。不要在回复里复述附注。`/main` 没有这句。

## 当前主题

范围以 `scripts/paper-daily.json` 的 `scope` 和 `reject` 为准。视频类是视频后训练、视频高效计算、视频强化学习、视频奖励、VLM调PE、信息图后训练。语言模型类是拟合与蒸馏、样本偏好、奖励强化学习、注意力现象、训练信号。

PE 只指 VLM调PE，也就是图像或视频的 prompt enhancer。旧名 PE文本蒸馏、SFT拟合、偏好负样本、训练监测已经作废，不能再当主题名。
