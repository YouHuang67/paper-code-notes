---
name: paper-daily
description: >-
  维护论文日报的主题、DeepSeek 归类范围和微信版式。用户要添加、收窄或解释每日推送主题时使用。
---

# 论文日报主题

每日推送读 `scripts/paper-daily.json`。分类由 DeepSeek 按每条 `scope` 和 `reject` 判断论文对象，完整流水线由 `scripts/daily_pipeline.py` 管理。`scripts/paper_daily.py` 是主题和 cron 管理兼容入口；定时任务调用 `scripts/daily_job.sh`，状态与审计保存在 `refs/scans/daily-runs/YYYY-MM-DD/`。日报只有在公开网页内容指纹和微信桥接 `ok=true` 回执都成立后才记入账本。

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

DeepSeek 使用 system prompt。其中每个主题有「收」和「不收」。模型接收完整摘要，每篇最多一个主题，并经过独立复核。入选论文继续读取 v1 PDF 正文，正文证据、结构化解读和审查结果全部缓存。全部主题一天合计不超过 `maxResults`（当前 30）；没有通过证据校验的结果进入失败状态，不生成空日报。

## 版式

消息以 `[论文]` 开头。日期、【主题】、每条「— 句子」各自成段，段之间空一行。主题用全角【】，论文用长横线。不要用圆点。

`scripts/paper_daily.py` 生成摘要正文。发送流程附加日报网页地址时，必须把“日报网页”单独放一行，下一段只放完整裸 URL。URL 必须从行首开始；链接前不加项目符号、编号、括号、Markdown 包装或“链接：”等前缀，行尾不加标点。发送前按 `docs/design/wechat-link-delivery-sop.md` 检查最终消息正文。

`/paper` 需要查询日报时读取 `refs/scans/daily-runs/YYYY-MM-DD/report.md` 与 `state.json`，并以公开 `docs/daily/` 页面为最终链接。历史 `refs/scans/daily/` 目录仅作兼容读取，不再作为生成目标。

## 当前主题

范围以 `scripts/paper-daily.json` 的 `scope` 和 `reject` 为准。视频类是视频后训练、视频高效计算、视频强化学习、视频奖励、VLM调PE、信息图后训练。语言模型类是拟合与蒸馏、样本偏好、奖励强化学习、注意力现象、训练信号。

PE 只指 VLM调PE，也就是图像或视频的 prompt enhancer。旧名 PE文本蒸馏、SFT拟合、偏好负样本、训练监测已经作废，不能再当主题名。
