# 日报生成、发布与微信交付

## 交付目标

北京时间 09:30 推送简文和当天完整日报网页。网页包含入选论文的完整标题、作者、原文摘要、v1 论文链接、问题、方法、实验范围、判断及可核验引句。微信最多显示八条短摘要，其余入选论文可在网页阅读；一天最多入选三十篇，按主题轮流选取。

美国东部时间周日至周四 20:00 的 arXiv 公告对应北京周一至周五。夏令时北京 08:00、冬令时 09:00 开始可见。任务在公告后五分钟开始准备，08:00–12:55 每五分钟重试。发送门槛为 09:30；来源、模型或部署延迟时，任务记录延迟并继续尝试。本机、cron、微信桥接和网络需保持运行。

## 今天问题复盘

2026-10-12 旧流程以 904 个输入、46 个 DeepSeek 批次选出 30 篇，微信发送了 3061 字，但未生成对应网页。模型只接收截断摘要，语言模型稀疏计算和驾驶蒸馏混入其他主题。2026-10-13 OAI 来源返回零条后，旧流程直接发送了十四字的空日报，未确认这是正常无公告还是来源失败。10:02 与 12:41 的微信交互回复成功，完成了问题描述和计划，未完成流程修复。

官方公布美国东部时间 2026-10-12 为假日，因此对应北京 10-13 需要跳过。这支持节假日解释；单凭旧 OAI 零条记录不能证明全部故障原因。此次实现分别处理公告日期、模型证据、网页部署和微信成功回执。

## 状态与门禁

状态文件位于 `refs/scans/daily-runs/YYYY-MM-DD/state.json`。`prepared` 表示本地报告已完成；`publishing` 表示 Git 已推送、公开网页仍待核验；`published` 表示内容指纹已公开；`delivered` 表示成功微信回执已经确认并写入账本。

`pending` 表示来源或部署尚未就绪，`failed` 表示来源、解析或模型处理失败，两者继续重试。周末和配置中的美国东部节假日为 `skip_no_announcement`；完整公告中没有配置类别论文，或全体候选均未通过复核时为 `no_matches`；候选均已在账本时为 `already_processed`。这些终态保留审计，不发送空日报。

DeepSeek 对每个候选完整摘要返回判定、理由、主题、连续原文引句和排除条件。接受项独立复核后，顺序下载 v1 PDF，正文按 24000 字符分块读取。所有分块证据进入解读和审查，引用必须存在于原文。输出截断、空对象、漏 ID、伪造引句和审查失败均阻止发布。成功输出按输入哈希和提示版本缓存，重复运行会复用结果。

发布使用专属 worktree `refs/scans/daily-runs/publisher`，只提交当天页面、索引和导航。发送前请求公开网页，验证 HTTP 成功且包含本次 SHA-256 指纹。微信使用日期加内容指纹构成稳定 `run_id`，桥接日志中的 `digest_sent ... ok=true` 才表示成功；RPC 正常返回本身不能证明已送达。相同回执可恢复账本并避免重发。

## 运维命令

在仓库根目录执行：

```bash
python3 scripts/paper_daily.py install
python3 scripts/daily_pipeline.py status --day 2026-10-14
python3 scripts/daily_pipeline.py prepare --day 2026-10-14
python3 scripts/daily_pipeline.py publish --day 2026-10-14
python3 scripts/daily_pipeline.py deliver --day 2026-10-14 --dry-run
python3 scripts/daily_pipeline.py tick
```

`prepare` 调用真实 DeepSeek 并产生本地文件；`--dry-run` 的准备阶段同样会调用模型。`publish --dry-run` 只读状态；`deliver --dry-run` 验证网页门禁但不发微信。手动 `deliver` 允许补发；自动 `tick` 执行 09:30 时间门禁。

`daily_job.sh` 在专属 `exec-paper-daily` tmux 中执行一次 `tick`，退出后会话自行结束。文件锁防止手动命令与 cron 并发。日志为 `refs/scans/paper-daily-cron.log`。检查 `crontab -l` 和 cron 服务状态；不要清理其他 tmux 会话。

## 审计与维护

运行目录保存 `source/`、`candidates.json`、`decisions.json`、`confirmed.json`、`papers/`、`ds/`、`report.md`、`report.json` 和 `wechat.txt`。同一天重试会复用已核验的候选快照；微信旧索引 `refs/scans/daily/index.md` 同步指向新运行目录。密钥从 `DEEPSEEK_API_KEY` 或被忽略的 `refs/deepseek_api` 读取；本机兼容读取 `.bashrc` 的 DeepSeek 专用 token，前提是配置的服务域名为 `api.deepseek.com`。密钥不写入审计。

每年按 [arXiv availability](https://info.arxiv.org/help/availability.html) 更新 `noAnnouncementDatesET`。arXiv robots 的当前最短请求间隔为十五秒，下载始终串行。机构原文可核实、名单规模尚未核实的条目使用“单位层级待核实”。当前页面请求最多显示 2000 条；列表不完整时阻止运行，后续需要完整分页支持。微信日志轮转后若丢失回执且本地账本尚未写入，人工核对历史发送记录再恢复状态。

回归验证执行 `python3 -m unittest discover -s scripts/tests -v`、`python3 scripts/validate_daily.py` 和 `.venv/bin/python -m mkdocs build`。详细规则见 [日报规范](daily-report-spec.md)，链接版式见 [微信链接交付 SOP](wechat-link-delivery-sop.md)。研究原始证据位于本机 `refs/research/daily-delivery-pipeline/`。

2026-10-14 实测：当天 `/list/cs/new` 请求仍返回 10-12 日期，系统进入 `pending` 并保持静默。`scripts/daily_smoke.py` 在已采集的官方快照上验证 OPD 正例、驾驶蒸馏负例和 KV cache 负例，随后处理 112686 字符论文正文并生成通过审核的单篇本地报告。该冒烟报告不作为当天日报发布。

本地临时 Git 仓库的发布回归确认只推送当天页面、索引和导航，并保留工作区未提交编辑；该测试使用本地 bare 仓库，不向外部发送。全部 18 项回归通过。微信成功/失败回执、旧页面阻止发送和成功回执恢复均通过故障注入验证；本次没有真实当天日报可发送，因此未执行新的微信日报端到端发送。
