from __future__ import annotations

"""Search arXiv papers by topic and filter with DeepSeek AI.

Three-stage pipeline:
  1. DeepSeek generates search queries from user topic
  2. arXiv API search + DeepSeek per-paper relevance filtering
  3. DeepSeek summarizes results into a Markdown report

Usage:
    python scripts/scan_arxiv.py "视频生成中的强化学习对齐"
    python scripts/scan_arxiv.py "sparse attention" --days 30
    python scripts/scan_arxiv.py "sparse attention" --from 20260301 --to 20260401
    python scripts/scan_arxiv.py "视频生成" --max-results 100
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import arxiv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT_DIR / "refs" / "papers"
SCANS_DIR = ROOT_DIR / "refs" / "scans"
API_KEY_FILE = ROOT_DIR / "refs" / "deepseek_api"


def load_deepseek_client() -> OpenAI:
    """Load DeepSeek API client from local key file."""
    key = API_KEY_FILE.read_text().strip()
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def chat(client: OpenAI, prompt: str, temperature: float = 0.3) -> str:
    """Call DeepSeek chat API and return response text."""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def get_existing_arxiv_ids() -> set[str]:
    """Scan refs/papers/ to collect already-downloaded arXiv base IDs."""
    ids = set()
    if not PAPERS_DIR.exists():
        return ids
    for item in PAPERS_DIR.iterdir():
        match = re.search(r"(\d{4}\.\d{4,5})", item.name)
        if match:
            ids.add(match.group(1))
    return ids


def parse_date_range(args) -> tuple[str, str] | None:
    """Build arXiv submittedDate range string from CLI args."""
    if args.date_from and args.date_to:
        return f"{args.date_from}0000", f"{args.date_to}2359"
    if args.days:
        end = datetime.now()
        start = end - timedelta(days=args.days)
        return start.strftime("%Y%m%d") + "0000", end.strftime("%Y%m%d") + "2359"
    return None


# =========================================================================
# Stage 1: DeepSeek generates arXiv search queries
# =========================================================================

STAGE1_PROMPT = """\
你是一个学术检索助手。用户想在 arXiv 上搜索以下研究主题的论文：

【研究主题】：{topic}
{exclude_section}
请根据这个主题，生成适合 arXiv API 的搜索查询。要求：
1. 生成 3-6 组不同角度的查询关键词组合，覆盖该主题的同义词、相关术语和子方向
2. 查询语法使用 arXiv API 格式：all:"xxx" AND all:"yyy"（精确匹配用双引号）
3. 如果有排除关键词，每条查询末尾必须加上 ANDNOT all:"排除词" 来过滤不相关领域
4. 推荐最相关的 arXiv 分类（如 cs.CV, cs.LG, cs.CL 等）

请严格按以下 JSON 格式返回，不要输出任何其他内容：
```json
{{
  "queries": [
    "all:\\"keyword1\\" AND all:\\"keyword2\\"",
    "all:\\"keyword3\\" AND all:\\"keyword4\\""
  ],
  "categories": ["cs.CV", "cs.LG"]
}}
```
"""


def stage1_generate_queries(
    client: OpenAI, topic: str, work_dir: Path,
    exclude: list[str] | None = None,
) -> dict:
    """Use DeepSeek to generate arXiv search queries from topic."""
    print("[阶段1] 生成检索关键词...")
    if exclude:
        exclude_text = (
            "\n【排除方向】：" + exclude
            + "\n（涉及以上方向的论文不是用户的目标，必须在查询中用 ANDNOT 排除）\n"
        )
    else:
        exclude_text = ""
    prompt = STAGE1_PROMPT.format(topic=topic, exclude_section=exclude_text)
    raw = chat(client, prompt)

    json_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(1)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]

    result = json.loads(raw)
    out_path = work_dir / "step1_queries.json"
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"  查询组数: {len(result['queries'])}")
    print(f"  推荐分类: {result.get('categories', [])}")
    print(f"  保存: {out_path}")
    return result


# =========================================================================
# Stage 2: arXiv search + per-paper DeepSeek filtering
# =========================================================================

STAGE2_PROMPT = """\
你是一个学术论文筛选助手。用户正在研究以下主题：

【研究主题】：{topic}
{exclude_section}
请阅读下面这篇论文的信息，判断它与上述研究主题的相关程度。

【论文标题】：{title}
【发表时间】：{published}
【作者】：{authors}
【备注】：{comment}
【摘要】：{summary}

请严格按以下 JSON 格式返回，不要输出任何其他内容：
```json
{{
  "relevance": "高相关/中相关/低相关/不相关",
  "contribution": "一句话概括这篇论文的核心贡献（中文）",
  "reason": "简述判断理由（中文，2-3句话）",
  "team": "根据作者名单判断团队背景（中文，如：Google DeepMind / 清华大学 / 未知小团队 等）"
}}
```

判断标准：
- 高相关：论文直接针对该研究主题，提出了新方法/新架构/新发现
- 中相关：论文涉及该主题的相关技术或上下游问题，有参考价值
- 低相关：论文与该主题仅有表面关联，核心内容不同
- 不相关：论文明显属于排除方向或与主题完全无关，不纳入报告

团队判断说明：
- 根据你的知识判断作者所属机构（知名高校/研究院/大厂/明星团队）
- 如果无法确认，标注"未知"即可，不要编造
- 备注字段（comment）中可能包含机构或会议信息，注意利用{exclude_rule}
"""


def _run_one_query(
    arxiv_client: arxiv.Client,
    full_query: str,
    max_results: int,
    all_papers: dict[str, dict],
) -> int:
    """Execute a single arXiv query with application-level retry + backoff."""
    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    max_retries = 3
    for attempt in range(max_retries):
        try:
            count = 0
            for result in arxiv_client.results(search):
                arxiv_id = result.get_short_id().split("v")[0]
                if arxiv_id not in all_papers:
                    all_papers[arxiv_id] = {
                        "arxiv_id": arxiv_id,
                        "title": result.title,
                        "authors": [a.name for a in result.authors],
                        "comment": result.comment or "",
                        "published": result.published.strftime("%Y-%m-%d"),
                        "summary": result.summary.replace("\n", " "),
                        "pdf_url": result.pdf_url,
                        "categories": result.categories,
                    }
                    count += 1
                print(".", end="", flush=True)
            return count
        except Exception as e:
            wait = 30 * (attempt + 1)
            if attempt < max_retries - 1:
                print(
                    f" 被限流({e.__class__.__name__})，"
                    f"等待 {wait}s 后重试 ({attempt + 2}/{max_retries})...",
                    end="", flush=True,
                )
                time.sleep(wait)
            else:
                print(f" 重试耗尽: {e}")
                return -1
    return -1


def fetch_arxiv_papers(
    queries: list[str],
    categories: list[str],
    date_range: tuple[str, str] | None,
    max_results_per_query: int,
) -> dict[str, dict]:
    """Run multiple arXiv queries, merge and deduplicate by base ID."""
    all_papers: dict[str, dict] = {}
    arxiv_client = arxiv.Client(
        page_size=20, delay_seconds=5.0, num_retries=0,
    )

    for i, query in enumerate(queries):
        if i > 0:
            print(f"    查询间隔等待 15s...", flush=True)
            time.sleep(15)

        if categories:
            cat_filter = " OR ".join(f"cat:{c}" for c in categories)
            full_query = f"({query}) AND ({cat_filter})"
        else:
            full_query = query

        if date_range:
            full_query += (
                f" AND submittedDate:[{date_range[0]} TO {date_range[1]}]"
            )

        print(f"  查询 {i + 1}/{len(queries)}: {full_query[:80]}...")
        print(f"    ", end="", flush=True)

        count = _run_one_query(
            arxiv_client, full_query, max_results_per_query, all_papers,
        )
        if count >= 0:
            print(f" {count} 篇新增（累计 {len(all_papers)} 篇）")
        else:
            print(f"    跳过此查询，继续...")

    return all_papers


def stage2_search_and_filter(
    client: OpenAI,
    topic: str,
    queries_data: dict,
    date_range: tuple[str, str] | None,
    max_results: int,
    work_dir: Path,
    exclude: str | None = None,
) -> list[dict]:
    """Search arXiv and filter each paper with DeepSeek."""
    print("\n[阶段2] arXiv 检索 + 逐篇筛选...")

    all_papers = fetch_arxiv_papers(
        queries_data["queries"],
        queries_data.get("categories", []),
        date_range,
        max_results,
    )
    print(f"\n  去重后共 {len(all_papers)} 篇")

    existing_ids = get_existing_arxiv_ids()
    skipped = 0
    to_filter = {}
    for aid, paper in all_papers.items():
        if aid in existing_ids:
            skipped += 1
        else:
            to_filter[aid] = paper
    if skipped:
        print(f"  已入库跳过: {skipped} 篇")
    print(f"  待筛选: {len(to_filter)} 篇")

    results_path = work_dir / "step2_results.jsonl"
    already_done = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                already_done.add(rec["arxiv_id"])
        if already_done:
            print(f"  已有中间结果: {len(already_done)} 篇（断点续传）")

    results = []
    total = len(to_filter)
    for idx, (aid, paper) in enumerate(to_filter.items(), 1):
        if aid in already_done:
            for line in results_path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec["arxiv_id"] == aid:
                    results.append(rec)
                    break
            continue

        print(f"  [{idx}/{total}] {paper['title'][:60]}...")
        if exclude:
            exclude_sec = "\n【排除方向】：" + exclude + "\n"
            exclude_rule = (
                "\n- 强制规则：如果论文的核心应用场景或研究对象属于排除方向"
                "（如机器人操控、世界模型预测等），无论使用了什么技术方法，"
                "必须判定为「不相关」。仅当论文的主要贡献完全不涉及排除方向时，"
                "才考虑高/中/低相关"
            )
        else:
            exclude_sec = ""
            exclude_rule = ""
        prompt = STAGE2_PROMPT.format(
            topic=topic,
            title=paper["title"],
            published=paper["published"],
            authors=", ".join(paper["authors"]),
            comment=paper.get("comment", "") or "无",
            summary=paper["summary"],
            exclude_section=exclude_sec,
            exclude_rule=exclude_rule,
        )

        try:
            raw = chat(client, prompt, temperature=0.2)
            json_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
            if json_match:
                raw = json_match.group(1)
            analysis = json.loads(raw.strip().strip("`"))
        except Exception as e:
            print(f"    分析失败: {e}")
            analysis = {
                "relevance": "低相关",
                "contribution": "分析失败",
                "reason": str(e),
            }

        record = {**paper, **analysis}
        results.append(record)

        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        rel = analysis.get("relevance", "?")
        print(f"    → {rel}: {analysis.get('contribution', '')[:50]}")
        time.sleep(0.5)

    stats = {"total": len(all_papers), "skipped": skipped, "filtered": total}
    return results, stats


# =========================================================================
# Stage 3: Generate Markdown report
# =========================================================================

STAGE3_PROMPT = """\
你是一个学术总结助手。以下是围绕研究主题「{topic}」从 arXiv 检索并筛选出的论文列表。

{paper_list}

请完成以下两个任务：

1. 写一段 100-200 字的中文总结，概括这批论文的整体研究趋势和热点方向。

2. 根据研究主题，给出一个简洁的中文文件名关键词（3-6个汉字，不含日期前缀，\
不含空格，用下划线连接），用于命名检索报告。

请严格按以下 JSON 格式返回：
```json
{{
  "summary": "总体趋势分析文本...",
  "filename_keyword": "视频生成RL对齐"
}}
```
"""


def stage3_generate_report(
    client: OpenAI,
    topic: str,
    results: list[dict],
    stats: dict,
    date_range_str: str,
    work_dir: Path,
) -> Path:
    """Generate final Markdown report from filtered results."""
    print("\n[阶段3] 生成检索报告...")

    groups = {"高相关": [], "中相关": [], "低相关": []}
    excluded_count = 0
    for r in results:
        rel = r.get("relevance", "低相关")
        if rel == "不相关":
            excluded_count += 1
            continue
        if rel not in groups:
            rel = "低相关"
        groups[rel].append(r)

    paper_summary_lines = []
    for r in results:
        if r.get("relevance") == "不相关":
            continue
        paper_summary_lines.append(
            f"- [{r.get('relevance', '?')}] {r['title']}: "
            f"{r.get('contribution', '')}"
        )
    paper_list_text = "\n".join(paper_summary_lines[:30])

    prompt = STAGE3_PROMPT.format(topic=topic, paper_list=paper_list_text)
    try:
        raw = chat(client, prompt, temperature=0.3)
        json_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1)
        meta = json.loads(raw.strip().strip("`"))
    except Exception as e:
        print(f"  汇总分析失败: {e}")
        meta = {"summary": "（汇总生成失败）", "filename_keyword": "检索报告"}

    today = datetime.now().strftime("%Y%m%d")
    keyword = meta.get("filename_keyword", "检索报告")
    report_name = f"{today}_{keyword}.md"
    report_path = SCANS_DIR / report_name

    high_count = len(groups["高相关"])
    mid_count = len(groups["中相关"])
    low_count = len(groups["低相关"])
    included = high_count + mid_count + low_count

    lines = [
        f"# arXiv 检索报告：{topic}\n",
        f"- 检索时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 时间范围：{date_range_str}",
        f"- 检索结果：{stats['total']} 篇（去重后），"
        f"已入库跳过 {stats['skipped']} 篇",
        f"- 有效筛选：{stats['filtered']} 篇 → "
        f"收录 {included} 篇，排除不相关 {excluded_count} 篇",
        f"  - 高相关 {high_count} / 中相关 {mid_count} / 低相关 {low_count}",
        f"- 中间结果：`{work_dir.name}/`\n",
        "## 总体趋势\n",
        meta["summary"],
        "",
    ]

    if groups["高相关"]:
        lines.append("\n## 高相关\n")
        for r in groups["高相关"]:
            lines.extend(format_paper_detail(r))

    if groups["中相关"]:
        lines.append("\n## 中相关\n")
        for r in groups["中相关"]:
            lines.extend(format_paper_detail(r))

    if groups["低相关"]:
        lines.append("\n## 低相关\n")
        lines.append("| 标题 | arXiv ID | 发表时间 | 团队 | 核心贡献 |")
        lines.append("|------|----------|----------|------|----------|")
        for r in groups["低相关"]:
            title = r["title"].replace("|", "\\|")
            contrib = r.get("contribution", "").replace("|", "\\|")
            team = r.get("team", "未知").replace("|", "\\|")
            lines.append(
                f"| {title} | {r['arxiv_id']} "
                f"| {r['published']} | {team} | {contrib} |"
            )

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  报告: {report_path}")
    return report_path


def format_paper_detail(r: dict) -> list[str]:
    """Format a single paper entry for high/mid relevance sections."""
    return [
        f"### [{r['title']}](https://arxiv.org/abs/{r['arxiv_id']})\n",
        f"- **arXiv ID**: {r['arxiv_id']} | "
        f"**发表时间**: {r['published']} | "
        f"**团队**: {r.get('team', '未知')}",
        f"- **核心贡献**: {r.get('contribution', '')}",
        f"- **分析**: {r.get('reason', '')}",
        "",
    ]


# =========================================================================
# CLI entry point
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="arXiv 论文检索 + DeepSeek AI 初筛工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
三阶段流水线:
  1. DeepSeek 根据主题自动生成多组 arXiv 检索关键词
  2. 批量调用 arXiv API 检索，逐篇调 DeepSeek 判定相关度
  3. 汇总生成 Markdown 报告（高/中/低相关分组）

输出:
  中间结果 → refs/scans/{时间戳}/  (支持断点续传)
  最终报告 → refs/scans/{日期}_{主题}.md

示例:
  %(prog)s "视频生成中的强化学习对齐"
  %(prog)s "sparse attention" --days 30
  %(prog)s "sparse attention" --from 20260301 --to 20260401
  %(prog)s "视频生成" --max-results 100
  %(prog)s "reinforcement learning" --exclude "robotics, medical, 机器人"

依赖: pip install arxiv openai
API Key: refs/deepseek_api (一行纯文本)""",
    )
    parser.add_argument(
        "topic",
        help="研究主题，支持中文或英文",
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="检索最近 N 天的论文 (默认: 不限时间)",
    )
    parser.add_argument(
        "--from", dest="date_from", default=None, metavar="YYYYMMDD",
        help="起始日期，需配合 --to 使用",
    )
    parser.add_argument(
        "--to", dest="date_to", default=None, metavar="YYYYMMDD",
        help="截止日期，需配合 --from 使用",
    )
    parser.add_argument(
        "--max-results", type=int, default=50, metavar="N",
        help="每组查询的最大结果数 (默认: 50)",
    )
    parser.add_argument(
        "--exclude", default=None, metavar="TEXT",
        help="排除条件，自由文本描述不想看到的方向 "
             '(如: --exclude "机器人、医学、自动驾驶")',
    )
    args = parser.parse_args()

    SCANS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = SCANS_DIR / timestamp
    work_dir.mkdir(parents=True, exist_ok=True)

    date_range = parse_date_range(args)
    if date_range:
        date_range_str = (
            f"{date_range[0][:8]} ~ {date_range[1][:8]}"
        )
    else:
        date_range_str = "不限"

    print(f"主题: {args.topic}")
    if args.exclude:
        print(f"排除: {args.exclude}")
    print(f"时间范围: {date_range_str}")
    print(f"单查询最大结果数: {args.max_results}")
    print(f"工作目录: {work_dir}\n")

    ds_client = load_deepseek_client()

    queries_data = stage1_generate_queries(
        ds_client, args.topic, work_dir, exclude=args.exclude,
    )
    results, stats = stage2_search_and_filter(
        ds_client, args.topic, queries_data, date_range,
        args.max_results, work_dir, exclude=args.exclude,
    )
    report_path = stage3_generate_report(
        ds_client, args.topic, results, stats, date_range_str, work_dir,
    )

    print(f"\n完成! 报告: {report_path}")


if __name__ == "__main__":
    main()
