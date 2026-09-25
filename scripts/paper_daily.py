#!/usr/bin/env python3
"""Configure and run the local arXiv daily digest.

Topics and the daily clock live in paper-daily.json.
The seen-paper ledger stays under refs/scans/ and is not committed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "paper-daily.json"
LEDGER_PATH = ROOT / "refs" / "scans" / "daily-ledger.json"
BRIDGE_DIR = Path("/mnt/d/projects/cli-wechat-bridge")
NODE_BIN = Path("/home/hy/.local/opt/node-v24.19.0-linux-x64/bin/node")
BRIDGE_DATA_DIR = Path("/home/hy/.local/share/cli-wechat-bridge")
CRON_MARK = "# paper-daily"
RSS_CATEGORIES = ("cs.CV", "cs.LG", "cs.CL", "cs.AI", "cs.MM")
MIN_GAP_SECONDS = 4
_last_request_at = 0.0


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"timezone": "Asia/Shanghai", "checkTime": "14:30", "maxResults": 8, "topics": []}
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def save_config(config: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_ledger() -> dict:
    if not LEDGER_PATH.exists():
        return {"seen": {}}
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def save_ledger(ledger: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_clock(value: str) -> tuple[int, int]:
    hour, minute = value.split(":", 1)
    hour_n, minute_n = int(hour), int(minute)
    if not (0 <= hour_n <= 23 and 0 <= minute_n <= 59):
        raise ValueError(value)
    return hour_n, minute_n


def cmd_topic_add(args: argparse.Namespace) -> None:
    config = load_config()
    name = args.name.strip()
    if any(topic["name"] == name for topic in config["topics"]):
        raise SystemExit(f"topic already exists: {name}")
    config["topics"].append({
        "name": name,
        "query": args.query or f'all:"{name}"',
        "exclude": args.exclude or "",
        "enabled": True,
    })
    save_config(config)
    print(f"added {name}")


def cmd_topic_remove(args: argparse.Namespace) -> None:
    config = load_config()
    name = args.name.strip()
    kept = [topic for topic in config["topics"] if topic["name"] != name]
    if len(kept) == len(config["topics"]):
        raise SystemExit(f"topic not found: {name}")
    config["topics"] = kept
    save_config(config)
    print(f"removed {name}")


def cmd_time(args: argparse.Namespace) -> None:
    parse_clock(args.clock)
    config = load_config()
    config["checkTime"] = args.clock
    save_config(config)
    sync_cron(config)
    print(f"check time {args.clock} {config.get('timezone', 'Asia/Shanghai')}")


def cmd_list(_: argparse.Namespace) -> None:
    config = load_config()
    print(f"time {config.get('checkTime', '14:30')} {config.get('timezone', 'Asia/Shanghai')}")
    topics = config.get("topics") or []
    if not topics:
        print("topics (none)")
        return
    for topic in topics:
        state = "on" if topic.get("enabled", True) else "off"
        print(f"- {topic['name']} [{state}] {topic.get('query', '')}")


def request_direct(url: str) -> bytes:
    global _last_request_at
    wait = MIN_GAP_SECONDS - (time.monotonic() - _last_request_at)
    if _last_request_at and wait > 0:
        time.sleep(wait)
    request = urllib.request.Request(url, headers={"User-Agent": "python-urllib/3.12"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        error.close()
        raise
    finally:
        _last_request_at = time.monotonic()


def query_groups(query: str) -> list[list[str]]:
    groups = []
    for part in re.split(r"\s+AND\s+", query):
        found = re.findall(r'"([^"]+)"|(?:all|ti|abs):([A-Za-z0-9.+-]+)', part)
        terms = [quoted or bare for quoted, bare in found]
        terms = [term.strip() for term in terms if term.strip()]
        if terms:
            groups.append(terms)
    return groups


def paper_matches(paper: dict, groups: list[list[str]]) -> bool:
    text = f"{paper['title']} {paper.get('summary', '')}".lower()
    return all(any(term.lower() in text for term in group) for group in groups)


def load_announcements() -> list[dict]:
    papers = []
    seen_ids: set[str] = set()
    for category in RSS_CATEGORIES:
        print(f"rss {category}", file=sys.stderr, flush=True)
        try:
            payload = request_direct(f"https://rss.arxiv.org/rss/{category}")
        except Exception as error:
            print(f"rss {category} failed: {error}", file=sys.stderr, flush=True)
            continue
        root = ET.fromstring(payload)
        for item in root.findall(".//item"):
            link = (item.findtext("link") or "").rstrip("/")
            base_id = link.rsplit("/", 1)[-1].split("v")[0]
            title = " ".join((item.findtext("title") or "").split())
            summary = " ".join((item.findtext("description") or "").split())
            published = ""
            pub = item.findtext("pubDate") or ""
            if pub:
                try:
                    published = parsedate_to_datetime(pub).date().isoformat()
                except (TypeError, ValueError, IndexError):
                    published = ""
            if not base_id or not title or base_id in seen_ids:
                continue
            seen_ids.add(base_id)
            papers.append({"id": base_id, "title": title, "published": published, "summary": summary})
    return papers


def load_deepseek_key() -> str:
    key_file = ROOT / "refs" / "deepseek_api"
    if key_file.is_file():
        key = key_file.read_text(encoding="utf-8").strip()
        if key:
            return key
    bashrc = Path.home().joinpath(".bashrc").read_text(encoding="utf-8")
    match = re.search(r'^export ANTHROPIC_AUTH_TOKEN="([^"]+)"', bashrc, re.M)
    if not match:
        raise SystemExit("deepseek key missing")
    return match.group(1)


def topic_guide(config: dict) -> str:
    lines = []
    for topic in config.get("topics") or []:
        if topic.get("enabled", True):
            lines.append(f"{topic['name']}：{topic.get('scope', '')}")
    return "\n".join(lines)


def topic_names(config: dict) -> list[str]:
    return [topic["name"] for topic in config.get("topics") or [] if topic.get("enabled", True)]


def ask_deepseek(prompt: str) -> dict:
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps({
            "model": "deepseek-chat",
            "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}],
        }).encode(),
        headers={
            "Authorization": f"Bearer {load_deepseek_key()}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode())
    content = payload["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return {}
    return json.loads(match.group(0))


def classify_papers(topic_names: list[str], papers: list[dict], guide: str) -> dict[str, list[tuple[str, str]]]:
    grouped = {name: [] for name in topic_names}
    allowed = {paper["id"]: paper for paper in papers}
    batch_size = 20
    for start in range(0, len(papers), batch_size):
        batch = papers[start:start + batch_size]
        blocks = []
        for paper in batch:
            summary = (paper.get("summary") or "")[:280]
            blocks.append(f"id={paper['id']}\n{paper['title']}\n{summary}")
        prompt = (
            "下面这些论文请你归类。先看论文对象是不是主题规定的那个，不要按标题里的词硬塞。每篇最多一个主题。\n"
            "若一句解释里不得不写「非…」或「不是…」，这篇不要放进 picks。\n"
            f"主题说明：\n{guide}\n"
            "line 保留关键英文术语，后面用很短的中文补方法和洞察。不要写「属于某主题」，不要把术语整句翻译掉。\n"
            "每个主题最多 8 篇。只返回 JSON："
            "{\"picks\":[{\"id\":\"\",\"topic\":\"\",\"line\":\"\"}]}\n\n"
            + "\n\n".join(blocks)
        )
        print(f"ds batch {start // batch_size + 1} n={len(batch)}", file=sys.stderr, flush=True)
        for item in ask_deepseek(prompt).get("picks") or []:
            paper_id = str(item.get("id") or "").split("v")[0]
            topic = str(item.get("topic") or "")
            line = " ".join(str(item.get("line") or "").split())
            if paper_id not in allowed or topic not in grouped or not line:
                continue
            if "非" in line or "不是" in line:
                continue
            if any(kept_id == paper_id for kept_id, _line in grouped[topic]):
                continue
            if len(grouped[topic]) >= 8:
                continue
            grouped[topic].append((paper_id, line))
    return grouped


def render_wechat(day_label: str, sections: list[tuple[str, list[str]]], empty_names: list[str]) -> str:
    blocks = ["", day_label, ""]
    for name, sentences in sections:
        blocks.append(f"【{name}】")
        blocks.append("")
        for sentence in sentences:
            blocks.append(f"— {sentence}")
            blocks.append("")
    if empty_names:
        blocks.append("无更新：" + "、".join(empty_names))
    return "\n".join(blocks).strip("\n") + "\n"


def compose_day(day_label: str, topic_names: list[str], papers: list[dict], guide: str) -> tuple[str, list[str]]:
    grouped = classify_papers(topic_names, papers, guide)
    sections = []
    empty_names = []
    new_ids: list[str] = []
    used: set[str] = set()
    for name in topic_names:
        fresh = []
        for paper_id, line in grouped.get(name) or []:
            if paper_id in used:
                continue
            used.add(paper_id)
            fresh.append(line)
            new_ids.append(paper_id)
        if fresh:
            sections.append((name, fresh))
        else:
            empty_names.append(name)
    return render_wechat(day_label, sections, empty_names), new_ids


def day_label_from(iso_day: str) -> str:
    year, month, day = iso_day.split("-")
    return f"{int(month)}月{int(day)}日"


def build_digest(config: dict, ledger: dict) -> tuple[str, list[str]]:
    seen = ledger.setdefault("seen", {})
    papers = [paper for paper in load_announcements() if paper["id"] not in seen]
    if not papers:
        raise SystemExit("arXiv RSS 没有未读论文")
    dates = sorted({paper["published"] for paper in papers if paper.get("published")})
    label = day_label_from(dates[-1]) if dates else datetime.now().strftime("%m-%d")
    return compose_day(label, topic_names(config), papers, topic_guide(config))


def fetch_oai_new(day: str) -> list[dict]:
    ns = {
        "o": "http://www.openarchives.org/OAI/2.0/",
        "dc": "http://purl.org/dc/elements/1.1/",
    }
    papers = []
    seen_ids: set[str] = set()
    token = ""
    page = 0
    while page < 6:
        if token:
            query = urllib.parse.urlencode({"verb": "ListRecords", "resumptionToken": token})
        else:
            query = urllib.parse.urlencode({
                "verb": "ListRecords",
                "metadataPrefix": "oai_dc",
                "set": "cs",
                "from": day,
                "until": day,
            })
        print(f"oai {day} page {page}", file=sys.stderr, flush=True)
        payload = request_direct(f"https://oaipmh.arxiv.org/oai?{query}")
        root = ET.fromstring(payload)
        for record in root.findall(".//o:record", ns):
            header = record.find("o:header", ns)
            if header is None or header.get("status") == "deleted":
                continue
            if (header.findtext("o:datestamp", default="", namespaces=ns) or "") != day:
                continue
            raw = header.findtext("o:identifier", default="", namespaces=ns) or ""
            paper_id = raw.rsplit(":", 1)[-1].split("v")[0]
            if not paper_id.startswith("2609.") or paper_id in seen_ids:
                continue
            title = " ".join((record.findtext(".//dc:title", default="", namespaces=ns) or "").split())
            summary = " ".join((record.findtext(".//dc:description", default="", namespaces=ns) or "").split())
            if not title:
                continue
            seen_ids.add(paper_id)
            papers.append({"id": paper_id, "title": title, "published": day, "summary": summary})
        token_node = root.find(".//o:resumptionToken", ns)
        token = (token_node.text or "").strip() if token_node is not None else ""
        page += 1
        if not token:
            break
    papers.sort(key=lambda paper: paper["id"], reverse=True)
    return papers[:80]


def cmd_recent(args: argparse.Namespace) -> None:
    config = load_config()
    names = topic_names(config)
    days = [day.strip() for day in args.days.split(",") if day.strip()]
    for day in days:
        papers = fetch_oai_new(day)
        print(f"{day} papers {len(papers)}", file=sys.stderr, flush=True)
        if not papers:
            print(f"skip {day}")
            continue
        digest, new_ids = compose_day(day_label_from(day), names, papers, topic_guide(config))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + day.replace("-", "")
        out_dir = ROOT / "refs" / "scans" / "daily" / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        digest_path = out_dir / "digest.txt"
        digest_path.write_text(digest, encoding="utf-8")
        print(digest)
        if args.dry_run:
            continue
        send_heartbeat(stamp, digest_path, new_ids)
        print(f"sent {day} {len(new_ids)}")


def send_heartbeat(run_id: str, digest_path: Path, paper_ids: list[str]) -> None:
    command = [
        str(NODE_BIN), "--no-warnings", "--experimental-strip-types",
        "src/daemon/paper-heartbeat.ts",
        "--run-id", run_id,
        "--digest-file", str(digest_path),
    ]
    for paper_id in paper_ids:
        command.extend(["--paper-id", paper_id])
    env = os.environ.copy()
    env["CLI_BRIDGE_DATA_DIR"] = str(BRIDGE_DATA_DIR)
    env["PATH"] = f"{NODE_BIN.parent}{os.pathsep}{env.get('PATH', '')}"
    subprocess.run(command, cwd=BRIDGE_DIR, check=True, env=env)


def cmd_run(args: argparse.Namespace) -> None:
    config = load_config()
    topics = [topic for topic in config.get("topics") or [] if topic.get("enabled", True)]
    if not topics:
        print("no enabled topics")
        return
    if not args.dry_run:
        if not NODE_BIN.is_file():
            raise SystemExit(f"node not found: {NODE_BIN}")
        if not (BRIDGE_DATA_DIR / "daemon-endpoint.json").is_file():
            raise SystemExit("wechat daemon endpoint missing")
    ledger = load_ledger()
    digest, new_ids = build_digest(config, ledger)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "refs" / "scans" / "daily" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    digest_path = out_dir / "digest.txt"
    digest_path.write_text(digest, encoding="utf-8")
    print(digest)
    if args.dry_run:
        print(f"dry-run wrote {digest_path}")
        return
    if not new_ids:
        print("nothing new to send")
        return
    send_heartbeat(stamp, digest_path, new_ids)
    now = datetime.now(timezone.utc).isoformat()
    for paper_id in new_ids:
        ledger["seen"][paper_id] = now
    save_ledger(ledger)
    print(f"sent {len(new_ids)}")


def cron_line(config: dict) -> str:
    hour, minute = parse_clock(config.get("checkTime") or "14:30")
    script = Path(__file__).resolve()
    return f"{minute} {hour} * * 1-5 {sys.executable} {script} run {CRON_MARK}"


def sync_cron(config: dict) -> None:
    try:
        current = subprocess.run(["crontab", "-l"], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        print("crontab is not installed")
        return
    lines = [] if current.returncode else current.stdout.splitlines()
    kept = [line for line in lines if CRON_MARK not in line and line.strip()]
    kept.append(cron_line(config))
    subprocess.run(["crontab", "-"], input="\n".join(kept) + "\n", text=True, check=True)


def cmd_install(_: argparse.Namespace) -> None:
    config = load_config()
    parse_clock(config.get("checkTime") or "14:30")
    sync_cron(config)
    print(cron_line(config))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper daily topics and arXiv push")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add")
    add.add_argument("name")
    add.add_argument("--query")
    add.add_argument("--exclude", default="")
    add.set_defaults(func=cmd_topic_add)

    remove = sub.add_parser("remove")
    remove.add_argument("name")
    remove.set_defaults(func=cmd_topic_remove)

    clock = sub.add_parser("time")
    clock.add_argument("clock", help="HH:MM in Asia/Shanghai")
    clock.set_defaults(func=cmd_time)

    listing = sub.add_parser("list")
    listing.set_defaults(func=cmd_list)

    run = sub.add_parser("run")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=cmd_run)

    recent = sub.add_parser("recent")
    recent.add_argument("--days", default="2026-09-22,2026-09-23,2026-09-24")
    recent.add_argument("--dry-run", action="store_true")
    recent.set_defaults(func=cmd_recent)

    install = sub.add_parser("install")
    install.set_defaults(func=cmd_install)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
