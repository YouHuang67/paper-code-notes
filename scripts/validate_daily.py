#!/usr/bin/env python3
"""Validate generated daily-report fields and institution type labels."""

from __future__ import annotations

import html
import itertools
import json
import re
import sys
from pathlib import Path
import daily_pipeline
from daily_pipeline import affiliation_is_address, affiliation_type_rule, author_front_matter

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "docs" / "daily"
CARD_RE = re.compile(r'<article\s+markdown="1"\s+class="daily-paper-card"(?P<attrs>[^>]*)>')
ATTR_RE = re.compile(r'(?P<name>data-(?:topic|org|level))="(?P<value>[^"]+)">?')
REQUIRED_SECTIONS = ("**问题**：", "**方法**：", "**实验范围**：", "**判断**：")
ORG_TYPES = ("高校", "企业", "科研机构", "个人", "其他", "待核实")
ALLOWED_ORG = {"单位未列出"}
for size in range(1, len(ORG_TYPES) + 1):
    ALLOWED_ORG.update("+".join(combo) for combo in itertools.combinations(ORG_TYPES, size))
ALLOWED_LEVEL = {"通用", "垂类"}


def validate(path: Path, expected_rows: list[dict] | None = None,
             paper_dir: Path | None = None) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    if 'data-filter-group="task"' in text or 'data-task=' in text or '任务类型=`' in text:
        errors.append(f"{path}: obsolete task dimension remains")
    if 'data-filter-group="coverage"' in text or 'data-coverage=' in text or '覆盖面=`' in text:
        errors.append(f"{path}: obsolete coverage dimension remains")
    cards = list(CARD_RE.finditer(text))
    expected = {row.get("id"): row for row in expected_rows or []}
    if not cards:
        if 'daily-content-' in text:
            errors.append(f'{path}: generated daily has no paper cards')
        return errors
    grouped = [heading for heading in re.findall(r'^##\s+(.+)$', text, re.M)
               if heading.strip() != '当日概览']
    if grouped:
        errors.append(f"{path}: papers must be listed as independent cards, found grouped headings")
    for number, card in enumerate(cards, 1):
        attrs = {match.group("name"): html.unescape(match.group("value"))
                 for match in ATTR_RE.finditer(card.group("attrs"))}
        missing = [name for name in ("data-topic", "data-org", "data-level") if not attrs.get(name)]
        if missing:
            errors.append(f"{path}:{number}: missing {', '.join(missing)}")
        if attrs.get("data-org") and attrs["data-org"] not in ALLOWED_ORG:
            errors.append(f"{path}:{number}: unsupported data-org {attrs['data-org']}")
        if attrs.get("data-level") and attrs["data-level"] not in ALLOWED_LEVEL:
            errors.append(f"{path}:{number}: unsupported data-level {attrs['data-level']}")
        end = text.find("</article>", card.end())
        if end == -1 or (number < len(cards) and end > cards[number].start()):
            errors.append(f'{path}:{number}: missing article closure')
        body = text[card.end():end if end != -1 else len(text)]
        paper_id_match = re.search(r'\[arXiv:(\d{4}\.\d{4,5})(?:v\d+)?\]', body)
        paper_id = paper_id_match.group(1) if paper_id_match else None
        source_row = expected.get(paper_id) if expected else None
        for section in REQUIRED_SECTIONS:
            if section not in body:
                errors.append(f"{path}:{number}: missing {section}")
        if "**标签**：" not in body:
            errors.append(f"{path}:{number}: missing **标签**")
        author_line = next((line for line in body.splitlines() if line.startswith("**作者单位**：")), "")
        type_line = next((line for line in body.splitlines() if line.startswith("**单位分类**：")), "")
        label_line = html.unescape(next((line for line in body.splitlines() if line.startswith("**标签**：")), ""))
        if not author_line:
            errors.append(f"{path}:{number}: missing **作者单位**")
        if not type_line:
            errors.append(f"{path}:{number}: missing **单位分类**")
        if attrs.get("data-org") and f"机构类型=`{attrs['data-org']}`" not in label_line:
            errors.append(f"{path}:{number}: 机构类型 label differs from data-org")
        if attrs.get("data-level") and f"层级=`{attrs['data-level']}`" not in label_line:
            errors.append(f"{path}:{number}: 层级 label differs from data-level")
        if attrs.get("data-topic") and f"主题=`{attrs['data-topic']}`" not in label_line:
            errors.append(f"{path}:{number}: 主题 label differs from data-topic")
        if author_line and type_line:
            author_units = html.unescape(author_line.removeprefix("**作者单位**："))
            types_value = html.unescape(type_line.removeprefix("**单位分类**："))
            if author_units == "论文首页未列出作者单位":
                expected_org = "单位未列出"
                if types_value != "未列出":
                    errors.append(f"{path}:{number}: unlisted affiliations have a type")
            else:
                units = [unit.strip() for unit in author_units.split("；") if unit.strip()]
                entries = [entry.strip().split("=", 1) for entry in types_value.split("；") if entry.strip()]
                if len(entries) != len(units) or any(len(entry) != 2 for entry in entries):
                    errors.append(f"{path}:{number}: unit type list does not align with affiliations")
                    expected_org = ""
                else:
                    indexed = []
                    for index, (pair) in enumerate(entries, 1):
                        if pair[0] != str(index) or pair[1] not in ORG_TYPES:
                            errors.append(f"{path}:{number}: invalid unit type entry {entries[index - 1]}")
                        unit = units[index - 1]
                        inferred = affiliation_type_rule(unit)
                        if affiliation_is_address(unit):
                            errors.append(f"{path}:{number}: postal address is not an institution: {unit}")
                        if inferred and pair[1] != inferred:
                            errors.append(f"{path}:{number}: {unit} should be {inferred}, got {pair[1]}")
                        indexed.append(pair[1])
                    expected_org = "+".join(kind for kind in ORG_TYPES if kind in indexed)
            if expected_org and attrs.get("data-org") != expected_org:
                errors.append(f"{path}:{number}: affiliations imply {expected_org}, got {attrs.get('data-org')}")
            if expected_rows is not None:
                if source_row is None:
                    errors.append(f"{path}:{number}: paper ID is absent from the approved report data")
                else:
                    source_units = source_row.get("affiliations", [])
                    source_author_units = "；".join(source_units) or "论文首页未列出作者单位"
                    if author_units != source_author_units:
                        errors.append(f"{path}:{number}: author units differ from the approved report data")
                    source_types = source_row.get("affiliation_types", [])
                    source_type_line = "；".join(f"{index}={item['type']}"
                                                   for index, item in enumerate(source_types, 1)) or "未列出"
                    if types_value != source_type_line:
                        errors.append(f"{path}:{number}: unit types differ from the approved report data")
                    if attrs.get("data-org") != organization_summary(source_row):
                        errors.append(f"{path}:{number}: institution summary differs from the approved report data")
                    if attrs.get("data-topic") != source_row.get("topic"):
                        errors.append(f"{path}:{number}: topic differs from the approved report data")
                    if attrs.get("data-level") != source_row.get("level"):
                        errors.append(f"{path}:{number}: research level differs from the approved report data")
                    if paper_dir is not None and paper_id_match:
                        version_match = re.search(r'v(\d+)\]', paper_id_match.group(0))
                        version = version_match.group(1) if version_match else "latest"
                        raw_path = paper_dir / f"{paper_id}v{version}-raw.txt" if version != "latest" else paper_dir / f"{paper_id}latest-raw.txt"
                        if not raw_path.is_file():
                            errors.append(f"{path}:{number}: source PDF text is missing for {paper_id}")
                        else:
                            paper_text = author_front_matter(raw_path.read_text(encoding="utf-8", errors="replace"))
                            for unit in source_units:
                                if unit.casefold() not in paper_text:
                                    errors.append(f"{path}:{number}: author unit is absent from the source PDF: {unit}")
    return errors


def organization_summary(row: dict) -> str:
    if not row.get("affiliations"):
        return "单位未列出"
    order = ("高校", "企业", "科研机构", "个人", "其他", "待核实")
    types = {item.get("type") for item in row.get("affiliation_types", [])}
    return "+".join(kind for kind in order if kind in types)


def main() -> int:
    errors: list[str] = []
    for path in sorted(DAILY_DIR.glob("*.md")):
        day = path.stem
        run_dir = ROOT / "refs" / "scans" / "daily-runs" / day
        report_path = run_dir / "report.json"
        state_path = run_dir / "state.json"
        if report_path.is_file() and state_path.is_file():
            rows = json.loads(report_path.read_text(encoding="utf-8"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            try:
                daily_pipeline.validate_artifacts(day, run_dir, state)
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                errors.append(f"{path}: approved run artifacts failed validation: {error}")
            errors.extend(validate(path, rows, run_dir / "papers"))
            generated, content_id = daily_pipeline.render_page(day, rows)
            marker = re.search(r'id="daily-content-([a-f0-9]+)"',
                               path.read_text(encoding="utf-8"))
            if not marker or marker.group(1) != content_id or content_id != state.get("content_id"):
                errors.append(f"{path}: content fingerprint differs from the approved run")
            if path.read_text(encoding="utf-8") != generated:
                errors.append(f"{path}: page content differs from the approved run artifacts")
        else:
            errors.extend(validate(path))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("daily reports: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
