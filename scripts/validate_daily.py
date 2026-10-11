#!/usr/bin/env python3
"""Validate the structured fields used by published daily reports."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "docs" / "daily"
CARD_RE = re.compile(r'<article\s+markdown="1"\s+class="daily-paper-card"(?P<attrs>[^>]*)>')
ATTR_RE = re.compile(r'(?P<name>data-(?:topic|org|coverage|task))="(?P<value>[^"]+)"')
REQUIRED_SECTIONS = ("**问题**：", "**方法**：", "**实验范围**：", "**判断**：")
ALLOWED_ORG = {"大型机构独占", "大型机构+高校/非大型机构", "高校/非大型机构独占", "单位未列出"}
ALLOWED_COVERAGE = {"跨任务通用", "任务内通用", "细分任务"}
ORG_SCALE_PATH = ROOT / "scripts" / "daily-institution-scale.json"


def expected_org_category(author_units: str, large_institutions: set[str], other_institutions: set[str]) -> str:
    if "论文首页未列出作者单位" in author_units:
        return "单位未列出"
    units = [unit.strip().rstrip("。.") for unit in author_units.split("；") if unit.strip()]
    normalized_large = {unit.rstrip("。.") for unit in large_institutions}
    normalized_other = {unit.rstrip("。.") for unit in other_institutions}
    unknown = [unit for unit in units if unit not in normalized_large and unit not in normalized_other]
    if unknown:
        return "单位层级待核实"
    has_large = any(unit in normalized_large for unit in units)
    has_other = any(unit in normalized_other for unit in units)
    if has_large and not has_other:
        return "大型机构独占"
    if has_large:
        return "大型机构+高校/非大型机构"
    return "高校/非大型机构独占"


def validate(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    scale_data = json.loads(ORG_SCALE_PATH.read_text(encoding="utf-8"))
    large_institutions = set(scale_data["large_institutions"])
    other_institutions = set(scale_data["other_institutions"])
    errors: list[str] = []
    cards = list(CARD_RE.finditer(text))
    if not cards:
        return errors
    for number, card in enumerate(cards, 1):
        attrs = {match.group("name"): match.group("value") for match in ATTR_RE.finditer(card.group("attrs"))}
        missing = [name for name in ("data-topic", "data-org", "data-coverage", "data-task") if not attrs.get(name)]
        if missing:
            errors.append(f"{path}:{number}: missing {', '.join(missing)}")
        if attrs.get("data-org") and attrs["data-org"] not in ALLOWED_ORG:
            errors.append(f"{path}:{number}: unsupported data-org {attrs['data-org']}")
        if attrs.get("data-coverage") and attrs["data-coverage"] not in ALLOWED_COVERAGE:
            errors.append(f"{path}:{number}: unsupported data-coverage {attrs['data-coverage']}")
        end = text.find("</article>", card.end())
        body = text[card.end():end if end != -1 else len(text)]
        if "**标签**：" not in body:
            errors.append(f"{path}:{number}: missing **标签**")
        if "**作者单位**：" not in body:
            errors.append(f"{path}:{number}: missing **作者单位**")
        label_line = next((line for line in body.splitlines() if line.startswith("**标签**：")), "")
        author_units_line = next((line for line in body.splitlines() if line.startswith("**作者单位**：")), "")
        author_units = author_units_line.removeprefix("**作者单位**：")
        if attrs.get("data-org") and f"机构组合=`{attrs['data-org']}`" not in label_line:
            errors.append(f"{path}:{number}: 机构组合标签与 data-org 不一致")
        if author_units and attrs.get("data-org"):
            expected = expected_org_category(author_units, large_institutions, other_institutions)
            if attrs["data-org"] != expected:
                errors.append(f"{path}:{number}: author affiliations imply {expected}, got {attrs['data-org']}")
        for section in REQUIRED_SECTIONS:
            if section not in body:
                errors.append(f"{path}:{number}: missing {section}")
    return errors


def main() -> int:
    errors: list[str] = []
    for path in sorted(DAILY_DIR.glob("*.md")):
        errors.extend(validate(path))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("daily reports: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
