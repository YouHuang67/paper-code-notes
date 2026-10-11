#!/usr/bin/env python3
"""Validate the structured fields used by published daily reports."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "docs" / "daily"
CARD_RE = re.compile(r'<article\s+markdown="1"\s+class="daily-paper-card"(?P<attrs>[^>]*)>')
ATTR_RE = re.compile(r'(?P<name>data-(?:topic|org|coverage|task))="(?P<value>[^"]+)"')
REQUIRED_SECTIONS = ("**问题**：", "**方法**：", "**实验范围**：", "**判断**：")
ALLOWED_ORG = {"企业/研究机构参与", "高校参与", "企业/研究机构+高校", "单位未列出"}
ALLOWED_COVERAGE = {"跨任务通用", "任务内通用", "细分任务"}


def validate(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
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
        if attrs.get("data-org") and f"论文单位=`{attrs['data-org']}`" not in label_line:
            errors.append(f"{path}:{number}: 论文单位标签与 data-org 不一致")
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
