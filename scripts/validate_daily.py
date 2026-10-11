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
        end = text.find("</article>", card.end())
        body = text[card.end():end if end != -1 else len(text)]
        if "**标签**：" not in body:
            errors.append(f"{path}:{number}: missing **标签**")
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
