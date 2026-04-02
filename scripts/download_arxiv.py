from __future__ import annotations

"""Download arXiv paper PDF and run split_pdf.py preprocessing.

Accepts one or more arXiv URLs (abstract, PDF, or bare ID), downloads
each PDF to refs/papers/, then invokes split_pdf.py to split and extract text.

Usage:
    python scripts/download_arxiv.py https://arxiv.org/abs/1706.03762
    python scripts/download_arxiv.py 2504.13074 2501.12345 2403.99999
    python scripts/download_arxiv.py url1 url2 url3
"""

import re
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

PAPERS_DIR = Path(__file__).resolve().parent.parent / "refs" / "papers"
SPLIT_SCRIPT = Path(__file__).resolve().parent / "split_pdf.py"


def extract_arxiv_id(url_or_id: str) -> str:
    """Extract arXiv paper ID from URL or bare ID string."""
    match = re.search(r'(\d{4}\.\d{4,5})(v\d+)?', url_or_id)
    if match:
        return match.group(0)
    raise ValueError(f"Cannot parse arXiv ID from: {url_or_id}")


def extract_base_id(arxiv_id: str) -> str:
    """Extract base arXiv ID without version suffix."""
    match = re.search(r'(\d{4}\.\d{4,5})', arxiv_id)
    return match.group(1) if match else arxiv_id


def get_existing_arxiv_ids(papers_dir: Path) -> set[str]:
    """Scan papers directory to collect already-downloaded base IDs."""
    ids = set()
    if not papers_dir.exists():
        return ids
    for item in papers_dir.iterdir():
        match = re.search(r"(\d{4}\.\d{4,5})", item.name)
        if match:
            ids.add(match.group(1))
    return ids


def download_pdf(arxiv_id: str, save_dir: Path) -> Path:
    """Download PDF from arXiv and return the saved file path."""
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{arxiv_id}.pdf"
    save_path = save_dir / filename

    if save_path.exists():
        print(f"  PDF already exists: {save_path}")
        return save_path

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    print(f"  Downloading: {url}")

    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urlopen(req, timeout=60)
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} (paper may be withdrawn or ID invalid)")

    data = resp.read()
    if len(data) < 1024:
        raise RuntimeError("response too small, check the arXiv ID")

    save_path.write_bytes(data)
    size_mb = len(data) / (1024 * 1024)
    print(f"  Saved: {save_path} ({size_mb:.1f} MB)")
    return save_path


def process_one(raw: str, existing_ids: set[str]) -> str:
    """Process a single arXiv URL/ID. Returns 'ok', 'skipped', or 'failed'."""
    try:
        arxiv_id = extract_arxiv_id(raw)
    except ValueError as e:
        print(f"  SKIP: {e}")
        return "failed"

    base_id = extract_base_id(arxiv_id)
    if base_id in existing_ids:
        print(f"  已入库，跳过: {base_id}")
        return "skipped"

    print(f"  arXiv ID: {arxiv_id}")

    try:
        pdf_path = download_pdf(arxiv_id, PAPERS_DIR)
    except RuntimeError as e:
        print(f"  FAIL: download error - {e}")
        return "failed"

    print(f"  Running split_pdf.py ...")
    result = subprocess.run(
        [sys.executable, str(SPLIT_SCRIPT), pdf_path.name],
    )
    if result.returncode != 0:
        print(f"  WARN: split_pdf.py exited with code {result.returncode}")
    return "ok"


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("arXiv 论文下载 + 预处理工具")
        print("")
        print("用法: python scripts/download_arxiv.py <url_or_id> [url_or_id ...]")
        print("")
        print("功能:")
        print("  1. 从 arXiv 下载 PDF 到 refs/papers/")
        print("  2. 自动调用 split_pdf.py 拆分 + 提取文本")
        print("  3. 自动跳过已入库论文（按 base ID 去重，忽略版本号）")
        print("")
        print("支持的输入格式:")
        print("  arXiv URL   https://arxiv.org/abs/2412.14167")
        print("  PDF URL     https://arxiv.org/pdf/2412.14167")
        print("  带版本号    2412.14167v2")
        print("  纯 ID       2412.14167")
        print("")
        print("示例:")
        print("  python scripts/download_arxiv.py 2412.14167")
        print("  python scripts/download_arxiv.py 2412.14167 2501.13918 2502.13923")
        print("  python scripts/download_arxiv.py https://arxiv.org/abs/2412.14167")
        sys.exit(0)

    args = sys.argv[1:]
    total = len(args)
    ok = 0
    skipped = 0
    failed = 0

    existing_ids = get_existing_arxiv_ids(PAPERS_DIR)
    if existing_ids:
        print(f"已入库论文: {len(existing_ids)} 篇")

    for i, raw in enumerate(args, 1):
        print(f"\n[{i}/{total}] {raw}")
        status = process_one(raw, existing_ids)
        if status == "ok":
            ok += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1

    parts = []
    if ok:
        parts.append(f"{ok} 下载成功")
    if skipped:
        parts.append(f"{skipped} 已入库跳过")
    if failed:
        parts.append(f"{failed} 失败")
    print(f"\nDone: {', '.join(parts)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
