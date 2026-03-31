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
        raise RuntimeError(f"HTTP {e.code}")

    data = resp.read()
    if len(data) < 1024:
        raise RuntimeError("response too small, check the arXiv ID")

    save_path.write_bytes(data)
    size_mb = len(data) / (1024 * 1024)
    print(f"  Saved: {save_path} ({size_mb:.1f} MB)")
    return save_path


def process_one(raw: str) -> bool:
    """Process a single arXiv URL/ID. Returns True on success."""
    try:
        arxiv_id = extract_arxiv_id(raw)
    except ValueError as e:
        print(f"  SKIP: {e}")
        return False

    print(f"  arXiv ID: {arxiv_id}")

    try:
        pdf_path = download_pdf(arxiv_id, PAPERS_DIR)
    except RuntimeError as e:
        print(f"  FAIL: download error - {e}")
        return False

    print(f"  Running split_pdf.py ...")
    result = subprocess.run(
        [sys.executable, str(SPLIT_SCRIPT), pdf_path.name],
    )
    if result.returncode != 0:
        print(f"  WARN: split_pdf.py exited with code {result.returncode}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/download_arxiv.py <url_or_id> [url_or_id ...]")
        sys.exit(1)

    args = sys.argv[1:]
    total = len(args)
    ok = 0

    for i, raw in enumerate(args, 1):
        print(f"\n[{i}/{total}] {raw}")
        if process_one(raw):
            ok += 1

    print(f"\nDone: {ok}/{total} succeeded")
    if ok < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
