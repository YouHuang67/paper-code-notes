"""Split large PDF files and extract text for paper reading workflow.

Each part is bounded by max file size. Pages accumulate until adding the next
page would exceed the limit; oversized single pages get image compression.

Idempotent: skips processing if a valid meta.json already exists with all
expected output files present. Use --force to re-process.

Output structure (same directory as source PDF):
    refs/papers/<stem>/
    ├── meta.json        # page count, split info, processing time
    ├── full_text.txt    # extracted plain text with page markers
    ├── part_001.pdf     # split PDFs
    ├── part_002.pdf
    └── ...

Usage:
    python scripts/split_pdf.py foo.pdf                # process specific file
    python scripts/split_pdf.py                        # auto-detect latest PDF
    python scripts/split_pdf.py foo.pdf --max-size 10  # max 10 MB per split
    python scripts/split_pdf.py foo.pdf --force         # re-process even if done
"""

import argparse
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import fitz
except ImportError:
    print("PyMuPDF not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyMuPDF"])
    import fitz

try:
    import PIL.Image
except ImportError:
    print("Pillow not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    import PIL.Image

PAPERS_DIR = Path(__file__).resolve().parent.parent / "refs" / "papers"
DEFAULT_MAX_SIZE_MB = 5


def find_latest_pdf(papers_dir: Path) -> Path:
    """Find the most recently modified PDF in the papers directory."""
    pdfs = sorted(
        papers_dir.glob("*.pdf"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not pdfs:
        print(f"No PDF files found in {papers_dir}")
        sys.exit(1)
    return pdfs[0]


def check_existing(src: Path) -> bool:
    """Check if a valid processing result already exists.

    Returns True if meta.json exists, is well-formed, and all referenced
    output files (parts + full_text.txt) are present on disk.
    """
    out_dir = src.parent / src.stem
    meta_path = out_dir / "meta.json"

    if not meta_path.exists():
        return False

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    required_keys = {"source", "total_pages", "max_size_mb", "parts", "processed_at"}
    if not required_keys.issubset(meta.keys()):
        return False

    if meta.get("source") != src.name:
        return False

    if not (out_dir / "full_text.txt").exists():
        return False

    parts = meta.get("parts", [])
    if not parts:
        return False

    for part in parts:
        part_file = part.get("file")
        if not part_file or not (out_dir / part_file).exists():
            return False

    return True


def extract_pdf_metadata(doc: fitz.Document) -> dict:
    """Extract basic metadata from PDF document."""
    raw = doc.metadata or {}
    return {k: v for k, v in raw.items() if v}


def estimate_part_size(doc: fitz.Document, start: int, end: int) -> int:
    """Estimate PDF size in bytes by writing to memory buffer."""
    part_doc = fitz.open()
    part_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
    buf = io.BytesIO()
    part_doc.save(buf)
    size = buf.tell()
    part_doc.close()
    return size


def compress_images(doc: fitz.Document, scale: float) -> None:
    """Downsample all images in a document by the given scale factor."""
    for page in doc:
        image_list = page.get_images(full=True)
        for img_info in image_list:
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue
            if not base_image or "image" not in base_image:
                continue

            img_bytes = base_image["image"]
            pix = fitz.Pixmap(img_bytes)

            new_w = max(1, int(pix.width * scale))
            new_h = max(1, int(pix.height * scale))

            if pix.alpha:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            resized = fitz.Pixmap(pix, 0)
            resized = fitz.Pixmap(fitz.csRGB, resized) if resized.n > 3 else resized

            pil_img = PIL.Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pil_img = pil_img.resize((new_w, new_h), PIL.Image.LANCZOS)

            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=75)
            buf.seek(0)

            new_pix = fitz.Pixmap(buf)
            page.replace_image(xref, pixmap=new_pix)


def save_part(
    doc: fitz.Document, start: int, end: int, path: Path, max_size_bytes: int,
) -> int:
    """Save a range of pages as a new PDF with image compression if needed."""
    part_doc = fitz.open()
    part_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
    part_doc.save(str(path), garbage=4, deflate=True)
    part_doc.close()

    file_size = path.stat().st_size
    if file_size <= max_size_bytes:
        return file_size

    tmp_path = path.with_suffix(".tmp.pdf")
    for scale in (0.5, 0.3, 0.2):
        print(f"    Compressing images (scale={scale})...")
        part_doc = fitz.open(str(path))
        compress_images(part_doc, scale)
        part_doc.save(str(tmp_path), garbage=4, deflate=True)
        part_doc.close()
        tmp_path.replace(path)
        file_size = path.stat().st_size
        if file_size <= max_size_bytes:
            break

    return file_size


def split_pdf(src: Path, max_size_bytes: int) -> None:
    """Split a PDF into parts bounded by max file size."""
    out_dir = src.parent / src.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(src)
    total = len(doc)
    max_size_mb = max_size_bytes / (1024 * 1024)
    print(f"Processing: {src.name} ({total} pages)")
    print(f"Limit: max {max_size_mb:.0f} MB per part")

    all_text = []
    parts = []
    page_idx = 0

    while page_idx < total:
        part_start = page_idx
        part_end = page_idx

        while part_end < total:
            next_end = part_end + 1
            size = estimate_part_size(doc, part_start, next_end)

            if size > max_size_bytes and part_end > part_start:
                break

            part_end = next_end

            if part_end == part_start + 1 and size > max_size_bytes:
                break

        part_num = len(parts) + 1
        part_path = out_dir / f"part_{part_num:03d}.pdf"
        file_size = save_part(doc, part_start, part_end, part_path, max_size_bytes)

        parts.append({
            "file": part_path.name,
            "pages": f"{part_start + 1}-{part_end}",
            "page_count": part_end - part_start,
            "size_mb": round(file_size / (1024 * 1024), 2),
        })
        print(
            f"  part_{part_num:03d}.pdf: "
            f"pages {part_start + 1}-{part_end}, "
            f"{file_size / (1024 * 1024):.1f} MB"
        )

        for pi in range(part_start, part_end):
            text = doc[pi].get_text()
            all_text.append(f"--- Page {pi + 1} ---\n{text}")

        page_idx = part_end

    text_path = out_dir / "full_text.txt"
    text_path.write_text("\n\n".join(all_text), encoding="utf-8")
    print(f"  Text: {text_path.name}")

    meta = {
        "source": src.name,
        "total_pages": total,
        "max_size_mb": max_size_mb,
        "parts": parts,
        "pdf_metadata": extract_pdf_metadata(doc),
        "processed_at": datetime.now().isoformat(),
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Meta: {meta_path.name}")

    doc.close()
    print(f"Done: {len(parts)} parts -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split PDF and extract text")
    parser.add_argument(
        "file", nargs="?", default=None,
        help="PDF filename in refs/papers/ (default: latest by mtime)",
    )
    parser.add_argument(
        "--max-size", type=float, default=DEFAULT_MAX_SIZE_MB,
        help=f"Max MB per split (default: {DEFAULT_MAX_SIZE_MB})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process even if valid results already exist",
    )
    args = parser.parse_args()

    if args.file:
        src = PAPERS_DIR / args.file
        if not src.exists():
            print(f"File not found: {src}")
            sys.exit(1)
    else:
        src = find_latest_pdf(PAPERS_DIR)
        print(f"No file specified, using latest: {src.name}")

    if not args.force and check_existing(src):
        out_dir = src.parent / src.stem
        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        n_parts = len(meta["parts"])
        print(
            f"Already processed: {src.name} "
            f"({meta['total_pages']} pages, {n_parts} parts)"
        )
        print(f"  Output: {out_dir}")
        print(f"  Use --force to re-process")
        return

    max_size_bytes = int(args.max_size * 1024 * 1024)
    split_pdf(src, max_size_bytes)


if __name__ == "__main__":
    main()
