from __future__ import annotations

from pathlib import Path

import pymupdf


def render_pdf_pages(path: Path, output_dir: Path, dpi: int = 180):
    output_dir.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(path)
    scale = dpi / 72
    try:
        for index, page in enumerate(document):
            out = output_dir / f"page-{index + 1:05d}.png"
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            pixmap.save(out)
            yield index + 1, out, pixmap.width, pixmap.height
    finally:
        document.close()
