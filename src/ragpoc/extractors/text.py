from __future__ import annotations

from pathlib import Path

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


def extract_text(path: Path) -> str:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        raise ValueError(f"Unsupported text file: {path.suffix}")
    return path.read_text(encoding="utf-8", errors="replace")
