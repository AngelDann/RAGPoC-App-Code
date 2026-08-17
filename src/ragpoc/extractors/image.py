from __future__ import annotations

from pathlib import Path

from PIL import Image


def inspect_image(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size
    except Exception as error:
        raise ValueError(f"Unreadable image {path.name}: {error}") from error
