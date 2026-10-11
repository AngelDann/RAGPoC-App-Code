"""Let the concentric radius rule actually govern the Ajustes controls.

The settings controls carry `border-radius: var(--radius-sm)` as an inline
style, and inline styles beat any stylesheet rule. That leaves every nested
box at 6px inside a 20px card with 16px padding, where the concentric rule
requires 20 - 16 = 4px.

The inline radius is removed only within the settings dialog, so the rest of
the app keeps whatever it is doing.
"""

from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "src" / "ragpoc" / "templates" / "console.html"

START = '<div class="craft-dialog craft-dialog-settings">'
END = '<div class="modal-backdrop-custom"'  # the next dialog after Ajustes
INLINE_RADIUS = "border-radius: var(--radius-sm); "


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    start = text.find(START)
    end = text.find(END, start + len(START))
    if start < 0 or end < 0:
        print("no se encontro el dialogo de ajustes", file=sys.stderr)
        return 1

    region = text[start:end]
    removed = region.count(INLINE_RADIUS)
    if not removed:
        print("no habia border-radius inline que quitar")
        return 0

    updated = text[:start] + region.replace(INLINE_RADIUS, "") + text[end:]
    TARGET.write_text(updated, encoding="utf-8")
    print(f"inline border-radius eliminado en {removed} controles del dialogo de ajustes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
