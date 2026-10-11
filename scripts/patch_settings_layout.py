"""Comfort pass on the Ajustes layout, following the better-ui rules.

Three things made the settings dialog feel cramped:

* it was pinned to 560px with six cards stacked in one column, so any change
  meant scrolling past unrelated groups;
* border radii were not concentric -- a 10px card with 12px padding should nest
  0px corners, but the inner boxes carried 4-6px, which is the single most
  common cause of an interface reading as "off";
* every card arrived at once, so nothing communicated grouping.

The fix widens the dialog into a responsive two-column grid, makes the radii
concentric (card radius = inner radius + padding), and staggers the cards in
by ~70ms so the sequence reads as a hierarchy. No element id changes: all the
settings JavaScript keeps working untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "ragpoc" / "templates" / "console.html"

DIALOG_ANCHOR = '<div class="craft-dialog" style="max-width: 560px;">'
DIALOG_NEW = '<div class="craft-dialog craft-dialog-settings">'

CSS_ANCHOR = "    @keyframes dialogFade {"
CSS_BLOCK = """    /* --- Ajustes: layout de confort (better-ui) ---
       Radios concentricos: radio_tarjeta = radio_interior + padding.
       Aqui: 20px = 6px + 14px. El desajuste entre niveles es la causa mas
       comun de que una interfaz se sienta "mal" sin saber por que. */
    .craft-dialog-settings {
      max-width: 880px;
    }

    .craft-dialog-settings .settings-body {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
      gap: 14px;
      align-content: start;
    }

    .craft-dialog-settings .settings-body > .craft-card {
      border-radius: var(--radius-xl);
      padding: 14px;
      /* Elevation lives in the shadow; the border only keeps the edge warm. */
      box-shadow: 0 1px 2px rgba(28, 27, 24, 0.04), 0 6px 16px rgba(28, 27, 24, 0.05);
      border-color: var(--border-warm);
      transition: box-shadow 0.18s ease-out, transform 0.18s ease-out, border-color 0.18s ease-out;
    }

    .craft-dialog-settings .settings-body > .craft-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 2px 4px rgba(28, 27, 24, 0.05), 0 12px 28px rgba(28, 27, 24, 0.08);
      border-color: #d1ccc0;
    }

    /* Nested boxes and controls sit one level in, so 20px - 14px = 6px. */
    .craft-dialog-settings .settings-body .rounded,
    .craft-dialog-settings .settings-body .form-select,
    .craft-dialog-settings .settings-body .form-control {
      border-radius: var(--radius-sm);
    }

    .craft-dialog-settings .settings-body .progress,
    .craft-dialog-settings .settings-body .progress-bar {
      border-radius: var(--radius-full);
    }

    /* Optical alignment: the title row holds a fixed icon column so titles line
       up regardless of each glyph's own advance width. Targets the existing
       header markup rather than requiring a new class. */
    .craft-dialog-settings .settings-body > .craft-card > .d-flex.align-items-center:first-child {
      gap: 9px;
      min-height: 22px;
    }

    .craft-dialog-settings .settings-body > .craft-card > .d-flex.align-items-center:first-child > i {
      width: 18px;
      text-align: center;
      font-size: 13px;
      line-height: 1;
      transform: translateY(-0.5px);
      flex: 0 0 18px;
    }

    /* Staged entrance: sequence communicates grouping, so cards arrive in
       order at ~70ms apart. Unstaggered would read as one flat block. */
    .modal-backdrop-custom.active .craft-dialog-settings .settings-body > .craft-card {
      animation: settingsCardIn 0.26s cubic-bezier(0.2, 0, 0, 1) both;
    }

    .modal-backdrop-custom.active .craft-dialog-settings .settings-body > .craft-card:nth-child(1) { animation-delay: 0ms; }
    .modal-backdrop-custom.active .craft-dialog-settings .settings-body > .craft-card:nth-child(2) { animation-delay: 70ms; }
    .modal-backdrop-custom.active .craft-dialog-settings .settings-body > .craft-card:nth-child(3) { animation-delay: 140ms; }
    .modal-backdrop-custom.active .craft-dialog-settings .settings-body > .craft-card:nth-child(4) { animation-delay: 210ms; }
    .modal-backdrop-custom.active .craft-dialog-settings .settings-body > .craft-card:nth-child(5) { animation-delay: 280ms; }
    .modal-backdrop-custom.active .craft-dialog-settings .settings-body > .craft-card:nth-child(6) { animation-delay: 350ms; }
    .modal-backdrop-custom.active .craft-dialog-settings .settings-body > .craft-card:nth-child(n+7) { animation-delay: 420ms; }

    @keyframes settingsCardIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Comfort for a long form: a taller dialog eats less scrolling, and the
       body keeps a visible bottom edge instead of clipping mid-card. */
    .craft-dialog-settings .settings-body {
      padding: 18px;
      max-height: calc(85vh - 68px);
    }

    @media (max-width: 720px) {
      .craft-dialog-settings {
        max-width: 100%;
      }

      .craft-dialog-settings .settings-body {
        grid-template-columns: 1fr;
        padding: 14px;
        gap: 12px;
      }

      /* On one column the stagger would just feel like latency. */
      .modal-backdrop-custom.active .craft-dialog-settings .settings-body > .craft-card {
        animation: none;
      }
    }

    @media (prefers-reduced-motion: reduce) {
      .modal-backdrop-custom.active .craft-dialog-settings .settings-body > .craft-card {
        animation: none;
      }

      .craft-dialog-settings .settings-body > .craft-card {
        transition: none;
      }
    }

"""

BODY_ANCHOR = '<div class="p-4 d-flex flex-column gap-3 overflow-auto">\n        <!-- Language Selector Card -->'
BODY_NEW = '<div class="settings-body overflow-auto">\n        <!-- Language Selector Card -->'


def apply(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        print(f"  [FALLA] {label}: {count} coincidencias (se esperaba 1)")
        return text
    print(f"  [ok]    {label}")
    return text.replace(old, new, 1)


def main() -> int:
    if not TARGET.exists():
        print(f"no hay plantilla {TARGET}", file=sys.stderr)
        return 1
    text = TARGET.read_text(encoding="utf-8")
    print(f"plantilla: {TARGET} ({len(text)} caracteres)")

    text = apply(text, DIALOG_ANCHOR, DIALOG_NEW, "dialogo sin ancho fijo")
    text = apply(text, CSS_ANCHOR, CSS_BLOCK + CSS_ANCHOR, "bloque CSS de confort")
    text = apply(text, BODY_ANCHOR, BODY_NEW, "cuerpo como rejilla")

    TARGET.write_text(text, encoding="utf-8")
    print(f"escrito: {len(text)} caracteres")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
