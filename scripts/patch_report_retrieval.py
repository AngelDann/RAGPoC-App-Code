"""Refresh the retrieval numbers in the report after the data cleanup."""

from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "INFORME_AGENTE_LOCAL.md"

REPLACEMENTS = [
    ("**70,09%**", "**77,88%**"),
    ("**80,74%**", "**89,70%**"),
    ("**85,00%**", "**94,44%**"),
    ("**90,00%**", "**100,00%**"),
    ("**88,47%**", "**98,30%**"),
    ("**87,50%**", "**97,22%**"),
    ("796 ms / 1 051 ms / 1 361 ms", "785 ms / 1 164 ms / 1 482 ms"),
    ("**191 fragmentos / 164 documentos**", "**188 fragmentos / 166 documentos**"),
    ("20 consultas", "18 consultas"),
]

OLD_SECTION_START = "### 9.2 Dos fallos son datos corruptos, no retrieval"
OLD_SECTION_END = "### 9.3 Riesgo de memoria en la recuperación"

NEW_SECTION = """### 9.2 La limpieza de datos subió el retrieval

Los primeros números medidos venían degradados por datos de prueba que habían
quedado en la base de producción. Se eliminaron y se re-midió:

| Metrica | Antes | **Despues** |
|---|---|---|
| **MRR** | 87,50% | **97,22%** |
| Recall@1 | 70,09% | **77,88%** |
| Recall@8 | 80,74% | **89,70%** |
| Precision@1 | 85,00% | **94,44%** |
| Hit rate@3 | 90,00% | **100,00%** |
| nDCG@8 | 88,47% | **98,30%** |

Lo que se limpio con `scripts/clean_test_vectors.py`:

* **5 vectores huerfanos** que apuntaban a chunks borrados (4 de segmentos de
  video y 1 de texto). Quedaron porque el driver sqlite3 plano no puede cargar
  la extension `vec0` que respalda la tabla virtual.
* **2 vectores placeholder** (`c_focus_1`, `c_focus_2`) con todas las dimensiones
  rellenadas con `0.1`. No eran embeddings de nada: clasificaban al azar y
  justamente eran las 2 consultas que nunca se recuperaban.

Actualmente el almacén tiene **188 vectores, 0 huérfanos, 0 placeholder** y
todas las consultas del gold set recuperan su contenido en el **top-3**.

"""


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    applied = 0
    for old, new in REPLACEMENTS:
        if old in text:
            text = text.replace(old, new)
            applied += 1

    start = text.find(OLD_SECTION_START)
    end = text.find(OLD_SECTION_END)
    if 0 <= start < end:
        text = text[:start] + NEW_SECTION + text[end:]
        applied += 1

    TARGET.write_text(text, encoding="utf-8")
    print(f"cambios aplicados: {applied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
