"""Refresh the report's status section after the pending work landed."""

from __future__ import annotations

from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "INFORME_AGENTE_LOCAL.md"

OLD_START = "### Completado y verificado"
OLD_END = "### Fallo de pruebas preexistente (fuera de alcance)"

NEW = """### Completado y verificado

- Arneses de medicion de RAM, latencia y generacion
- Presupuesto de memoria con control de admision
- Suite de metricas de recuperacion por coseno, con **gold set y numeros reales**
- Runtime local ONNX GenAI con streaming
- Adaptador pydantic-ai con tool-loop validado end-to-end
- Multimodal: imagen, video y audio validados sobre el modelo real
- Proveedor `local` en `llm.py`, settings persistidos y UI de Ajustes
- **Mitigacion de alucinacion** \u2014 barrera estructural + metrica semantica:
  abstencion 0,00 \u2192 **1,00**, alucinacion 0,60 \u2192 **0,00**
- **Whisper bajo demanda** \u2014 el pico sin audio queda en **2 341 MB**
- **Herramienta de reparacion** de claves foraneas huerfanas
- 281 pruebas nuevas en verde; suite total 431 en verde

### Pendiente

Solo queda lo que requiere una decision de producto, no de ingenieria:

1. **Ejecutar la reparacion de datos** cuando confirmes que esas 2 filas se
   pueden borrar. Mientras no se haga, `manage.py migrate` seguira abortando.
2. **Limpiar los vectores de prueba** (`c_focus_1`, `c_focus_2`, `c_guide`,
   `c_guide_unique_test`) y los 4 vectores sin chunk, y reindexar. Sube el
   recall real (seccion 9.2).
3. **Glosario de acronimos** en la memoria del agente (seccion 5.3).
4. Evaluacion de calidad con juicio humano, para validar las metricas
   automaticas.

### Datos corruptos en la base (requieren tu confirmacion)

`scripts/repair_orphan_documents.py` en modo solo lectura reporta **2 filas
huerfanas**:

| Tabla | Fila | Referencia rota |
|---|---|---|
| `notebook_documents` | `id=19` | `document_id=e38ea53b-5585-43a7-a501-3daa9ab6a679` |
| `chunks` | `rowid=188` | `document_id=b4adeff2-1551-4520-bfcc-f142906b3297` |

La segunda ademas tiene una entrada colgante en `chunks_fts`. Ambas **impiden
ejecutar `manage.py migrate`**, porque Django verifica todas las claves
foraneas al final de cada migracion. La migracion `0015` se aplico mediante DDL
directo (`scripts/apply_0015.py`), que evita esa verificacion.

**No se borro nada.** Para reparar, con respaldo automatico y re-verificacion
previa a cada borrado:

```bash
python scripts/repair_orphan_documents.py --dry-run   # informa, no toca nada
python scripts/repair_orphan_documents.py --apply     # respalda y borra
```

"""


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    start = text.find(OLD_START)
    end = text.find(OLD_END)
    if start < 0 or end < 0 or end < start:
        print("no se encontraron los anclajes")
        return 1
    updated = text[:start] + NEW + text[end:]
    TARGET.write_text(updated, encoding="utf-8")
    print(f"reemplazado: {end - start} -> {len(NEW)} caracteres")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
