"""Insert the anti-hallucination directive into both agent system prompts.

The prompts contain accented Spanish that does not round-trip through the
shell, so the insertion is done from a script with explicit UTF-8.

The directive exists because a measured defect: asked for a fact that is not
in the knowledge base, the local model answered with a fabricated price
("$18.50") instead of abstaining. For an agent that answers from the user's
own documents, a invented figure is worse than an admitted gap.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "knowledge" / "pydantic_agent.py"

ES_ANCHOR = "- **Despu\u00e9s de `create_workspace_page`"
EN_ANCHOR = "- **After `create_workspace_page`"

ES_BLOCK = (
    "- **Exactitud y Abstenci\u00f3n (cr\u00edtico):** Responde SOLO con lo que est\u00e9 respaldado por la "
    "evidencia recuperada o por lo que el usuario te haya dado. Si un dato concreto (precio, fecha, "
    "cantidad, n\u00famero de identificaci\u00f3n, porcentaje, nombre propio) NO aparece en los fragmentos "
    "devueltos por `search_knowledge_base`, en el contenido le\u00eddo con `fetch_web_page`, ni en el "
    "historial, NO lo inventes: di expl\u00edcitamente que ese dato no est\u00e1 disponible en la base de "
    "conocimiento y ofrece buscarlo o pedir que se indexe la fuente. Es preferible decir \"no lo s\u00e9\" "
    "a entregar una cifra, fecha o nombre fabricado.\n"
    "- **Antes de afirmar un dato num\u00e9rico, una fecha o un nombre propio:** verifica que puedes "
    "citar el fragmento concreto del que sale. Si no puedes citarlo, no lo afirmes; ind\u00edcalo como "
    "no verificado.\n"
)

EN_BLOCK = (
    "- **Accuracy and Abstention (critical):** Answer ONLY from what is supported by the retrieved "
    "evidence or by what the user gave you. If a concrete value (price, date, quantity, identifier, "
    "percentage, proper name) does NOT appear in the chunks returned by `search_knowledge_base`, in "
    "content read with `fetch_web_page`, or in the history, do NOT invent it: say explicitly that the "
    "value is not available in the knowledge base, and offer to search for it or index the source. "
    "Saying \"I don't know\" is better than a fabricated figure, date or name.\n"
    "- **Before asserting a number, a date or a proper name:** check that you can cite the specific "
    "chunk it comes from. If you cannot cite it, do not assert it; mark it as unverified.\n"
)


def insert(text: str, anchor: str, block: str, label: str) -> str:
    if block.splitlines()[0][:40] in text:
        print(f"  [skip]  {label}: ya presente")
        return text
    index = text.find(anchor)
    if index < 0:
        print(f"  [FALLA] {label}: no se encontro el ancla")
        return text
    print(f"  [ok]    {label}")
    return text[:index] + block + text[index:]


def main() -> int:
    if not TARGET.exists():
        print(f"no hay archivo {TARGET}", file=sys.stderr)
        return 1
    text = TARGET.read_text(encoding="utf-8")
    print(f"archivo: {TARGET} ({len(text)} caracteres)")
    text = insert(text, ES_ANCHOR, ES_BLOCK, "directiva ES")
    text = insert(text, EN_ANCHOR, EN_BLOCK, "directiva EN")
    TARGET.write_text(text, encoding="utf-8")
    print(f"escrito: {len(text)} caracteres")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
