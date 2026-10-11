"""End-to-end smoke: pydantic-ai Agent + local ONNX model + one tool."""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knowledge.local_agent_model import build_local_agent_model  # noqa: E402
from ragpoc.budget import MemoryBudget  # noqa: E402
from ragpoc.local_llm import LocalLlmEngine  # noqa: E402
from ragpoc.metrics import PeakMemoryTracker, process_rss_bytes  # noqa: E402

MB = 1024 * 1024
OUT = ROOT / "scripts" / "agent_e2e.txt"
lines: list[str] = []


def w(s: object = "") -> None:
    lines.append(str(s))


async def main() -> None:
    model_key = sys.argv[1] if len(sys.argv) > 1 else "qwen3-0.6b"
    engine = LocalLlmEngine(
        ROOT / "data" / "models" / model_key,
        dtype="q4f16",
        device="cpu",
        context_length=1024,
        max_new_tokens=200,
        budget=MemoryBudget(),
    )

    from pydantic_ai import Agent

    with PeakMemoryTracker(interval=0.05) as tracker:
        engine.load()
        w(f"model            : {model_key}")
        w(f"rss after load   : {process_rss_bytes() / MB:.1f} MB")

        adapter = build_local_agent_model(engine, name="local-smoke")
        agent = Agent(adapter.model, deps_type=dict)

        @agent.tool_plain
        def search_knowledge_base(query: str) -> str:
            """Busca en la base de conocimiento local y devuelve fragmentos.

            Args:
                query: consulta en lenguaje natural.
            """
            return f"fragmento_simulado: la similitud coseno compara vectores (consulta={query})"

        w("")
        w("--- caso 1: pregunta directa, sin tools ---")
        result = await agent.run("Que es la similitud coseno? Responde en una frase.")
        w("respuesta: " + str(result.output)[:400])
        w("")

        w("--- caso 2: la tool debe invocarse ---")
        result2 = await agent.run("Busca en la base de conocimiento sobre similitud coseno.")
        w("respuesta: " + str(result2.output)[:400])
        w("")
        calls = []
        try:
            for message in result2.all_messages():
                for part in getattr(message, "parts", []):
                    if type(part).__name__ == "ToolCallPart":
                        calls.append(part.tool_name)
                    if type(part).__name__ == "ToolReturnPart":
                        calls.append("return:" + part.tool_name)
        except Exception as exc:
            w(f"(no se pudieron leer las tool calls: {exc})")
        w("tool trace: " + json.dumps(calls))

    w("")
    w(f"peak rss         : {tracker.peak_mb:.1f} MB")
    w(f"delta rss        : {tracker.delta_mb:.1f} MB")
    engine.unload()

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT)


asyncio.run(main())
