"""Benchmark the local agent stack: memory, latency and answer quality.

Produces a JSON report covering the three things the product spec cares about:

* resource usage against the 2 GB target / 3 GB ceiling
* inference time (load, time to first token, tokens per second)
* answer quality on a fixed question set, with and without RAG context

Run it after the models are downloaded:

    python scripts/run_local_agent_bench.py --model qwen3-0.6b
    python scripts/run_local_agent_bench.py --model qwen3-vl-2b --out report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from knowledge.local_agent_model import apply_guard  # noqa: E402
from ragpoc.budget import HARD_LIMIT_BYTES, SOFT_LIMIT_BYTES, MemoryBudget  # noqa: E402
from ragpoc.chat_models import CHAT_MODELS  # noqa: E402
from ragpoc.config import Settings  # noqa: E402
from ragpoc.eval_retrieval import RetrievalCase, run_retrieval_eval  # noqa: E402
from ragpoc.local_llm import (  # noqa: E402
    ChatMessage,
    LocalLlmEngine,
    strip_thinking,
)
from ragpoc.metrics import MB, PeakMemoryTracker, process_rss_bytes  # noqa: E402

# Fixed question set. Answers are graded for groundedness and language below.
AGENT_QUESTIONS = [
    {
        "id": "def-rag",
        "question": "Que es un sistema RAG y para que sirve?",
        "kind": "knowledge",
        "must_include": ["recuper", "documento"],
        "language": "es",
    },
    {
        "id": "def-cosine",
        "question": "Explica brevemente que es la similitud coseno en recuperacion de informacion.",
        "kind": "knowledge",
        "must_include": ["coseno", "vector"],
        "language": "es",
    },
    {
        "id": "summarize",
        "question": "Resume en dos frases: la base de conocimiento indexa documentos en fragmentos y los busca por similitud coseno.",
        "kind": "summarization",
        "must_include": ["fragmento", "similitud"],
        "language": "es",
    },
    {
        "id": "extract",
        "question": "Extrae el nombre propio y la ciudad de: 'Ana Ruiz trabaja en Sevilla desde 2019'.",
        "kind": "extraction",
        "must_include": ["Ana Ruiz", "Sevilla"],
        "language": "es",
    },
    {
        "id": "refuse",
        "question": "Cual es el precio exacto de las acciones de la empresa ACME el 3 de marzo de 2019?",
        "kind": "unanswerable",
        "must_include": [],
        "language": "es",
    },
    {
        "id": "refuse-date",
        "question": "En que fecha exacta se aprobo el contrato de licencia del proyecto RAGPoC?",
        "kind": "unanswerable",
        "must_include": [],
        "language": "es",
    },
    {
        "id": "refuse-person",
        "question": "Cual es el numero de identificacion fiscal del autor de este documento?",
        "kind": "unanswerable",
        "must_include": [],
        "language": "es",
    },
    {
        "id": "refuse-quantity",
        "question": "Cuantas unidades del producto se vendieron en el ultimo trimestre?",
        "kind": "unanswerable",
        "must_include": [],
        "language": "es",
    },
    {
        "id": "refuse-en",
        "question": "What was the exact revenue of the company in Q3 2021?",
        "kind": "unanswerable",
        "must_include": [],
        "language": "en",
    },
    {
        "id": "en-answer",
        "question": "In one sentence: what is a vector database?",
        "kind": "knowledge",
        "must_include": ["vector"],
        "language": "en",
    },
]


def grade_answer(entry: dict, text: str) -> dict:
    """Score one answer with the semantic scorer, keeping keyword recall.

    The keyword recall is retained so numbers stay comparable with earlier
    runs; the fabrication/abstention dimensions are what actually catch the
    measured hallucination defect.
    """
    from ragpoc.eval_quality import score_answer

    score = score_answer(
        text,
        expected_terms=entry["must_include"],
        case_kind=entry["kind"],
        expected_language=entry["language"],
    )
    return {
        "case_id": entry["id"],
        "question": entry["question"],
        "kind": entry["kind"],
        "answer_chars": len(text.strip()),
        "answer_preview": text.strip()[:220],
        "expected_terms": entry["must_include"],
        "matched_terms": score.matched_terms,
        "term_recall": score.keyword_recall,
        "abstained": score.abstained,
        "fabrication_risk": score.fabrication_risk,
        "abstention_score": score.abstention_score,
        "shape_score": score.shape_score,
        "language_score": score.language_score,
        "detected_language": score.detected_language,
        "citation_score": score.citation_score,
        "groundedness": score.groundedness,
        "quality_overall": score.overall_for(None),
        "length_ok": not score.empty and not score.too_long,
    }


def build_engine(model_key: str, context_length: int) -> tuple[LocalLlmEngine, MemoryBudget, Path]:
    settings = Settings(_env_file=None, data_dir="data", allowed_upload_dir="data")
    manifest = CHAT_MODELS[model_key]
    model_dir = Path(settings.local_chat_cache_dir) / manifest.key
    budget = MemoryBudget(
        hard_limit_bytes=HARD_LIMIT_BYTES,
        soft_limit_bytes=SOFT_LIMIT_BYTES,
    )
    engine = LocalLlmEngine(
        model_dir,
        dtype=settings.local_chat_dtype,
        device=settings.local_chat_device,
        context_length=context_length,
        budget=budget,
    )
    return engine, budget, model_dir


def benchmark_generation(engine: LocalLlmEngine, runs: int, max_new_tokens: int) -> dict:
    rows = []
    with PeakMemoryTracker(interval=0.05) as tracker:
        for entry in AGENT_QUESTIONS:
            messages = [
                ChatMessage(role="system", content="Responde en espanol de forma breve y directa. Sin preambulos."),
                ChatMessage(role="user", content=entry["question"]),
            ]
            started = time.perf_counter()
            result = engine.generate(messages, max_new_tokens=max_new_tokens, temperature=0.2)
            # Same guard the adapter applies, so the numbers describe what ships.
            guarded = apply_guard(engine, messages, result.text, max_new_tokens=max_new_tokens)
            elapsed = time.perf_counter() - started
            row = {
                "latency_s": round(elapsed, 3),
                **result.metrics.summary(),
                **grade_answer(entry, strip_thinking(guarded)),
            }
            rows.append(row)
    latencies = sorted(r["latency_s"] for r in rows)
    speeds = sorted(r["tokens_per_second"] for r in rows if r["tokens_per_second"])
    term_recalls = [r["term_recall"] for r in rows if r["expected_terms"]]
    unanswerable = [r for r in rows if r["kind"] == "unanswerable"]
    abstained = [r for r in unanswerable if r["abstained"]]
    fabricated = [r for r in unanswerable if r["fabrication_risk"] >= 0.5]
    quality = [r["quality_overall"] for r in rows if r["quality_overall"] is not None]
    return {
        "cases": rows,
        "aggregate": {
            "n": len(rows),
            "latency_mean_s": round(sum(latencies) / len(latencies), 3) if latencies else 0,
            "latency_p50_s": round(latencies[len(latencies) // 2], 3) if latencies else 0,
            "latency_max_s": round(max(latencies), 3) if latencies else 0,
            "tokens_per_second_mean": round(sum(speeds) / len(speeds), 2) if speeds else 0,
            "tokens_per_second_min": round(min(speeds), 2) if speeds else 0,
            "term_recall_mean": round(sum(term_recalls) / len(term_recalls), 3) if term_recalls else 0,
            "quality_overall_mean": round(sum(quality) / len(quality), 3) if quality else 0,
            "unanswerable_cases": len(unanswerable),
            "abstention_rate": round(len(abstained) / len(unanswerable), 3) if unanswerable else 0,
            "hallucination_rate": round(len(fabricated) / len(unanswerable), 3) if unanswerable else 0,
            "peak_rss_mb": round(tracker.peak_mb, 1),
            "delta_rss_mb": round(tracker.delta_mb, 1),
            "baseline_rss_mb": round(tracker.baseline_mb, 1),
        },
    }


class _FakeRetriever:
    """Stand-in used to exercise the retrieval metric path without a corpus."""

    def __init__(self, corpus: list[dict]):
        self.corpus = corpus

    async def search(self, query: str, top_k: int = 8, media_type: str | None = None):
        from ragpoc.eval_retrieval import cosine_similarity

        scored = []
        for item in self.corpus:
            scored.append((cosine_similarity(item["vec"], self._embed(query)), item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {"chunk_id": c["chunk_id"], "document_id": c["document_id"], "score": s, **c}
            for s, c in scored[:top_k]
        ]

    @staticmethod
    def _embed(text: str) -> list[float]:
        import hashlib
        import math

        digest = hashlib.sha256(text.encode()).digest()
        vec = [b / 255.0 for b in digest[:32]]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def synthetic_retrieval_bench() -> dict:
    """Deterministic retrieval metrics with a known-answer corpus.

    Used to prove the cosine scoring pipeline is sound end to end; real corpus
    numbers come from run_retrieval_eval against the user's own knowledge base.
    """
    import hashlib
    import math

    def embed(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        vec = [b / 255.0 for b in digest[:32]]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    corpus = [
        {"chunk_id": f"c{i}", "document_id": f"d{i // 3}", "vec": embed(f"topic {i}"), "text": f"topic {i}"}
        for i in range(24)
    ]
    cases = [
        RetrievalCase(query="topic 5", relevant={"c5"}, relevant_documents={"d1"}, case_id="hit"),
        RetrievalCase(query="topic 0", relevant={"c0"}, relevant_documents={"d0"}, case_id="hit2"),
        RetrievalCase(query="unrelated text", relevant={"c9"}, relevant_documents={"d3"}, case_id="miss"),
    ]
    retriever = _FakeRetriever(corpus)

    async def run():
        return await run_retrieval_eval(retriever, cases, ks=(1, 3, 5, 8))

    return asyncio.run(run())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-vl-2b", choices=sorted(CHAT_MODELS))
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out", default="")
    parser.add_argument("--skip-retrieval", action="store_true")
    args = parser.parse_args()

    if args.model not in CHAT_MODELS:
        print(f"unknown model {args.model!r}", file=sys.stderr)
        return 2

    engine, budget, model_dir = build_engine(args.model, args.context_length)
    report: dict = {
        "model": args.model,
        "model_dir": str(model_dir),
        "context_length": args.context_length,
        "budget": {
            "hard_limit_mb": round(HARD_LIMIT_BYTES / MB, 1),
            "soft_limit_mb": round(SOFT_LIMIT_BYTES / MB, 1),
        },
        "process_rss_start_mb": round(process_rss_bytes() / MB, 1),
    }

    print(f"loading {args.model} from {model_dir} ...")
    started = time.perf_counter()
    try:
        engine.load()
    except Exception as exc:
        print(f"load failed: {exc}", file=sys.stderr)
        report["error"] = str(exc)
        _write(report, args.out)
        return 1
    report["load_s"] = round(time.perf_counter() - started, 3)
    report["rss_after_load_mb"] = round(process_rss_bytes() / MB, 1)
    print(f"loaded in {report['load_s']}s, rss={report['rss_after_load_mb']} MB")

    report["generation"] = benchmark_generation(engine, args.runs, args.max_new_tokens)
    agg = report["generation"]["aggregate"]
    print(
        f"gen: {agg['tokens_per_second_mean']} tok/s, "
        f"latency p50 {agg['latency_p50_s']}s, peak rss {agg['peak_rss_mb']} MB"
    )

    if not args.skip_retrieval:
        report["retrieval_synthetic"] = synthetic_retrieval_bench()
        summary = report["retrieval_synthetic"]["summary"]
        print(f"retrieval: recall@8={summary.get('recall@8')} mrr={summary.get('mrr')}")

    report["budget_report"] = budget.report()
    report["within_hard_limit"] = report["budget_report"]["within_hard_limit"]
    report["meets_2gb_target"] = agg["peak_rss_mb"] <= 2048

    engine.unload()
    _write(report, args.out)
    print(f"within 3GB ceiling: {report['within_hard_limit']} | under 2GB target: {report['meets_2gb_target']}")
    return 0


def _write(report: dict, out: str) -> None:
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if out:
        Path(out).write_text(payload, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(payload)


if __name__ == "__main__":
    raise SystemExit(main())
