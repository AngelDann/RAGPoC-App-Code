"""Retrieval quality evaluation on cosine similarity.

The retriever ranks by cosine similarity (``ragpoc.retrieval._cosine``), so all
ranking metrics here are computed over that ordering. A benchmark case is a
query paired with the chunk ids a correct system should surface; nothing else
about the corpus is assumed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

# Ranking depths reported by default. K=8 matches the agent's search_knowledge_base
# default, so the numbers describe what the agent actually sees.
DEFAULT_KS = (1, 3, 5, 8)


@dataclass
class RetrievalCase:
    """One labelled query.

    ``relevant`` holds chunk ids. Use ``relevant_documents`` when the gold label
    is at document granularity (common for video/audio, where any segment of the
    right recording counts).
    """

    query: str
    relevant: set[str] = field(default_factory=set)
    relevant_documents: set[str] = field(default_factory=set)
    case_id: str = ""
    media_type: str | None = None


@dataclass
class RetrievalResult:
    case_id: str
    query: str
    retrieved: list[str]
    scores: list[float]
    latency_s: float
    relevant: set[str]
    relevant_documents: set[str]
    retrieved_document_ids: list[str] = field(default_factory=list)

    def hits(self, k: int, mode: str = "chunk") -> int:
        if mode == "document":
            gold = self.relevant_documents
            retrieved = self.retrieved_document_ids
        else:
            gold = self.relevant
            retrieved = self.retrieved
        return len(set(retrieved[:k]) & gold)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_l = math.sqrt(sum(a * a for a in left))
    norm_r = math.sqrt(sum(b * b for b in right))
    if not norm_l or not norm_r:
        return 0.0
    return dot / (norm_l * norm_r)


def recall_at_k(result: RetrievalResult, k: int, mode: str = "chunk") -> float:
    gold = result.relevant_documents if mode == "document" else result.relevant
    if not gold:
        return 0.0
    return result.hits(k, mode) / len(gold)


def precision_at_k(result: RetrievalResult, k: int, mode: str = "chunk") -> float:
    if k == 0:
        return 0.0
    return result.hits(k, mode) / k


def hit_rate_at_k(result: RetrievalResult, k: int, mode: str = "chunk") -> float:
    return 1.0 if result.hits(k, mode) > 0 else 0.0


def reciprocal_rank(result: RetrievalResult, mode: str = "chunk") -> float:
    gold = result.relevant_documents if mode == "document" else result.relevant
    retrieved = result.retrieved_document_ids if mode == "document" else result.retrieved
    for index, item in enumerate(retrieved, start=1):
        if item in gold:
            return 1.0 / index
    return 0.0


def ndcg_at_k(result: RetrievalResult, k: int, mode: str = "chunk") -> float:
    gold = result.relevant_documents if mode == "document" else result.relevant
    retrieved = result.retrieved_document_ids if mode == "document" else result.retrieved
    if not gold:
        return 0.0
    dcg = 0.0
    seen: set[str] = set()
    for index, item in enumerate(retrieved[:k], start=1):
        if item in gold and item not in seen:
            dcg += 1.0 / math.log2(index + 1)
            seen.add(item)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def score_case(
    result: RetrievalResult,
    ks: tuple[int, ...] = DEFAULT_KS,
    mode: str = "chunk",
) -> dict[str, float]:
    metrics: dict[str, float] = {"mrr": reciprocal_rank(result, mode), "latency_s": result.latency_s}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(result, k, mode)
        metrics[f"precision@{k}"] = precision_at_k(result, k, mode)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(result, k, mode)
        metrics[f"ndcg@{k}"] = ndcg_at_k(result, k, mode)
    return metrics


def aggregate(
    results: list[RetrievalResult],
    ks: tuple[int, ...] = DEFAULT_KS,
    mode: str = "chunk",
) -> dict[str, Any]:
    if not results:
        return {"cases": 0, "mode": mode}
    per_case = [score_case(r, ks, mode) for r in results]
    keys = per_case[0].keys()
    averaged = {key: _mean([row[key] for row in per_case]) for key in keys}
    latencies = [r.latency_s for r in results]
    averaged["latency_p50_s"] = _percentile(latencies, 0.5)
    averaged["latency_p95_s"] = _percentile(latencies, 0.95)
    averaged["latency_max_s"] = max(latencies)
    averaged["cases"] = len(results)
    averaged["mode"] = mode
    averaged["queries_with_no_relevant"] = sum(
        1 for r in results if not r.relevant and not r.relevant_documents
    )
    return averaged


async def run_retrieval_eval(
    retriever: Any,
    cases: list[RetrievalCase],
    ks: tuple[int, ...] = DEFAULT_KS,
    mode: str = "chunk",
    top_k: int | None = None,
) -> dict[str, Any]:
    """Execute every case against ``retriever`` and score the cosine ranking.

    ``retriever`` is anything with an async ``search(query, top_k=...)`` method
    returning dicts that carry ``chunk_id``, ``document_id`` and ``score``.
    """
    depth = top_k or max(ks)
    results: list[RetrievalResult] = []
    for case in cases:
        started = time.perf_counter()
        hits = await retriever.search(case.query, top_k=depth, media_type=case.media_type)
        elapsed = time.perf_counter() - started
        results.append(
            RetrievalResult(
                case_id=case.case_id or case.query[:48],
                query=case.query,
                retrieved=[str(h["chunk_id"]) for h in hits],
                scores=[float(h.get("score", 0.0)) for h in hits],
                latency_s=elapsed,
                relevant=set(case.relevant),
                relevant_documents=set(case.relevant_documents),
                retrieved_document_ids=[str(h["document_id"]) for h in hits],
            )
        )
    return {
        "summary": aggregate(results, ks, mode),
        "per_case": [
            {
                "case_id": r.case_id,
                "query": r.query,
                "retrieved": r.retrieved[: max(ks)],
                "scores": [round(s, 4) for s in r.scores[: max(ks)]],
                "latency_s": round(r.latency_s, 4),
                **{k: round(v, 4) for k, v in score_case(r, ks, mode).items()},
            }
            for r in results
        ],
    }
