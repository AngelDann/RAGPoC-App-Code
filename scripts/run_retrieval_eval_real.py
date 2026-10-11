"""Measure retrieval quality against the real knowledge base.

Scores the production Retriever's cosine ranking on the gold set produced by
build_retrieval_gold, so the numbers describe what the agent actually sees. The
default HashingEmbedder keeps the benchmark deterministic and offline;
--provider local swaps in the app's configured embedding provider for semantic
numbers (network and credentials permitting).

    python scripts/run_retrieval_eval_real.py
    python scripts/run_retrieval_eval_real.py --provider local --ks 1,3,5,8
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sqlite3
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_retrieval_gold import load_gold  # noqa: E402

from ragpoc.config import Settings  # noqa: E402
from ragpoc.db import configured_dimension, connect  # noqa: E402
from ragpoc.embeddings import get_provider  # noqa: E402
from ragpoc.eval_retrieval import DEFAULT_KS, RetrievalCase, run_retrieval_eval  # noqa: E402
from ragpoc.retrieval import Retriever, _decode_vector  # noqa: E402

DEFAULT_DIMENSION = 768
DEFAULT_SEED = 13

HASHING_NOTE = (
    "hashing embedder: query vectors are deterministic hashed bag-of-words while corpus vectors "
    "come from the indexed model, so these numbers exercise the retrieval pipeline and bound a "
    "worst case; run --provider local for semantic retrieval quality."
)


class GoldSetError(ValueError):
    """Gold set missing or malformed; the message is the user-facing fix."""


class HashingEmbedder:
    """Deterministic bag-of-words embedder: no network, no model, reproducible.

    The dimension must match the indexed vectors because retrieval._cosine zips
    the two strictly and raises on a length mismatch; callers pass the database's
    stored dimension (the 768 default covers fresh corpora).
    """

    _TOKENS = re.compile(r"\w+")

    def __init__(self, dimension: int = DEFAULT_DIMENSION, seed: int = DEFAULT_SEED):
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension, self.seed = dimension, seed

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    async def embed_image(self, path: Path) -> list[float]:
        raise NotImplementedError(self._media_error("image"))

    async def embed_video(self, path: Path) -> list[float]:
        raise NotImplementedError(self._media_error("video"))

    async def embed_audio(self, path: Path) -> list[float]:
        raise NotImplementedError(self._media_error("audio"))

    def _embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimension
        for token in self._TOKENS.findall(text.casefold()):
            digest = hashlib.sha256(f"{self.seed}:{token}".encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            values[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values] if norm else values

    @staticmethod
    def _media_error(kind: str) -> str:
        return f"HashingEmbedder is text-only; use --provider local to embed {kind}."


def to_retrieval_case(case: dict, index: int = 0) -> RetrievalCase:
    return RetrievalCase(
        query=case["query"],
        relevant={str(chunk_id) for chunk_id in case["relevant_chunk_ids"]},
        relevant_documents={str(document_id) for document_id in case["relevant_document_ids"]},
        case_id=case.get("case_id") or f"g{index:03d}",
    )


def load_gold_cases(path: Path) -> tuple[list[RetrievalCase], dict]:
    if not path.exists():
        raise GoldSetError(
            f"gold set not found at {path}; run python scripts/build_retrieval_gold.py first"
        )
    try:
        cases, meta = load_gold(path)
    except (ValueError, OSError) as error:
        raise GoldSetError(f"gold set at {path} is not usable: {error}") from error
    return [to_retrieval_case(case, index) for index, case in enumerate(cases)], meta


def build_report(
    eval_output: dict,
    *,
    gold_meta: dict,
    provider_name: str,
    database: str,
    dimension: int | None,
    ks: Sequence[int],
    top_k: int,
) -> dict:
    """Fold an eval run into the persisted report so the numbers stay self-describing."""
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "database": database,
        "provider": provider_name,
        "embedding_dimension": dimension,
        "ks": list(ks),
        "top_k": top_k,
        "notes": [HASHING_NOTE] if provider_name == "hashing" else [],
        "gold": gold_meta,
        "summary": eval_output.get("summary", {}),
        "per_case": eval_output.get("per_case", []),
    }


def format_summary(report: dict) -> str:
    """Percentages per depth; retrieval quality is reported as a percentage to the user."""
    summary = report["summary"]
    ks = report["ks"]
    lines = [
        f"retrieval eval — {summary.get('cases', 0)} cases | provider {report['provider']}"
        f" | db {report['database']}",
        f"{'metric':<10}" + "".join(f"{f'K={k}':<10}" for k in ks),
    ]
    for label, template in (
        ("recall", "recall@{}"),
        ("precision", "precision@{}"),
        ("hit_rate", "hit_rate@{}"),
        ("ndcg", "ndcg@{}"),
    ):
        cells = "".join(f"{_pct(summary.get(template.format(k), 0.0)):<10}" for k in ks)
        lines.append(f"{label:<10}{cells}")
    lines.append(f"{'mrr':<10}{_pct(summary.get('mrr', 0.0))}")
    lines.append(
        f"{'latency':<10}p50 {_ms(summary.get('latency_p50_s', 0.0))}"
        f" | p95 {_ms(summary.get('latency_p95_s', 0.0))}"
        f" | max {_ms(summary.get('latency_max_s', 0.0))}"
    )
    return "\n".join(lines)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}ms"


def _parse_ks(value: str) -> tuple[int, ...]:
    try:
        ks = tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid --ks {value!r}; expected e.g. 1,3,5,8") from error
    if not ks or any(depth < 1 for depth in ks):
        raise argparse.ArgumentTypeError(f"invalid --ks {value!r}; every depth must be >= 1")
    return ks


def _index_dimension(connection: sqlite3.Connection) -> int | None:
    try:
        row = connection.execute("SELECT embedding FROM chunk_vectors LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        return None
    return len(_decode_vector(row[0])) if row else None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on the real knowledge base.")
    parser.add_argument("--gold", default=None, help="gold set JSON (default: data/retrieval_gold.json)")
    parser.add_argument("--db", default=None, help="SQLite database (default: the app's data/ragpoc.db)")
    parser.add_argument(
        "--ks", type=_parse_ks, default=DEFAULT_KS, help="comma-separated depths (default 1,3,5,8)"
    )
    parser.add_argument("--top-k", type=int, default=None, help="depth passed to search (default: max K)")
    parser.add_argument(
        "--out", default=None, help="report JSON (default: data/retrieval_eval_report.json)"
    )
    parser.add_argument(
        "--provider",
        choices=("hashing", "local"),
        default="hashing",
        help="hashing: deterministic offline embedder; local: the app's configured embedding provider",
    )
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N gold cases")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = Settings()
    gold_path = Path(args.gold) if args.gold else settings.data_dir / "retrieval_gold.json"
    out_path = Path(args.out) if args.out else settings.data_dir / "retrieval_eval_report.json"
    db_path = Path(args.db) if args.db else settings.database_path

    try:
        cases, gold_meta = load_gold_cases(gold_path)
    except GoldSetError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not db_path.exists():
        print(f"error: no database at {db_path}; ingest documents first", file=sys.stderr)
        return 2
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("error: the gold set has no cases to evaluate", file=sys.stderr)
        return 2

    try:
        connection = connect(db_path)
    except sqlite3.Error as error:
        print(f"error: cannot open database {db_path}: {error}", file=sys.stderr)
        return 2
    try:
        dimension = _index_dimension(connection) or configured_dimension(connection) or DEFAULT_DIMENSION
        provider = (
            HashingEmbedder(dimension=dimension) if args.provider == "hashing" else get_provider(settings)
        )
        retriever = Retriever(connection, provider, max_top_k=settings.max_top_k)
        eval_output = asyncio.run(run_retrieval_eval(retriever, cases, ks=args.ks, top_k=args.top_k))
    except Exception as error:
        print(f"error: retrieval evaluation failed: {error}", file=sys.stderr)
        return 1
    finally:
        connection.close()

    report = build_report(
        eval_output,
        gold_meta=gold_meta,
        provider_name=args.provider,
        database=str(db_path),
        dimension=dimension,
        ks=args.ks,
        top_k=args.top_k or max(args.ks),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(format_summary(report))
    for note in report["notes"]:
        print(f"note: {note}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
