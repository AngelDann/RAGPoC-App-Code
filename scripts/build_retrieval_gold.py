"""Derive a gold retrieval set from the real knowledge base.

Queries are generated from each chunk's own text so the chunk is the known
correct answer: no manual labelling and no external model, which keeps the set
deterministic and reproducible. Chunks sharing identical text are labelled
relevant together, because a retriever that returns any duplicate is behaving
correctly and must not be penalised for it.

    python scripts/build_retrieval_gold.py
    python scripts/build_retrieval_gold.py --min-chars 30 --limit 50
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragpoc.config import Settings  # noqa: E402
from ragpoc.db import connect  # noqa: E402

DEFAULT_OUT = "data/retrieval_gold.json"
# The corpus's real notes average ~40 chars; 30 keeps every distinct text while
# dropping degenerate fragments. Raise it for prose corpora.
DEFAULT_MIN_CHARS = 30
DEFAULT_SEED = 13
MIN_QUERY_WORDS = 2
MID_PHRASE_WORDS = 5

STRATEGIES = ("first_sentence", "mid_phrase", "title_filename")
STRATEGY_DESCRIPTION = (
    "queries derived from chunk text (first sentence, mid-chunk phrase) and informative filenames; "
    "identical texts share relevance"
)


class GoldBuildError(RuntimeError):
    """The database cannot yield gold cases yet (empty, partial or unindexed)."""


def first_sentence_query(text: str) -> str | None:
    """Opening sentence of the chunk, skipping slug-style headers without real words."""
    for line in text.splitlines():
        fragment = line.strip()
        match = re.search(r"[.!?](?=\s|$)", fragment)
        if match:
            fragment = fragment[: match.end()]
        query = " ".join(fragment.split())
        if len(query.split()) >= MIN_QUERY_WORDS:
            return query
    return None


def mid_phrase_query(text: str) -> str | None:
    """A phrase from the middle of the chunk, where wording is most distinctive."""
    words = text.split()
    if len(words) < MID_PHRASE_WORDS + 3:
        return None
    start = (len(words) - MID_PHRASE_WORDS) // 2
    phrase = " ".join(words[start : start + MID_PHRASE_WORDS]).strip(" .,;:!?()[]-\"'")
    return phrase if len(phrase.split()) >= MIN_QUERY_WORDS else None


def title_query(filename: str) -> str | None:
    """Filename minus extension and hash prefixes, when enough real words remain."""
    tokens = re.split(r"[\s_\-]+", Path(filename).stem)
    words = [token for token in tokens if len(token) >= 2 and not _noise_token(token)]
    query = " ".join(words)
    return query if len(query.split()) >= 2 else None


def _noise_token(token: str) -> bool:
    return token.isdigit() or bool(re.fullmatch(r"[0-9a-fA-F]{6,}", token))


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def collect_chunks(db_path: Path) -> tuple[list[dict], dict[str, int]]:
    """Retrievable text rows plus corpus counts.

    Only chunks with a stored vector count: an unvectorised chunk can never be
    returned by the Retriever, so labelling it relevant would poison recall.
    """
    connection = connect(db_path)
    try:
        try:
            rows = connection.execute(
                """
                SELECT c.id AS chunk_id, c.document_id AS document_id, c.text_content AS text,
                       d.original_filename AS filename
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                JOIN chunk_vectors v ON v.chunk_id = c.id
                WHERE d.status = 'indexed' AND c.text_content IS NOT NULL
                ORDER BY c.id
                """
            ).fetchall()
            counts = {
                "corpus_documents": _count(connection, "documents"),
                "corpus_chunks": _count(connection, "chunks"),
                "corpus_chunks_with_vectors": _count(connection, "chunk_vectors"),
            }
        except sqlite3.OperationalError as error:
            raise GoldBuildError(f"database schema is incomplete ({error}); ingest documents first") from error
        return [dict(row) for row in rows], counts
    finally:
        connection.close()


def _count(connection: sqlite3.Connection, table: str) -> int:
    return connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def build_cases(
    rows: Sequence[dict],
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    limit: int | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict], dict[str, int]]:
    """Derive gold cases from raw chunk rows. Pure and deterministic per input."""
    considered = [row for row in rows if len((row.get("text") or "").strip()) >= min_chars]
    groups: dict[str, list[dict]] = {}
    for row in considered:
        groups.setdefault(_normalize(row["text"]), []).append(row)

    by_query: dict[str, dict] = {}
    for text_key in sorted(groups):
        members = groups[text_key]
        chunk_ids = sorted({member["chunk_id"] for member in members})
        document_ids = sorted({member["document_id"] for member in members})
        for strategy, query in (
            ("first_sentence", first_sentence_query(members[0]["text"])),
            ("mid_phrase", mid_phrase_query(members[0]["text"])),
        ):
            if query:
                _add_case(by_query, query, strategy, chunk_ids, document_ids)

    by_document: dict[str, list[dict]] = {}
    for row in considered:
        by_document.setdefault(row["document_id"], []).append(row)
    for document_id in sorted(by_document):
        members = by_document[document_id]
        query = title_query(members[0].get("filename") or "")
        if query:
            chunk_ids = sorted({member["chunk_id"] for member in members})
            _add_case(by_query, query, "title_filename", chunk_ids, [document_id])

    def order(case: dict) -> tuple[str, str]:
        return case["strategy"], case["query"].casefold()

    cases = sorted(by_query.values(), key=order)
    if limit and len(cases) > limit:
        cases = sorted(random.Random(seed).sample(cases, limit), key=order)
    for index, case in enumerate(cases):
        case["case_id"] = f"g{index:03d}"
    stats = {
        "chunks_considered": len(considered),
        "unique_texts": len(groups),
        "total_cases": len(cases),
    }
    return cases, stats


def _add_case(
    by_query: dict[str, dict],
    query: str,
    strategy: str,
    chunk_ids: list[str],
    document_ids: list[str],
) -> None:
    existing = by_query.get(query.casefold())
    if existing:
        existing["relevant_chunk_ids"] = sorted(set(existing["relevant_chunk_ids"]) | set(chunk_ids))
        existing["relevant_document_ids"] = sorted(
            set(existing["relevant_document_ids"]) | set(document_ids)
        )
        existing["strategy"] = "+".join(sorted(set(existing["strategy"].split("+")) | {strategy}))
        return
    by_query[query.casefold()] = {
        "query": query,
        "relevant_chunk_ids": list(chunk_ids),
        "relevant_document_ids": list(document_ids),
        "strategy": strategy,
    }


def validate_gold_set(payload: dict) -> list[str]:
    """Return schema problems for a gold payload; empty list means usable."""
    if not isinstance(payload, dict):
        return ["gold set must be a JSON object"]
    problems: list[str] = []
    if not isinstance(payload.get("meta"), dict):
        problems.append("missing 'meta' object")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        return [*problems, "'cases' must be a list"]
    if not cases:
        problems.append("'cases' is empty")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            problems.append(f"case {index}: must be an object")
            continue
        query = case.get("query")
        if not isinstance(query, str) or not query.strip():
            problems.append(f"case {index}: missing or empty 'query'")
        else:
            key = query.casefold()
            if key in seen:
                problems.append(f"case {index}: duplicate query {query!r}")
            seen.add(key)
        for field in ("relevant_chunk_ids", "relevant_document_ids"):
            ids = case.get(field)
            if not isinstance(ids, list) or not ids or not all(isinstance(item, str) and item for item in ids):
                problems.append(f"case {index}: '{field}' must be a non-empty list of ids")
    return problems


def write_gold(path: Path, cases: list[dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "cases": cases}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_gold(path: Path) -> tuple[list[dict], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    problems = validate_gold_set(payload)
    if problems:
        raise ValueError(f"invalid gold set {path}: " + "; ".join(problems))
    return payload["cases"], payload["meta"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive a gold retrieval set from the knowledge base.")
    parser.add_argument("--db", default=None, help="SQLite database (default: the app's data/ragpoc.db)")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"output JSON (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help=f"minimum chunk text length (default: {DEFAULT_MIN_CHARS})",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap the number of cases")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for --limit sampling")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    db_path = Path(args.db) if args.db else Settings().database_path
    out_path = Path(args.out)
    if not db_path.exists():
        print(f"no database at {db_path}; ingest documents first — nothing to build")
        return 0
    try:
        rows, corpus = collect_chunks(db_path)
        cases, stats = build_cases(rows, min_chars=args.min_chars, limit=args.limit, seed=args.seed)
    except (GoldBuildError, sqlite3.Error) as error:
        print(f"cannot build a gold set from {db_path}: {error}")
        return 0
    if not cases:
        print(f"no usable chunks (min_chars={args.min_chars}) in {db_path}; nothing written")
        return 0
    meta = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "database": str(db_path),
        "total_cases": len(cases),
        "chunks_considered": stats["chunks_considered"],
        "unique_texts": stats["unique_texts"],
        **corpus,
        "min_chars": args.min_chars,
        "seed": args.seed,
        "limit": args.limit,
        "strategies": list(STRATEGIES),
        "strategy": STRATEGY_DESCRIPTION,
    }
    write_gold(out_path, cases, meta)
    _print_summary(cases, meta, out_path)
    return 0


def _print_summary(cases: list[dict], meta: dict, out_path: Path) -> None:
    print(
        f"gold set: {meta['total_cases']} cases from {meta['chunks_considered']} chunks "
        f"({meta['unique_texts']} unique texts) — corpus {meta['corpus_chunks']} chunks / "
        f"{meta['corpus_documents']} documents"
    )
    counts = Counter(case["strategy"] for case in cases)
    print("  strategies: " + " ".join(f"{name}={counts[name]}" for name in sorted(counts)))
    print(f"  written to {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
