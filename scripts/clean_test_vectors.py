"""Remove vector rows that cannot represent real content.

Two kinds of damage were found in the production store while building the
retrieval benchmark:

* **orphan vectors** -- a ``chunk_vectors`` row whose chunk was deleted, left
  behind because the plain sqlite3 driver cannot load the ``vec0`` extension
  that backs the virtual table;
* **placeholder vectors** -- rows written by test fixtures, every dimension
  filled with the same constant (0.1). They are not embeddings of anything, so
  they rank randomly and silently depress every retrieval metric.

Dry-run by default, matching the repair tool. Deleting a vector is recoverable
by reindexing the document, but it is still data, so nothing moves without
``--apply``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragpoc.db import connect  # noqa: E402

EXIT_OK = 0
EXIT_ISSUES_REMAIN = 1
EXIT_CANNOT_RUN = 2

# A vector whose every dimension is equal is not an embedding: real vectors
# never have zero variance.
PLACEHOLDER_TOLERANCE = 1e-9


def _decode(value: bytes) -> list[float]:
    import struct

    if isinstance(value, (bytes, bytearray)):
        return list(struct.unpack(f"<{len(value) // 4}f", bytes(value)))
    return list(value)


def _is_placeholder(vector: list[float]) -> bool:
    if len(vector) < 2:
        return True
    first = vector[0]
    return all(abs(component - first) < PLACEHOLDER_TOLERANCE for component in vector)


def find_issues(connection) -> dict:
    chunks = {row["id"] for row in connection.execute("SELECT id FROM chunks")}
    rows = connection.execute("SELECT chunk_id, embedding FROM chunk_vectors").fetchall()

    orphans: list[str] = []
    placeholders: list[dict] = []
    for row in rows:
        chunk_id = row["chunk_id"]
        if chunk_id not in chunks:
            orphans.append(chunk_id)
            continue
        vector = _decode(row["embedding"])
        if _is_placeholder(vector):
            placeholders.append(
                {
                    "chunk_id": chunk_id,
                    "dimension": len(vector),
                    "constant": round(vector[0], 6),
                }
            )
    return {"orphan_vectors": orphans, "placeholder_vectors": placeholders, "total_vectors": len(rows)}


def apply(connection, issues: dict) -> dict:
    deleted = {"orphan_vectors": [], "placeholder_vectors": []}
    for chunk_id in issues["orphan_vectors"]:
        connection.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
        deleted["orphan_vectors"].append(chunk_id)
    for entry in issues["placeholder_vectors"]:
        connection.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (entry["chunk_id"],))
        deleted["placeholder_vectors"].append(entry["chunk_id"])
    connection.commit()
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="")
    parser.add_argument("--apply", action="store_true", help="delete the offending rows")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    from ragpoc.config import Settings

    settings = Settings(_env_file=None, data_dir="data", allowed_upload_dir="data")
    db_path = Path(args.db) if args.db else Path(settings.database_path)
    if not db_path.exists():
        print(f"error: no database at {db_path}", file=sys.stderr)
        return EXIT_CANNOT_RUN

    connection = connect(db_path)
    try:
        issues = find_issues(connection)
        result = {
            "database": str(db_path),
            "mode": "apply" if args.apply else "dry-run",
            **issues,
            "deleted": {"orphan_vectors": [], "placeholder_vectors": []},
        }
        if args.apply and (issues["orphan_vectors"] or issues["placeholder_vectors"]):
            result["deleted"] = apply(connection, issues)

        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"database: {db_path}")
            print(f"mode: {result['mode']}")
            print(f"vectors: {issues['total_vectors']}")
            print(f"orphan vectors: {len(issues['orphan_vectors'])}")
            for chunk_id in issues["orphan_vectors"]:
                print(f"  orphan: {chunk_id}")
            print(f"placeholder vectors: {len(issues['placeholder_vectors'])}")
            for entry in issues["placeholder_vectors"]:
                print(
                    f"  placeholder: {entry['chunk_id']} "
                    f"(dim={entry['dimension']}, all={entry['constant']})"
                )
            if args.apply:
                deleted = result["deleted"]
                print(
                    f"apply: {len(deleted['orphan_vectors'])} orphan + "
                    f"{len(deleted['placeholder_vectors'])} placeholder vector(s) deleted"
                )
            elif issues["orphan_vectors"] or issues["placeholder_vectors"]:
                print("dry run: nothing changed (use --apply to delete them)")
            else:
                print("vector store is clean")

        remaining = len(result["deleted"]["orphan_vectors"]) == 0 and not args.apply and (
            issues["orphan_vectors"] or issues["placeholder_vectors"]
        )
        return EXIT_ISSUES_REMAIN if remaining else EXIT_OK
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
