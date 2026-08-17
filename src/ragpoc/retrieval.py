from __future__ import annotations

import json
import math
import sqlite3
import struct

from ragpoc.embeddings import EmbeddingProvider


def _decode_vector(value: bytes | str) -> list[float]:
    if isinstance(value, str):
        return json.loads(value)
    if len(value) % 4:
        raise ValueError("Invalid vector blob length.")
    return list(struct.unpack(f"<{len(value) // 4}f", value))


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    return sum(x * y for x, y in zip(left, right, strict=True)) / denominator if denominator else 0.0


class Retriever:
    def __init__(self, connection: sqlite3.Connection, provider: EmbeddingProvider, max_top_k: int = 50):
        self.connection, self.provider, self.max_top_k = connection, provider, max_top_k

    async def search(
        self,
        query: str,
        top_k: int = 8,
        media_type: str | None = None,
        notebook_id: str | None = None,
        notebook_ids: list[str] | None = None,
    ) -> list[dict]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if not 1 <= top_k <= self.max_top_k:
            raise ValueError(f"top_k must be between 1 and {self.max_top_k}.")
        query_vector = (await self.provider.embed_texts([query]))[0]
        sql = """
            SELECT c.*, d.source_path, d.original_filename, d.media_type, v.embedding
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN chunk_vectors v ON v.chunk_id = c.id
        """
        params: list[str] = []
        if notebook_id:
            sql += " AND d.id IN (SELECT document_id FROM notebook_documents WHERE notebook_id = ?)"
            params.append(notebook_id)
        elif notebook_ids:
            placeholders = ", ".join("?" for _ in notebook_ids)
            sql += f" AND d.id IN (SELECT document_id FROM notebook_documents WHERE notebook_id IN ({placeholders}))"
            params.extend(notebook_ids)
        sql += " AND d.status = 'indexed'"
        try:
            rows = self.connection.execute(sql, params).fetchall()
        except sqlite3.OperationalError as error:
            if "no such table: chunk_vectors" in str(error):
                return []
            raise
        results = []
        for row in rows:
            score = _cosine(query_vector, _decode_vector(row["embedding"]))
            results.append({
                "chunk_id": row["id"], "document_id": row["document_id"],
                "score": score, "source_path": row["source_path"],
                "filename": row["original_filename"], "media_type": row["media_type"],
                "kind": row["kind"], "text": row["text_content"],
                "page_number": row["page_number"], "start_seconds": row["start_seconds"],
                "end_seconds": row["end_seconds"], "derived_path": row["derived_path"],
                "metadata": json.loads(row["metadata_json"]),
            })
        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]

    def get_document(self, document_id: str) -> dict | None:
        row = self.connection.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return dict(row) if row else None

    def get_chunk(self, chunk_id: str) -> dict | None:
        row = self.connection.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        return dict(row) if row else None

    def list_documents(self, limit: int = 100, offset: int = 0) -> list[dict]:
        limit = min(max(limit, 1), 100)
        return [dict(row) for row in self.connection.execute("SELECT * FROM documents ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, max(offset, 0)))]
