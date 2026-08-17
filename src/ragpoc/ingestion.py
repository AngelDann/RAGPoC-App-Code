from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ragpoc.chunking import ApproximateTokenCounter, chunk_text
from ragpoc.config import Settings
from ragpoc.db import persist_dimension
from ragpoc.embeddings import EmbeddingProvider
from ragpoc.extractors.image import inspect_image
from ragpoc.extractors.pdf_pages import render_pdf_pages
from ragpoc.extractors.text import TEXT_SUFFIXES, extract_text
from ragpoc.extractors.video import extract_video_segments, timecode, video_duration

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Ingestor:
    def __init__(self, connection: sqlite3.Connection, settings: Settings, provider: EmbeddingProvider):
        self.connection, self.settings, self.provider = connection, settings, provider
        settings.ensure_directories()

    async def ingest(self, source: Path) -> dict:
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        suffix = source.suffix.lower()
        size, digest = source.stat().st_size, _hash(source)
        if suffix == ".pdf" and size > self.settings.max_pdf_bytes:
            raise ValueError(f"PDF is {size} bytes; maximum is {self.settings.max_pdf_bytes}.")
        existing = self.connection.execute("SELECT * FROM documents WHERE source_path = ?", (str(source),)).fetchone()
        if existing and existing["content_hash"] == digest and existing["status"] == "indexed":
            return {"document_id": existing["id"], "status": "skipped", "reason": "unchanged"}
        media = (
            "text" if suffix in TEXT_SUFFIXES
            else "pdf" if suffix == ".pdf"
            else "image" if suffix in IMAGE_SUFFIXES
            else "video" if suffix in {".mp4", ".mov", ".mkv", ".webm"}
            else "audio" if suffix in AUDIO_SUFFIXES
            else None
        )
        if not media:
            raise ValueError(f"Unsupported file type: {suffix}")
        if media == "video" and video_duration(source) > self.settings.max_video_seconds:
            raise ValueError(f"Video exceeds {self.settings.max_video_seconds} seconds.")
        doc_id = existing["id"] if existing else str(uuid.uuid4())
        self.connection.execute(
            "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)",
            (doc_id,),
        )
        self.connection.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        self.connection.execute("INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, NULL, NULL)", (doc_id, str(source), source.name, media, digest, size, _now()))
        self.connection.commit()
        try:
            if media == "text": count = await self._text(doc_id, source)
            elif media == "image": count = await self._image(doc_id, source, 0, "image", None)
            elif media == "pdf": count = await self._pdf(doc_id, source)
            elif media == "audio": count = await self._audio(doc_id, source)
            else: count = await self._video(doc_id, source)
            self.connection.execute("UPDATE documents SET status='indexed', indexed_at=? WHERE id=?", (_now(), doc_id))
            self.connection.commit()
            return {"document_id": doc_id, "status": "indexed", "chunk_count": count}
        except Exception as error:
            self.connection.execute("UPDATE documents SET status='failed', error_message=? WHERE id=?", (str(error)[:1000], doc_id))
            self.connection.commit()
            raise

    async def _store(self, chunk_id: str, document_id: str, kind: str, ordinal: int, vector: list[float], **data: object) -> None:
        persist_dimension(self.connection, len(vector))
        self.connection.execute("INSERT INTO chunks(id, document_id, kind, ordinal, text_content, token_count, page_number, start_seconds, end_seconds, derived_path, width, height, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (chunk_id, document_id, kind, ordinal, data.get("text"), data.get("tokens"), data.get("page"), data.get("start"), data.get("end"), data.get("derived"), data.get("width"), data.get("height"), json.dumps(data.get("metadata", {}))))
        self.connection.execute("INSERT INTO chunk_vectors(chunk_id, embedding) VALUES (?, ?)", (chunk_id, json.dumps(vector)))
        if data.get("text"):
            self.connection.execute("INSERT INTO chunks_fts(chunk_id, text_content) VALUES (?, ?)", (chunk_id, data["text"]))
        self.connection.commit()

    async def _text(self, doc_id: str, source: Path) -> int:
        chunks = chunk_text(extract_text(source), ApproximateTokenCounter())
        for offset in range(0, len(chunks), self.settings.embedding_batch_size):
            group = chunks[offset:offset + self.settings.embedding_batch_size]
            vectors = await self.provider.embed_texts([x.content for x in group])
            for chunk, vector in zip(group, vectors, strict=True):
                await self._store(f"{doc_id}:text:{chunk.ordinal:05d}", doc_id, "text_chunk", chunk.ordinal, vector, text=chunk.content, tokens=chunk.token_count, metadata={"tokenizer": "regex-approx-v1"})
        return len(chunks)

    async def _image(self, doc_id: str, source: Path, ordinal: int, kind: str, page: int | None, **meta: object) -> int:
        width, height = inspect_image(source)
        vector = await self.provider.embed_image(source)
        extra_metadata = meta.pop("metadata", {})
        if not isinstance(extra_metadata, dict):
            raise ValueError("metadata must be a dictionary.")
        start = meta.pop("start", None)
        end = meta.pop("end", None)
        merged_metadata = {**meta, **extra_metadata}
        await self._store(
            f"{doc_id}:{kind}:{ordinal:05d}", doc_id, kind, ordinal, vector,
            page=page,
            start=start,
            end=end,
            derived=str(source) if kind != "image" else None,
            width=width,
            height=height,
            metadata=merged_metadata,
        )
        return 1

    async def _pdf(self, doc_id: str, source: Path) -> int:
        out = self.settings.renders_dir / doc_id
        count = 0
        for page, image, _, _ in render_pdf_pages(source, out, self.settings.pdf_render_dpi):
            count += await self._image(doc_id, image, page - 1, "pdf_page_image", page, original_pdf=str(source))
        return count

    async def _video(self, doc_id: str, source: Path) -> int:
        out = self.settings.frames_dir / doc_id
        segments = await asyncio.to_thread(extract_video_segments, source, out, self.settings.video_segment_seconds)
        async def one(segment):
            vector = await self.provider.embed_video(segment.clip_path)
            await self._store(
                f"{doc_id}:video_segment:{segment.ordinal:05d}",
                doc_id,
                "video_segment",
                segment.ordinal,
                vector,
                start=segment.start_seconds,
                end=segment.end_seconds,
                derived=str(segment.clip_path),
                metadata={
                    "start_timecode": timecode(segment.start_seconds),
                    "end_timecode": timecode(segment.end_seconds),
                    "source_video": str(source),
                    "frame_path": str(segment.frame_path),
                    "clip_path": str(segment.clip_path),
                    "embedding_modality": "video",
                },
            )
            return 1
        semaphore = asyncio.Semaphore(self.settings.embedding_concurrency)
        async def bounded(item):
            async with semaphore: return await one(item)
        return sum(await asyncio.gather(*(bounded(segment) for segment in segments)))

    async def _audio(self, doc_id: str, source: Path) -> int:
        vector = await self.provider.embed_audio(source)
        await self._store(
            f"{doc_id}:audio:00000",
            doc_id,
            "audio_chunk",
            0,
            vector,
            derived=str(source),
            metadata={"source_audio": str(source), "embedding_modality": "audio"},
        )
        return 1
