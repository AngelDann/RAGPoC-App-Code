import asyncio
from pathlib import Path

import httpx

from ragpoc.config import Settings
from ragpoc.db import initialize_database
from ragpoc.embeddings import FakeEmbeddingProvider, OpenRouterEmbeddingProvider
from ragpoc.ingestion import Ingestor
from ragpoc.retrieval import Retriever


def test_ingest_text_and_search(tmp_path: Path):
    data = tmp_path / "data"
    source = tmp_path / "notes.txt"
    source.write_text("Redis and Celery process asynchronous payment validation.")
    settings = Settings(_env_file=None, data_dir=data, allowed_upload_dir=data / "uploads")
    connection = initialize_database(settings.database_path)
    provider = FakeEmbeddingProvider()
    report = asyncio.run(Ingestor(connection, settings, provider).ingest(source))
    assert report["status"] == "indexed"
    assert report["chunk_count"] == 1
    results = asyncio.run(Retriever(connection, provider).search("payment validation"))
    assert results[0]["source_path"] == str(source.resolve())

def test_unchanged_text_is_skipped(tmp_path: Path):
    data = tmp_path / "data"
    source = tmp_path / "notes.txt"
    source.write_text("A small source file.")
    settings = Settings(_env_file=None, data_dir=data, allowed_upload_dir=data / "uploads")
    connection = initialize_database(settings.database_path)
    ingestor = Ingestor(connection, settings, FakeEmbeddingProvider())
    asyncio.run(ingestor.ingest(source))
    assert asyncio.run(ingestor.ingest(source))["status"] == "skipped"


class _NetworkErrorProvider:
    """Stands in for OpenRouterEmbeddingProvider hitting a real connectivity failure (no
    internet, DNS failure, timeout) -- those surface as raw httpx.TransportError, uncaught by
    _post()'s own retry loop (which only handles HTTP error status codes), unlike a missing key
    (EmbeddingError, covered by test_missing_api_key_marks_unindexed below)."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise httpx.ConnectError("no internet")

    async def embed_image(self, path: Path) -> list[float]:
        raise httpx.ConnectError("no internet")

    async def embed_video(self, path: Path) -> list[float]:
        raise httpx.ConnectError("no internet")

    async def embed_audio(self, path: Path) -> list[float]:
        raise httpx.ConnectError("no internet")


def test_connectivity_failure_marks_unindexed_without_deleting_and_retry_succeeds(tmp_path: Path):
    # A user reported losing an upload attempt entirely (file deleted, hard 400) whenever there
    # was no internet or no OpenRouter key configured -- not specific to video, every media type
    # went through the same embedding step. ingest() should keep the file and mark the document
    # 'unindexed' instead of 'failed' for connectivity problems specifically, so a later retry
    # (e.g. from the attachment panel's retry button, see document_detail_dispatch's POST branch)
    # can pick the same document back up once there's a connection.
    data = tmp_path / "data"
    source = tmp_path / "notes.txt"
    source.write_text("Redis and Celery process asynchronous payment validation.")
    settings = Settings(_env_file=None, data_dir=data, allowed_upload_dir=data / "uploads")
    connection = initialize_database(settings.database_path)

    report = asyncio.run(Ingestor(connection, settings, _NetworkErrorProvider()).ingest(source))
    assert report["status"] == "unindexed"
    assert source.exists()  # Ingestor never touches the source file itself either way

    # Retrying (a fresh Ingestor against the same connection/source_path, exactly what
    # document_detail_dispatch's reindex endpoint does) once connectivity is back must reuse the
    # same document row, not create a duplicate.
    retry_report = asyncio.run(Ingestor(connection, settings, FakeEmbeddingProvider()).ingest(source))
    assert retry_report["status"] == "indexed"
    assert retry_report["document_id"] == report["document_id"]


def test_missing_api_key_marks_unindexed(tmp_path: Path):
    data = tmp_path / "data"
    source = tmp_path / "notes.txt"
    source.write_text("Some content with no API key configured.")
    settings = Settings(_env_file=None, data_dir=data, allowed_upload_dir=data / "uploads")
    connection = initialize_database(settings.database_path)
    provider = OpenRouterEmbeddingProvider(settings)  # settings.openrouter_api_key is unset

    report = asyncio.run(Ingestor(connection, settings, provider).ingest(source))
    assert report["status"] == "unindexed"
    assert "OpenRouter API key" in report["reason"]
    assert source.exists()
