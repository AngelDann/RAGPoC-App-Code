import asyncio
from pathlib import Path

from ragpoc.config import Settings
from ragpoc.db import initialize_database
from ragpoc.embeddings import FakeEmbeddingProvider
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
