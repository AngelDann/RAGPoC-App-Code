import asyncio
from pathlib import Path

from ragpoc.config import Settings
from ragpoc.db import configured_dimension, initialize_database, persist_dimension, reindex_database
from ragpoc.embeddings import FakeEmbeddingProvider
from ragpoc.ingestion import Ingestor


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        allowed_upload_dir=tmp_path / "data" / "uploads",
    )


def test_reindex_database_clears_vectors_and_documents(tmp_path):
    settings = make_settings(tmp_path)
    connection = initialize_database(settings.database_path)
    provider = FakeEmbeddingProvider()
    source = tmp_path / "notes.txt"
    source.write_text("Some content for testing reindex.")
    report = asyncio.run(Ingestor(connection, settings, provider).ingest(source))
    assert report["status"] == "indexed"
    assert configured_dimension(connection) == 8

    source_paths = reindex_database(connection)
    assert source_paths == [str(source.resolve())]

    docs = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    vecs = connection.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
    assert docs == 0
    assert chunks == 0
    assert vecs == 0
    assert configured_dimension(connection) is None


def test_reindex_database_updates_dimension(tmp_path):
    settings = make_settings(tmp_path)
    connection = initialize_database(settings.database_path)
    provider = FakeEmbeddingProvider()
    source = tmp_path / "notes.txt"
    source.write_text("Test.")
    asyncio.run(Ingestor(connection, settings, provider).ingest(source))
    assert configured_dimension(connection) == 8

    reindex_database(connection, new_dimension=768)
    assert configured_dimension(connection) == 768


def test_reindex_database_returns_all_source_paths(tmp_path):
    settings = make_settings(tmp_path)
    connection = initialize_database(settings.database_path)
    provider = FakeEmbeddingProvider()
    sources = []
    for i in range(3):
        p = tmp_path / f"doc{i}.txt"
        p.write_text(f"Document number {i}.")
        asyncio.run(Ingestor(connection, settings, provider).ingest(p))
        sources.append(str(p.resolve()))

    result = reindex_database(connection)
    assert sorted(result) == sorted(sources)


def test_reindex_database_preserves_workspaces_and_notebooks(tmp_path):
    settings = make_settings(tmp_path)
    connection = initialize_database(settings.database_path)
    connection.execute(
        "INSERT INTO workspaces (id, name, created_at, updated_at) VALUES ('w1', 'WS', '2025', '2025')"
    )
    connection.execute(
        "INSERT INTO notebooks (id, workspace_id, name, description, position, created_at, updated_at)"
        " VALUES ('n1', 'w1', 'NB', '', 0, '2025', '2025')"
    )
    connection.commit()

    reindex_database(connection)

    ws = connection.execute("SELECT COUNT(*) FROM workspaces").fetchone()[0]
    nb = connection.execute("SELECT COUNT(*) FROM notebooks").fetchone()[0]
    assert ws == 1
    assert nb == 1


def test_persist_dimension_after_reindex_allows_new_dimension(tmp_path):
    settings = make_settings(tmp_path)
    connection = initialize_database(settings.database_path)
    provider = FakeEmbeddingProvider()
    source = tmp_path / "notes.txt"
    source.write_text("Content.")
    asyncio.run(Ingestor(connection, settings, provider).ingest(source))

    reindex_database(connection)
    assert configured_dimension(connection) is None

    persist_dimension(connection, 768)
    assert configured_dimension(connection) == 768
