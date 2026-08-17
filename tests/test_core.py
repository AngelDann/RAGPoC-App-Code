from pathlib import Path

from ragpoc.chunking import ApproximateTokenCounter, chunk_text
from ragpoc.config import BASE_DIR, Settings
from ragpoc.db import configured_dimension, initialize_database, persist_dimension


def test_settings_default_to_local_data_directory():
    settings = Settings(_env_file=None)
    # Anchored to BASE_DIR (the .exe's folder when frozen, project root in dev)
    # rather than the launch cwd, so data/.env travel with the app.
    assert settings.database_path == BASE_DIR / "data" / "ragpoc.db"
    assert settings.allowed_upload_dir == BASE_DIR / "data" / "uploads"


def test_database_persists_embedding_dimension(tmp_path):
    connection = initialize_database(tmp_path / "ragpoc.db")
    persist_dimension(connection, 3)
    assert configured_dimension(connection) == 3
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
    assert {"documents", "chunks", "chunks_fts", "chunk_vectors"} <= tables


def test_text_is_chunked_near_target_with_overlap():
    counter = ApproximateTokenCounter()
    text = "\n\n".join(f"Sentence {i}. " * 12 for i in range(12))
    chunks = chunk_text(text, counter, target_tokens=80, overlap_tokens=15)
    assert len(chunks) > 1
    assert all(chunk.token_count <= 110 for chunk in chunks)
    assert chunks[0].content.split("\n\n")[-1] in chunks[1].content


def test_empty_text_has_no_chunks():
    assert chunk_text(" \n ", ApproximateTokenCounter()) == []
