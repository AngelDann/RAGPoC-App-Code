from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_path TEXT UNIQUE NOT NULL,
    original_filename TEXT NOT NULL,
    media_type TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    indexed_at TEXT,
    error_message TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    text_content TEXT,
    token_count INTEGER,
    page_number INTEGER,
    start_seconds REAL,
    end_seconds REAL,
    derived_path TEXT,
    width INTEGER,
    height INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(document_id, ordinal, kind)
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, text_content);
CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notebooks (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    position INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pages (
    id TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    content_json TEXT NOT NULL DEFAULT '{"type":"doc","content":[]}',
    plain_text TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL DEFAULT '',
    position INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notebook_documents (
    notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    attached_at TEXT NOT NULL,
    PRIMARY KEY (notebook_id, document_id)
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    return connection


def initialize_database(path: Path, dimension: int | None = None) -> sqlite3.Connection:
    connection = connect(path)
    connection.executescript(SCHEMA)
    existing = connection.execute(
        "SELECT value FROM schema_metadata WHERE key = 'embedding_dimension'"
    ).fetchone()
    if existing and dimension and int(existing["value"]) != dimension:
        raise ValueError(
            f"Database vector dimension is {existing['value']}; configured dimension is {dimension}. "
            "Use a new database or reindex."
        )
    if dimension and not existing:
        connection.execute(
            "INSERT INTO schema_metadata(key, value) VALUES ('embedding_dimension', ?)",
            (str(dimension),),
        )
        connection.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(chunk_id TEXT PRIMARY KEY, embedding float[{dimension}])"
        )
    connection.commit()
    return connection


def persist_dimension(connection: sqlite3.Connection, dimension: int) -> None:
    current = connection.execute(
        "SELECT value FROM schema_metadata WHERE key = 'embedding_dimension'"
    ).fetchone()
    if current and int(current["value"]) != dimension:
        raise ValueError("Embedding dimension changed; reindex into a new database.")
    if not current:
        connection.execute(
            "INSERT INTO schema_metadata(key, value) VALUES ('embedding_dimension', ?)",
            (str(dimension),),
        )
        connection.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(chunk_id TEXT PRIMARY KEY, embedding float[{dimension}])"
        )
        connection.commit()


def configured_dimension(connection: sqlite3.Connection) -> int | None:
    row = connection.execute("SELECT value FROM schema_metadata WHERE key = 'embedding_dimension'").fetchone()
    return int(row["value"]) if row else None
