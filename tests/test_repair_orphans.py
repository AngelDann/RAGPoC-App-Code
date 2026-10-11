"""Tests for scripts/repair_orphan_documents.py.

Why the script is driven through ``main``: the contract under test is the CLI's
report and exit-code behaviour, including that expected failures never surface as
tracebacks and that non-orphan rows are never touched.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repair_orphan_documents.py"

# Mirrors the real column names from src/ragpoc/db.py and the Django-era variant
# found in data/ragpoc.db (notebook_documents with a surrogate ``id`` key).
SCHEMA = """
CREATE TABLE documents (
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
CREATE TABLE notebooks (id TEXT PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE notebook_documents (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    attached_at TEXT NOT NULL,
    document_id TEXT NOT NULL REFERENCES documents(id),
    notebook_id TEXT NOT NULL REFERENCES notebooks(id)
);
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    ordinal INTEGER NOT NULL
);
CREATE TABLE chunks_fts (chunk_id TEXT, text_content TEXT);
CREATE TABLE chunk_vectors (chunk_id TEXT);
"""

# The other flavour in the wild: the composite-key layout from src/ragpoc/db.py.
COMPOSITE_KEY_SCHEMA = """
CREATE TABLE documents (id TEXT PRIMARY KEY);
CREATE TABLE notebooks (id TEXT PRIMARY KEY);
CREATE TABLE notebook_documents (
    notebook_id TEXT NOT NULL REFERENCES notebooks(id),
    document_id TEXT NOT NULL REFERENCES documents(id),
    attached_at TEXT NOT NULL,
    PRIMARY KEY (notebook_id, document_id)
);
"""


def load_repair_module():
    spec = importlib.util.spec_from_file_location("repair_orphan_documents", SCRIPT)
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


repair = load_repair_module()


def make_database(tmp_path: Path) -> Path:
    """Seed good rows and orphans; returns the database path."""
    db_path = tmp_path / "ragpoc.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(SCHEMA)
    connection.execute("INSERT INTO notebooks (id, name) VALUES ('nb-1', 'Demo')")
    for doc_id in ("doc-good-1", "doc-good-2"):
        connection.execute(
            "INSERT INTO documents (id, source_path, original_filename, media_type,"
            " content_hash, byte_size, status, created_at)"
            " VALUES (?, ?, ?, 'application/pdf', 'h', 1, 'ready', '2026-01-01')",
            (doc_id, f"/docs/{doc_id}.pdf", f"{doc_id}.pdf"),
        )
    for document_id in ("doc-good-1", "doc-gone-1", "doc-good-2"):
        connection.execute(
            "INSERT INTO notebook_documents (attached_at, document_id, notebook_id)"
            " VALUES ('2026-01-01', ?, 'nb-1')",
            (document_id,),
        )
    for chunk_id, document_id in (("chunk-good", "doc-good-1"), ("chunk-orphan", "doc-gone-2")):
        connection.execute(
            "INSERT INTO chunks (id, document_id, kind, ordinal) VALUES (?, ?, 'text_chunk', 0)",
            (chunk_id, document_id),
        )
        connection.execute(
            "INSERT INTO chunks_fts (chunk_id, text_content) VALUES (?, 'text')", (chunk_id,)
        )
        connection.execute(
            "INSERT INTO chunk_vectors (chunk_id) VALUES (?)", (chunk_id,)
        )
    connection.commit()
    connection.close()
    return db_path


def table_rows(db_path: Path, sql: str) -> list[tuple]:
    connection = sqlite3.connect(db_path)
    rows = connection.execute(sql).fetchall()
    connection.close()
    return rows


def run_cli(argv: list[str]) -> int:
    return repair.main(argv)


# --- dry run ---


def test_dry_run_reports_orphans_and_deletes_nothing(tmp_path, capsys):
    db_path = make_database(tmp_path)
    code = run_cli(["--db", str(db_path)])
    out = capsys.readouterr().out

    assert code == 1
    assert "orphans found: 2" in out
    assert "table=notebook_documents" in out
    assert "pk=id=2" in out
    assert "document_id=doc-gone-1" in out
    assert "notebook_id=nb-1" in out
    assert "table=chunks" in out
    assert "pk=id=chunk-orphan" in out
    assert "0 deleted" in out
    assert table_rows(db_path, "SELECT id FROM notebook_documents ORDER BY id") == [
        (1,),
        (2,),
        (3,),
    ]
    assert table_rows(db_path, "SELECT id FROM chunks ORDER BY id") == [
        ("chunk-good",),
        ("chunk-orphan",),
    ]


def test_dry_run_json_report_shape(tmp_path, capsys):
    db_path = make_database(tmp_path)
    code = run_cli(["--db", str(db_path), "--json"])
    report = json.loads(capsys.readouterr().out)

    assert code == 1
    assert report["mode"] == "dry-run"
    assert report["status"] == "orphans_found"
    assert report["backup"] is None
    assert report["deleted"] == []
    tables = {entry["table"] for entry in report["orphans"]}
    assert tables == {"notebook_documents", "chunks"}
    notebook_entry = next(e for e in report["orphans"] if e["table"] == "notebook_documents")
    assert notebook_entry["pk"] == {"id": 2}
    assert notebook_entry["document_id"] == "doc-gone-1"
    assert notebook_entry["context"] == {"notebook_id": "nb-1"}
    note_text = " ".join(report["notes"])
    assert "page_documents" in note_text
    assert "workspace_documents" in note_text


def test_clean_database_exits_zero(tmp_path, capsys):
    db_path = make_database(tmp_path)
    connection = sqlite3.connect(db_path)
    connection.execute("DELETE FROM notebook_documents WHERE document_id = 'doc-gone-1'")
    connection.execute("DELETE FROM chunks_fts WHERE chunk_id = 'chunk-orphan'")
    connection.execute("DELETE FROM chunk_vectors WHERE chunk_id = 'chunk-orphan'")
    connection.execute("DELETE FROM chunks WHERE document_id = 'doc-gone-2'")
    connection.commit()
    connection.close()

    code = run_cli(["--db", str(db_path)])
    out = capsys.readouterr().out

    assert code == 0
    assert "orphans found: 0" in out
    assert "PRAGMA foreign_key_check: clean" in out


# --- apply ---


def test_apply_deletes_only_orphans_and_cleans_side_tables(tmp_path, capsys):
    db_path = make_database(tmp_path)
    backup_path = tmp_path / "backup.json"
    code = run_cli(["--db", str(db_path), "--apply", "--backup", str(backup_path)])
    out = capsys.readouterr().out

    assert code == 0
    assert "deleted: table=notebook_documents" in out
    assert "deleted: table=chunks" in out
    assert "refused:" not in out
    # Non-orphan rows survive untouched.
    assert table_rows(db_path, "SELECT document_id FROM notebook_documents ORDER BY id") == [
        ("doc-good-1",),
        ("doc-good-2",),
    ]
    assert table_rows(db_path, "SELECT id FROM chunks") == [("chunk-good",)]
    assert table_rows(db_path, "SELECT chunk_id FROM chunks_fts ORDER BY chunk_id") == [
        ("chunk-good",),
    ]
    assert table_rows(db_path, "SELECT chunk_id FROM chunk_vectors ORDER BY chunk_id") == [
        ("chunk-good",),
    ]


def test_backup_file_contains_the_deleted_rows(tmp_path, capsys):
    db_path = make_database(tmp_path)
    backup_path = tmp_path / "backup.json"
    code = run_cli(["--db", str(db_path), "--apply", "--backup", str(backup_path)])
    capsys.readouterr()

    assert code == 0
    payload = json.loads(backup_path.read_text(encoding="utf-8"))
    assert payload["database"] == str(db_path)
    backed_up = {(e["table"], e["document_id"]) for e in payload["orphans"]}
    assert backed_up == {
        ("notebook_documents", "doc-gone-1"),
        ("chunks", "doc-gone-2"),
    }
    ids = [e["pk"] for e in payload["orphans"] if e["table"] == "notebook_documents"]
    assert ids == [{"id": 2}]


def test_foreign_key_check_passes_after_repair(tmp_path, capsys):
    db_path = make_database(tmp_path)
    code = run_cli(["--db", str(db_path), "--apply", "--no-backup"])
    out = capsys.readouterr().out

    assert code == 0
    assert "PRAGMA foreign_key_check: clean" in out
    connection = sqlite3.connect(db_path)
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    connection.close()


def test_default_backup_is_written_next_to_the_database(tmp_path, capsys):
    db_path = make_database(tmp_path)
    code = run_cli(["--db", str(db_path), "--apply"])
    capsys.readouterr()

    assert code == 0
    backups = list(tmp_path.glob("orphan_backup_*.json"))
    assert len(backups) == 1
    payload = json.loads(backups[0].read_text(encoding="utf-8"))
    assert {e["table"] for e in payload["orphans"]} == {"notebook_documents", "chunks"}


def test_no_backup_writes_no_backup_file(tmp_path, capsys):
    db_path = make_database(tmp_path)
    code = run_cli(["--db", str(db_path), "--apply", "--no-backup"])
    capsys.readouterr()

    assert code == 0
    assert list(tmp_path.glob("orphan_backup_*.json")) == []
    assert list(tmp_path.glob("backup*")) == []


def test_refused_rows_are_reported_and_exit_nonzero(tmp_path, capsys, monkeypatch):
    db_path = make_database(tmp_path)
    stale = repair.Orphan(
        table="notebook_documents",
        pk={"id": 424242},
        document_id="doc-ghost",
        context={"notebook_id": "nb-1"},
        row={"id": 424242, "attached_at": "2026-01-01", "document_id": "doc-ghost",
             "notebook_id": "nb-1"},
    )
    monkeypatch.setattr(repair, "find_orphans", lambda connection: ([stale], []))

    code = run_cli(["--db", str(db_path), "--apply", "--no-backup"])
    out = capsys.readouterr().out

    assert code == 1
    assert "refused: table=notebook_documents" in out
    assert "reason=row disappeared before deletion" in out
    assert "0 deleted" in out


# --- safety guards ---


def test_row_that_is_no_longer_an_orphan_is_never_deleted(tmp_path):
    db_path = make_database(tmp_path)
    connection = sqlite3.connect(db_path)
    orphans, _ = repair.find_orphans(connection)
    assert len(orphans) == 2

    # Simulate a concurrent writer restoring the document before deletion.
    connection.execute(
        "INSERT INTO documents (id, source_path, original_filename, media_type,"
        " content_hash, byte_size, status, created_at)"
        " VALUES ('doc-gone-1', '/x', 'x', 'application/pdf', 'h', 1, 'ready', '2026-01-01')"
    )
    result = repair.delete_orphans(connection, orphans)
    connection.commit()
    connection.close()

    assert [refusal.reason for refusal in result.refused] == ["document is no longer missing"]
    assert [orphan.table for orphan in result.deleted] == ["chunks"]
    # The re-attached row survives; only the still-orphaned chunk was removed.
    assert table_rows(db_path, "SELECT document_id FROM notebook_documents ORDER BY id") == [
        ("doc-good-1",),
        ("doc-gone-1",),
        ("doc-good-2",),
    ]


def test_row_that_disappeared_between_scan_and_delete_is_refused(tmp_path):
    db_path = make_database(tmp_path)
    connection = sqlite3.connect(db_path)
    orphans, _ = repair.find_orphans(connection)
    connection.execute("DELETE FROM notebook_documents WHERE document_id = 'doc-gone-1'")
    result = repair.delete_orphans(connection, orphans)
    connection.commit()
    connection.close()

    reasons = {refusal.orphan.table: refusal.reason for refusal in result.refused}
    assert reasons["notebook_documents"] == "row disappeared before deletion"
    assert [orphan.table for orphan in result.deleted] == ["chunks"]


# --- expected failures never traceback ---


def test_missing_database_is_reported_without_traceback(tmp_path, capsys):
    code = run_cli(["--db", str(tmp_path / "nope.db")])
    captured = capsys.readouterr()

    assert code == 2
    assert "database not found" in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err


def test_missing_documents_table_is_reported_without_traceback(tmp_path, capsys):
    db_path = tmp_path / "bare.db"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()

    code = run_cli(["--db", str(db_path)])
    captured = capsys.readouterr()

    assert code == 2
    assert "Traceback" not in captured.out + captured.err


def test_missing_referencing_tables_are_noted_not_fatal(tmp_path, capsys):
    db_path = make_database(tmp_path)
    connection = sqlite3.connect(db_path)
    connection.execute("DROP TABLE chunks_fts")
    connection.execute("DROP TABLE chunk_vectors")
    connection.execute("DROP TABLE chunks")
    connection.commit()
    connection.close()

    code = run_cli(["--db", str(db_path)])
    out = capsys.readouterr().out

    assert code == 1
    assert "table chunks not present in this database; skipped" in out
    assert "table=notebook_documents" in out
    assert "Traceback" not in out


def test_composite_primary_key_table_is_supported(tmp_path, capsys):
    db_path = tmp_path / "composite.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(COMPOSITE_KEY_SCHEMA)
    connection.execute("INSERT INTO notebooks (id) VALUES ('nb-1')")
    connection.execute("INSERT INTO documents (id) VALUES ('doc-good-1')")
    for document_id in ("doc-good-1", "doc-gone-1"):
        connection.execute(
            "INSERT INTO notebook_documents (notebook_id, document_id, attached_at)"
            " VALUES ('nb-1', ?, '2026-01-01')",
            (document_id,),
        )
    connection.commit()
    connection.close()

    code = run_cli(["--db", str(db_path), "--apply", "--no-backup"])
    out = capsys.readouterr().out

    assert code == 0
    assert "pk=notebook_id=nb-1&document_id=doc-gone-1" in out
    assert table_rows(db_path, "SELECT document_id FROM notebook_documents") == [("doc-good-1",)]
