"""Audit and repair rows that reference deleted documents.

Why this exists: Django re-checks every foreign key after each migration and aborts
on the first violation, so a single dangling row blocks all schema changes. The tool
is therefore dry-run by default, re-verifies every row immediately before deleting it,
and backs up exactly what it removes.

Exit codes:
    0  the database is clean: no orphan remains unrepaired and ``PRAGMA
       foreign_key_check`` passes
    1  orphans remain: dry run reported them, or a delete was refused or failed
    2  cannot run: database file missing or unreadable, ``documents`` table absent,
       or the backup could not be written before an ``--apply``
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Make the script runnable as `python scripts/...` without setting PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

EXIT_OK = 0
EXIT_ORPHANS_REMAIN = 1
EXIT_CANNOT_RUN = 2


@dataclass(frozen=True)
class RefTable:
    """A table whose ``document_id`` references ``documents(id)``.

    ``cleanup`` lists side tables (search index, vector store) as
    ``(table, column, source_column)`` so deleting an orphan does not leave
    dangling index entries behind.
    """

    name: str
    fk_column: str = "document_id"
    context: tuple[str, ...] = ()
    cleanup: tuple[tuple[str, str, str], ...] = ()


REF_TABLES: tuple[RefTable, ...] = (
    RefTable("notebook_documents", context=("notebook_id",)),
    RefTable("page_documents", context=("page_id",)),
    RefTable("workspace_documents", context=("workspace_id",)),
    RefTable(
        "chunks",
        context=("kind", "ordinal"),
        cleanup=(("chunks_fts", "chunk_id", "id"), ("chunk_vectors", "chunk_id", "id")),
    ),
)


@dataclass(frozen=True)
class Orphan:
    """One referencing row whose ``document_id`` points at a missing document."""

    table: str
    pk: dict[str, Any]
    document_id: Any
    context: dict[str, Any]
    row: dict[str, Any]


@dataclass(frozen=True)
class Refusal:
    orphan: Orphan
    reason: str


@dataclass(frozen=True)
class Cleanup:
    table: str
    column: str
    value: Any
    rows: int


@dataclass(frozen=True)
class RepairResult:
    deleted: list[Orphan]
    refused: list[Refusal]
    cleanup: list[Cleanup]
    warnings: list[str]
    errors: list[str]


def default_database() -> Path:
    """Resolve the database exactly like the app does so the tool targets the same file."""
    from ragpoc.config import Settings  # imported lazily so --db runs never need the app

    settings = Settings(_env_file=None, data_dir="data", allowed_upload_dir="data")
    return Path(settings.database_path)


def find_orphans(connection: sqlite3.Connection) -> tuple[list[Orphan], list[str]]:
    """Return referencing rows whose ``document_id`` is dangling, plus notes for skipped tables."""
    orphans: list[Orphan] = []
    notes: list[str] = []
    for spec in REF_TABLES:
        if not _table_exists(connection, spec.name):
            notes.append(f"table {spec.name} not present in this database; skipped")
            continue
        columns = _table_columns(connection, spec.name)
        if spec.fk_column not in columns:
            notes.append(f"table {spec.name} has no {spec.fk_column} column; skipped")
            continue
        pk_columns = [
            name for name, rank in sorted(columns.items(), key=lambda item: item[1]) if rank > 0
        ]
        rowid_select = "" if pk_columns else "rowid AS _rowid_, "
        order_by = ", ".join(_quote(name) for name in pk_columns) or "r.rowid"
        sql = (
            f"SELECT {rowid_select}* FROM {_quote(spec.name)} AS r "
            f"WHERE r.{_quote(spec.fk_column)} IS NOT NULL "
            f"AND NOT EXISTS (SELECT 1 FROM documents AS d WHERE d.id = r.{_quote(spec.fk_column)}) "
            f"ORDER BY {order_by}"
        )
        cursor = connection.execute(sql)
        for row in cursor.fetchall():
            data = _row_dict(cursor, row)
            rowid = data.pop("_rowid_", None)
            pk = {name: data[name] for name in pk_columns} if pk_columns else {"rowid": rowid}
            orphans.append(
                Orphan(
                    table=spec.name,
                    pk=pk,
                    document_id=data.get(spec.fk_column),
                    context={name: data[name] for name in spec.context if name in data},
                    row=data,
                )
            )
    return orphans, notes


def delete_orphans(connection: sqlite3.Connection, orphans: list[Orphan]) -> RepairResult:
    """Delete orphans one by one inside a transaction, re-checking each row first.

    Re-verification closes the race between scan and delete: a row that was fixed
    or changed meanwhile is refused and left untouched.
    """
    specs = {spec.name: spec for spec in REF_TABLES}
    deleted: list[Orphan] = []
    refused: list[Refusal] = []
    cleanup: list[Cleanup] = []
    warnings: list[str] = []
    errors: list[str] = []
    connection.execute("PRAGMA foreign_keys = ON")
    # Callers may already hold an implicit transaction; BEGIN must only open one when needed.
    if not connection.in_transaction:
        connection.execute("BEGIN IMMEDIATE")
    try:
        for orphan in orphans:
            reason = _refusal_reason(connection, orphan)
            if reason:
                refused.append(Refusal(orphan, reason))
                continue
            clause, values = _pk_match(orphan.pk)
            connection.execute(
                f"DELETE FROM {_quote(orphan.table)} WHERE {clause}", values
            )
            deleted.append(orphan)
            cleanup.extend(_cleanup_side_tables(connection, specs[orphan.table], orphan, warnings))
        connection.commit()
    except sqlite3.Error as exc:
        connection.rollback()
        deleted, cleanup = [], []
        errors.append(f"delete failed, transaction rolled back: {exc}")
    return RepairResult(deleted, refused, cleanup, warnings, errors)


def write_backup(path: Path, database: Path, orphans: list[Orphan]) -> None:
    """Persist the rows about to be deleted so a repair stays auditable and reversible."""
    payload = {
        "database": str(database),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "orphans": [asdict(orphan) for orphan in orphans],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repair_orphan_documents.py",
        description=(
            "Report rows that reference documents which no longer exist. "
            "Dry-run by default; pass --apply to delete them."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="path to the SQLite database (default: the app's data/ragpoc.db)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete the orphan rows (default is a read-only dry run)",
    )
    backup = parser.add_mutually_exclusive_group()
    backup.add_argument("--backup", type=Path, help="write the deleted rows to this JSON file")
    backup.add_argument(
        "--no-backup",
        action="store_true",
        help="do not write a backup before deleting (default writes one next to the database)",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser


def run(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {
        "database": "",
        "mode": "apply" if args.apply else "dry-run",
        "status": "",
        "orphans": [],
        "deleted": [],
        "refused": [],
        "cleanup": [],
        "backup": None,
        "foreign_key_check": [],
        "notes": [],
        "warnings": [],
        "errors": [],
    }

    def finish(code: int, status: str) -> int:
        report["status"] = status
        report["exit_code"] = code
        _emit(report, args.json)
        return code

    if args.db is not None:
        db_path = Path(args.db)
    else:
        try:
            db_path = default_database()
        except Exception as exc:  # pragma: no cover - depends on the local environment
            report["errors"].append(f"cannot resolve the default database path: {exc}")
            return finish(EXIT_CANNOT_RUN, "error")
    report["database"] = str(db_path)

    if not db_path.is_file():
        report["errors"].append(f"database not found: {db_path}")
        return finish(EXIT_CANNOT_RUN, "error")

    try:
        connection = sqlite3.connect(db_path)
    except sqlite3.Error as exc:
        report["errors"].append(f"cannot open database {db_path}: {exc}")
        return finish(EXIT_CANNOT_RUN, "error")

    try:
        connection.row_factory = sqlite3.Row
        # The scan must tolerate the very violations we are looking for.
        connection.execute("PRAGMA foreign_keys = OFF")
        if not _table_exists(connection, "documents"):
            report["errors"].append(
                "table 'documents' is missing; this does not look like a RAGPoC database"
            )
            return finish(EXIT_CANNOT_RUN, "error")

        orphans, notes = find_orphans(connection)
        report["notes"].extend(notes)
        report["orphans"] = [asdict(orphan) for orphan in orphans]
        result = RepairResult([], [], [], [], [])

        if args.apply and orphans:
            if args.no_backup:
                report["notes"].append("backup skipped (--no-backup)")
            else:
                backup_path = args.backup or _default_backup_path(db_path)
                try:
                    write_backup(backup_path, db_path, orphans)
                except OSError as exc:
                    report["errors"].append(f"cannot write backup {backup_path}: {exc}")
                    return finish(EXIT_CANNOT_RUN, "error")
                report["backup"] = str(backup_path)
            result = delete_orphans(connection, orphans)

        report["deleted"] = [asdict(orphan) for orphan in result.deleted]
        report["refused"] = [
            {**asdict(refusal.orphan), "reason": refusal.reason} for refusal in result.refused
        ]
        report["cleanup"] = [asdict(entry) for entry in result.cleanup]
        report["warnings"].extend(result.warnings)
        report["errors"].extend(result.errors)
        report["foreign_key_check"] = [
            dict(row) for row in connection.execute("PRAGMA foreign_key_check").fetchall()
        ]
    except sqlite3.Error as exc:
        report["errors"].append(f"sqlite error: {exc}")
        return finish(EXIT_CANNOT_RUN, "error")
    finally:
        connection.close()

    unrepaired = len(report["orphans"]) - len(report["deleted"])
    if report["errors"]:
        return finish(EXIT_ORPHANS_REMAIN, "error" if args.apply else "orphans_found")
    if unrepaired or report["foreign_key_check"]:
        status = "orphans_found" if not args.apply else "incomplete"
        return finish(EXIT_ORPHANS_REMAIN, status)
    return finish(EXIT_OK, "repaired" if args.apply and orphans else "clean")


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


# --- internals ---


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _row_dict(cursor: sqlite3.Cursor, row: sqlite3.Row | tuple[Any, ...]) -> dict[str, Any]:
    return {description[0]: row[index] for index, description in enumerate(cursor.description)}


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ? AND type IN ('table', 'view')", (table,)
        ).fetchone()
        is not None
    )


def _table_columns(connection: sqlite3.Connection, table: str) -> dict[str, int]:
    """Map column name to primary-key rank (0 means the column is not part of the key)."""
    cursor = connection.execute(f"PRAGMA table_info({_quote(table)})")
    return {row[1]: row[5] for row in cursor.fetchall()}


def _pk_match(pk: dict[str, Any]) -> tuple[str, list[Any]]:
    if set(pk) == {"rowid"}:
        return "rowid = ?", [pk["rowid"]]
    clause = " AND ".join(f"{_quote(name)} = ?" for name in pk)
    return clause, list(pk.values())


def _document_exists(connection: sqlite3.Connection, document_id: Any) -> bool:
    found = connection.execute("SELECT 1 FROM documents WHERE id = ?", (document_id,)).fetchone()
    return found is not None


def _refusal_reason(connection: sqlite3.Connection, orphan: Orphan) -> str | None:
    clause, values = _pk_match(orphan.pk)
    cursor = connection.execute(f"SELECT * FROM {_quote(orphan.table)} WHERE {clause}", values)
    row = cursor.fetchone()
    if row is None:
        return "row disappeared before deletion"
    if _row_dict(cursor, row) != orphan.row:
        return "row changed since scan"
    if _document_exists(connection, orphan.document_id):
        return "document is no longer missing"
    return None


def _cleanup_side_tables(
    connection: sqlite3.Connection,
    spec: RefTable,
    orphan: Orphan,
    warnings: list[str],
) -> list[Cleanup]:
    results: list[Cleanup] = []
    for table, column, source in spec.cleanup:
        if source not in orphan.pk or not _table_exists(connection, table):
            continue
        value = orphan.pk[source]
        try:
            cursor = connection.execute(
                f"DELETE FROM {_quote(table)} WHERE {_quote(column)} = ?", (value,)
            )
        except sqlite3.Error as exc:
            warnings.append(f"could not clean {table} for {column}={value!r}: {exc}")
            continue
        results.append(Cleanup(table, column, value, cursor.rowcount))
    return results


def _default_backup_path(db_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return db_path.parent / f"orphan_backup_{stamp}.json"


def _orphan_line(prefix: str, entry: dict[str, Any]) -> str:
    pk = "&".join(f"{key}={value}" for key, value in entry["pk"].items())
    parts = [f"table={entry['table']}", f"pk={pk}", f"document_id={entry['document_id']}"]
    parts.extend(f"{key}={value}" for key, value in entry["context"].items())
    return f"{prefix}: " + " ".join(parts)


def _emit(report: dict[str, Any], json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(report, indent=2, default=str))
        return
    lines = [f"database: {report['database']}", f"mode: {report['mode']}"]
    lines.extend(f"note: {note}" for note in report["notes"])
    orphans = report["orphans"]
    lines.append(f"orphans found: {len(orphans)}")
    lines.extend(_orphan_line("orphan", entry) for entry in orphans)
    if report["mode"] == "apply":
        lines.extend(_orphan_line("deleted", entry) for entry in report["deleted"])
        for entry in report["refused"]:
            lines.append(_orphan_line("refused", entry) + f" reason={entry['reason']}")
        for entry in report["cleanup"]:
            lines.append(
                f"cleanup: table={entry['table']} column={entry['column']} "
                f"value={entry['value']} rows_deleted={entry['rows']}"
            )
        if report["backup"]:
            lines.append(f"backup: {report['backup']}")
        lines.append(
            f"apply: {len(report['deleted'])} deleted, {len(report['refused'])} refused"
        )
    else:
        lines.append(
            f"dry run: {len(orphans)} orphan row(s) reported, 0 deleted; "
            "nothing changed (use --apply to delete them)"
        )
    violations = report["foreign_key_check"]
    if violations:
        lines.append(f"PRAGMA foreign_key_check: {len(violations)} violation(s) remain")
        for violation in violations:
            lines.append(
                f"fk violation: table={violation['table']} rowid={violation['rowid']} "
                f"references {violation['parent']} (fk #{violation['fkid']})"
            )
    else:
        lines.append("PRAGMA foreign_key_check: clean")
    lines.extend(f"warning: {warning}" for warning in report["warnings"])
    lines.extend(f"error: {error}" for error in report["errors"])
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
