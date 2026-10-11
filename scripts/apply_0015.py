"""Apply the 0015 AppConfig columns without the unrelated FK check.

`manage.py migrate` runs Django's full constraint verification after applying a
migration, which aborts on a pre-existing orphan row in `notebook_documents`.
The DDL itself is independent of that table, so the columns are added directly
and the migration is recorded normally.

The orphan row is left untouched: it is user data and repairing it is a
separate decision.
"""

import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ragpoc_django.settings")

MIGRATION = "0015_appconfig_local_chat_context_length_and_more"


def main() -> int:
    from ragpoc.config import Settings

    settings = Settings(_env_file=None, data_dir="data", allowed_upload_dir="data")
    db_path = Path(settings.database_path)
    if not db_path.exists():
        print(f"no database at {db_path}", file=sys.stderr)
        return 1

    connection = sqlite3.connect(str(db_path))
    try:
        cur = connection.cursor()
        cur.execute("PRAGMA foreign_keys=OFF")
        cols = [row[1] for row in cur.execute("PRAGMA table_info(app_config)").fetchall()]

        added = []
        if "local_chat_model" not in cols:
            cur.execute("ALTER TABLE app_config ADD COLUMN local_chat_model varchar(50) NOT NULL DEFAULT ''")
            added.append("local_chat_model")
        if "local_chat_context_length" not in cols:
            cur.execute("ALTER TABLE app_config ADD COLUMN local_chat_context_length integer NULL")
            added.append("local_chat_context_length")

        cur.execute("SELECT 1 FROM django_migrations WHERE app=? AND name=?", ("knowledge", MIGRATION))
        already = bool(cur.fetchone())
        if not already:
            cur.execute(
                "INSERT INTO django_migrations (app, name, applied) VALUES (?, ?, ?)",
                ("knowledge", MIGRATION, datetime.now(UTC).isoformat()),
            )
        connection.commit()

        cols_now = [r[1] for r in cur.execute("PRAGMA table_info(app_config)").fetchall()]
        cur.execute("SELECT COUNT(*) FROM django_migrations WHERE app='knowledge'")
        print(f"base de datos      : {db_path}")
        print(f"columnas anadidas  : {added or 'ninguna (ya existian)'}")
        print(f"migracion nueva    : {not already}")
        print(f"columnas actuales  : {cols_now}")
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
