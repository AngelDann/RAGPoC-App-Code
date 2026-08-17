import sqlite3
from pathlib import Path

from django.core.management.base import BaseCommand

from knowledge.models import Document, Notebook, NotebookDocument, Page, Workspace
from ragpoc.config import get_settings


class Command(BaseCommand):
    help = "Import relational data from a raw ragpoc SQLite database into Django models."

    def add_arguments(self, parser):
        parser.add_argument("--source-db", type=str, default=None, help="Path to raw sqlite database")

    def handle(self, *args, **options):
        settings = get_settings()
        db_path = Path(options["source_db"]) if options["source_db"] else settings.database_path

        if not db_path.exists():
            self.stdout.write(self.style.WARNING(f"Database at {db_path} does not exist. Nothing to import."))
            return

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Workspaces
        try:
            workspaces = conn.execute("SELECT * FROM workspaces").fetchall()
            for w in workspaces:
                Workspace.objects.update_or_create(
                    id=w["id"],
                    defaults={
                        "name": w["name"],
                        "created_at": w["created_at"],
                        "updated_at": w["updated_at"],
                    }
                )
            self.stdout.write(self.style.SUCCESS(f"Imported {len(workspaces)} workspaces."))
        except sqlite3.OperationalError:
            self.stdout.write("No workspaces table found.")

        # Notebooks
        try:
            notebooks = conn.execute("SELECT * FROM notebooks").fetchall()
            for n in notebooks:
                Notebook.objects.update_or_create(
                    id=n["id"],
                    defaults={
                        "workspace_id": n["workspace_id"],
                        "name": n["name"],
                        "description": n["description"],
                        "position": n["position"],
                        "created_at": n["created_at"],
                        "updated_at": n["updated_at"],
                    }
                )
            self.stdout.write(self.style.SUCCESS(f"Imported {len(notebooks)} notebooks."))
        except sqlite3.OperationalError:
            self.stdout.write("No notebooks table found.")

        # Pages
        try:
            pages = conn.execute("SELECT * FROM pages").fetchall()
            for p in pages:
                import json
                try:
                    c_json = json.loads(p["content_json"])
                except Exception:
                    c_json = {"type": "doc", "content": []}
                Page.objects.update_or_create(
                    id=p["id"],
                    defaults={
                        "notebook_id": p["notebook_id"],
                        "title": p["title"],
                        "content_json": c_json,
                        "plain_text": p["plain_text"],
                        "content_hash": p["content_hash"],
                        "position": p["position"],
                        "created_at": p["created_at"],
                        "updated_at": p["updated_at"],
                    }
                )
            self.stdout.write(self.style.SUCCESS(f"Imported {len(pages)} pages."))
        except sqlite3.OperationalError:
            self.stdout.write("No pages table found.")

        # Documents
        try:
            documents = conn.execute("SELECT * FROM documents").fetchall()
            for d in documents:
                Document.objects.update_or_create(
                    id=d["id"],
                    defaults={
                        "source_path": d["source_path"],
                        "original_filename": d["original_filename"],
                        "media_type": d["media_type"],
                        "content_hash": d["content_hash"],
                        "byte_size": d["byte_size"],
                        "status": d["status"],
                        "created_at": d["created_at"],
                        "indexed_at": d["indexed_at"],
                        "error_message": d["error_message"],
                    }
                )
            self.stdout.write(self.style.SUCCESS(f"Imported {len(documents)} documents."))
        except sqlite3.OperationalError:
            self.stdout.write("No documents table found.")

        # PageDocuments — the legacy source only tracks page-level links, so each one is
        # resolved to its page's notebook and imported as a NotebookDocument instead (sources are
        # notebook-scoped only in this app now).
        try:
            pdocs = conn.execute("SELECT * FROM page_documents").fetchall()
            imported = 0
            for pd in pdocs:
                page = Page.objects.filter(id=pd["page_id"]).first()
                if not page:
                    continue
                NotebookDocument.objects.update_or_create(
                    notebook_id=page.notebook_id,
                    document_id=pd["document_id"],
                    defaults={
                        "attached_at": pd["attached_at"],
                    }
                )
                imported += 1
            self.stdout.write(self.style.SUCCESS(f"Imported {imported} page_document links as notebook_document links."))
        except sqlite3.OperationalError:
            self.stdout.write("No page_documents table found.")

        conn.close()
