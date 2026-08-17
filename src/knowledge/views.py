from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from asgiref.sync import sync_to_async
from django.http import HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from pydantic_ai import BinaryContent

from knowledge.models import (
    AgentMemory,
    AgentSkill,
    ApiUsageLog,
    ChatMessage,
    ChatThread,
    Document,
    Notebook,
    NotebookArtifact,
    NotebookDocument,
    Page,
    Workspace,
    calculate_content_hash,
)
from knowledge.pydantic_agent import AgentDeps, create_pydantic_rag_agent
from knowledge.services import get_rag_service, reset_rag_service
from knowledge.settings_store import (
    get_api_key_status,
    set_app_settings,
    set_byok_api_key,
)
from knowledge.settings_store import get_effective_settings as get_settings
from knowledge.usage import fetch_openrouter_key_status, record_usage, usage_summary
from ragpoc.artifact_pdf import render_artifact_pdf, render_page_pdf


def console_view(request: HttpRequest) -> HttpResponse:
    return render(request, "console.html")


def health_view(request: HttpRequest) -> JsonResponse:
    settings = get_settings()
    return JsonResponse({
        "status": "ok",
        "database": str(settings.database_path),
        "embedding_model": settings.embedding_model,
        "chat_model": settings.chat_model,
    })


@csrf_exempt
def app_settings_view(request: HttpRequest) -> JsonResponse:
    if request.method == "GET":
        status = get_api_key_status()
        return JsonResponse(status)

    if request.method == "POST":
        try:
            body = json.loads(request.body.decode("utf-8"))
        except Exception:
            return JsonResponse({"detail": "Invalid JSON"}, status=400)

        api_key = body.get("openrouter_api_key")
        set_app_settings(api_key=api_key.strip() if isinstance(api_key, str) else None)
        reset_rag_service()
        return JsonResponse(get_api_key_status())

    return JsonResponse({"detail": "Method not allowed"}, status=405)


@csrf_exempt
def create_workspace(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)

    name = (body.get("name") or "").strip()
    if not name:
        return JsonResponse({"detail": "Workspace name cannot be empty."}, status=422)

    workspace = Workspace.objects.create(name=name)
    return JsonResponse({
        "id": workspace.id,
        "name": workspace.name,
        "created_at": workspace.created_at.isoformat(),
        "updated_at": workspace.updated_at.isoformat(),
    }, status=201)


@csrf_exempt
def create_notebook(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)

    workspace_id = body.get("workspace_id")
    name = (body.get("name") or "").strip()
    description = body.get("description", "")

    if not name or not workspace_id:
        return JsonResponse({"detail": "name and workspace_id are required."}, status=422)

    try:
        workspace = Workspace.objects.get(id=workspace_id)
    except Workspace.DoesNotExist:
        return JsonResponse({"detail": "Workspace not found."}, status=404)

    last_notebook = Notebook.objects.filter(workspace=workspace).order_by("-position").first()
    position = (last_notebook.position + 1) if last_notebook else 0

    notebook = Notebook.objects.create(
        workspace=workspace,
        name=name,
        description=description,
        position=position,
    )
    return JsonResponse({
        "id": notebook.id,
        "workspace_id": notebook.workspace_id,
        "name": notebook.name,
        "description": notebook.description,
        "position": notebook.position,
        "created_at": notebook.created_at.isoformat(),
        "updated_at": notebook.updated_at.isoformat(),
    }, status=201)


@csrf_exempt
def create_page(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)

    notebook_id = body.get("notebook_id")
    title = (body.get("title") or "").strip() or "Untitled"
    content_json = body.get("content_json") or {"type": "doc", "content": []}

    if not notebook_id:
        return JsonResponse({"detail": "notebook_id is required."}, status=422)

    try:
        notebook = Notebook.objects.get(id=notebook_id)
    except Notebook.DoesNotExist:
        return JsonResponse({"detail": "Notebook not found."}, status=404)

    last_page = Page.objects.filter(notebook=notebook).order_by("-position").first()
    position = (last_page.position + 1) if last_page else 0

    page = Page.objects.create(
        notebook=notebook,
        title=title,
        content_json=content_json,
        plain_text="",
        content_hash=calculate_content_hash(title, ""),
        position=position,
    )
    return JsonResponse({
        "id": page.id,
        "notebook_id": page.notebook_id,
        "title": page.title,
        "content_json": page.content_json,
        "plain_text": page.plain_text,
        "content_hash": page.content_hash,
        "position": page.position,
        "created_at": page.created_at.isoformat(),
        "updated_at": page.updated_at.isoformat(),
    }, status=201)


@csrf_exempt
def page_detail_dispatch(request: HttpRequest, page_id: str) -> JsonResponse:
    if request.method == "GET":
        return get_page(request, page_id)
    elif request.method == "PUT":
        return update_page(request, page_id)
    elif request.method == "DELETE":
        return delete_page(request, page_id)
    return JsonResponse({"detail": "Method not allowed"}, status=405)


@csrf_exempt
def delete_page(request: HttpRequest, page_id: str) -> JsonResponse:
    try:
        page = Page.objects.get(id=page_id)
    except Page.DoesNotExist:
        return JsonResponse({"detail": "Page not found."}, status=404)
    page.delete()
    return JsonResponse({"status": "deleted", "id": page_id})


@csrf_exempt
def notebook_detail_dispatch(request: HttpRequest, notebook_id: str) -> JsonResponse:
    try:
        notebook = Notebook.objects.get(id=notebook_id)
    except Notebook.DoesNotExist:
        return JsonResponse({"detail": "Notebook not found."}, status=404)

    if request.method == "DELETE":
        notebook.delete()
        return JsonResponse({"status": "deleted", "id": notebook_id})
    return JsonResponse({"detail": "Method not allowed"}, status=405)


@csrf_exempt
def workspace_detail_dispatch(request: HttpRequest, workspace_id: str) -> JsonResponse:
    try:
        workspace = Workspace.objects.get(id=workspace_id)
    except Workspace.DoesNotExist:
        return JsonResponse({"detail": "Workspace not found."}, status=404)

    if request.method == "DELETE":
        workspace.delete()
        return JsonResponse({"status": "deleted", "id": workspace_id})
    return JsonResponse({"detail": "Method not allowed"}, status=405)


def get_page(request: HttpRequest, page_id: str) -> JsonResponse:
    try:
        page = Page.objects.get(id=page_id)
    except Page.DoesNotExist:
        return JsonResponse({"detail": "Page not found."}, status=404)

    return JsonResponse({
        "id": page.id,
        "notebook_id": page.notebook_id,
        "title": page.title,
        "content_json": page.content_json,
        "plain_text": page.plain_text,
        "content_hash": page.content_hash,
        "position": page.position,
        "created_at": page.created_at.isoformat(),
        "updated_at": page.updated_at.isoformat(),
    })


@csrf_exempt
def update_page(request: HttpRequest, page_id: str) -> JsonResponse:
    if request.method != "PUT":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    try:
        page = Page.objects.get(id=page_id)
    except Page.DoesNotExist:
        return JsonResponse({"detail": "Page not found."}, status=404)

    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)

    title = (body.get("title") or "").strip() or "Untitled"
    content_json = body.get("content_json") or {"type": "doc", "content": []}
    plain_text = body.get("plain_text", "")

    page.title = title
    page.content_json = content_json
    page.plain_text = plain_text
    page.content_hash = calculate_content_hash(title, plain_text)
    page.save()

    return JsonResponse({
        "id": page.id,
        "notebook_id": page.notebook_id,
        "title": page.title,
        "content_json": page.content_json,
        "plain_text": page.plain_text,
        "content_hash": page.content_hash,
        "position": page.position,
        "created_at": page.created_at.isoformat(),
        "updated_at": page.updated_at.isoformat(),
    })


def workspace_tree(request: HttpRequest, workspace_id: str) -> JsonResponse:
    try:
        workspace = Workspace.objects.get(id=workspace_id)
    except Workspace.DoesNotExist:
        return JsonResponse({"detail": "Workspace not found."}, status=404)

    notebooks_data = []
    for nb in workspace.notebooks.all().order_by("position", "name"):
        pages_data = [
            {
                "id": p.id,
                "notebook_id": p.notebook_id,
                "title": p.title,
                "position": p.position,
                "updated_at": p.updated_at.isoformat(),
            }
            for p in nb.pages.all().order_by("position", "title")
        ]
        notebooks_data.append({
            "id": nb.id,
            "workspace_id": nb.workspace_id,
            "name": nb.name,
            "description": nb.description,
            "position": nb.position,
            "created_at": nb.created_at.isoformat(),
            "updated_at": nb.updated_at.isoformat(),
            "pages": pages_data,
        })

    return JsonResponse({
        "id": workspace.id,
        "name": workspace.name,
        "created_at": workspace.created_at.isoformat(),
        "updated_at": workspace.updated_at.isoformat(),
        "notebooks": notebooks_data,
    })


@csrf_exempt
def attach_document(request: HttpRequest, page_id: str, document_id: str) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    try:
        page = Page.objects.get(id=page_id)
    except Page.DoesNotExist:
        return JsonResponse({"detail": "Page not found."}, status=404)

    doc, _ = Document.objects.get_or_create(
        id=document_id,
        defaults={
            "source_path": f"linked://{document_id}",
            "original_filename": document_id,
            "media_type": "external",
            "content_hash": document_id,
            "byte_size": 0,
            "status": "indexed",
        }
    )

    _, created = NotebookDocument.objects.get_or_create(
        notebook=page.notebook,
        document=doc,
        defaults={"attached_at": timezone.now()}
    )
    return JsonResponse({"created": created}, status=201 if created else 200)


def page_documents(request: HttpRequest, page_id: str) -> JsonResponse:
    try:
        page = Page.objects.get(id=page_id)
    except Page.DoesNotExist:
        return JsonResponse({"detail": "Page not found."}, status=404)

    docs = []
    for nd in page.notebook.notebook_documents.select_related("document").order_by("-attached_at"):
        d = nd.document
        docs.append({
            "id": d.id,
            "source_path": d.source_path,
            "original_filename": d.original_filename,
            "media_type": d.media_type,
            "content_hash": d.content_hash,
            "byte_size": d.byte_size,
            "status": d.status,
            "created_at": d.created_at.isoformat(),
            "indexed_at": d.indexed_at.isoformat() if d.indexed_at else None,
            "error_message": d.error_message,
        })
    return JsonResponse(docs, safe=False)


@csrf_exempt
def upload_page_attachment(request: HttpRequest, page_id: str) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    try:
        page = Page.objects.get(id=page_id)
    except Page.DoesNotExist:
        return JsonResponse({"detail": "Page not found."}, status=404)

    file_obj = request.FILES.get("file")
    if not file_obj:
        return JsonResponse({"detail": "No file uploaded."}, status=400)

    raw = file_obj.read()
    digest = hashlib.sha256(raw).hexdigest()

    existing = Document.objects.filter(content_hash=digest).first()
    if existing:
        _, created = NotebookDocument.objects.get_or_create(notebook=page.notebook, document=existing)
        doc_dict = {
            "id": existing.id,
            "source_path": existing.source_path,
            "original_filename": existing.original_filename,
            "media_type": existing.media_type,
            "content_hash": existing.content_hash,
            "byte_size": existing.byte_size,
            "status": existing.status,
            "created_at": existing.created_at.isoformat(),
            "indexed_at": existing.indexed_at.isoformat() if existing.indexed_at else None,
            "error_message": existing.error_message,
        }
        return JsonResponse({"linked": True, "reused": True, "document": doc_dict}, status=201)

    settings = get_settings()
    filename = Path(file_obj.name or "upload").name
    settings.allowed_upload_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.allowed_upload_dir / f"{digest[:16]}-{filename}"
    destination.write_bytes(raw)

    rag = get_rag_service()
    try:
        report = asyncio.run(rag.ingestor.ingest(destination))
        # Ensure Document exists in Django ORM (sync from database if needed or create)
        doc = Document.objects.filter(content_hash=digest).first()
        if not doc:
            doc_row = rag.retriever.get_document(report["document_id"])
            media_type = doc_row["media_type"] if doc_row else "text"
            doc = Document.objects.create(
                id=report["document_id"],
                source_path=str(destination),
                original_filename=filename,
                media_type=media_type,
                content_hash=digest,
                byte_size=len(raw),
                status="indexed",
                indexed_at=timezone.now(),
            )
        NotebookDocument.objects.get_or_create(notebook=page.notebook, document=doc)
        doc_dict = {
            "id": doc.id,
            "source_path": doc.source_path,
            "original_filename": doc.original_filename,
            "media_type": doc.media_type,
            "content_hash": doc.content_hash,
            "byte_size": doc.byte_size,
            "status": doc.status,
            "created_at": doc.created_at.isoformat(),
            "indexed_at": doc.indexed_at.isoformat() if doc.indexed_at else None,
            "error_message": doc.error_message,
        }
        return JsonResponse({"linked": True, "reused": False, "document": doc_dict, "report": report}, status=201)
    except Exception as error:
        destination.unlink(missing_ok=True)
        return JsonResponse({"detail": str(error)}, status=400)


@csrf_exempt
def ingest_file(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    file_obj = request.FILES.get("file")
    if not file_obj:
        return JsonResponse({"detail": "No file uploaded."}, status=400)

    settings = get_settings()
    filename = Path(file_obj.name or "upload").name
    settings.allowed_upload_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.allowed_upload_dir / filename
    with destination.open("wb") as target:
        for chunk in file_obj.chunks():
            target.write(chunk)

    rag = get_rag_service()
    try:
        report = asyncio.run(rag.ingestor.ingest(destination))
        return JsonResponse(report)
    except (ValueError, FileNotFoundError) as error:
        return JsonResponse({"detail": str(error)}, status=400)


@csrf_exempt
def search_view(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)

    query = body.get("query", "")
    top_k = int(body.get("top_k", 8))
    media_type = body.get("media_type")

    rag = get_rag_service()
    try:
        results = asyncio.run(rag.retriever.search(query, top_k, media_type))
        return JsonResponse({"results": results})
    except ValueError as error:
        return JsonResponse({"detail": str(error)}, status=400)


# --- Notebook-Scoped Sources Management ---
@csrf_exempt
def list_or_add_sources(request: HttpRequest) -> JsonResponse:
    page_id = request.GET.get("page_id") or request.POST.get("page_id")
    notebook_id = request.GET.get("notebook_id") or request.POST.get("notebook_id")

    if request.method == "POST":
        # Handle JSON body if POST
        body = {}
        if request.content_type == "application/json":
            try:
                body = json.loads(request.body.decode("utf-8"))
            except Exception:
                pass
            page_id = body.get("page_id", page_id)
            notebook_id = body.get("notebook_id", notebook_id)

        source_type = body.get("source_type") or request.POST.get("source_type", "file")
        title = body.get("title") or request.POST.get("title", "")
        content = body.get("content") or request.POST.get("content", "")
        url = body.get("url") or request.POST.get("url", "")
        file_obj = request.FILES.get("file")

        settings = get_settings()
        raw_bytes: bytes = b""
        filename = ""
        media_type = "text"

        if source_type == "web" or url:
            # Fetch URL or text from URL
            import urllib.request
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (RAGPoC Assistant)"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw_bytes = resp.read()
                filename = url.split("://")[-1].replace("/", "_")[:60] + ".html"
                # If HTML, extract text or keep raw
                media_type = "text"
            except Exception:
                # Fallback: store URL and summary
                raw_bytes = f"URL: {url}\nTitulo: {title}\n\n{content}".encode()
                filename = (title[:40].replace(" ", "_") or "web_source") + ".txt"
        elif source_type == "text":
            raw_bytes = f"{title}\n\n{content}".encode()
            filename = (title[:40].replace(" ", "_") or "nota_fuente") + ".txt"
        elif file_obj:
            raw_bytes = file_obj.read()
            filename = Path(file_obj.name or "upload").name
        else:
            return JsonResponse({"detail": "No content, url or file provided."}, status=400)

        digest = hashlib.sha256(raw_bytes).hexdigest()
        existing_doc = Document.objects.filter(content_hash=digest).first()

        doc = existing_doc
        if not doc:
            settings.allowed_upload_dir.mkdir(parents=True, exist_ok=True)
            dest = settings.allowed_upload_dir / f"{digest[:16]}-{filename}"
            dest.write_bytes(raw_bytes)
            rag = get_rag_service()
            try:
                report = asyncio.run(rag.ingestor.ingest(dest))
                # rag.ingestor.ingest() already inserts this row via raw SQL into the same
                # physical `documents` table Django's ORM reads (Document.Meta.db_table).
                # Re-check before creating, or Document.objects.create() hits a UNIQUE
                # constraint violation on `id` (matches the working pattern in upload_page_attachment).
                doc = Document.objects.filter(content_hash=digest).first()
                if not doc:
                    doc_row = rag.retriever.get_document(report["document_id"])
                    media_type = doc_row["media_type"] if doc_row else "text"
                    doc = Document.objects.create(
                        id=report["document_id"],
                        source_path=str(dest),
                        original_filename=filename,
                        media_type=media_type,
                        content_hash=digest,
                        byte_size=len(raw_bytes),
                        status="indexed",
                        indexed_at=timezone.now(),
                    )
            except Exception as e:
                dest.unlink(missing_ok=True)
                return JsonResponse({"detail": f"Error al indexar: {str(e)}"}, status=400)

        # Attach to the notebook (resolving it via page_id when only a page is given)
        nb_id = notebook_id
        if not nb_id and page_id:
            p = Page.objects.filter(id=page_id).first()
            if p:
                nb_id = p.notebook_id
        if not nb_id:
            return JsonResponse({"detail": "notebook_id is required."}, status=400)
        try:
            nb = Notebook.objects.get(id=nb_id)
            NotebookDocument.objects.get_or_create(notebook=nb, document=doc)
        except Notebook.DoesNotExist:
            return JsonResponse({"detail": "Notebook not found."}, status=404)

        return JsonResponse({
            "id": doc.id,
            "filename": doc.original_filename,
            "media_type": doc.media_type,
            "byte_size": doc.byte_size,
            "notebook_id": nb_id,
            "reused": bool(existing_doc),
        }, status=201)

    # GET: List this notebook's sources (resolving the notebook via page_id if needed)
    nb_id_resolved = notebook_id
    if not nb_id_resolved and page_id:
        p = Page.objects.filter(id=page_id).first()
        if p: nb_id_resolved = p.notebook_id

    if not nb_id_resolved:
        return JsonResponse([], safe=False)

    docs = Document.objects.filter(notebook_documents__notebook_id=nb_id_resolved)
    sources_list = [
        {"id": d.id, "filename": d.original_filename, "media_type": d.media_type, "byte_size": d.byte_size}
        for d in docs
    ]
    return JsonResponse(sources_list, safe=False)


@csrf_exempt
def document_detail_dispatch(request: HttpRequest, document_id: str) -> JsonResponse:
    try:
        doc = Document.objects.get(id=document_id)
    except Document.DoesNotExist:
        return JsonResponse({"detail": "Document not found."}, status=404)

    if request.method == "PATCH" or request.method == "PUT":
        try:
            body = json.loads(request.body.decode("utf-8"))
        except Exception:
            return JsonResponse({"detail": "Invalid JSON"}, status=400)
        
        new_name = (body.get("filename") or body.get("name") or "").strip()
        if not new_name:
            return JsonResponse({"detail": "Filename cannot be empty."}, status=422)
        
        doc.original_filename = new_name
        doc.save()

        # Also update in sqlite raw database for RAG retriever consistency
        rag = get_rag_service()
        try:
            rag.connection.execute("UPDATE documents SET original_filename = ? WHERE id = ?", (new_name, doc.id))
            rag.connection.commit()
        except Exception:
            pass

        return JsonResponse({
            "id": doc.id,
            "filename": doc.original_filename,
            "media_type": doc.media_type,
            "byte_size": doc.byte_size,
            "status": doc.status,
            "updated_at": timezone.now().isoformat(),
        })

    elif request.method == "DELETE":
        # Clean up RAG-only artifacts (chunks/vectors/FTS) that have no Django model, via raw SQL.
        # Do NOT delete the `documents` row here: `notebook_documents` was added via a Django
        # migration, which only enforces the FK (DEFERRABLE INITIALLY DEFERRED) without an
        # SQL-level cascade — Django cascades it in Python instead. Deleting `documents` via raw
        # SQL here would leave orphaned notebook links, and the FK violation surfaces at commit()
        # and gets swallowed by the bare except, which was leaving the shared rag.connection
        # wedged and blocking every later write with "database is locked" until the process
        # restarted. doc.delete() below (Django ORM) correctly cascades notebook_documents instead.
        rag = get_rag_service()
        try:
            rag.connection.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)", (doc.id,))
            rag.connection.execute("DELETE FROM chunk_vectors WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)", (doc.id,))
            rag.connection.execute("DELETE FROM chunks WHERE document_id = ?", (doc.id,))
            rag.connection.commit()
        except Exception:
            rag.connection.rollback()

        # Also unlink physical file if exists
        try:
            file_path = Path(doc.source_path)
            if file_path.exists() and file_path.is_file():
                file_path.unlink(missing_ok=True)
        except Exception:
            pass

        doc.delete()
        return JsonResponse({"status": "deleted", "id": document_id})

    return JsonResponse({"detail": "Method not allowed"}, status=405)


def _collect_notebook_context(notebook: Notebook, custom_instructions: str, rag) -> str:
    """Gathers notebook page text plus top RAG matches into a single context blob, used by every
    artifact-generation pipeline (sync text artifacts and the streaming media artifacts alike)."""
    context_chunks = []
    for page in notebook.pages.all():
        if page.plain_text:
            context_chunks.append(f"Página '{page.title}':\n{page.plain_text[:1200]}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        sources = loop.run_until_complete(
            rag.retriever.search(query=f"conceptos clave {custom_instructions}", top_k=6, notebook_id=notebook.id)
        )
        for s in sources:
            s_name = s.get("filename", "")
            s_text = s.get("text") or ""
            if s_text:
                context_chunks.append(f"Fuente '{s_name}':\n{s_text[:1200]}")
    finally:
        loop.close()

    return "\n\n---\n\n".join(context_chunks) if context_chunks else "Sin fuentes específicas. Utiliza las instrucciones del usuario."


ARTIFACT_TITLES = {
    "diagram": "Diagrama de Arquitectura",
    "mindmap": "Mapa Mental",
    "quiz": "Cuestionario Interactivo",
    "study_guide": "Guía de Estudio",
    "flashcards": "Flashcards de Repaso",
    "summary": "Resumen Ejecutivo",
    "infographic": "Infografía",
    "timeline": "Línea de Tiempo",
    "podcast": "Podcast",
}

ARTIFACT_SYSTEM_PROMPTS = {
    "diagram": (
        "Eres un arquitecto de software y diseñador de diagramas. Genera un diagrama Mermaid.js "
        "claro y profesional (flowchart TD o sequenceDiagram o classDiagram) que resuma la arquitectura, "
        "flujo de datos o conceptos de las notas proporcionadas. Devuelve SOLO el bloque de código Mermaid "
        "encapsulado entre ```mermaid y ``` sin texto adicional innecesario."
    ),
    "mindmap": (
        "Eres un experto en organización visual del conocimiento. Genera un mapa mental jerárquico en formato "
        "Mermaid.js (tipo `mindmap`) que organice los conceptos clave de las notas en una estructura de árbol "
        "clara (raíz = tema central, ramas = subtemas, hojas = detalles). Devuelve SOLO el bloque de código "
        "Mermaid encapsulado entre ```mermaid y ``` sin texto adicional."
    ),
    "quiz": (
        "Eres un evaluador pedagógico. Genera un cuestionario interactivo de {count} preguntas de opción múltiple "
        "con respuestas basadas en las notas. Devuelve la respuesta en formato JSON estrictamente válido con la estructura: "
        '{{"title": "...", "questions": [{{"id": 1, "question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct_index": 0, "explanation": "..."}}]}}'
    ),
    "study_guide": (
        "Eres un tutor experto. Genera una Guía de Estudio estructurada y completa en Markdown "
        "con las siguientes secciones: 1. Resumen Ejecutivo, 2. Conceptos Clave & Definiciones, "
        "3. Decisiones Técnicas & Flujos, 4. Preguntas de Autoevaluación. Formatea con negritas, listas y llamadas a la acción."
    ),
    "flashcards": (
        "Eres un especialista en aprendizaje espaciado. Genera un set de {count} tarjetas de memoria (Flashcards) "
        "con conceptos clave de las notas. Devuelve formato JSON: "
        '{{"cards": [{{"front": "Concepto o Pregunta", "back": "Explicación clara y concisa"}}]}}'
    ),
    "summary": (
        "Genera un Resumen Ejecutivo de 1 página en formato Markdown con viñetas concisas de alto impacto "
        "y una tabla comparativa o matriz de decisiones si aplica."
    ),
}


@csrf_exempt
def generate_artifact_view(request: HttpRequest) -> JsonResponse:
    """Genera artefactos de conocimiento basados en texto (Diagramas Mermaid, Mapas Mentales Mermaid,
    Quizzes, Guías de Estudio, Flashcards, Resúmenes) a nivel de Cuaderno. Los tipos basados en imagen/audio
    (infografía, timeline, mapa mental ilustrado, podcast) se generan vía generate_artifact_stream_view."""
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)

    artifact_type = body.get("artifact_type", "study_guide")
    notebook_id = body.get("notebook_id")
    page_id = body.get("page_id")
    custom_instructions = (body.get("instructions") or "").strip()
    preferences = body.get("settings") or {}

    if not notebook_id and page_id:
        p = Page.objects.filter(id=page_id).first()
        if p: notebook_id = p.notebook_id

    if not notebook_id:
        return JsonResponse({"detail": "notebook_id is required to generate notebook artifacts."}, status=422)

    try:
        notebook = Notebook.objects.get(id=notebook_id)
    except Notebook.DoesNotExist:
        return JsonResponse({"detail": "Notebook not found."}, status=404)

    settings = get_settings()
    rag = get_rag_service()
    full_context = _collect_notebook_context(notebook, custom_instructions, rag)

    from knowledge.artifact_media import describe_artifact_settings
    settings_directive = describe_artifact_settings(artifact_type, preferences)

    # quiz/flashcards' base prompts hardcode a default item count via {count} — the settings
    # chip's value (already a plain number, e.g. "8") slots in directly, no phrase lookup needed.
    default_counts = {"quiz": "4", "flashcards": "5"}
    count = preferences.get("count") or default_counts.get(artifact_type, "")
    chosen_system = ARTIFACT_SYSTEM_PROMPTS.get(artifact_type, ARTIFACT_SYSTEM_PROMPTS["study_guide"]).format(count=count)
    user_prompt = f"Contenido del cuaderno '{notebook.name}':\n{full_context}\n\nInstrucciones adicionales del usuario: {custom_instructions}"
    if settings_directive:
        user_prompt += f"\n\nPreferencias de formato: {settings_directive}"

    started = time.perf_counter()
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )
        completion = client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": chosen_system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        result_content = completion.choices[0].message.content or ""
        usage = completion.usage
        record_usage(
            category="artifact",
            action=f"artifact:{artifact_type}",
            model=settings.chat_model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            duration_ms=int((time.perf_counter() - started) * 1000),
            metadata={"notebook_id": notebook.id, "artifact_type": artifact_type},
        )

        # Save to NotebookArtifact gallery automatically
        artifact_title = f"{ARTIFACT_TITLES.get(artifact_type, 'Artefacto')} · {notebook.name}"
        metadata = {"render_mode": "mermaid"} if artifact_type == "mindmap" else {}
        saved_artifact = NotebookArtifact.objects.create(
            notebook=notebook,
            artifact_type=artifact_type,
            title=artifact_title,
            content=result_content,
            metadata_json=metadata,
        )

        return JsonResponse({
            "status": "success",
            "artifact": {
                "id": saved_artifact.id,
                "notebook_id": notebook.id,
                "notebook_name": notebook.name,
                "artifact_type": saved_artifact.artifact_type,
                "title": saved_artifact.title,
                "content": saved_artifact.content,
                "metadata_json": saved_artifact.metadata_json,
                "created_at": saved_artifact.created_at.isoformat(),
            }
        })
    except Exception as e:
        record_usage(
            category="artifact",
            action=f"artifact:{artifact_type}",
            model=settings.chat_model,
            duration_ms=int((time.perf_counter() - started) * 1000),
            status="error",
            error_message=str(e),
            metadata={"notebook_id": notebook.id, "artifact_type": artifact_type},
        )
        return JsonResponse({"detail": f"Error al generar artefacto: {str(e)}"}, status=500)


@csrf_exempt
def notebook_artifacts_list_view(request: HttpRequest, notebook_id: str) -> JsonResponse:
    try:
        notebook = Notebook.objects.get(id=notebook_id)
    except Notebook.DoesNotExist:
        return JsonResponse({"detail": "Notebook not found."}, status=404)

    artifacts = notebook.artifacts.all()
    data = [
        {
            "id": a.id,
            "notebook_id": a.notebook_id,
            "artifact_type": a.artifact_type,
            "title": a.title,
            "content": a.content,
            "metadata_json": a.metadata_json,
            "created_at": a.created_at.isoformat(),
        }
        for a in artifacts
    ]
    return JsonResponse(data, safe=False)


@csrf_exempt
def delete_notebook_artifact_view(request: HttpRequest, artifact_id: str) -> JsonResponse:
    try:
        artifact = NotebookArtifact.objects.get(id=artifact_id)
    except NotebookArtifact.DoesNotExist:
        return JsonResponse({"detail": "Artifact not found."}, status=404)

    if request.method == "DELETE":
        artifact.delete()
        return JsonResponse({"status": "deleted", "id": artifact_id})
    return JsonResponse({"detail": "Method not allowed"}, status=405)


def download_artifact_pdf_view(request: HttpRequest, artifact_id: str) -> HttpResponse:
    try:
        artifact = NotebookArtifact.objects.get(id=artifact_id)
    except NotebookArtifact.DoesNotExist:
        return JsonResponse({"detail": "Artifact not found."}, status=404)

    pdf_bytes = render_artifact_pdf(artifact.title, artifact.artifact_type, artifact.content, artifact.metadata_json)

    slug = re.sub(r"[^a-z0-9]+", "-", (artifact.title or "artefacto").lower()).strip("-")[:60] or "artefacto"
    response = HttpResponse(pdf_bytes, content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{slug}.pdf"'
    return response


_ARTIFACT_MEDIA_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
}


def serve_artifact_media_view(request: HttpRequest, artifact_id: str) -> HttpResponse:
    try:
        artifact = NotebookArtifact.objects.get(id=artifact_id)
    except NotebookArtifact.DoesNotExist:
        return JsonResponse({"detail": "Artifact not found."}, status=404)

    media_path = (artifact.metadata_json or {}).get("media_path")
    if not media_path:
        return JsonResponse({"detail": "This artifact has no media file."}, status=404)

    file_path = Path(media_path)
    if not file_path.exists() or not file_path.is_file():
        return JsonResponse({"detail": "Media file not found on disk."}, status=404)

    content_type = artifact.metadata_json.get("mime_type") or _ARTIFACT_MEDIA_CONTENT_TYPES.get(
        file_path.suffix.lower(), "application/octet-stream"
    )
    file_size = file_path.stat().st_size
    range_header = request.headers.get("Range")

    if range_header:
        # <audio>/<video> elements probe with a Range request to determine duration/seek support;
        # without a 206 response here, Chrome never populates the element's duration/metadata.
        try:
            _, _, range_spec = range_header.partition("=")
            start_str, _, end_str = range_spec.partition("-")
            start = int(start_str) if start_str else 0
            end = min(int(end_str), file_size - 1) if end_str else file_size - 1
        except ValueError:
            start, end = 0, file_size - 1

        with file_path.open("rb") as f:
            f.seek(start)
            chunk = f.read(end - start + 1)

        response = HttpResponse(chunk, status=206, content_type=content_type)
        response["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        response["Content-Length"] = str(len(chunk))
    else:
        response = HttpResponse(file_path.read_bytes(), content_type=content_type)

    response["Accept-Ranges"] = "bytes"
    response["Content-Disposition"] = f'inline; filename="{file_path.name}"'
    return response


@csrf_exempt
def generate_artifact_stream_view(request: HttpRequest) -> HttpResponse:
    """Genera artefactos basados en imagen/audio (infografía, timeline, mapa mental ilustrado, podcast)
    en un hilo de fondo, transmitiendo eventos de progreso por SSE (mismo patrón que chat_stream_view)
    porque estas llamadas a modelos de imagen/TTS pueden tardar bastante más que una llamada de texto."""
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)

    artifact_type = body.get("artifact_type")
    if artifact_type not in {"infographic", "timeline", "podcast", "mindmap"}:
        return JsonResponse({"detail": "Unsupported artifact_type for streaming generation."}, status=422)

    notebook_id = body.get("notebook_id")
    page_id = body.get("page_id")
    custom_instructions = (body.get("instructions") or "").strip()
    preferences = body.get("settings") or {}

    if not notebook_id and page_id:
        p = Page.objects.filter(id=page_id).first()
        if p: notebook_id = p.notebook_id
    if not notebook_id:
        return JsonResponse({"detail": "notebook_id is required to generate notebook artifacts."}, status=422)
    try:
        notebook = Notebook.objects.get(id=notebook_id)
    except Notebook.DoesNotExist:
        return JsonResponse({"detail": "Notebook not found."}, status=404)

    settings = get_settings()
    rag = get_rag_service()

    def event_stream():
        import queue
        import threading

        q: queue.Queue = queue.Queue()

        def background_task():
            try:
                q.put({"type": "status", "message": "Analizando notas del cuaderno…"})
                # _collect_notebook_context spins up (and closes) its own event loop, so it must
                # run to completion here, BEFORE the media pipeline's own loop starts below —
                # asyncio forbids starting a new loop while another is already running on this thread.
                full_context = _collect_notebook_context(notebook, custom_instructions, rag)

                from knowledge.artifact_media import build_media_artifact

                media_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(media_loop)
                try:
                    title, content, metadata = media_loop.run_until_complete(
                        build_media_artifact(
                            artifact_type=artifact_type,
                            notebook=notebook,
                            full_context=full_context,
                            custom_instructions=custom_instructions,
                            settings=settings,
                            on_progress=lambda message: q.put({"type": "status", "message": message}),
                            preferences=preferences,
                        )
                    )
                finally:
                    media_loop.close()

                saved_artifact = NotebookArtifact.objects.create(
                    notebook=notebook,
                    artifact_type=artifact_type,
                    title=title,
                    content=content,
                    metadata_json=metadata,
                )
                q.put({
                    "type": "done",
                    "artifact": {
                        "id": saved_artifact.id,
                        "notebook_id": notebook.id,
                        "notebook_name": notebook.name,
                        "artifact_type": saved_artifact.artifact_type,
                        "title": saved_artifact.title,
                        "content": saved_artifact.content,
                        "metadata_json": saved_artifact.metadata_json,
                        "created_at": saved_artifact.created_at.isoformat(),
                    },
                })
            except Exception as exc:
                q.put({"type": "error", "detail": str(exc)})
            finally:
                q.put(None)

        worker = threading.Thread(target=background_task, daemon=True)
        worker.start()

        while True:
            try:
                item = q.get(timeout=180)
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'error', 'detail': 'Timeout esperando generación de medio.'}, ensure_ascii=False)}\n\n"
                break

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


@csrf_exempt
def download_page_pdf_view(request: HttpRequest, page_id: str) -> HttpResponse:
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    try:
        page = Page.objects.get(id=page_id)
    except Page.DoesNotExist:
        return JsonResponse({"detail": "Page not found."}, status=404)

    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)

    title = (body.get("title") or page.title or "Untitled").strip()
    html = body.get("html") or ""

    pdf_bytes = render_page_pdf(title, html)

    slug = re.sub(r"[^a-z0-9]+", "-", (title or "pagina").lower()).strip("-")[:60] or "pagina"
    response = HttpResponse(pdf_bytes, content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{slug}.pdf"'
    return response


@csrf_exempt
def inline_ai_action_view(request: HttpRequest) -> JsonResponse:
    """Endpoint para acciones de IA en línea estilo Notion AI (redactar, continuar, resumir, tareas, mejorar)."""
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)

    action = body.get("action", "ask")  # ask, continue, summarize, tasks, improve, explain
    prompt = (body.get("prompt") or "").strip()
    context_text = (body.get("context_text") or "").strip()
    selected_text = (body.get("selected_text") or "").strip()

    settings = get_settings()
    if not settings.openrouter_api_key:
        return JsonResponse({"detail": "OPENROUTER_API_KEY no configurada."}, status=500)

    system_prompts = {
        "ask": (
            "Eres el asistente de redacción en línea integrado en el editor (estilo Notion AI). "
            "Escribe directamente el contenido solicitado en formato HTML limpio (usando <p>, <h1>, <h2>, <ul>, <li>, <blockquote>, <pre><code>) "
            "o Markdown estándar, listo para ser insertado en la nota. NO añadas introducciones como 'Aquí tienes:' ni saludos. "
            "Sé conciso, técnico y directo."
        ),
        "continue": (
            "Eres un copiloto de redacción. Lee el contexto previo de la nota y continúa escribiendo el siguiente párrafo o sección "
            "de forma natural, manteniendo el mismo tono, estilo y vocabulario. Devuelve únicamente el texto de continuación."
        ),
        "summarize": (
            "Genera un resumen conciso en 3-5 viñetas clave del texto proporcionado. Devuelve listas <ul><li> estructuradas."
        ),
        "tasks": (
            "Analiza el texto y extrae todos los elementos de acción o tareas pendientes. "
            "Devuelve una lista de tareas estructurada con viñetas claras y responsables/pasos si los hay."
        ),
        "improve": (
            "Reescribe el texto proporcionado para mejorar su claridad, redacción, flujo y profesionalismo, "
            "manteniendo intacto el significado técnico original. Devuelve únicamente el texto mejorado."
        ),
        "explain": (
            "Explica de forma clara, didáctica y estructurada el concepto o fragmento de código proporcionado."
        ),
    }

    chosen_system = system_prompts.get(action, system_prompts["ask"])

    # Prepare user content
    parts = []
    if context_text and action == "continue":
        parts.append(f"Contexto previo de la nota:\n{context_text[-1500:]}")
    elif selected_text:
        parts.append(f"Texto seleccionado:\n{selected_text}")
    elif context_text:
        parts.append(f"Contexto de la nota:\n{context_text[:1500]}")

    if prompt:
        parts.append(f"Instrucción del usuario: {prompt}")
    elif action == "ask":
        parts.append("Escribe una sección informativa relevante.")

    user_message = "\n\n".join(parts)

    started = time.perf_counter()
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )
        completion = client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": chosen_system},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )
        generated_text = completion.choices[0].message.content or ""
        usage = completion.usage
        record_usage(
            category="inline_ai",
            action=f"inline_ai:{action}",
            model=settings.chat_model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        return JsonResponse({
            "status": "success",
            "action": action,
            "content": generated_text,
        })
    except Exception as e:
        record_usage(
            category="inline_ai",
            action=f"inline_ai:{action}",
            model=settings.chat_model,
            duration_ms=int((time.perf_counter() - started) * 1000),
            status="error",
            error_message=str(e),
        )
        return JsonResponse({"detail": f"Error en generación de IA: {str(e)}"}, status=500)


@csrf_exempt
def omni_search_view(request: HttpRequest) -> JsonResponse:
    """Buscador universal Ctrl+K sobre páginas, cuadernos y documentos indexados."""
    q = (request.GET.get("q") or "").strip()
    if not q:
        return JsonResponse({"results": []})

    results = []
    
    # 1. Search in Pages
    pages = Page.objects.filter(title__icontains=q)[:8]
    for p in pages:
        nb_name = p.notebook.name if p.notebook else "Sin cuaderno"
        results.append({
            "type": "page",
            "id": p.id,
            "title": p.title,
            "subtitle": f"Página en '{nb_name}'",
            "icon": "bi-file-earmark-text",
            "notebook_id": p.notebook_id,
        })

    # 2. Search in Notebooks
    notebooks = Notebook.objects.filter(name__icontains=q)[:5]
    for n in notebooks:
        results.append({
            "type": "notebook",
            "id": n.id,
            "title": n.name,
            "subtitle": f"Cuaderno ({n.pages.count()} páginas)",
            "icon": "bi-journal-bookmark",
        })

    # 3. Search in Documents
    docs = Document.objects.filter(original_filename__icontains=q)[:6]
    for d in docs:
        results.append({
            "type": "document",
            "id": d.id,
            "title": d.original_filename,
            "subtitle": f"Documento {d.media_type.upper()} ({(d.byte_size/1024).toFixed(1) if hasattr(d.byte_size, 'toFixed') else round(d.byte_size/1024, 1)} KB)",
            "icon": "bi-paperclip",
            "media_type": d.media_type,
        })

    return JsonResponse({"results": results})


@csrf_exempt
def list_or_create_memories_view(request: HttpRequest) -> JsonResponse:
    """API para listar o crear memorias declarativas persistentes (estilo Hermes)."""
    if request.method == "POST":
        try:
            body = json.loads(request.body.decode("utf-8"))
            content = (body.get("content") or "").strip()
            category = body.get("category", "user_preference")
            if not content:
                return JsonResponse({"detail": "El contenido es requerido."}, status=422)
            mem = AgentMemory.objects.create(category=category, content=content, source="user_explicit")
            return JsonResponse({"id": mem.id, "category": mem.category, "content": mem.content, "source": mem.source}, status=201)
        except Exception as e:
            return JsonResponse({"detail": str(e)}, status=400)
    
    mems = list(AgentMemory.objects.all().values("id", "category", "content", "source", "updated_at"))
    return JsonResponse(mems, safe=False)


@csrf_exempt
def delete_memory_view(request: HttpRequest, memory_id: str) -> JsonResponse:
    if request.method != "DELETE":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    AgentMemory.objects.filter(id=memory_id).delete()
    return JsonResponse({"status": "success", "id": memory_id})


@csrf_exempt
def list_or_create_skills_view(request: HttpRequest) -> JsonResponse:
    """API para listar o crear habilidades/skills procedimentales (estilo Hermes)."""
    if request.method == "POST":
        try:
            body = json.loads(request.body.decode("utf-8"))
            name = (body.get("name") or "").strip().lower().replace(" ", "-")
            description = (body.get("description") or "").strip()
            instructions = (body.get("instructions") or "").strip()
            category = body.get("category", "general")
            if not name or not instructions:
                return JsonResponse({"detail": "name e instructions son requeridos."}, status=422)
            skill, _ = AgentSkill.objects.update_or_create(
                name=name,
                defaults={
                    "description": description or name,
                    "instructions": instructions,
                    "category": category,
                    "is_active": True,
                }
            )
            return JsonResponse({"id": skill.id, "name": skill.name, "description": skill.description, "instructions": skill.instructions}, status=201)
        except Exception as e:
            return JsonResponse({"detail": str(e)}, status=400)
    
    skills = list(AgentSkill.objects.filter(is_active=True).values("id", "name", "category", "description", "instructions", "updated_at"))
    return JsonResponse(skills, safe=False)


@csrf_exempt
def delete_skill_view(request: HttpRequest, skill_id: str) -> JsonResponse:
    if request.method != "DELETE":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    AgentSkill.objects.filter(id=skill_id).delete()
    return JsonResponse({"status": "success", "id": skill_id})


# --- Chat Threads & History Views ---
@csrf_exempt
def list_or_create_threads(request: HttpRequest) -> JsonResponse:
    if request.method == "POST":
        try:
            body = json.loads(request.body.decode("utf-8"))
        except Exception:
            body = {}
        
        scope = body.get("scope", "workspace")
        page_id = body.get("page_id")
        notebook_id = body.get("notebook_id")
        workspace_id = body.get("workspace_id")
        title = body.get("title", "Nueva conversación").strip() or "Nueva conversación"

        thread = ChatThread.objects.create(
            scope=scope,
            page_id=page_id,
            notebook_id=notebook_id,
            workspace_id=workspace_id,
            title=title,
        )
        return JsonResponse({
            "id": thread.id,
            "title": thread.title,
            "scope": thread.scope,
            "page_id": thread.page_id,
            "created_at": thread.created_at.isoformat(),
            "updated_at": thread.updated_at.isoformat(),
        }, status=201)

    # GET: List threads optionally filtered by page/notebook/workspace
    page_id = request.GET.get("page_id")
    notebook_id = request.GET.get("notebook_id")
    workspace_id = request.GET.get("workspace_id")
    qs = ChatThread.objects.all()
    if page_id:
        qs = qs.filter(page_id=page_id)
    elif notebook_id:
        qs = qs.filter(notebook_id=notebook_id)
    elif workspace_id:
        qs = qs.filter(workspace_id=workspace_id)
    
    threads_data = [
        {
            "id": t.id,
            "title": t.title,
            "scope": t.scope,
            "page_id": t.page_id,
            "created_at": t.created_at.isoformat(),
            "updated_at": t.updated_at.isoformat(),
        }
        for t in qs[:30]
    ]
    return JsonResponse(threads_data, safe=False)


def get_thread_messages(request: HttpRequest, thread_id: str) -> JsonResponse:
    try:
        thread = ChatThread.objects.get(id=thread_id)
    except ChatThread.DoesNotExist:
        return JsonResponse({"detail": "Thread not found."}, status=404)

    messages = [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "sources_json": m.sources_json,
            "attachments_json": m.attachments_json,
            "created_at": m.created_at.isoformat(),
        }
        for m in thread.messages.all()
    ]
    return JsonResponse({
        "thread": {
            "id": thread.id,
            "title": thread.title,
            "scope": thread.scope,
            "page_id": thread.page_id,
        },
        "messages": messages,
    })


@csrf_exempt
def chat_stream_view(request: HttpRequest) -> HttpResponse:
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)

    question = (body.get("question") or "").strip()
    if not question:
        return JsonResponse({"detail": "La pregunta no puede estar vacía."}, status=422)

    scope = body.get("scope", "workspace")
    if scope not in {"page", "notebook", "workspace"}:
        return JsonResponse({"detail": "Invalid scope."}, status=422)

    page_id = body.get("page_id")
    notebook_id = body.get("notebook_id")
    workspace_id = body.get("workspace_id")
    thread_id = body.get("thread_id")
    attachments = body.get("attachments") or []  # List of {id, name, url, text}

    # Sources only ever live at notebook level now, so every chat scope bottoms out at one or
    # more notebook_ids: "page" resolves to its own notebook, "notebook" is direct, "workspace"
    # searches across every notebook in that workspace (an empty list here is intentionally
    # falsy — it falls through to Retriever.search()'s unrestricted global search, same as when
    # workspace_id was omitted before this scope model existed).
    selected: dict[str, str | list[str]]
    if scope == "page":
        if not page_id:
            return JsonResponse({"detail": "page_id is required for this scope."}, status=422)
        p = Page.objects.filter(id=page_id).first()
        if not p:
            return JsonResponse({"detail": "Page not found."}, status=404)
        selected = {"notebook_id": p.notebook_id}
    elif scope == "notebook":
        if not notebook_id:
            return JsonResponse({"detail": "notebook_id is required for this scope."}, status=422)
        selected = {"notebook_id": notebook_id}
    else:
        nb_ids = list(Notebook.objects.filter(workspace_id=workspace_id).values_list("id", flat=True)) if workspace_id else []
        selected = {"notebook_ids": nb_ids}

    # Get or create persistent ChatThread if provided
    thread = None
    if thread_id:
        try:
            thread = ChatThread.objects.get(id=thread_id)
        except ChatThread.DoesNotExist:
            thread = None
    
    if not thread:
        # Create thread automatically with the title of the question (truncated)
        thread_title = (question[:50] + "…") if len(question) > 50 else question
        thread = ChatThread.objects.create(
            scope=scope,
            page_id=page_id,
            notebook_id=notebook_id,
            workspace_id=workspace_id,
            title=thread_title,
        )

    # Save User message in DB
    ChatMessage.objects.create(
        thread=thread,
        role="user",
        content=question,
        attachments_json=attachments,
    )

    rag = get_rag_service()
    settings = get_settings()

    def event_stream():
        import queue
        import threading

        q = queue.Queue()

        # 1. Fetch direct attachments and text from the active page if scope is 'page'
        page_direct_text = ""
        if page_id:
            try:
                p = Page.objects.get(id=page_id)
                if p.plain_text:
                    page_direct_text = f"\n[TEXTO DE LA NOTA ACTUAL (Título: {p.title})]:\n{p.plain_text}\n"
            except Exception:
                pass

        def background_task():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            started = time.perf_counter()
            try:
                # Fetch relevant RAG sources for citations card
                sources = loop.run_until_complete(rag.retriever.search(question, top_k=5, **selected))
                refs = [
                    {
                        "citation": f"[{i}]",
                        "label": s["filename"],
                        "filename": s["filename"],
                        "media_type": s.get("media_type"),
                    }
                    for i, s in enumerate(sources, 1)
                ]
                if refs:
                    q.put({"type": "sources", "sources": refs})

                # Check if OPENROUTER_API_KEY is present
                if not settings.openrouter_api_key:
                    q.put({"type": "token", "text": "OPENROUTER_API_KEY no está configurada."})
                    q.put({"type": "done"})
                    return

                agent_instance = create_pydantic_rag_agent(settings)
                deps = AgentDeps(
                    retriever=rag.retriever,
                    settings=settings,
                    page_id=page_id,
                    # Derived from `selected` (the same scope resolution used for the citations
                    # search above), not the raw body value — otherwise a stale/irrelevant
                    # notebook_id sent alongside scope="workspace" would wrongly narrow the
                    # agent's own search_knowledge_base tool calls to a single notebook.
                    notebook_id=selected.get("notebook_id"),
                    workspace_id=workspace_id,
                    attached_docs_context="",
                )
                
                prompt_parts: list[Any] = []
                
                # Inyectar Memorias y Skills del sistema activas (Hermes Style)
                mems = list(AgentMemory.objects.all().values_list("category", "content"))
                skills = list(AgentSkill.objects.filter(is_active=True).values_list("name", "description", "instructions"))
                
                if mems or skills:
                    mem_context = "══════════════════════════════════════════════\n"
                    if mems:
                        mem_context += "PERSISTENT AGENT MEMORY (HECHOS Y PREFERENCIAS GUARDADAS):\n"
                        for cat, cnt in mems:
                            mem_context += f"- [{cat}] {cnt}\n"
                    if skills:
                        mem_context += "\nACTIVE PROCEDURAL SKILLS (HABILIDADES DISPONIBLES):\n"
                        for s_name, s_desc, s_inst in skills:
                            mem_context += f"• Skill '{s_name}': {s_desc}\n  Instrucciones: {s_inst}\n"
                    mem_context += "══════════════════════════════════════════════\n\n"
                    prompt_parts.append(mem_context)
                
                if page_direct_text:
                    prompt_parts.append(page_direct_text)
                
                # If there are sources found in this scope, pre-inject evidence snippets AND BinaryContent for images
                if sources:
                    rag_context_text = "\n[EVIDENCIA RECUPERADA DE LA BASE DE CONOCIMIENTO]:\n"
                    for idx, s in enumerate(sources, 1):
                        s_text = s.get("text") or ""
                        s_name = s.get("filename") or "Documento"
                        s_type = s.get("media_type") or "desconocido"
                        rag_context_text += f"[{idx}] Fuente: {s_name} (Tipo: {s_type})"
                        if s_text:
                            rag_context_text += f":\n{s_text[:1000]}\n\n"
                        else:
                            rag_context_text += " (Archivo binario/visual indexado)\n\n"
                    prompt_parts.append(rag_context_text)

                    # INJECT MULTIMODAL BYTES FOR IMAGES IN EVIDENCE
                    for _idx, s in enumerate(sources, 1):
                        s_type = s.get("media_type")
                        s_path = s.get("source_path")
                        if s_type == "image" and s_path:
                            p_img = Path(s_path)
                            if p_img.exists() and p_img.is_file():
                                try:
                                    img_bytes = p_img.read_bytes()
                                    suffix = p_img.suffix.lower()
                                    mime = "image/png" if suffix == ".png" else "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/webp"
                                    prompt_parts.append(f"\n[Imagen recuperada de la fuente {s.get('filename')}]:\n")
                                    prompt_parts.append(BinaryContent(data=img_bytes, media_type=mime))
                                except Exception:
                                    pass

                prompt_parts.append(f"\nPregunta del usuario: {question}")
                
                if attachments:
                    for att in attachments:
                        att_name = att.get("name", "adjunto")
                        att_type = att.get("type", "")
                        att_text = att.get("text")
                        att_data = att.get("data")
                        
                        if att_text:
                            prompt_parts.append(f"\n\n[Documento de texto adjunto en chat: {att_name}]\n{att_text}")
                        elif att_data:
                            raw_bin = base64.b64decode(att_data)
                            media_mime = att_type or "image/png"
                            prompt_parts.append(BinaryContent(data=raw_bin, media_type=media_mime))

                async def stream_all():
                    assistant_tokens = []
                    async with agent_instance.run_stream(prompt_parts if len(prompt_parts) > 1 else question, deps=deps) as result:
                        async for token in result.stream_text(delta=True):
                            assistant_tokens.append(token)
                            q.put({"type": "token", "text": token})
                        usage = result.usage

                    final_text = "".join(assistant_tokens)

                    @sync_to_async
                    def save_assistant_msg():
                        ChatMessage.objects.create(
                            thread=thread,
                            role="assistant",
                            content=final_text,
                            sources_json=refs,
                        )
                        thread.save()
                        record_usage(
                            category="chat",
                            action="chat_stream",
                            model=settings.chat_model,
                            input_tokens=usage.input_tokens or 0,
                            output_tokens=usage.output_tokens or 0,
                            total_tokens=usage.total_tokens or 0,
                            cost_usd=usage.cost,
                            duration_ms=int((time.perf_counter() - started) * 1000),
                            metadata={"scope": scope, "thread_id": thread.id},
                        )

                    await save_assistant_msg()
                    q.put({"type": "done"})

                loop.run_until_complete(stream_all())
            except Exception as exc:
                loop.run_until_complete(sync_to_async(record_usage)(
                    category="chat",
                    action="chat_stream",
                    model=settings.chat_model,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    status="error",
                    error_message=str(exc),
                    metadata={"scope": scope, "thread_id": thread.id},
                ))
                q.put({"type": "error", "message": str(exc)})
            finally:
                q.put(None)  # Sentinel to end stream
                loop.close()

        # Start background worker thread
        worker = threading.Thread(target=background_task, daemon=True)
        worker.start()

        # Yield initial metadata event with thread_id
        yield f"data: {json.dumps({'type': 'thread_init', 'thread_id': thread.id, 'title': thread.title}, ensure_ascii=False)}\n\n"

        while True:
            try:
                item = q.get(timeout=60)
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Timeout esperando respuesta del modelo.'}, ensure_ascii=False)}\n\n"
                break

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


def serve_document_file(request: HttpRequest, document_id: str) -> HttpResponse:
    try:
        doc = Document.objects.get(id=document_id)
    except Document.DoesNotExist:
        return JsonResponse({"detail": "Document not found."}, status=404)

    file_path = Path(doc.source_path)
    if not file_path.exists() or not file_path.is_file():
        # Check if stored in uploads directory by content_hash or filename
        settings = get_settings()
        matches = list(settings.allowed_upload_dir.glob(f"{doc.content_hash[:16]}-*"))
        if matches:
            file_path = matches[0]
        else:
            return JsonResponse({"detail": "File not found on disk."}, status=404)

    # Determine content type based on media_type / suffix
    suffix = file_path.suffix.lower()
    content_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".pdf": "application/pdf",
        ".txt": "text/plain; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
    }
    content_type = content_types.get(suffix, "application/octet-stream")

    response = HttpResponse(file_path.read_bytes(), content_type=content_type)
    response["Content-Disposition"] = f'inline; filename="{doc.original_filename}"'
    return response


def documents_list(request: HttpRequest) -> JsonResponse:
    limit = int(request.GET.get("limit", 100))
    offset = int(request.GET.get("offset", 0))
    rag = get_rag_service()
    return JsonResponse({"documents": rag.retriever.list_documents(limit, offset)})


def usage_summary_view(request: HttpRequest) -> JsonResponse:
    """Aggregated API/embedding usage stats + live OpenRouter key status, for the Uso & Monitoreo panel."""
    days = int(request.GET.get("days", 7))
    settings = get_settings()
    return JsonResponse({
        "summary": usage_summary(days=days),
        "key_status": fetch_openrouter_key_status(settings),
        "api_key": get_api_key_status(),
        "embedding_model": settings.embedding_model,
        "chat_model": settings.chat_model,
    })


def usage_logs_view(request: HttpRequest) -> JsonResponse:
    """Paginated raw usage log entries, optionally filtered by category/status."""
    limit = min(int(request.GET.get("limit", 50)), 200)
    offset = int(request.GET.get("offset", 0))
    category = request.GET.get("category")
    status_filter = request.GET.get("status")

    qs = ApiUsageLog.objects.all()
    if category:
        qs = qs.filter(category=category)
    if status_filter:
        qs = qs.filter(status=status_filter)

    total = qs.count()
    rows = list(qs[offset:offset + limit].values(
        "id", "created_at", "category", "action", "provider", "model", "status",
        "error_message", "input_tokens", "output_tokens", "total_tokens", "cost_usd",
        "request_count", "duration_ms", "metadata_json",
    ))
    for row in rows:
        row["created_at"] = row["created_at"].isoformat()

    return JsonResponse({"logs": rows, "total": total, "limit": limit, "offset": offset})
