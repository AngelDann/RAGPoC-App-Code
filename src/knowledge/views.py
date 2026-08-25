from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import mimetypes
import re
import sys
import time
from pathlib import Path
from typing import Any

from asgiref.sync import sync_to_async
from django.http import HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.clickjacking import xframe_options_sameorigin
from django.views.decorators.csrf import csrf_exempt
from django.views.static import serve as serve_static_file
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest
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
from knowledge.services import get_rag_service, reset_rag_service
from knowledge.settings_store import (
    get_api_key_status,
    set_app_settings,
    set_byok_api_key,
)
from knowledge.settings_store import get_effective_settings as get_settings
from knowledge.usage import fetch_openrouter_key_status, record_usage, usage_summary
from ragpoc.updater import (
    UpdateError,
    UpdateState,
    apply_update,
    check_for_update,
    updater_manager,
)


def console_view(request: HttpRequest) -> HttpResponse:
    return render(request, "console.html")


def favicon_view(request: HttpRequest) -> HttpResponse:
    # Dev/source mode: assets/ lives at the repo root. Frozen PyInstaller build: the assets/
    # datas entry in ragpoc.spec gets extracted to sys._MEIPASS at runtime, not next to the .exe
    # itself and not one level above _MEIPASS -- those would silently 404 here.
    base_dirs = [Path(__file__).resolve().parent.parent.parent / "assets"]
    if getattr(sys, "frozen", False):
        base_dirs.append(Path(sys._MEIPASS) / "assets")
    for bdir in base_dirs:
        fav = bdir / "favicon.ico"
        if fav.exists():
            with open(fav, "rb") as f:
                return HttpResponse(f.read(), content_type="image/x-icon")
    return HttpResponse(status=404)


# Windows resolves file types through the registry, and what it answers for ".js" depends on
# what the user happens to have installed -- plenty of machines report "text/plain". Chromium
# (and therefore WebView2) enforces a strict MIME check on ES modules and refuses to execute a
# module served as anything other than a JavaScript type, so a registry quirk on the user's
# machine would break the editor exactly the way the CDN outage this vendoring replaced did.
# Pinning the types here overrides the registry lookup for every type we actually serve; fonts
# get an entry too since Windows has no registry mapping for them at all.
for _ext, _type in (
    (".js", "text/javascript"),
    (".mjs", "text/javascript"),
    (".css", "text/css"),
    (".woff2", "font/woff2"),
    (".woff", "font/woff"),
    (".ttf", "font/ttf"),
    (".svg", "image/svg+xml"),
):
    mimetypes.add_type(_type, _ext)


def _vendor_root() -> Path:
    """Where the bundled frontend libraries live. Same split as favicon_view: from source they
    sit next to the package, in a frozen build PyInstaller extracts the spec's static/ datas
    entry into sys._MEIPASS (i.e. the _internal folder, for the onedir build)."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "ragpoc" / "static" / "vendor"
    return Path(__file__).resolve().parent.parent / "ragpoc" / "static" / "vendor"


def vendor_view(request: HttpRequest, relpath: str) -> HttpResponse:
    """Serves the frontend's third-party libraries (Bootstrap, KaTeX, mermaid, the bundled
    TipTap/marked/DOMPurify module, the web fonts) off the local server.

    These used to be <script>/<link> tags pointing at cdn.jsdelivr.net, esm.sh and Google
    Fonts, which quietly made a desktop app unusable without internet: the whole UI lives in
    one <script type="module">, and an ES module whose imports fail never executes a single
    line. The window still opened and still painted -- so it looked like a working app that had
    frozen, with nothing in ragpoc.log because the failure was entirely browser-side.

    django.views.static.serve rather than a hand-rolled read: it already does the path-traversal
    containment, the Last-Modified/If-Modified-Since handling and the 404, and the "don't use
    this in production" caveat in Django's docs is about throughput on a public site, not
    correctness -- here it serves a few dozen files to one local window."""
    return serve_static_file(request, relpath, document_root=str(_vendor_root()))


def health_view(request: HttpRequest) -> JsonResponse:
    settings = get_settings()
    return JsonResponse({
        # desktop_launcher.py probes this endpoint to tell an already-running RAGPoC apart from
        # an unrelated server that happens to hold the port it wanted, so this marker has to
        # stay put: without it a second launch cannot recognise the first one.
        "app": "ragpoc",
        "status": "ok",
        "database": str(settings.database_path),
        "embedding_model": settings.embedding_model,
        "chat_model": settings.chat_model,
    })


async def check_update_view(request: HttpRequest) -> JsonResponse:
    target_platform = request.GET.get("platform") or None
    try:
        info = await updater_manager.check(target_os=target_platform)
    except Exception as e:
        return JsonResponse({"detail": f"No se pudo verificar actualizaciones: {e}"}, status=502)
    return JsonResponse(info)


def update_status_view(request: HttpRequest) -> JsonResponse:
    return JsonResponse(updater_manager.get_status())


@csrf_exempt
async def start_download_update_view(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
    except Exception:
        body = {}

    download_url = body.get("download_url")
    target_version = body.get("target_version")
    try:
        status = await updater_manager.start_download(download_url=download_url, target_version=target_version)
        return JsonResponse(status)
    except UpdateError as e:
        return JsonResponse({"detail": str(e)}, status=400)
    except Exception as e:
        return JsonResponse({"detail": f"Error al iniciar descarga: {e}"}, status=502)


@csrf_exempt
async def cancel_download_update_view(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    status = await updater_manager.cancel_download()
    return JsonResponse(status)


@csrf_exempt
async def apply_update_view(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
    except Exception:
        body = {}

    download_url = body.get("download_url") or updater_manager.download_url

    # If the update has already been downloaded in the background and is ready to install, apply staged directly
    if updater_manager.state == UpdateState.READY_TO_INSTALL:
        try:
            await updater_manager.apply_staged_update()
            return JsonResponse({"status": "restarting"})
        except UpdateError as e:
            return JsonResponse({"detail": str(e)}, status=400)
        except Exception as e:
            return JsonResponse({"detail": f"Error al aplicar la actualización: {e}"}, status=502)

    if not download_url:
        return JsonResponse({"detail": "download_url is required."}, status=422)

    try:
        await apply_update(download_url)
    except UpdateError as e:
        return JsonResponse({"detail": str(e)}, status=400)
    except Exception as e:
        return JsonResponse({"detail": f"Error al descargar la actualización: {e}"}, status=502)

    # apply_update() only schedules the process exit ~1.5s from now (see its docstring), so this
    # response still has time to reach the client before the connection drops out from under it.
    return JsonResponse({"status": "restarting"})


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
        language = body.get("language")
        set_app_settings(
            api_key=api_key.strip() if isinstance(api_key, str) else None,
            language=language.strip() if isinstance(language, str) else None,
        )
        reset_rag_service()
        return JsonResponse(get_api_key_status())

    return JsonResponse({"detail": "Method not allowed"}, status=405)


@csrf_exempt
def create_workspace(request: HttpRequest) -> JsonResponse:
    if request.method == "GET":
        # The frontend's initWorkspace() falls back here when it doesn't already know a
        # workspace id (e.g. first launch after the browser/webview profile's localStorage
        # was reset) -- without this, a user with real data but no remembered workspace id
        # had no way to ever discover it again, and the app would look permanently empty.
        workspaces = Workspace.objects.all().order_by("created_at")
        return JsonResponse([
            {
                "id": ws.id,
                "name": ws.name,
                "created_at": ws.created_at.isoformat(),
                "updated_at": ws.updated_at.isoformat(),
            }
            for ws in workspaces
        ], safe=False)

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

    if request.method in ("PUT", "PATCH"):
        try:
            body = json.loads(request.body.decode("utf-8"))
        except Exception:
            return JsonResponse({"detail": "Invalid JSON"}, status=400)
        name = (body.get("name") or "").strip()
        if not name:
            return JsonResponse({"detail": "El nombre no puede estar vacío."}, status=422)
        notebook.name = name
        if "description" in body:
            notebook.description = body.get("description") or ""
        notebook.save()
        return JsonResponse({
            "id": notebook.id,
            "workspace_id": notebook.workspace_id,
            "name": notebook.name,
            "description": notebook.description,
            "updated_at": notebook.updated_at.isoformat(),
        })
    elif request.method == "DELETE":
        notebook.delete()
        return JsonResponse({"status": "deleted", "id": notebook_id})
    return JsonResponse({"detail": "Method not allowed"}, status=405)


@csrf_exempt
def workspace_detail_dispatch(request: HttpRequest, workspace_id: str) -> JsonResponse:
    try:
        workspace = Workspace.objects.get(id=workspace_id)
    except Workspace.DoesNotExist:
        return JsonResponse({"detail": "Workspace not found."}, status=404)

    if request.method in ("PUT", "PATCH"):
        try:
            body = json.loads(request.body.decode("utf-8"))
        except Exception:
            return JsonResponse({"detail": "Invalid JSON"}, status=400)
        name = (body.get("name") or "").strip()
        if not name:
            return JsonResponse({"detail": "El nombre no puede estar vacío."}, status=422)
        workspace.name = name
        workspace.save()
        return JsonResponse({
            "id": workspace.id,
            "name": workspace.name,
            "updated_at": workspace.updated_at.isoformat(),
        })
    elif request.method == "DELETE":
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


@csrf_exempt
def append_html_to_page_view(request: HttpRequest, page_id: str) -> JsonResponse:
    """Appends already-rendered HTML straight to a page's content through the API.

    This is the inline "/ IA" bar's fallback for when its generation finishes after the user has
    already navigated to a different page: that page's live editor (and the in-memory position it
    was streaming into) no longer exists in the browser at that point, so the frontend can't just
    insert the result anymore -- it has to be saved server-side instead, the same way a chat
    page-write survives navigation because the backend persists it directly rather than relying on
    whatever happens to be on screen when the stream ends.

    Takes HTML rather than raw markdown (unlike create_workspace_page/update_page_notes, which go
    through markdown_to_tiptap_json) because the caller already ran the generated markdown through
    the browser's own `marked.parse()` for the live on-page render -- reusing that exact HTML via
    html_to_tiptap_json keeps this fallback's formatting identical to what would have landed had
    the user just stayed on the page, instead of re-deriving it from the raw markdown a second time
    with a different renderer."""
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

    html_text = (body.get("html") or "").strip()
    if not html_text:
        return JsonResponse({"detail": "html no puede estar vacío."}, status=422)
    plain_text_addition = (body.get("plain_text") or "").strip() or html_text

    from knowledge.markdown_tiptap import html_to_tiptap_json

    new_nodes = html_to_tiptap_json(html_text)["content"]
    doc_json = page.content_json if isinstance(page.content_json, dict) and "content" in page.content_json else {"type": "doc", "content": []}
    doc_json["content"] = [*doc_json["content"], *new_nodes]
    page.content_json = doc_json
    page.plain_text = (page.plain_text or "").strip() + "\n\n" + plain_text_addition
    page.content_hash = calculate_content_hash(page.title, page.plain_text)
    page.save()

    return JsonResponse({
        "id": page.id,
        "notebook_id": page.notebook_id,
        "title": page.title,
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

    # Only short-circuit onto a prior document if it actually finished indexing and its file is
    # still on disk. A same-hash match on a 'failed' or 'pending' row (e.g. a video that errored
    # out for lack of an OpenRouter key, which also deletes the uploaded file -- see the except
    # branch below) used to get "linked" here unconditionally: the page would show the attachment
    # as if the upload succeeded, but /api/documents/<id>/file would 404 forever since nothing was
    # ever indexed and the file was gone. Falling through instead re-runs the full ingest path,
    # which retries against the same document id (Ingestor.ingest matches by source_path).
    existing = Document.objects.filter(content_hash=digest).first()
    if existing and existing.status == "indexed" and Path(existing.source_path).is_file():
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
        # Ensure Document exists in Django ORM (sync from database if needed or create). Looked up
        # by the id ingest() just reported, not by content_hash: a retry after an earlier failed
        # attempt under a different filename (different source_path, so Ingestor treats it as a
        # separate row instead of reusing the old one by source_path match) can leave more than one
        # document sharing this content_hash, and a hash-only lookup has no way to prefer the one
        # that was actually just indexed over an unrelated stale row.
        doc = Document.objects.filter(id=report["document_id"]).first()
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
                status=report["status"],
                indexed_at=timezone.now() if report["status"] == "indexed" else None,
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
            from knowledge.pydantic_agent import extract_clean_text_from_html, fetch_remote_resource
            try:
                raw_bytes, ctype, fname, ext = fetch_remote_resource(url)
                filename = fname
                if ext == ".pdf" or "pdf" in ctype or raw_bytes.startswith(b"%PDF-"):
                    media_type = "pdf"
                elif ext in {".docx", ".xlsx", ".pptx"}:
                    media_type = "office"
                else:
                    if ext in {".html", ".htm"} or "html" in ctype or b"<html" in raw_bytes[:1000].lower():
                        clean_text = extract_clean_text_from_html(raw_bytes.decode("utf-8", errors="replace"))
                        if clean_text:
                            raw_bytes = clean_text.encode("utf-8")
                            filename = (Path(filename).stem or "web_article") + ".txt"
                    media_type = "text"
            except Exception as ex:
                if content.strip():
                    raw_bytes = f"URL: {url}\nTitulo: {title}\n\n{content}".encode()
                    filename = (title[:40].replace(" ", "_") or "web_source") + ".txt"
                    media_type = "text"
                else:
                    return JsonResponse({"detail": f"No se pudo descargar la URL: {str(ex)}"}, status=400)
        elif source_type == "text":
            raw_bytes = f"{title}\n\n{content}".encode()
            filename = (title[:40].replace(" ", "_") or "nota_fuente") + ".txt"
            media_type = "text"
        elif file_obj:
            raw_bytes = file_obj.read()
            filename = Path(file_obj.name or "upload").name
        else:
            return JsonResponse({"detail": "No content, url or file provided."}, status=400)

        digest = hashlib.sha256(raw_bytes).hexdigest()
        # Only reuse a prior document if it actually finished indexing and its file is still on
        # disk -- same fix as upload_page_attachment: a same-hash match on an 'unindexed' or
        # 'failed' row (e.g. no OpenRouter key/connection at the time, which also means the file
        # may since have been cleared) used to get linked here unconditionally instead of
        # retrying, silently reusing a document that was never actually searchable.
        existing_doc = Document.objects.filter(content_hash=digest).first()
        reusable = existing_doc if existing_doc and existing_doc.status == "indexed" and Path(existing_doc.source_path).is_file() else None

        doc = reusable
        if not doc:
            settings.allowed_upload_dir.mkdir(parents=True, exist_ok=True)
            dest = settings.allowed_upload_dir / f"{digest[:16]}-{filename}"
            dest.write_bytes(raw_bytes)
            rag = get_rag_service()
            try:
                report = asyncio.run(rag.ingestor.ingest(dest))
                # rag.ingestor.ingest() already inserts this row via raw SQL into the same
                # physical `documents` table Django's ORM reads (Document.Meta.db_table).
                # Looked up by the id ingest() just reported, not by content_hash -- a retry
                # after an earlier unindexed/failed attempt under a different filename (different
                # source_path, so Ingestor treats it as a separate row) can leave more than one
                # document sharing this content_hash (matches the fix in upload_page_attachment).
                doc = Document.objects.filter(id=report["document_id"]).first()
                if not doc:
                    doc_row = rag.retriever.get_document(report["document_id"])
                    m_type = doc_row["media_type"] if doc_row else media_type
                    doc = Document.objects.create(
                        id=report["document_id"],
                        source_path=str(dest),
                        original_filename=filename,
                        media_type=m_type,
                        content_hash=digest,
                        byte_size=len(raw_bytes),
                        status=report["status"],
                        indexed_at=timezone.now() if report["status"] == "indexed" else None,
                    )
                else:
                    doc_row = rag.retriever.get_document(doc.id)
                    if doc_row and doc_row.get("media_type") and doc.media_type != doc_row["media_type"]:
                        doc.media_type = doc_row["media_type"]
                        doc.save(update_fields=["media_type"])
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
            "reused": bool(reusable),
            "status": doc.status,
            "error_message": doc.error_message,
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
        {
            "id": d.id,
            "filename": d.original_filename,
            "media_type": d.media_type,
            "byte_size": d.byte_size,
            "status": d.status,
            "error_message": d.error_message,
        }
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

    elif request.method == "POST":
        # Retries indexing an 'unindexed' (or 'failed') document from its already-stored file --
        # no re-upload needed. Ingestor.ingest() matches by source_path, so this reuses the same
        # doc_id/chunks instead of creating a duplicate (see ingestion.py).
        file_path = Path(doc.source_path)
        if not file_path.is_file():
            return JsonResponse({"detail": "El archivo original ya no está en disco; hay que volver a subirlo."}, status=404)

        rag = get_rag_service()
        try:
            asyncio.run(rag.ingestor.ingest(file_path))
        except Exception:
            # A hard failure (bad file) leaves ingest()'s own 'failed' status/error_message
            # already persisted -- just reflect it below rather than erroring the request too.
            pass
        doc.refresh_from_db()
        return JsonResponse({
            "id": doc.id,
            "filename": doc.original_filename,
            "media_type": doc.media_type,
            "byte_size": doc.byte_size,
            "status": doc.status,
            "error_message": doc.error_message,
            "indexed_at": doc.indexed_at.isoformat() if doc.indexed_at else None,
        })

    return JsonResponse({"detail": "Method not allowed"}, status=405)


@csrf_exempt
def document_source_guide_view(request: HttpRequest, document_id: str) -> JsonResponse:
    """Generate or retrieve a NotebookLM-style Source Guide (Summary, Key Topics, Suggested Questions) for a document."""
    try:
        doc = Document.objects.get(id=document_id)
    except Document.DoesNotExist:
        return JsonResponse({"detail": "Document not found."}, status=404)

    settings = get_settings()
    rag = get_rag_service()

    # Extract text from chunks or source file
    text_content = ""
    try:
        chunks = rag.connection.execute(
            "SELECT text_content FROM chunks WHERE document_id = ? AND text_content IS NOT NULL ORDER BY ordinal ASC",
            (document_id,),
        ).fetchall()
        extracted_chunks = []
        for c in chunks:
            val = c["text_content"] if (isinstance(c, dict) or hasattr(c, "keys")) else c[0]
            if val:
                extracted_chunks.append(str(val))
        text_content = "\n\n".join(extracted_chunks)
    except Exception:
        pass

    if not text_content and Path(doc.source_path).is_file():
        from ragpoc.extractors.text import extract_text
        try:
            text_content = extract_text(Path(doc.source_path))
        except Exception:
            pass

    if not text_content.strip():
        return JsonResponse({
            "status": "success",
            "document_id": document_id,
            "filename": doc.original_filename,
            "media_type": doc.media_type,
            "guide": {
                "summary": f"Documento '{doc.original_filename}' ({doc.media_type}) indexado en la base de conocimiento.",
                "key_topics": [doc.media_type.upper(), "Documento"],
                "suggested_questions": [f"¿De qué trata {doc.original_filename}?"],
            },
        })

    # Generate guide using LLM
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )
        sys_prompt = (
            "Eres un analista de conocimiento especializado estilo Google NotebookLM. "
            "Genera una ficha 'Source Guide' completa, profesional y estructurada para el documento suministrado. "
            "Debes responder estrictamente un objeto JSON con las claves:\n"
            "- 'summary': Resumen ejecutivo claro y sustancial (2 a 3 párrafos).\n"
            "- 'key_topics': Lista de 4 a 8 temas, conceptos o entidades centrales.\n"
            "- 'suggested_questions': Lista de 3 a 5 preguntas concretas y sugerentes que un usuario podría hacer sobre este documento.\n"
            "Responde únicamente el JSON."
        )
        sample_text = text_content[:15000]
        completion = client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Documento: {doc.original_filename}\nTipo: {doc.media_type}\n\nContenido:\n{sample_text}"},
            ],
            temperature=0.2,
        )
        raw_guide = completion.choices[0].message.content or "{}"
        cleaned = raw_guide.replace("```json", "").replace("```", "").strip()
        guide_data = json.loads(cleaned)
    except Exception:
        guide_data = {
            "summary": f"Documento '{doc.original_filename}' indexado con {len(text_content)} caracteres.",
            "key_topics": [doc.media_type.capitalize(), "Análisis de fuente"],
            "suggested_questions": [
                f"¿Cuáles son los puntos principales de {doc.original_filename}?",
                f"¿Qué conclusiones destaca {doc.original_filename}?",
            ],
        }

    return JsonResponse({
        "status": "success",
        "document_id": document_id,
        "filename": doc.original_filename,
        "media_type": doc.media_type,
        "guide": guide_data,
    })


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
        retriever = getattr(rag, "retriever", rag)
        sources = loop.run_until_complete(
            retriever.search(query=f"conceptos clave {custom_instructions}", top_k=6, notebook_id=notebook.id)
        )
        for s in sources:
            s_name = s.get("filename", "")
            s_text = s.get("text") or ""
            if s_text:
                context_chunks.append(f"Fuente '{s_name}':\n{s_text[:1200]}")
    finally:
        loop.close()

    return "\n\n---\n\n".join(context_chunks) if context_chunks else "Sin fuentes específicas. Utiliza las instrucciones del usuario."


ARTIFACT_TITLES_ES = {
    "diagram": "Diagrama de Arquitectura",
    "mindmap": "Mapa Mental",
    "quiz": "Cuestionario Interactivo",
    "study_guide": "Guía de Estudio",
    "flashcards": "Flashcards de Repaso",
    "summary": "Resumen Ejecutivo",
    "infographic": "Infografía",
    "timeline": "Línea de Tiempo",
    "podcast": "Podcast",
    "table": "Tabla de Datos",
}

ARTIFACT_TITLES_EN = {
    "diagram": "Architecture Diagram",
    "mindmap": "Mind Map",
    "quiz": "Interactive Quiz",
    "study_guide": "Study Guide",
    "flashcards": "Review Flashcards",
    "summary": "Executive Summary",
    "infographic": "Infographic",
    "timeline": "Timeline",
    "podcast": "Podcast",
    "table": "Data Table",
}

ARTIFACT_TITLES = ARTIFACT_TITLES_ES

ARTIFACT_SYSTEM_PROMPTS_ES = {
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

ARTIFACT_SYSTEM_PROMPTS_EN = {
    "diagram": (
        "You are a software architect and diagram designer. Generate a clear and professional Mermaid.js diagram "
        "(flowchart TD, sequenceDiagram, or classDiagram) summarizing the architecture, data flow, or concepts "
        "from the provided notes. Return ONLY the Mermaid code block enclosed in ```mermaid and ``` with no extra text."
    ),
    "mindmap": (
        "You are a visual knowledge organization expert. Generate a hierarchical mind map in Mermaid.js "
        "format (`mindmap` type) organizing the key concepts of the notes in a clear tree structure "
        "(root = central topic, branches = subtopics, leaves = details). Return ONLY the Mermaid code "
        "block enclosed in ```mermaid and ``` with no extra text."
    ),
    "quiz": (
        "You are a pedagogical evaluator. Generate an interactive quiz with {count} multiple-choice questions "
        "based on the notes. Return strictly valid JSON with the structure: "
        '{{"title": "...", "questions": [{{"id": 1, "question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct_index": 0, "explanation": "..."}}]}}'
    ),
    "study_guide": (
        "You are an expert tutor. Generate a comprehensive and structured Study Guide in Markdown "
        "with the following sections: 1. Executive Summary, 2. Key Concepts & Definitions, "
        "3. Technical Decisions & Flows, 4. Self-Assessment Questions. Format with bold text, lists, and callouts."
    ),
    "flashcards": (
        "You are a spaced-repetition specialist. Generate a set of {count} flashcards "
        "covering key concepts from the notes. Return valid JSON format: "
        '{{"cards": [{{"front": "Concept or Question", "back": "Clear and concise explanation"}}]}}'
    ),
    "summary": (
        "Generate a 1-page Executive Summary in Markdown with high-impact concise bullet points "
        "and a comparison table or decision matrix if applicable."
    ),
}

ARTIFACT_SYSTEM_PROMPTS = ARTIFACT_SYSTEM_PROMPTS_ES


def get_artifact_title(artifact_type: str, notebook_name: str = "", language: str = "es") -> str:
    lang = (language or "es").strip().lower()
    titles = ARTIFACT_TITLES_EN if lang == "en" else ARTIFACT_TITLES_ES
    fallback = "Artifact" if lang == "en" else "Artefacto"
    type_title = titles.get(artifact_type, fallback)
    return f"{type_title} · {notebook_name}" if notebook_name else type_title


def get_artifact_system_prompt(artifact_type: str, count: str = "", language: str = "es") -> str:
    lang = (language or "es").strip().lower()
    prompts = ARTIFACT_SYSTEM_PROMPTS_EN if lang == "en" else ARTIFACT_SYSTEM_PROMPTS_ES
    fallback = prompts.get("study_guide", "")
    raw = prompts.get(artifact_type, fallback)
    return raw.format(count=count) if "{count}" in raw else raw


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

    from knowledge.settings_store import get_current_language

    artifact_type = body.get("artifact_type", "study_guide")
    notebook_id = body.get("notebook_id")
    page_id = body.get("page_id")
    custom_instructions = (body.get("instructions") or "").strip()
    preferences = body.get("settings") or {}
    language = (body.get("language") or get_current_language() or "es").strip().lower()
    is_en = language == "en"

    if not notebook_id and page_id:
        p = Page.objects.filter(id=page_id).first()
        if p: notebook_id = p.notebook_id

    if not notebook_id:
        detail = "notebook_id is required to generate notebook artifacts." if is_en else "notebook_id es requerido para generar artefactos de cuaderno."
        return JsonResponse({"detail": detail}, status=422)

    try:
        notebook = Notebook.objects.get(id=notebook_id)
    except Notebook.DoesNotExist:
        detail = "Notebook not found." if is_en else "Cuaderno no encontrado."
        return JsonResponse({"detail": detail}, status=404)

    settings = get_settings()
    rag = get_rag_service()
    full_context = _collect_notebook_context(notebook, custom_instructions, rag)

    from knowledge.artifact_media import describe_artifact_settings
    settings_directive = describe_artifact_settings(artifact_type, preferences, language=language)

    # quiz/flashcards' base prompts hardcode a default item count via {count} — the settings
    # chip's value (already a plain number, e.g. "8") slots in directly, no phrase lookup needed.
    default_counts = {"quiz": "4", "flashcards": "5"}
    count = preferences.get("count") or default_counts.get(artifact_type, "")
    chosen_system = get_artifact_system_prompt(artifact_type, count=count, language=language)

    if is_en:
        user_prompt = f"Notebook content for '{notebook.name}':\n{full_context}\n\nAdditional user instructions: {custom_instructions}"
        if settings_directive:
            user_prompt += f"\n\nFormatting preferences: {settings_directive}"
    else:
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
        artifact_title = get_artifact_title(artifact_type, notebook.name, language=language)
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

    if request.method in ("PUT", "PATCH"):
        try:
            body = json.loads(request.body.decode("utf-8"))
        except Exception:
            return JsonResponse({"detail": "Invalid JSON"}, status=400)
        title = (body.get("title") or "").strip()
        if not title:
            return JsonResponse({"detail": "El título no puede estar vacío."}, status=422)
        artifact.title = title
        artifact.save()
        return JsonResponse({"id": artifact.id, "title": artifact.title})
    elif request.method == "DELETE":
        artifact.delete()
        return JsonResponse({"status": "deleted", "id": artifact_id})
    return JsonResponse({"detail": "Method not allowed"}, status=405)


@xframe_options_sameorigin
def download_artifact_pdf_view(request: HttpRequest, artifact_id: str) -> HttpResponse:
    try:
        artifact = NotebookArtifact.objects.get(id=artifact_id)
    except NotebookArtifact.DoesNotExist:
        return JsonResponse({"detail": "Artifact not found."}, status=404)

    from ragpoc.artifact_pdf import render_artifact_pdf

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


@xframe_options_sameorigin
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

    from knowledge.settings_store import get_current_language
    language = (body.get("language") or get_current_language() or "es").strip().lower()
    is_en = language == "en"

    # `async def` (not a plain generator) so StreamingHttpResponse's ASGI path treats this as a
    # genuine async iterator and forwards each `yield` to the client as it happens. A synchronous
    # generator here hits Django's `__aiter__` fallback (see django/http/response.py), which
    # buffers the ENTIRE generator into a list via sync_to_async before sending anything -- i.e.
    # no streaming at all, just a long wait followed by the whole response landing in one burst.
    async def event_stream():
        import queue
        import threading

        q: queue.Queue = queue.Queue()
        loop = asyncio.get_running_loop()

        def background_task():
            try:
                init_msg = "Analyzing notebook notes…" if is_en else "Analizando notas del cuaderno…"
                q.put({"type": "status", "message": init_msg})
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
                            language=language,
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
                # q.get() blocks the calling thread, so it runs in the default executor instead
                # of directly on the ASGI event loop -- otherwise it would stall every other
                # request this server is handling until an item shows up.
                item = await loop.run_in_executor(None, lambda: q.get(timeout=180))
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
@xframe_options_sameorigin
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

    from ragpoc.artifact_pdf import render_page_pdf

    pdf_bytes = render_page_pdf(title, html)

    slug = re.sub(r"[^a-z0-9]+", "-", (title or "pagina").lower()).strip("-")[:60] or "pagina"
    response = HttpResponse(pdf_bytes, content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{slug}.pdf"'
    return response


_INLINE_AI_SYSTEM_PROMPTS = {
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


def _build_inline_ai_prompt(body: dict) -> tuple[str, str, str]:
    """Resuelve (action, system_prompt, user_message) a partir del payload del endpoint inline-action."""
    action = body.get("action", "ask")  # ask, continue, summarize, tasks, improve, explain
    prompt = (body.get("prompt") or "").strip()
    context_text = (body.get("context_text") or "").strip()
    selected_text = (body.get("selected_text") or "").strip()

    chosen_system = _INLINE_AI_SYSTEM_PROMPTS.get(action, _INLINE_AI_SYSTEM_PROMPTS["ask"])

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
    return action, chosen_system, user_message


@csrf_exempt
def inline_ai_action_view(request: HttpRequest) -> JsonResponse:
    """Endpoint para acciones de IA en línea estilo Notion AI (redactar, continuar, resumir, tareas, mejorar)."""
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)

    settings = get_settings()
    if not settings.openrouter_api_key:
        return JsonResponse({"detail": "OPENROUTER_API_KEY no configurada."}, status=500)

    action, chosen_system, user_message = _build_inline_ai_prompt(body)

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
def inline_ai_action_stream_view(request: HttpRequest) -> HttpResponse:
    """Variante SSE de inline_ai_action_view: transmite tokens a medida que el modelo los genera,
    para que el editor pueda mostrar el efecto de escritura en vivo en vez de esperar la respuesta completa."""
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)

    settings = get_settings()
    if not settings.openrouter_api_key:
        return JsonResponse({"detail": "OPENROUTER_API_KEY no configurada."}, status=500)

    action, chosen_system, user_message = _build_inline_ai_prompt(body)

    # `async def` (not a plain generator) so StreamingHttpResponse's ASGI path treats this as a
    # genuine async iterator and sends each `yield` to the client as it happens. A synchronous
    # generator here would hit Django's `__aiter__` fallback (see django/http/response.py), which
    # buffers the ENTIRE generator into a list via sync_to_async before sending anything -- i.e.
    # no streaming at all, just a long wait followed by the whole response landing in one burst.
    # AsyncOpenAI (not the sync client) is what makes `async for chunk in stream` possible below
    # without blocking the event loop on the underlying HTTP read.
    async def event_stream():
        started = time.perf_counter()
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.openrouter_api_key,
            )
            stream = await client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {"role": "system", "content": chosen_system},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                stream=True,
                stream_options={"include_usage": True},
            )

            input_tokens = 0
            output_tokens = 0
            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield f"data: {json.dumps({'type': 'token', 'text': delta}, ensure_ascii=False)}\n\n"
                usage = getattr(chunk, "usage", None)
                if usage:
                    input_tokens = usage.prompt_tokens or 0
                    output_tokens = usage.completion_tokens or 0

            await sync_to_async(record_usage)(
                category="inline_ai",
                action=f"inline_ai:{action}",
                model=settings.chat_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            await sync_to_async(record_usage)(
                category="inline_ai",
                action=f"inline_ai:{action}",
                model=settings.chat_model,
                duration_ms=int((time.perf_counter() - started) * 1000),
                status="error",
                error_message=str(e),
            )
            yield f"data: {json.dumps({'type': 'error', 'message': f'Error en generación de IA: {str(e)}'}, ensure_ascii=False)}\n\n"

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


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
        for t in qs.order_by("-updated_at")
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
def thread_detail_dispatch(request: HttpRequest, thread_id: str) -> JsonResponse:
    try:
        thread = ChatThread.objects.get(id=thread_id)
    except ChatThread.DoesNotExist:
        return JsonResponse({"detail": "Thread not found."}, status=404)

    if request.method in ("PUT", "PATCH"):
        try:
            body = json.loads(request.body.decode("utf-8"))
        except Exception:
            return JsonResponse({"detail": "Invalid JSON"}, status=400)
        title = (body.get("title") or "").strip()
        if not title:
            return JsonResponse({"detail": "El título no puede estar vacío."}, status=422)
        thread.title = title
        thread.save()
        return JsonResponse({"id": thread.id, "title": thread.title})
    elif request.method == "DELETE":
        thread.delete()
        return JsonResponse({"status": "deleted", "id": thread_id})
    return JsonResponse({"detail": "Method not allowed"}, status=405)


MAX_HISTORY_TURNS = 12


def _compact_tool_return_content(content: Any) -> Any:
    """Compact bulky tool return payloads for historical turns (observation masking)."""
    if isinstance(content, str):
        if len(content) > 1000:
            return content[:800] + "\n... [contenido truncado para optimizar el contexto]"
        return content
    elif isinstance(content, dict):
        compacted = dict(content)
        for key in ("content_preview", "content", "raw_text", "text", "body"):
            if key in compacted and isinstance(compacted[key], str) and len(compacted[key]) > 1000:
                compacted[key] = compacted[key][:800] + "\n... [truncado para optimizar contexto]"
        if "chunks" in compacted and isinstance(compacted["chunks"], list) and len(compacted["chunks"]) > 4:
            compacted["chunks"] = compacted["chunks"][:3] + [{"text": f"... y {len(compacted['chunks']) - 3} fragmentos adicionales consultados"}]
        if "results" in compacted and isinstance(compacted["results"], list) and len(compacted["results"]) > 5:
            compacted["results"] = compacted["results"][:4]
        return compacted
    return content


def _trim_message_history(messages: list, max_turns: int = MAX_HISTORY_TURNS) -> list:
    """Keep only the last `max_turns` user turns, cut on whole-turn boundaries,
    and compact bulky tool returns in prior turns (observation masking).
    """
    boundaries = [
        i for i, m in enumerate(messages)
        if isinstance(m, ModelRequest) and any(getattr(p, "part_kind", None) == "user-prompt" for p in m.parts)
    ]
    if len(boundaries) > max_turns:
        messages = messages[boundaries[-max_turns]:]

    # Compact bulky tool return payloads in past messages
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if getattr(part, "part_kind", None) == "tool-return" and hasattr(part, "content"):
                    part.content = _compact_tool_return_content(part.content)
    return messages


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
    if scope not in {"notebook", "workspace", "page"}:
        return JsonResponse({"detail": "Invalid scope."}, status=422)

    page_id = body.get("page_id")
    notebook_id = body.get("notebook_id")
    workspace_id = body.get("workspace_id")
    thread_id = body.get("thread_id")
    selected_source_ids = body.get("selected_source_ids") or []
    attachments = body.get("attachments") or []  # List of {id, name, url, text}

    # Sources only ever live at notebook level, so every chat scope bottoms out at one or more
    # notebook_ids: "notebook" is direct, "workspace" searches across every notebook in that
    # workspace (an empty list here is intentionally falsy — it falls through to
    # Retriever.search()'s unrestricted global search, same as when workspace_id was omitted
    # before this scope model existed). page_id, if present, is only used below to fold the
    # active page's own text into context — it no longer restricts the search scope itself.
    selected: dict[str, str | list[str]]
    if scope == "notebook":
        if not notebook_id:
            return JsonResponse({"detail": "notebook_id is required for this scope."}, status=422)
        selected = {"notebook_id": notebook_id}
    elif scope == "page":
        if notebook_id:
            selected = {"notebook_id": notebook_id}
        elif page_id:
            p_nb = Page.objects.filter(id=page_id).values_list("notebook_id", flat=True).first()
            selected = {"notebook_id": p_nb} if p_nb else {}
        else:
            selected = {}
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

    # `async def` (not a plain generator) so StreamingHttpResponse's ASGI path treats this as a
    # genuine async iterator and forwards each `yield` to the client as it happens. A synchronous
    # generator here hits Django's `__aiter__` fallback (see django/http/response.py), which
    # buffers the ENTIRE generator into a list via sync_to_async before sending anything -- i.e.
    # no streaming at all, just a long wait followed by the whole response landing in one burst.
    async def event_stream():
        import queue
        import threading

        q = queue.Queue()
        asgi_loop = asyncio.get_running_loop()

        def background_task():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Fold in what the user is actually looking at right now: the open page's own text
            # (independent of chat scope — this fires whenever a page happens to be open), plus
            # whatever's selected in the editor or, failing that, the text immediately around the
            # cursor, so the agent knows what part of a possibly-long page the question is about.
            # Computed here (a plain OS thread) rather than in event_stream's own body -- that body
            # now runs directly on the ASGI event loop, where a synchronous ORM call like this one
            # would raise Django's SynchronousOnlyOperation.
            page_direct_text = ""
            selected_text = (body.get("selected_text") or "").strip()
            cursor_text = (body.get("cursor_text") or "").strip()
            if page_id:
                try:
                    p = Page.objects.select_related("notebook").get(id=page_id)
                    if p.plain_text:
                        page_direct_text = f"\n[TEXTO DE LA NOTA ACTUAL (Título: {p.title}, Cuaderno: {p.notebook.name})]:\n{p.plain_text}\n"
                    if selected_text:
                        page_direct_text += f"\n[TEXTO SELECCIONADO POR EL USUARIO EN EL EDITOR AHORA MISMO]:\n{selected_text}\n"
                    elif cursor_text:
                        page_direct_text += f"\n[CONTEXTO ALREDEDOR DEL CURSOR DEL USUARIO EN EL EDITOR]:\n…{cursor_text}…\n"
                except Exception:
                    pass
            started = time.perf_counter()
            try:
                # Check if OPENROUTER_API_KEY is present
                if not settings.openrouter_api_key:
                    q.put({"type": "token", "text": "OPENROUTER_API_KEY no está configurada."})
                    q.put({"type": "done"})
                    return

                from pydantic_ai import BinaryContent
                from pydantic_ai.messages import ModelMessagesTypeAdapter
                from knowledge.pydantic_agent import AgentDeps, create_pydantic_rag_agent

                agent_instance = create_pydantic_rag_agent(settings)
                deps = AgentDeps(
                    retriever=rag.retriever,
                    settings=settings,
                    page_id=page_id,
                    notebook_id=selected.get("notebook_id"),
                    workspace_id=workspace_id,
                    thread_id=thread.id,
                    selected_source_ids=selected_source_ids or None,
                    attached_docs_context="",
                    on_tool_event=lambda evt: q.put(evt),
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
                    page_write_tokens: list[str] = []
                    # Set by create_workspace_page/update_page_notes (see pydantic_agent.py) the
                    # moment the model calls one of them -- from then on, every token of the
                    # model's own final text step IS the page content, so it gets mirrored into
                    # page_write_tokens (and a page_write_token SSE event) alongside the normal
                    # chat token. Captured once the tool fires and not re-read per token: a
                    # second write-tool call mid-turn is not a real flow this UI exposes today.
                    write_state: dict | None = None

                    prior_messages = []
                    if thread.history_json:
                        try:
                            prior_messages = _trim_message_history(
                                ModelMessagesTypeAdapter.validate_json(thread.history_json)
                            )
                        except Exception:
                            prior_messages = []

                    async with agent_instance.run_stream(
                        prompt_parts if len(prompt_parts) > 1 else question,
                        deps=deps,
                        message_history=prior_messages,
                    ) as result:
                        async for token in result.stream_text(delta=True):
                            assistant_tokens.append(token)
                            q.put({"type": "token", "text": token})
                            if deps.page_write_state and write_state is None:
                                write_state = deps.page_write_state
                            if write_state:
                                page_write_tokens.append(token)
                                q.put({"type": "page_write_token", "text": token, "page_id": write_state["page_id"]})
                        usage = result.usage

                    final_text = "".join(assistant_tokens)
                    new_history_json = result.all_messages_json().decode("utf-8")

                    sources_payload = (
                        {
                            "sources": deps.collected_sources,
                            "tools_trace": [
                                {
                                    "tool": t.get("tool"),
                                    "label": t.get("label"),
                                    "icon": t.get("icon", "wrench"),
                                    "summary": t.get("summary", t.get("label")),
                                    "duration_ms": t.get("duration_ms", 0),
                                    "status": t.get("status", "done"),
                                }
                                for t in deps.executed_tools
                            ],
                        }
                        if (deps.executed_tools or deps.collected_sources)
                        else []
                    )

                    @sync_to_async
                    def save_assistant_msg():
                        ChatMessage.objects.create(
                            thread=thread,
                            role="assistant",
                            content=final_text,
                            sources_json=sources_payload,
                        )
                        thread.history_json = new_history_json
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

                    if write_state and page_write_tokens:
                        content = "".join(page_write_tokens).strip()

                        @sync_to_async
                        def persist_page_write():
                            from knowledge.markdown_tiptap import markdown_to_tiptap_json

                            p = Page.objects.filter(id=write_state["page_id"]).first()
                            if not p:
                                return None
                            new_nodes = markdown_to_tiptap_json(content)["content"]
                            if write_state["mode"] == "append":
                                p.plain_text = (p.plain_text or "").strip() + "\n\n" + content
                                doc_json = p.content_json if isinstance(p.content_json, dict) and "content" in p.content_json else {"type": "doc", "content": []}
                                doc_json["content"].extend(new_nodes)
                                p.content_json = doc_json
                            else:
                                p.plain_text = content
                                # Keep the H1 title node create_workspace_page seeded the page with.
                                heading = p.content_json["content"][:1] if isinstance(p.content_json, dict) and p.content_json.get("content") else []
                                p.content_json = {"type": "doc", "content": [*heading, *new_nodes]}
                            p.save()
                            return p.title

                        title = await persist_page_write()
                        if title is not None:
                            q.put({
                                "type": "page_written",
                                "action": "created" if write_state["mode"] == "create" else "updated",
                                "page_id": write_state["page_id"],
                                "notebook_id": write_state["notebook_id"],
                                "title": title,
                            })

                    q.put({
                        "type": "done",
                        "tools_trace": [
                            {
                                "tool": t.get("tool"),
                                "label": t.get("label"),
                                "icon": t.get("icon", "wrench"),
                                "summary": t.get("summary", t.get("label")),
                                "duration_ms": t.get("duration_ms", 0),
                                "status": t.get("status", "done"),
                            }
                            for t in deps.executed_tools
                        ],
                        "sources": deps.collected_sources,
                    })

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
                # q.get() blocks the calling thread, so it runs in the default executor instead
                # of directly on the ASGI event loop -- otherwise it would stall every other
                # request this server is handling until a token shows up.
                item = await asgi_loop.run_in_executor(None, lambda: q.get(timeout=60))
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


@xframe_options_sameorigin
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
