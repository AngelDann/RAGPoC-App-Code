import hashlib
import io
import json
from base64 import b64encode

import pytest
from django.core.management import call_command
from django.test import Client

from knowledge.models import Document
from knowledge.services import get_rag_service
from ragpoc.config import Settings
from ragpoc.embeddings import EmbeddingError, FakeEmbeddingProvider


def auth_header() -> dict[str, str]:
    return {"HTTP_AUTHORIZATION": "Basic " + b64encode(b"operator:test-password").decode()}


def get_stream_text(resp) -> str:
    content = resp.streaming_content
    if hasattr(content, "__aiter__"):
        import asyncio
        async def _collect():
            parts = []
            async for chunk in content:
                parts.append(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8"))
            return b"".join(parts).decode("utf-8")
        return asyncio.run(_collect())
    return b"".join(content).decode("utf-8")


@pytest.fixture(autouse=True)
def setup_django_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir = data_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    test_settings = Settings(
        data_dir=data_dir,
        allowed_upload_dir=uploads_dir,
        ui_password="test-password",
    )
    from ragpoc import config
    monkeypatch.setattr(config, "get_settings", lambda: test_settings)

    # Re-initialize RAG service with FakeEmbeddingProvider for tests
    rag_service = get_rag_service(test_settings, FakeEmbeddingProvider())
    from knowledge import views
    monkeypatch.setattr(views, "get_rag_service", lambda: rag_service)

    call_command("migrate", verbosity=0)
    yield


@pytest.mark.django_db
def test_django_requires_basic_authentication():
    client = Client()
    assert client.get("/").status_code == 401
    assert client.get("/health").status_code == 401
    assert client.get("/", **auth_header()).status_code == 200
    assert client.get("/health", **auth_header()).status_code == 200


@pytest.mark.django_db
def test_django_incorrect_basic_authentication_is_rejected():
    client = Client()
    wrong_auth = {"HTTP_AUTHORIZATION": "Basic " + b64encode(b"operator:wrong-password").decode()}
    response = client.get("/api/workspaces", **wrong_auth)
    assert response.status_code == 401
    assert response["WWW-Authenticate"] == 'Basic realm="RAGPoC"'


@pytest.mark.django_db
def test_django_authentication_required_by_default(tmp_path, monkeypatch):
    # BasicAuthMiddleware resolves `get_settings` from knowledge.middleware's own module
    # namespace at call time (`from ragpoc.config import get_settings` there is a one-time name
    # binding, not a live alias) — patching ragpoc.config.get_settings, like the fixture above
    # does for the rest of this file, would not reliably reach it. Patch the name directly on
    # knowledge.middleware instead so this test controls what the middleware actually sees.
    import knowledge.middleware as middleware_module

    data_dir = tmp_path / "data-no-password"
    data_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir = data_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    settings_without_password = Settings(_env_file=None, data_dir=data_dir, allowed_upload_dir=uploads_dir)
    monkeypatch.setattr(middleware_module, "get_settings", lambda: settings_without_password)

    response = Client().get("/")
    assert response.status_code == 503
    assert "RAGPOC_UI_PASSWORD" in response.content.decode()


@pytest.mark.django_db
def test_django_workspace_notebook_and_page_crud():
    client = Client()
    resp = client.post(
        "/api/workspaces",
        data=json.dumps({"name": "Personal"}),
        content_type="application/json",
        **auth_header(),
    )
    assert resp.status_code == 201
    workspace = resp.json()

    resp = client.post(
        "/api/notebooks",
        data=json.dumps({"workspace_id": workspace["id"], "name": "Ideas"}),
        content_type="application/json",
        **auth_header(),
    )
    assert resp.status_code == 201
    notebook = resp.json()

    resp = client.post(
        "/api/pages",
        data=json.dumps({"notebook_id": notebook["id"], "title": "Primera página", "content_json": {"type": "doc", "content": []}}),
        content_type="application/json",
        **auth_header(),
    )
    assert resp.status_code == 201
    page = resp.json()

    assert workspace["name"] == "Personal"
    assert notebook["workspace_id"] == workspace["id"]
    assert page["notebook_id"] == notebook["id"]
    assert page["title"] == "Primera página"

    resp = client.get(f"/api/workspaces/{workspace['id']}/tree", **auth_header())
    assert resp.status_code == 200
    listing = resp.json()
    assert listing["notebooks"][0]["pages"][0]["id"] == page["id"]


@pytest.mark.django_db
def test_django_list_workspaces():
    # This is the fallback the frontend's initWorkspace() relies on when it doesn't already
    # know a workspace id (e.g. a fresh browser/webview profile) -- it must return every
    # workspace, not 405, or a user with real data but no remembered id can never find it again.
    client = Client()
    resp = client.get("/api/workspaces", **auth_header())
    assert resp.status_code == 200
    assert resp.json() == []

    created = client.post(
        "/api/workspaces",
        data=json.dumps({"name": "Personal"}),
        content_type="application/json",
        **auth_header(),
    ).json()

    resp = client.get("/api/workspaces", **auth_header())
    assert resp.status_code == 200
    listing = resp.json()
    assert len(listing) == 1
    assert listing[0]["id"] == created["id"]
    assert listing[0]["name"] == "Personal"


@pytest.mark.django_db
def test_django_page_content_update_with_plain_text():
    client = Client()
    ws = client.post("/api/workspaces", data=json.dumps({"name": "Personal"}), content_type="application/json", **auth_header()).json()
    nb = client.post("/api/notebooks", data=json.dumps({"workspace_id": ws["id"], "name": "Notas"}), content_type="application/json", **auth_header()).json()
    page = client.post("/api/pages", data=json.dumps({"notebook_id": nb["id"], "title": "Nota"}), content_type="application/json", **auth_header()).json()

    update = client.put(
        f"/api/pages/{page['id']}",
        data=json.dumps({
            "title": "Nota actualizada",
            "content_json": {"type": "doc", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Hola base de conocimiento"}]}]},
            "plain_text": "Hola base de conocimiento",
        }),
        content_type="application/json",
        **auth_header(),
    )
    assert update.status_code == 200
    assert update.json()["plain_text"] == "Hola base de conocimiento"


@pytest.mark.django_db
def test_django_same_document_link_to_multiple_pages():
    # Attachment is notebook-scoped now: linking the same document via two pages of the SAME
    # notebook resolves to a single notebook_documents row, so only the first link is newly
    # created (201) — the second is a no-op re-link (200), not a second independent link.
    client = Client()
    ws = client.post("/api/workspaces", data=json.dumps({"name": "Personal"}), content_type="application/json", **auth_header()).json()
    nb = client.post("/api/notebooks", data=json.dumps({"workspace_id": ws["id"], "name": "Notas"}), content_type="application/json", **auth_header()).json()
    first = client.post("/api/pages", data=json.dumps({"notebook_id": nb["id"], "title": "A"}), content_type="application/json", **auth_header()).json()
    second = client.post("/api/pages", data=json.dumps({"notebook_id": nb["id"], "title": "B"}), content_type="application/json", **auth_header()).json()

    attached_one = client.post(f"/api/pages/{first['id']}/documents/doc-shared", **auth_header())
    attached_two = client.post(f"/api/pages/{second['id']}/documents/doc-shared", **auth_header())

    assert attached_one.status_code == 201
    assert attached_two.status_code == 200
    assert client.get(f"/api/pages/{first['id']}/documents", **auth_header()).json()[0]["id"] == "doc-shared"
    assert client.get(f"/api/pages/{second['id']}/documents", **auth_header()).json()[0]["id"] == "doc-shared"
    assert client.post(f"/api/pages/{first['id']}/documents/doc-shared", **auth_header()).status_code == 200


@pytest.mark.django_db
def test_django_upload_attachment_and_deduplication():
    client = Client()
    ws = client.post("/api/workspaces", data=json.dumps({"name": "Personal"}), content_type="application/json", **auth_header()).json()
    nb = client.post("/api/notebooks", data=json.dumps({"workspace_id": ws["id"], "name": "Notas"}), content_type="application/json", **auth_header()).json()
    p1 = client.post("/api/pages", data=json.dumps({"notebook_id": nb["id"], "title": "A"}), content_type="application/json", **auth_header()).json()
    p2 = client.post("/api/pages", data=json.dumps({"notebook_id": nb["id"], "title": "B"}), content_type="application/json", **auth_header()).json()

    file_data1 = io.BytesIO(b"Usar Redis para tareas asincronas")
    file_data1.name = "decision.txt"

    resp1 = client.post(f"/api/pages/{p1['id']}/attachments", {"file": file_data1}, **auth_header())
    assert resp1.status_code == 201
    data1 = resp1.json()
    assert data1["linked"] is True
    assert data1["reused"] is False

    file_data2 = io.BytesIO(b"Usar Redis para tareas asincronas")
    file_data2.name = "decision.txt"

    resp2 = client.post(f"/api/pages/{p2['id']}/attachments", {"file": file_data2}, **auth_header())
    assert resp2.status_code == 201
    data2 = resp2.json()
    assert data2["linked"] is True
    assert data2["reused"] is True
    assert data2["document"]["id"] == data1["document"]["id"]


@pytest.mark.django_db
def test_django_reupload_after_failed_ingest_retries_instead_of_linking_a_dead_document(tmp_path):
    # Reproduces a real report: uploading a video with no OpenRouter key configured fails ingest
    # (which deletes the uploaded file and leaves the document row status='failed'), and
    # re-uploading the exact same file afterwards -- e.g. once the user sets a key -- used to hit
    # the content-hash "reused" shortcut unconditionally. That shortcut only checked the hash, not
    # status or whether the file still existed, so it linked the page to the same dead document
    # without ever retrying ingestion: the attachment showed up in the sidebar but its preview
    # 404'd forever, since /api/documents/<id>/file has nothing on disk to serve.
    #
    # The failed document is seeded directly via the Django ORM rather than by driving a real
    # failing upload through the endpoint: Ingestor writes through its own raw sqlite3 connection,
    # which -- inside this test's wrapping transaction -- Document.objects can't see anyway, so a
    # first failed request here would never reach the buggy shortcut either way and the test would
    # pass regardless of the fix. In the running app, with no such open transaction spanning
    # requests, that write is exactly what the second request's ORM query does see.
    client = Client()
    ws = client.post("/api/workspaces", data=json.dumps({"name": "Personal"}), content_type="application/json", **auth_header()).json()
    nb = client.post("/api/notebooks", data=json.dumps({"workspace_id": ws["id"], "name": "Notas"}), content_type="application/json", **auth_header()).json()
    page = client.post("/api/pages", data=json.dumps({"notebook_id": nb["id"], "title": "A"}), content_type="application/json", **auth_header()).json()

    content = b"contenido identico que ya fallo una vez y se reintenta despues"
    digest = hashlib.sha256(content).hexdigest()
    Document.objects.create(
        id="stale-failed-doc",
        source_path=str(tmp_path / "deleted-by-the-failed-attempt" / "nota.txt"),
        original_filename="nota.txt",
        media_type="text",
        content_hash=digest,
        byte_size=len(content),
        status="failed",
        error_message="Configura tu OpenRouter API key en Ajustes o en OPENROUTER_API_KEY.",
    )

    file = io.BytesIO(content)
    file.name = "nota.txt"
    resp = client.post(f"/api/pages/{page['id']}/attachments", {"file": file}, **auth_header())
    assert resp.status_code == 201
    data = resp.json()
    # Must actually retry ingestion, not silently link the still-failed row from before.
    assert data["reused"] is False
    assert data["document"]["status"] == "indexed"
    assert data["document"]["id"] != "stale-failed-doc"

    file_resp = client.get(f"/api/documents/{data['document']['id']}/file", **auth_header())
    assert file_resp.status_code == 200


@pytest.mark.django_db
def test_django_upload_survives_connectivity_failure_as_unindexed_and_can_be_retried():
    # A user reported losing an upload attempt entirely (file deleted, hard 400 Bad Request)
    # whenever there was no internet or no OpenRouter key configured -- not video-specific, every
    # media type goes through the same embedding step. A connectivity failure should now come
    # back as a normal 201 with the document marked 'unindexed', file kept on disk and servable,
    # and retriable later via POST /api/documents/<id> without re-uploading anything.
    client = Client()
    ws = client.post("/api/workspaces", data=json.dumps({"name": "Personal"}), content_type="application/json", **auth_header()).json()
    nb = client.post("/api/notebooks", data=json.dumps({"workspace_id": ws["id"], "name": "Notas"}), content_type="application/json", **auth_header()).json()
    page = client.post("/api/pages", data=json.dumps({"notebook_id": nb["id"], "title": "A"}), content_type="application/json", **auth_header()).json()

    from knowledge import views
    rag = views.get_rag_service()
    working_provider = rag.ingestor.provider

    class _NoConnectionProvider:
        async def embed_texts(self, texts):
            raise EmbeddingError("Configura tu OpenRouter API key en Ajustes o en OPENROUTER_API_KEY.")

    rag.ingestor.provider = _NoConnectionProvider()
    try:
        file = io.BytesIO(b"contenido subido sin conexion a internet")
        file.name = "sin_red.txt"
        resp = client.post(f"/api/pages/{page['id']}/attachments", {"file": file}, **auth_header())
    finally:
        rag.ingestor.provider = working_provider

    # Soft outcome: still a success response, not the 400 the file-deleting hard-failure path uses.
    assert resp.status_code == 201
    data = resp.json()["document"]
    assert data["status"] == "unindexed"

    # The file must have been kept, not deleted -- servable right away even though unindexed.
    file_resp = client.get(f"/api/documents/{data['id']}/file", **auth_header())
    assert file_resp.status_code == 200

    # Retrying (connectivity "back") via the reindex endpoint must succeed without re-uploading.
    retry_resp = client.post(f"/api/documents/{data['id']}", **auth_header())
    assert retry_resp.status_code == 200
    # doc.refresh_from_db() (inside the view) reads via Django's own DB connection, which --
    # only inside this test's wrapping transaction -- can't see a write just committed through
    # the raw sqlite connection Ingestor uses (same cross-connection visibility gap documented
    # in test_django_reupload_after_failed_ingest_retries_instead_of_linking_a_dead_document
    # above; a real request has no such open transaction and doesn't hit this). Verify success
    # via that same raw connection instead, which reflects its own write immediately.
    row = rag.connection.execute("SELECT status FROM documents WHERE id = ?", (data["id"],)).fetchone()
    assert row["status"] == "indexed"


@pytest.mark.django_db
def test_django_agent_skills_and_memory_persistence():
    client = Client()
    # 1. Create Memory
    m_resp = client.post(
        "/api/memories",
        data=json.dumps({"content": "El usuario prefiere respuestas concisas en español y código tipado.", "category": "user_preference"}),
        content_type="application/json",
        **auth_header(),
    )
    assert m_resp.status_code == 201
    mem_id = m_resp.json()["id"]

    # 2. List Memories
    list_m = client.get("/api/memories", **auth_header())
    assert list_m.status_code == 200
    assert any(m["id"] == mem_id for m in list_m.json())

    # 3. Create Skill
    s_resp = client.post(
        "/api/skills",
        data=json.dumps({
            "name": "resumir-reuniones",
            "description": "Plantilla para minutas técnicas",
            "instructions": "Estructurar en: 1) Asistentes, 2) Decisiones clave, 3) Próximos pasos con fecha.",
            "category": "productivity",
        }),
        content_type="application/json",
        **auth_header(),
    )
    assert s_resp.status_code == 201
    skill_id = s_resp.json()["id"]

    # 4. List Skills
    list_s = client.get("/api/skills", **auth_header())
    assert list_s.status_code == 200
    assert any(s["id"] == skill_id for s in list_s.json())

    # 5. Delete Memory and Skill
    del_m = client.delete(f"/api/memories/{mem_id}", **auth_header())
    assert del_m.status_code == 200
    del_s = client.delete(f"/api/skills/{skill_id}", **auth_header())
    assert del_s.status_code == 200


@pytest.mark.django_db
def test_django_inline_ai_actions():
    client = Client()
    resp = client.post(
        "/api/ai/inline-action",
        data=json.dumps({
            "action": "summarize",
            "context_text": "El sistema RAGPoC fue migrado exitosamente de FastAPI a Django. Se utiliza SQLite con sqlite-vec para almacenar embeddings y k-NN vectorial. Tiptap es el editor central.",
        }),
        content_type="application/json",
        **auth_header(),
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert len(resp.json()["content"]) > 0


@pytest.mark.django_db
def test_django_inline_ai_actions_stream():
    client = Client()
    resp = client.post(
        "/api/ai/inline-action-stream",
        data=json.dumps({
            "action": "summarize",
            "context_text": "El sistema RAGPoC fue migrado exitosamente de FastAPI a Django. Se utiliza SQLite con sqlite-vec para almacenar embeddings y k-NN vectorial. Tiptap es el editor central.",
        }),
        content_type="application/json",
        **auth_header(),
    )
    assert resp.status_code == 200
    assert resp["Content-Type"] == "text/event-stream"
    events = [
        json.loads(line[6:])
        for line in get_stream_text(resp).split("\n\n")
        if line.startswith("data: ")
    ]
    assert events[-1]["type"] == "done"
    full_text = "".join(e["text"] for e in events if e["type"] == "token")
    assert len(full_text) > 0


@pytest.mark.django_db
def test_django_omni_search_and_artifacts():
    client = Client()
    ws = client.post("/api/workspaces", data=json.dumps({"name": "OmniSpace"}), content_type="application/json", **auth_header()).json()
    nb = client.post("/api/notebooks", data=json.dumps({"workspace_id": ws["id"], "name": "Algoritmos"}), content_type="application/json", **auth_header()).json()
    client.post("/api/pages", data=json.dumps({"notebook_id": nb["id"], "title": "Búsqueda Binaria"}), content_type="application/json", **auth_header()).json()

    # Omni search
    res = client.get("/api/omni-search?q=Binaria", **auth_header()).json()
    assert len(res["results"]) >= 1
    assert res["results"][0]["title"] == "Búsqueda Binaria"
    assert res["results"][0]["type"] == "page"

    # Search notebook
    res_nb = client.get("/api/omni-search?q=Algoritmos", **auth_header()).json()
    assert len(res_nb["results"]) >= 1
    assert res_nb["results"][0]["type"] == "notebook"


@pytest.mark.django_db
def test_django_rename_and_delete_document_source():
    client = Client()
    ws = client.post("/api/workspaces", data=json.dumps({"name": "Personal"}), content_type="application/json", **auth_header()).json()
    nb = client.post("/api/notebooks", data=json.dumps({"workspace_id": ws["id"], "name": "Notas"}), content_type="application/json", **auth_header()).json()
    p1 = client.post("/api/pages", data=json.dumps({"notebook_id": nb["id"], "title": "A"}), content_type="application/json", **auth_header()).json()

    # Add source
    resp = client.post(
        "/api/sources",
        data=json.dumps({
            "source_type": "text",
            "title": "Nombre Original",
            "content": "Contenido para probar renombrado y borrado.",
            "scope": "page",
            "page_id": p1["id"],
        }),
        content_type="application/json",
        **auth_header(),
    )
    assert resp.status_code == 201
    doc_id = resp.json()["id"]

    # Rename Document
    patch_resp = client.patch(
        f"/api/documents/{doc_id}",
        data=json.dumps({"filename": "Nombre Actualizado.txt"}),
        content_type="application/json",
        **auth_header(),
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["filename"] == "Nombre Actualizado.txt"

    # Delete Document
    del_resp = client.delete(f"/api/documents/{doc_id}", **auth_header())
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # Verify not in sources list
    sources_after = client.get(f"/api/sources?page_id={p1['id']}", **auth_header()).json()
    assert not any(s["id"] == doc_id for s in sources_after)


@pytest.mark.django_db
def test_django_delete_page_and_notebook():
    client = Client()
    ws = client.post("/api/workspaces", data=json.dumps({"name": "Personal"}), content_type="application/json", **auth_header()).json()
    nb = client.post("/api/notebooks", data=json.dumps({"workspace_id": ws["id"], "name": "Notas"}), content_type="application/json", **auth_header()).json()
    p1 = client.post("/api/pages", data=json.dumps({"notebook_id": nb["id"], "title": "Para Borrar"}), content_type="application/json", **auth_header()).json()

    # Delete Page
    del_page_resp = client.delete(f"/api/pages/{p1['id']}", **auth_header())
    assert del_page_resp.status_code == 200
    assert del_page_resp.json()["status"] == "deleted"
    assert client.get(f"/api/pages/{p1['id']}", **auth_header()).status_code == 404

    # Delete Notebook
    del_nb_resp = client.delete(f"/api/notebooks/{nb['id']}", **auth_header())
    assert del_nb_resp.status_code == 200
    assert del_nb_resp.json()["status"] == "deleted"


@pytest.mark.django_db
def test_django_scoped_sources_api():
    # Sources are notebook-only now: no more scope/level tiers, and each notebook's sources stay
    # isolated from every other notebook — even within the same workspace.
    client = Client()
    ws = client.post("/api/workspaces", data=json.dumps({"name": "Personal"}), content_type="application/json", **auth_header()).json()
    nb1 = client.post("/api/notebooks", data=json.dumps({"workspace_id": ws["id"], "name": "Notas"}), content_type="application/json", **auth_header()).json()
    nb2 = client.post("/api/notebooks", data=json.dumps({"workspace_id": ws["id"], "name": "Otro cuaderno"}), content_type="application/json", **auth_header()).json()
    p1 = client.post("/api/pages", data=json.dumps({"notebook_id": nb1["id"], "title": "A"}), content_type="application/json", **auth_header()).json()

    # 1. Add a text source directly to the notebook
    resp = client.post(
        "/api/sources",
        data=json.dumps({
            "source_type": "text",
            "title": "Arquitectura Backend",
            "content": "Lineamientos de arquitectura Django + sqlite-vec para el proyecto.",
            "notebook_id": nb1["id"],
        }),
        content_type="application/json",
        **auth_header(),
    )
    assert resp.status_code == 201
    source_data = resp.json()
    assert source_data["notebook_id"] == nb1["id"]

    # 2. Listing via page_id resolves to that page's own notebook
    sources_resp = client.get(f"/api/sources?page_id={p1['id']}", **auth_header())
    assert sources_resp.status_code == 200
    sources = sources_resp.json()
    assert any(s["filename"] == "Arquitectura_Backend.txt" for s in sources)

    # 3. A file source uploaded via page_id fallback also resolves to that page's notebook
    file_nb_content = b"# Documento Cuaderno\n\nGuia especifica del cuaderno."
    file_nb_obj = io.BytesIO(file_nb_content)
    file_nb_obj.name = "guia_cuaderno.txt"
    file_nb_resp = client.post(
        "/api/sources",
        data={"page_id": p1["id"], "file": file_nb_obj},
        **auth_header(),
    )
    assert file_nb_resp.status_code == 201
    file_nb_data = file_nb_resp.json()
    assert file_nb_data["filename"] == "guia_cuaderno.txt"
    assert file_nb_data["notebook_id"] == nb1["id"]

    # 4. Listing directly by notebook_id returns the same sources
    sources_by_nb = client.get(f"/api/sources?notebook_id={nb1['id']}", **auth_header()).json()
    assert any(s["filename"] == "Arquitectura_Backend.txt" for s in sources_by_nb)
    assert any(s["filename"] == "guia_cuaderno.txt" for s in sources_by_nb)

    # 5. A different notebook in the same workspace sees none of them — sources don't leak
    sources_other_nb = client.get(f"/api/sources?notebook_id={nb2['id']}", **auth_header()).json()
    assert sources_other_nb == []


@pytest.mark.django_db
def test_django_chat_threads_and_messages_persistence():
    client = Client()
    ws = client.post("/api/workspaces", data=json.dumps({"name": "Personal"}), content_type="application/json", **auth_header()).json()
    nb = client.post("/api/notebooks", data=json.dumps({"workspace_id": ws["id"], "name": "Notas"}), content_type="application/json", **auth_header()).json()
    p1 = client.post("/api/pages", data=json.dumps({"notebook_id": nb["id"], "title": "A"}), content_type="application/json", **auth_header()).json()

    # Create thread
    thread_resp = client.post(
        "/api/threads",
        data=json.dumps({"title": "Hilo de Investigación", "scope": "page", "page_id": p1["id"]}),
        content_type="application/json",
        **auth_header(),
    )
    assert thread_resp.status_code == 201
    thread = thread_resp.json()
    assert thread["title"] == "Hilo de Investigación"

    # List threads
    threads_list = client.get(f"/api/threads?page_id={p1['id']}", **auth_header()).json()
    assert len(threads_list) >= 1
    assert threads_list[0]["id"] == thread["id"]

    # Stream a chat message into the thread
    resp = client.post(
        "/chat/stream",
        data=json.dumps({
            "question": "Pregunta de prueba en hilo",
            "scope": "page",
            "page_id": p1["id"],
            "thread_id": thread["id"],
            "attachments": [{"name": "extra.txt", "text": "Informacion extra"}],
        }),
        content_type="application/json",
        **auth_header(),
    )
    assert resp.status_code == 200
    content = get_stream_text(resp)
    assert "thread_init" in content

    # Check messages in thread
    msgs_resp = client.get(f"/api/threads/{thread['id']}/messages", **auth_header())
    assert msgs_resp.status_code == 200
    msgs = msgs_resp.json()["messages"]
    assert len(msgs) >= 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Pregunta de prueba en hilo"


@pytest.mark.django_db
def test_django_serve_document_file_endpoint():
    client = Client()
    ws = client.post("/api/workspaces", data=json.dumps({"name": "Personal"}), content_type="application/json", **auth_header()).json()
    nb = client.post("/api/notebooks", data=json.dumps({"workspace_id": ws["id"], "name": "Notas"}), content_type="application/json", **auth_header()).json()
    p1 = client.post("/api/pages", data=json.dumps({"notebook_id": nb["id"], "title": "A"}), content_type="application/json", **auth_header()).json()

    file_data = io.BytesIO(b"Document content for direct serving test")
    file_data.name = "sample_doc.txt"
    resp = client.post(f"/api/pages/{p1['id']}/attachments", {"file": file_data}, **auth_header())
    assert resp.status_code == 201
    doc = resp.json()["document"]

    file_resp = client.get(f"/api/documents/{doc['id']}/file", **auth_header())
    assert file_resp.status_code == 200
    assert file_resp.content == b"Document content for direct serving test"
    assert file_resp["Content-Type"].startswith("text/plain")


@pytest.mark.django_db
def test_django_chat_stream_and_scopes():
    client = Client()
    ws = client.post("/api/workspaces", data=json.dumps({"name": "Personal"}), content_type="application/json", **auth_header()).json()
    nb = client.post("/api/notebooks", data=json.dumps({"workspace_id": ws["id"], "name": "Notas"}), content_type="application/json", **auth_header()).json()
    p1 = client.post("/api/pages", data=json.dumps({"notebook_id": nb["id"], "title": "A"}), content_type="application/json", **auth_header()).json()

    file_data = io.BytesIO(b"Remotion se puede usar con Claude Code para automatizacion de video.")
    file_data.name = "remotion.txt"
    client.post(f"/api/pages/{p1['id']}/attachments", {"file": file_data}, **auth_header())

    resp = client.post(
        "/chat/stream",
        data=json.dumps({"question": "Remotion", "scope": "page", "page_id": p1["id"]}),
        content_type="application/json",
        **auth_header(),
    )
    assert resp.status_code == 200
    content = get_stream_text(resp)
    assert "data: " in content


@pytest.mark.django_db
def test_settings_no_longer_expose_cross_scope_promotion():
    """Cross-scope promotion (moving sources between page/notebook/workspace) was removed
    entirely once sources became notebook-only — verify it's actually gone from the settings
    contract, not just hidden, and that posting the old field is a harmless no-op."""
    client = Client()
    res_get = client.get("/api/settings", **auth_header()).json()
    assert "allow_cross_scope_promotion" not in res_get

    res_post = client.post(
        "/api/settings",
        data=json.dumps({"allow_cross_scope_promotion": True}),
        content_type="application/json",
        **auth_header(),
    )
    assert res_post.status_code == 200
    assert "allow_cross_scope_promotion" not in res_post.json()


@pytest.mark.django_db
@pytest.mark.asyncio
async def test_agent_notebook_scoped_tools():
    """promote_or_link_source (moving a source between page/notebook/workspace) was removed
    entirely once sources became notebook-only. Verify the tool is actually gone, and that
    add_source_to_knowledge_base — the tool that replaces it for the agent — attaches new
    sources to the current notebook without any scope parameter."""
    from knowledge import views
    from knowledge.models import Document, Notebook, NotebookDocument, Page, Workspace
    from knowledge.pydantic_agent import AgentDeps, create_pydantic_rag_agent
    from ragpoc.config import get_settings

    rag = views.get_rag_service()
    settings = get_settings()

    ws = await Workspace.objects.acreate(name="WS Notebook Only")
    nb = await Notebook.objects.acreate(workspace=ws, name="NB Notebook Only")
    p = await Page.objects.acreate(notebook=nb, title="Pagina con Glosario", plain_text="Definiciones...")

    agent = create_pydantic_rag_agent(settings)
    assert agent._function_toolset.tools.get("promote_or_link_source") is None

    add_tool = agent._function_toolset.tools.get("add_source_to_knowledge_base")
    assert add_tool is not None

    deps = AgentDeps(retriever=rag.retriever, settings=settings, page_id=p.id, notebook_id=nb.id, workspace_id=ws.id)
    import types
    ctx = types.SimpleNamespace(deps=deps)
    res = await add_tool.function(ctx, source_type="text", title_or_url="Nota agente", content="Contenido de prueba del agente.")
    assert res["status"] == "success"
    assert res["notebook"] == nb.name

    doc = await Document.objects.aget(id=res["document_id"])
    linked = await NotebookDocument.objects.filter(notebook=nb, document=doc).aexists()
    assert linked is True


@pytest.mark.django_db
@pytest.mark.asyncio
async def test_agent_page_write_tools_prepare_target_without_content_argument():
    # create_workspace_page/update_page_notes no longer take the page content as an argument
    # (see the "stream the agent's page writes live" fix): a tool-call argument only reaches
    # Python once the model has finished generating it, so there was no way to stream it
    # token-by-token into the page. Now they just set deps.page_write_state with a target page
    # id -- the actual content arrives separately, as the model's next normal text-streaming
    # step in chat_stream_view, which mirrors those tokens into the page as they arrive.
    import inspect
    import types

    from knowledge import views
    from knowledge.models import Notebook, Page, Workspace
    from knowledge.pydantic_agent import AgentDeps, create_pydantic_rag_agent
    from ragpoc.config import get_settings

    rag = views.get_rag_service()
    settings = get_settings()

    ws = await Workspace.objects.acreate(name="WS Page Write")
    nb = await Notebook.objects.acreate(workspace=ws, name="NB Page Write")
    existing_page = await Page.objects.acreate(notebook=nb, title="Nota existente", plain_text="Contenido viejo.")

    agent = create_pydantic_rag_agent(settings)
    create_tool = agent._function_toolset.tools.get("create_workspace_page")
    update_tool = agent._function_toolset.tools.get("update_page_notes")
    assert create_tool is not None
    assert update_tool is not None
    assert "content" not in inspect.signature(create_tool.function).parameters
    assert "content_to_append" not in inspect.signature(update_tool.function).parameters

    deps = AgentDeps(retriever=rag.retriever, settings=settings, page_id=existing_page.id, notebook_id=nb.id, workspace_id=ws.id)
    ctx = types.SimpleNamespace(deps=deps)

    result = await create_tool.function(ctx, title="Nota nueva del agente", notebook_id=nb.id)
    assert result["status"] == "ready_for_content"
    assert deps.page_write_state == {
        "page_id": result["page_id"],
        "notebook_id": nb.id,
        "title": result["title"],
        "mode": "create",
    }
    new_page = await Page.objects.aget(id=result["page_id"])
    assert new_page.plain_text == ""  # content arrives later, via the token stream

    deps.page_write_state = None
    result2 = await update_tool.function(ctx, page_id=existing_page.id)
    assert result2["status"] == "ready_for_content"
    assert deps.page_write_state["mode"] == "append"
    assert deps.page_write_state["page_id"] == existing_page.id
    unchanged = await Page.objects.aget(id=existing_page.id)
    assert unchanged.plain_text == "Contenido viejo."  # not touched yet either

