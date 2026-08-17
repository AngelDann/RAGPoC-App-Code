import io
import json
from base64 import b64encode

import pytest
from django.core.management import call_command
from django.test import Client

from knowledge.services import get_rag_service
from ragpoc.config import Settings
from ragpoc.embeddings import FakeEmbeddingProvider


def auth_header() -> dict[str, str]:
    return {"HTTP_AUTHORIZATION": "Basic " + b64encode(b"operator:test-password").decode()}


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
    content = b"".join(resp.streaming_content).decode("utf-8")
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
    content = b"".join(resp.streaming_content).decode("utf-8")
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

