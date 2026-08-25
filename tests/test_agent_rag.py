import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from asgiref.sync import sync_to_async
from django.core.management import call_command
from django.db import connections
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel

from knowledge.models import Notebook, NotebookArtifact, Workspace
from knowledge.pydantic_agent import AgentDeps, create_pydantic_rag_agent
from knowledge.services import get_rag_service
from ragpoc.config import Settings
from ragpoc.embeddings import FakeEmbeddingProvider
from ragpoc.retrieval import Retriever


@pytest.fixture(autouse=True)
def setup_test_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir = data_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    test_settings = Settings(
        data_dir=data_dir,
        allowed_upload_dir=uploads_dir,
        ui_password="test-password",
        openrouter_api_key="test-key",
    )
    from ragpoc import config
    monkeypatch.setattr(config, "get_settings", lambda: test_settings)

    rag_service = get_rag_service(test_settings, FakeEmbeddingProvider())
    from knowledge import views
    monkeypatch.setattr(views, "get_rag_service", lambda: rag_service)

    call_command("migrate", verbosity=0)
    yield
    connections.close_all()


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_agentic_rag_search_invocation_and_events():
    """Verify that when the agent invokes search_knowledge_base:
    1. The retriever is queried with the right parameters and scope.
    2. The 'sources' event is emitted live via on_tool_event.
    3. Retrieved sources are collected in deps.collected_sources.
    """
    mock_retriever = MagicMock(spec=Retriever)
    mock_retriever.search = AsyncMock(return_value=[
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "filename": "manual_usuario.pdf",
            "media_type": "pdf",
            "page_number": 3,
            "text": "Contenido del manual sobre configuración.",
            "derived_path": None,
            "source_path": None,
            "metadata": {},
        },
        {
            "chunk_id": "c2",
            "document_id": "d2",
            "filename": "notas_reunion.txt",
            "media_type": "text",
            "page_number": None,
            "text": "Acuerdos del equipo para el sprint.",
            "derived_path": None,
            "source_path": None,
            "metadata": {},
        },
    ])

    emitted_events = []
    deps = AgentDeps(
        retriever=mock_retriever,
        settings=Settings(),
        notebook_id="nb-123",
        on_tool_event=lambda evt: emitted_events.append(evt),
    )

    agent = create_pydantic_rag_agent(Settings())
    agent.model = TestModel(call_tools=["search_knowledge_base"])

    result = await agent.run("Consulta el manual del usuario", deps=deps)
    assert result is not None

    # 1. Verify retriever was called
    assert mock_retriever.search.await_count >= 1
    call_args = mock_retriever.search.call_args
    assert call_args.kwargs.get("notebook_id") == "nb-123"

    # 2. Verify sources event was emitted live to on_tool_event
    assert len(emitted_events) >= 1
    sources_event = next(e for e in emitted_events if e.get("type") == "sources")
    assert len(sources_event["sources"]) == 2
    assert sources_event["sources"][0]["label"] == "manual_usuario.pdf"
    assert sources_event["sources"][0]["citation"] == "[1]"

    # 3. Verify deps.collected_sources accumulated the sources
    assert len(deps.collected_sources) == 2
    assert deps.collected_sources[0]["filename"] == "manual_usuario.pdf"
    assert deps.collected_sources[1]["filename"] == "notas_reunion.txt"


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_conversational_query_does_not_invoke_search_tool():
    """Verify that pure conversational queries don't trigger vector search tool."""
    mock_retriever = MagicMock(spec=Retriever)
    mock_retriever.search = AsyncMock()

    emitted_events = []
    deps = AgentDeps(
        retriever=mock_retriever,
        settings=Settings(),
        notebook_id="nb-123",
        on_tool_event=lambda evt: emitted_events.append(evt),
    )

    agent = create_pydantic_rag_agent(Settings())
    agent.model = TestModel(call_tools=[])

    result = await agent.run("¡Hola! ¿Cómo estás?", deps=deps)
    assert result is not None

    # Retriever must not have been called
    assert mock_retriever.search.await_count == 0
    assert len(emitted_events) == 0
    assert len(deps.collected_sources) == 0


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_generate_notebook_artifact_tool_direct_and_events():
    """Verify that generate_notebook_artifact tool generates artifacts and emits SSE events."""
    ws = await sync_to_async(Workspace.objects.create)(name="Workspace Test")
    nb = await sync_to_async(Notebook.objects.create)(workspace=ws, name="Arquitectura")

    mock_retriever = MagicMock(spec=Retriever)
    mock_retriever.search = AsyncMock(return_value=[])

    emitted_events = []
    deps = AgentDeps(
        retriever=mock_retriever,
        settings=Settings(openrouter_api_key="test-key"),
        notebook_id=str(nb.id),
        on_tool_event=lambda evt: emitted_events.append(evt),
    )

    agent = create_pydantic_rag_agent(deps.settings)
    tool_def = agent._function_toolset.tools["generate_notebook_artifact"]

    ctx = RunContext(
        deps=deps,
        model=TestModel(),
        usage=MagicMock(),
        prompt="Genera un diagrama",
    )

    mock_choice = MagicMock()
    mock_choice.message.content = "```mermaid\ngraph TD; A-->B;\n```"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with patch("openai.resources.chat.AsyncCompletions.create", AsyncMock(return_value=mock_completion)):
        res = await tool_def.function(
            ctx,
            artifact_type="diagram",
            instructions="Diagrama de flujo principal",
        )

    assert res["status"] == "success"
    assert "Diagrama" in res["title"]

    # Check NotebookArtifact in DB
    artifact = await sync_to_async(lambda: NotebookArtifact.objects.filter(notebook=nb).first())()
    assert artifact is not None
    assert artifact.artifact_type == "diagram"
    assert "Arquitectura" in artifact.title

    # Check SSE event emitted
    assert len(emitted_events) >= 1
    art_event = next(e for e in emitted_events if e.get("type") == "artifact_created")
    assert art_event["artifact_id"] == str(artifact.id)
    assert art_event["artifact_type"] == "diagram"


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_extract_clean_text_from_html_helper():
    """Verify that extract_clean_text_from_html strips scripts, styles, and tags while preserving readable text."""
    from knowledge.pydantic_agent import extract_clean_text_from_html

    sample_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Test Page</title><style>body { color: red; }</style></head>
    <body>
      <nav><a href="#home">Home</a></nav>
      <script>console.log("secret tracker");</script>
      <h1>Documentación de Python 3.13</h1>
      <p>Python 3.13 incluye un <strong>GIL opcional</strong> y mejoras en el compilador JIT.</p>
      <footer>Pie de página</footer>
    </body>
    </html>
    """
    cleaned = extract_clean_text_from_html(sample_html)
    assert "Documentación de Python 3.13" in cleaned
    assert "GIL opcional" in cleaned
    assert "secret tracker" not in cleaned
    assert "color: red" not in cleaned


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_add_source_to_knowledge_base_text_and_event():
    """Verify that add_source_to_knowledge_base indexes text and emits source_added SSE event."""
    from knowledge.models import Document, Notebook, NotebookDocument, Workspace

    ws = await sync_to_async(Workspace.objects.create)(name="Workspace Test Sources")
    nb = await sync_to_async(Notebook.objects.create)(workspace=ws, name="Cuaderno RAG")

    mock_retriever = MagicMock(spec=Retriever)
    emitted_events = []
    deps = AgentDeps(
        retriever=mock_retriever,
        settings=Settings(openrouter_api_key="test-key"),
        notebook_id=str(nb.id),
        on_tool_event=lambda evt: emitted_events.append(evt),
    )

    agent = create_pydantic_rag_agent(deps.settings)
    tool_def = agent._function_toolset.tools["add_source_to_knowledge_base"]

    ctx = RunContext(
        deps=deps,
        model=TestModel(),
        usage=MagicMock(),
        prompt="Indexa esta nota",
    )

    res = await tool_def.function(
        ctx,
        source_type="text",
        title_or_url="Nota sobre Arquitectura",
        content="La arquitectura se basa en microservicios desacoplados con FastAPI y SQLite.",
    )

    assert res["status"] == "success"
    assert "Nota sobre Arquitectura.txt" in res["filename"]
    assert res["notebook"] == "Cuaderno RAG"

    # Check Document in DB
    doc = await sync_to_async(lambda: Document.objects.filter(id=res["document_id"]).first())()
    assert doc is not None
    assert doc.original_filename == "Nota sobre Arquitectura.txt"

    # Check NotebookDocument link
    nb_doc = await sync_to_async(lambda: NotebookDocument.objects.filter(notebook=nb, document=doc).first())()
    assert nb_doc is not None

    # Check SSE event emitted
    assert any(e.get("type") == "source_added" and e.get("filename") == "Nota sobre Arquitectura.txt" for e in emitted_events)


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_fetch_web_page_tool():
    """Verify that fetch_web_page tool fetches and cleans web page text."""
    emitted_events = []
    deps = AgentDeps(
        retriever=MagicMock(spec=Retriever),
        settings=Settings(openrouter_api_key="test-key"),
        on_tool_event=lambda evt: emitted_events.append(evt),
    )

    agent = create_pydantic_rag_agent(deps.settings)
    tool_def = agent._function_toolset.tools["fetch_web_page"]

    ctx = RunContext(
        deps=deps,
        model=TestModel(),
        usage=MagicMock(),
        prompt="Lee la página",
    )

    fake_html = b"<html><body><h1>FastAPI Framework</h1><p>FastAPI es un framework moderno y rapido.</p></body></html>"

    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return fake_html

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        res = await tool_def.function(ctx, url="https://fastapi.tiangolo.com")

    assert res["status"] == "success"
    assert "FastAPI Framework" in res["content_preview"]
    assert "FastAPI es un framework moderno" in res["content_preview"]


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_agent_deps_tool_tracing_and_events():
    """Verify that tool invocations emit structured tool_start and tool_end events and record executed_tools."""
    emitted_events = []
    deps = AgentDeps(
        retriever=MagicMock(spec=Retriever),
        settings=Settings(openrouter_api_key="test-key"),
        on_tool_event=lambda evt: emitted_events.append(evt),
    )

    agent = create_pydantic_rag_agent(deps.settings)
    tool_def = agent._function_toolset.tools["manage_memory"]

    ctx = RunContext(
        deps=deps,
        model=TestModel(),
        usage=MagicMock(),
        prompt="Guarda en memoria",
    )

    res = await tool_def.function(ctx, action="add", content="El usuario prefiere respuestas concisas.", category="preference")
    assert res["status"] == "success"

    # Check that tool_start and tool_end events were emitted
    assert any(e.get("type") == "tool_start" and e.get("tool") == "manage_memory" for e in emitted_events)
    assert any(e.get("type") == "tool_end" and e.get("tool") == "manage_memory" and e.get("status") == "done" for e in emitted_events)

    # Check that deps.executed_tools contains the step
    assert len(deps.executed_tools) == 1
    assert deps.executed_tools[0]["tool"] == "manage_memory"
    assert deps.executed_tools[0]["status"] == "done"
    assert "duration_ms" in deps.executed_tools[0]


@pytest.mark.django_db(transaction=True)
def test_trim_message_history_observation_masking():
    """Verify that _trim_message_history compacts bulky tool return payloads in past turns."""
    from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart, TextPart
    from knowledge.views import _trim_message_history

    req1 = ModelRequest(parts=[UserPromptPart(content="Explica este tema")])
    call1 = ModelResponse(parts=[ToolCallPart(tool_name="fetch_web_page", args="{}", tool_call_id="call_1")])
    ret1 = ModelRequest(parts=[ToolReturnPart(tool_name="fetch_web_page", content={"content_preview": "A" * 5000, "status": "success"}, tool_call_id="call_1")])
    resp1 = ModelResponse(parts=[TextPart(content="Resumen del tema...")])

    req2 = ModelRequest(parts=[UserPromptPart(content="Agrega otro ejemplo")])

    history = [req1, call1, ret1, resp1, req2]
    compacted = _trim_message_history(history, max_turns=5)

    assert len(compacted) == 5
    tool_ret = compacted[2].parts[0]
    assert isinstance(tool_ret, ToolReturnPart)
    assert len(tool_ret.content["content_preview"]) < 1000
    assert "truncado" in tool_ret.content["content_preview"]


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_past_conversations_notebook_scope_isolation():
    """Verify that search_past_conversations and get_conversation_messages strictly respect notebook scope."""
    from knowledge.models import ChatThread, ChatMessage, Notebook, Workspace

    ws = await sync_to_async(Workspace.objects.create)(name="Espacio Test")
    nb_aws = await sync_to_async(Notebook.objects.create)(workspace=ws, name="AWS ML")
    nb_roma = await sync_to_async(Notebook.objects.create)(workspace=ws, name="Historia Roma")

    # Create thread in Rome notebook
    t_roma = await sync_to_async(ChatThread.objects.create)(
        notebook=nb_roma,
        workspace=ws,
        title="Caída del Imperio Romano",
        scope="notebook",
    )
    await sync_to_async(ChatMessage.objects.create)(
        thread=t_roma,
        role="user",
        content="¿Cuándo cayó el imperio romano?",
    )
    await sync_to_async(ChatMessage.objects.create)(
        thread=t_roma,
        role="assistant",
        content="El imperio romano de occidente cayó en 476 d.C.",
    )

    # Create thread in AWS notebook
    t_aws = await sync_to_async(ChatThread.objects.create)(
        notebook=nb_aws,
        workspace=ws,
        title="AWS Certified Machine Learning",
        scope="notebook",
    )

    deps = AgentDeps(
        retriever=MagicMock(spec=Retriever),
        settings=Settings(openrouter_api_key="test-key"),
        notebook_id=str(nb_aws.id),
        workspace_id=str(ws.id),
        thread_id=str(t_aws.id),
    )

    agent = create_pydantic_rag_agent(deps.settings)
    search_tool = agent._function_toolset.tools["search_past_conversations"]
    get_tool = agent._function_toolset.tools["get_conversation_messages"]

    ctx = RunContext(
        deps=deps,
        model=TestModel(),
        usage=MagicMock(),
        prompt="Consulta conversaciones anteriores",
    )

    # Search without query in notebook scope should NOT find Rome thread
    results = await search_tool.function(ctx, query="")
    assert isinstance(results, list)
    assert not any(r["thread_id"] == str(t_roma.id) for r in results)

    # Attempting to load Rome thread directly under AWS notebook scope must fail with restricted access
    roma_res = await get_tool.function(ctx, thread_id=str(t_roma.id))
    assert "error" in roma_res
    assert "Acceso restringido" in roma_res["error"]


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_search_web_tool_results_and_limits():
    """Verify that search_web tool executes DDGS search with configurable max_results and clamps bounds."""
    emitted_events = []
    deps = AgentDeps(
        retriever=MagicMock(spec=Retriever),
        settings=Settings(openrouter_api_key="test-key"),
        on_tool_event=lambda evt: emitted_events.append(evt),
    )

    agent = create_pydantic_rag_agent(deps.settings)
    search_tool = agent._function_toolset.tools["search_web"]

    ctx = RunContext(
        deps=deps,
        model=TestModel(),
        usage=MagicMock(),
        prompt="Busca en internet",
    )

    fake_ddgs_results = [
        {"title": f"Resultado {i}", "href": f"https://example.com/{i}", "body": f"Snippet {i}"}
        for i in range(1, 11)
    ]

    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = fake_ddgs_results
    mock_ddgs_class = MagicMock()
    mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs_instance

    with patch("knowledge.pydantic_agent.DDGS", mock_ddgs_class):
        # 1. Test default max_results (10)
        res = await search_tool.function(ctx, query="python 3.13 novedades")
        assert len(res) == 10
        mock_ddgs_instance.text.assert_called_with("python 3.13 novedades", max_results=10)

        # 2. Test custom max_results within bounds
        await search_tool.function(ctx, query="django tutorial", max_results=15)
        mock_ddgs_instance.text.assert_called_with("django tutorial", max_results=15)

        # 3. Test clamping above upper limit (25)
        await search_tool.function(ctx, query="machine learning", max_results=100)
        mock_ddgs_instance.text.assert_called_with("machine learning", max_results=25)


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_fetch_web_page_pdf_support():
    """Verify that fetch_web_page cleanly extracts page text from remote PDFs via PyMuPDF."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Atencion es todo lo que necesitas: Arquitectura Transformer.")
    pdf_bytes = doc.tobytes()
    doc.close()

    emitted_events = []
    deps = AgentDeps(
        retriever=MagicMock(spec=Retriever),
        settings=Settings(openrouter_api_key="test-key"),
        on_tool_event=lambda evt: emitted_events.append(evt),
    )

    agent = create_pydantic_rag_agent(deps.settings)
    tool_def = agent._function_toolset.tools["fetch_web_page"]

    ctx = RunContext(
        deps=deps,
        model=TestModel(),
        usage=MagicMock(),
        prompt="Lee el PDF",
    )

    with patch("knowledge.pydantic_agent.fetch_remote_resource", return_value=(pdf_bytes, "application/pdf", "paper.pdf", ".pdf")):
        res = await tool_def.function(ctx, url="https://arxiv.org/pdf/1706.03762.pdf")

    assert res["status"] == "success"
    assert res["media_type"] == "pdf"
    assert res["page_count"] == 1
    assert "Atencion es todo lo que necesitas" in res["content_preview"]
    assert "stream" not in res["content_preview"]
    assert "endobj" not in res["content_preview"]


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_add_source_to_knowledge_base_remote_pdf():
    """Verify that add_source_to_knowledge_base downloads remote PDF as .pdf, runs native PDF ingestor, and links in DB."""
    import pymupdf
    from knowledge.models import Document, Notebook, NotebookDocument, Workspace

    ws = await sync_to_async(Workspace.objects.create)(name="Workspace PDF Test")
    nb = await sync_to_async(Notebook.objects.create)(workspace=ws, name="Cuaderno ML")

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Capacidades multimodales y agentes RAG.")
    pdf_bytes = doc.tobytes()
    doc.close()

    mock_retriever = MagicMock(spec=Retriever)
    emitted_events = []
    deps = AgentDeps(
        retriever=mock_retriever,
        settings=Settings(openrouter_api_key="test-key"),
        notebook_id=str(nb.id),
        on_tool_event=lambda evt: emitted_events.append(evt),
    )

    agent = create_pydantic_rag_agent(deps.settings)
    tool_def = agent._function_toolset.tools["add_source_to_knowledge_base"]

    ctx = RunContext(
        deps=deps,
        model=TestModel(),
        usage=MagicMock(),
        prompt="Indexa este PDF",
    )

    with patch("knowledge.pydantic_agent.fetch_remote_resource", return_value=(pdf_bytes, "application/pdf", "attention_paper.pdf", ".pdf")):
        res = await tool_def.function(
            ctx,
            source_type="web",
            title_or_url="https://arxiv.org/pdf/1706.03762.pdf",
        )

    assert res["status"] == "success"
    assert "attention_paper.pdf" in res["filename"]
    assert res["media_type"] == "pdf"
    assert res["notebook"] == "Cuaderno ML"

    # Check Document in DB has media_type == 'pdf'
    db_doc = await sync_to_async(lambda: Document.objects.filter(id=res["document_id"]).first())()
    assert db_doc is not None
    assert db_doc.media_type == "pdf"
    assert db_doc.original_filename == "attention_paper.pdf"

    # Check NotebookDocument link
    nb_doc = await sync_to_async(lambda: NotebookDocument.objects.filter(notebook=nb, document=db_doc).first())()
    assert nb_doc is not None

    # Check SSE event emitted with media_type
    assert any(e.get("type") == "source_added" and e.get("media_type") == "pdf" for e in emitted_events)


@pytest.mark.django_db(transaction=True)
def test_list_or_add_sources_remote_pdf_view():
    """Verify that Django view list_or_add_sources handles remote PDF URLs by saving .pdf and ingesting."""
    import pymupdf
    from django.test import RequestFactory
    from knowledge.models import Document, Notebook, NotebookDocument, Workspace
    from knowledge.views import list_or_add_sources

    ws = Workspace.objects.create(name="WS View Test")
    nb = Notebook.objects.create(workspace=ws, name="NB View Test")

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Texto de prueba vista Django.")
    pdf_bytes = doc.tobytes()
    doc.close()

    factory = RequestFactory()
    request = factory.post(
        "/api/sources/",
        data=json.dumps({
            "source_type": "web",
            "url": "https://example.com/research_doc.pdf",
            "notebook_id": str(nb.id),
        }),
        content_type="application/json",
    )

    with patch("knowledge.pydantic_agent.fetch_remote_resource", return_value=(pdf_bytes, "application/pdf", "research_doc.pdf", ".pdf")):
        resp = list_or_add_sources(request)

    assert resp.status_code == 200 or resp.status_code == 201
    body = json.loads(resp.content)
    assert body.get("media_type") == "pdf"
    assert body.get("filename") == "research_doc.pdf"
    assert body.get("notebook_id") == str(nb.id)
    assert body.get("status") == "indexed"


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_retriever_search_filtered_by_document_ids():
    """Verify that Retriever.search restricts results when document_ids are provided."""
    import uuid
    from knowledge.models import Document, Notebook, NotebookDocument, Workspace
    from knowledge.views import get_rag_service
    from ragpoc.config import Settings

    rag = get_rag_service()
    ws = await sync_to_async(Workspace.objects.create)(name="WS Grounding Focus")
    nb = await sync_to_async(Notebook.objects.create)(workspace=ws, name="NB Grounding Focus")

    u1 = uuid.uuid4().hex[:8]
    doc1_id = f"doc-focus-{u1}-1"
    doc2_id = f"doc-focus-{u1}-2"

    # Insert into rag.connection to satisfy SQLite foreign keys
    rag.connection.execute(
        "INSERT INTO workspaces (id, name, created_at, updated_at) VALUES (?, ?, datetime('now'), datetime('now'))",
        (str(ws.id), ws.name),
    )
    rag.connection.execute(
        "INSERT INTO notebooks (id, workspace_id, name, position, created_at, updated_at) VALUES (?, ?, ?, 0, datetime('now'), datetime('now'))",
        (str(nb.id), str(ws.id), nb.name),
    )
    rag.connection.execute(
        "INSERT INTO documents (id, source_path, original_filename, media_type, content_hash, byte_size, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        (doc1_id, f"c:/tmp/test_focus_doc1_{u1}.txt", "doc1.txt", "text", f"hash1_{u1}", 100, "indexed"),
    )
    rag.connection.execute(
        "INSERT INTO documents (id, source_path, original_filename, media_type, content_hash, byte_size, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        (doc2_id, f"c:/tmp/test_focus_doc2_{u1}.txt", "doc2.txt", "text", f"hash2_{u1}", 100, "indexed"),
    )
    rag.connection.execute(
        "INSERT INTO notebook_documents (notebook_id, document_id, attached_at) VALUES (?, ?, datetime('now'))",
        (str(nb.id), doc1_id),
    )
    rag.connection.execute(
        "INSERT INTO notebook_documents (notebook_id, document_id, attached_at) VALUES (?, ?, datetime('now'))",
        (str(nb.id), doc2_id),
    )

    c1_id = f"c_focus_{u1}_1"
    c2_id = f"c_focus_{u1}_2"

    # Insert mock chunks in SQLite
    rag.connection.execute(
        "INSERT INTO chunks (id, document_id, ordinal, kind, text_content, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
        (c1_id, doc1_id, 0, "text", "Contenido sobre arquitectura limpia.", "{}"),
    )
    rag.connection.execute(
        "INSERT INTO chunks (id, document_id, ordinal, kind, text_content, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
        (c2_id, doc2_id, 0, "text", "Contenido sobre finanzas y presupuestos.", "{}"),
    )
    from ragpoc.db import persist_dimension
    persist_dimension(rag.connection, 8)

    import json
    rag.connection.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (c1_id, json.dumps([0.1] * 8)),
    )
    rag.connection.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (c2_id, json.dumps([0.1] * 8)),
    )
    rag.connection.commit()

    # Search without filter (returns both)
    all_res = await rag.retriever.search(query="arquitectura", top_k=5, notebook_id=str(nb.id))
    assert len(all_res) == 2

    # Search with document_ids=[doc1_id] only
    filtered_res = await rag.retriever.search(
        query="arquitectura",
        top_k=5,
        document_ids=[doc1_id],
    )
    assert len(filtered_res) == 1
    assert filtered_res[0]["document_id"] == doc1_id
    assert filtered_res[0]["filename"] == "doc1.txt"


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_agent_search_knowledge_base_respects_selected_source_ids():
    """Verify that search_knowledge_base uses deps.selected_source_ids when provided."""
    mock_retriever = MagicMock(spec=Retriever)
    mock_retriever.search = AsyncMock(return_value=[
        {
            "chunk_id": "c1",
            "document_id": "doc-focus-1",
            "filename": "focus_doc.pdf",
            "media_type": "pdf",
            "page_number": 1,
            "text": "Evidencia de documento seleccionado.",
            "derived_path": None,
            "source_path": None,
            "metadata": {},
        }
    ])

    emitted_events = []
    deps = AgentDeps(
        retriever=mock_retriever,
        settings=Settings(),
        notebook_id="nb-1",
        selected_source_ids=["doc-focus-1"],
        on_tool_event=lambda evt: emitted_events.append(evt),
    )

    agent = create_pydantic_rag_agent(Settings())
    agent.model = TestModel(call_tools=["search_knowledge_base"])

    result = await agent.run("Busca información en las fuentes", deps=deps)
    assert result is not None

    # Verify search was called with document_ids
    assert mock_retriever.search.await_count >= 1
    assert mock_retriever.search.call_args.kwargs.get("document_ids") == ["doc-focus-1"]


@pytest.mark.django_db(transaction=True)
def test_youtube_transcript_and_metadata_extractor():
    """Verify YouTube transcript extractor parses video info and timedtext into structured Markdown."""
    from knowledge.pydantic_agent import extract_youtube_transcript_and_metadata

    sample_html = """
    <html>
      <head><title>Aprende Python en 10 Minutos - YouTube</title></head>
      <body>
        <script>
          var ytInitialPlayerResponse = {
            "videoDetails": {
              "title": "Aprende Python en 10 Minutos",
              "author": "Tech Academy",
              "videoId": "dQw4w9WgXcQ"
            },
            "captions": {
              "playerCaptionsTracklistRenderer": {
                "captionTracks": [
                  {"baseUrl": "https://www.youtube.com/api/timedtext?v=dQw4w9WgXcQ&lang=es", "languageCode": "es"}
                ]
              }
            }
          };
        </script>
      </body>
    </html>
    """

    sample_transcript_xml = """<?xml version="1.0" encoding="utf-8" ?>
    <transcript>
      <text start="0.5" dur="2.1">Bienvenidos al curso intensivo de Python.</text>
      <text start="65.0" dur="3.0">Veamos ahora cómo definir variables y funciones.</text>
    </transcript>
    """

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp1 = MagicMock()
        mock_resp1.read.return_value = sample_html.encode("utf-8")
        mock_resp1.__enter__.return_value = mock_resp1

        mock_resp2 = MagicMock()
        mock_resp2.read.return_value = sample_transcript_xml.encode("utf-8")
        mock_resp2.__enter__.return_value = mock_resp2

        mock_urlopen.side_effect = [mock_resp1, mock_resp2]

        result = extract_youtube_transcript_and_metadata("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    assert result is not None
    raw_bytes, mime, filename, ext = result
    assert mime == "text/plain"
    assert ext == ".txt"
    assert "YouTube_" in filename
    text = raw_bytes.decode("utf-8")
    assert "# Aprende Python en 10 Minutos" in text
    assert "**Canal/Autor:** Tech Academy" in text
    assert "**[00:00]** Bienvenidos al curso intensivo de Python." in text
    assert "**[01:05]** Veamos ahora cómo definir variables y funciones." in text


@pytest.mark.django_db(transaction=True)
def test_document_source_guide_view():
    """Verify document_source_guide_view returns executive summary, key topics and suggested questions."""
    from django.test import RequestFactory
    from knowledge.models import Document, Workspace
    from knowledge.views import document_source_guide_view, get_rag_service

    rag = get_rag_service()
    doc = Document.objects.create(
        original_filename="manual_arquitectura_guide.txt",
        source_path="c:/tmp/manual_arquitectura_guide.txt",
        media_type="text",
        content_hash="arch_hash_guide_1",
        status="indexed",
    )

    # Insert document and chunk in SQLite
    rag.connection.execute(
        "INSERT INTO documents (id, source_path, original_filename, media_type, content_hash, byte_size, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        (str(doc.id), doc.source_path, doc.original_filename, doc.media_type, doc.content_hash, 100, "indexed"),
    )
    rag.connection.execute(
        "INSERT INTO chunks (id, document_id, ordinal, kind, text_content, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
        ("c_guide_unique_test", str(doc.id), 0, "text", "Este manual describe los principios fundamentales de diseño desacoplado.", "{}"),
    )
    rag.connection.commit()

    factory = RequestFactory()
    request = factory.get(f"/api/documents/{doc.id}/guide")

    mock_guide_json = json.dumps({
        "summary": "Este manual describe los principios de arquitectura limpia y patrones desacoplados.",
        "key_topics": ["Clean Architecture", "Inversión de Dependencias", "Escalabilidad"],
        "suggested_questions": [
            "¿Cuáles son las capas principales de la arquitectura?",
            "¿Cómo se gestionan las dependencias externas?",
        ],
    })

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content=mock_guide_json))]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_cls.return_value = mock_client

        response = document_source_guide_view(request, str(doc.id))

    assert response.status_code == 200
    data = json.loads(response.content)
    assert data["status"] == "success"
    assert data["document_id"] == str(doc.id)
    assert "summary" in data["guide"]
    assert len(data["guide"]["key_topics"]) == 3
    assert len(data["guide"]["suggested_questions"]) == 2








