from __future__ import annotations

import asyncio
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from asgiref.sync import sync_to_async
from ddgs import DDGS
from openai import AsyncOpenAI
from pydantic_ai import Agent, BinaryContent, RunContext
from pydantic_ai.messages import ToolReturn
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from knowledge.settings_store import get_effective_settings as get_settings
from ragpoc.config import Settings
from ragpoc.embeddings import AUDIO_MIME_TYPES
from ragpoc.http import new_async_client
from ragpoc.retrieval import Retriever


@dataclass
class AgentDeps:
    retriever: Retriever
    settings: Settings
    page_id: str | None = None
    notebook_id: str | None = None
    workspace_id: str | None = None
    attached_docs_context: str = ""
    # Lets page-writing tools notify the SSE stream so the frontend can refresh the sidebar
    # tree and, if it's the page currently open in the editor, reload it — otherwise a
    # create_workspace_page/update_page_notes write lands silently in the DB and the open
    # editor keeps showing stale content until the user navigates away and back.
    on_tool_event: Callable[[dict], None] | None = None
    # Set by create_workspace_page/update_page_notes to hand off to chat_stream_view's own
    # token loop: those tools no longer take the page content as an argument (a tool-call
    # argument only reaches Python once the model has finished generating the whole thing, so
    # there was no way to stream it token-by-token into the page). Instead they prepare a
    # target page id and set this dict; the model's NEXT text-producing step (which pydantic-ai
    # already exposes token-by-token via stream_text) is mirrored live into that page as it
    # arrives, and persisted once the turn's text stream ends. See chat_stream_view.
    page_write_state: dict | None = None


SYSTEM_PROMPT = """Eres el asistente de conocimiento y copiloto operativo RAGPoC potenciado por PydanticAI (inspirado en Hermes Agent).
Cuentas con memoria declarativa persistente (AgentMemory), memoria procedimental/habilidades (AgentSkills) y herramientas de acción en el espacio de trabajo.

══════════════════════════════════════════════
HERRAMIENTAS DISPONIBLES:
══════════════════════════════════════════════
1. `search_knowledge_base`: Busca evidencia en notas, documentos (PDFs, imágenes, videos, texto) del cuaderno actual.
2. `search_web`: Búsquedas en tiempo real en internet.
3. `add_source_to_knowledge_base`: Guarda e indexa URLs o notas de texto en la base vectorial, a nivel de cuaderno.
4. `manage_memory`: Guarda, actualiza o elimina hechos duraderos sobre el usuario, preferencias, convenciones y lecciones aprendidas (estilo Hermes memory).
5. `manage_skill`: Crea, consulta o actualiza habilidades y flujos de trabajo reutilizables (estilo Hermes skills).
6. `create_workspace_page`: Prepara una página nueva (con título) dentro del cuaderno activo o especificado. NO recibe el contenido como argumento: después de llamarla, escribe el contenido en Markdown como tu siguiente respuesta de texto normal — se transmite en vivo a la página y se guarda automáticamente al terminar.
7. `update_page_notes`: Prepara la página actual (o una especificada) para recibir contenido adicional. Igual que la anterior: no lleva el contenido como argumento, escríbelo como tu siguiente respuesta de texto normal.
8. `get_workspace_structure`: Explora la jerarquía completa de cuadernos y páginas del espacio de trabajo.

══════════════════════════════════════════════
DIRECTIVAS DE OPERACIÓN (HERMES STYLE):
══════════════════════════════════════════════
- **Memoria Declarativa Proactiva:** Cuando el usuario exprese una preferencia estable o descubras un hecho clave del proyecto, invoca proactivamente `manage_memory(action='add', content=...)`.
- **Memoria Procedimental (Skills):** Si descubres o el usuario te enseña un flujo de trabajo complejo repetible, guárdalo con `manage_skill(action='create', name=..., instructions=...)`.
- Responde siempre en español claro y conciso con formato Markdown (negritas, listas, tablas).
- Cita las fuentes de la base de conocimiento usando [n].
- **Después de `create_workspace_page` o `update_page_notes`:** tu siguiente respuesta de texto ES el contenido que se guardará en la página. Escribe ÚNICAMENTE el contenido en Markdown — sin saludos, sin "aquí tienes", sin confirmaciones. Si quieres además comentarle algo al usuario en el chat, hazlo en un mensaje aparte, no mezclado con el contenido de la página.
"""


def evidence_media_parts(source: dict) -> list[str | BinaryContent]:
    """Turn one retrieval hit into the raw bytes Gemini needs to actually see/hear it.

    Video, audio and PDF-page chunks are indexed as embedding vectors only -- ingestion never
    runs a transcription or captioning step, so chunks.text_content stays NULL for them. Without
    handing Gemini the underlying file directly (it's multimodal, so this is cheaper and more
    accurate than adding a separate transcription pipeline), that evidence is search-only: the
    model can tell a matching chunk exists but not what's actually in it.
    """
    media_type = source.get("media_type")
    filename = source.get("filename") or "fuente"
    parts: list[str | BinaryContent] = []
    try:
        if media_type == "image" and source.get("source_path"):
            path = Path(source["source_path"])
            if path.is_file():
                suffix = path.suffix.lower()
                mime = "image/png" if suffix == ".png" else "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/webp"
                parts.append(f"\n[Imagen recuperada de la fuente {filename}]:\n")
                parts.append(BinaryContent(data=path.read_bytes(), media_type=mime))
        elif media_type == "pdf" and source.get("derived_path"):
            path = Path(source["derived_path"])
            if path.is_file():
                page = source.get("page_number")
                label = f" (página {page})" if page else ""
                parts.append(f"\n[Página de PDF recuperada de la fuente {filename}{label}]:\n")
                parts.append(BinaryContent(data=path.read_bytes(), media_type="image/png"))
        elif media_type == "video" and source.get("derived_path"):
            path = Path(source["derived_path"])
            if path.is_file():
                meta = source.get("metadata") or {}
                start_tc = meta.get("start_timecode")
                label = f" ({start_tc}-{meta.get('end_timecode')})" if start_tc else ""
                parts.append(f"\n[Clip de video recuperado de la fuente {filename}{label}]:\n")
                parts.append(BinaryContent(data=path.read_bytes(), media_type="video/mp4"))
        elif media_type == "audio" and source.get("source_path"):
            path = Path(source["source_path"])
            if path.is_file():
                mime = AUDIO_MIME_TYPES.get(path.suffix.lower(), "audio/mpeg")
                parts.append(f"\n[Audio recuperado de la fuente {filename}]:\n")
                parts.append(BinaryContent(data=path.read_bytes(), media_type=mime))
    except Exception:
        return []
    return parts


def create_pydantic_rag_agent(settings: Settings | None = None) -> Agent[AgentDeps, str]:
    settings = settings or get_settings()

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key or "sk-dummy",
        http_client=new_async_client(),
    )
    model = OpenAIChatModel(
        settings.chat_model,
        provider=OpenAIProvider(openai_client=client),
    )

    agent: Agent[AgentDeps, str] = Agent(
        model=model,
        deps_type=AgentDeps,
        system_prompt=SYSTEM_PROMPT,
    )

    @agent.tool
    async def search_knowledge_base(
        ctx: RunContext[AgentDeps],
        query: str,
        top_k: int = 5,
    ) -> list[dict] | ToolReturn:
        """Busca en la base de conocimiento local (notas, imágenes, PDFs, videos) del cuaderno actual."""
        try:
            from knowledge.models import Notebook, Page

            notebook_id = ctx.deps.notebook_id
            notebook_ids: list[str] | None = None
            if not notebook_id and ctx.deps.page_id:
                @sync_to_async
                def _resolve_notebook_id():
                    p = Page.objects.filter(id=ctx.deps.page_id).first()
                    return p.notebook_id if p else None

                notebook_id = await _resolve_notebook_id()
            elif not notebook_id and ctx.deps.workspace_id:
                @sync_to_async
                def _resolve_notebook_ids():
                    return list(Notebook.objects.filter(workspace_id=ctx.deps.workspace_id).values_list("id", flat=True))

                notebook_ids = await _resolve_notebook_ids()

            results = await ctx.deps.retriever.search(
                query=query,
                top_k=top_k,
                notebook_id=notebook_id,
                notebook_ids=notebook_ids,
            )
            formatted = []
            media_parts: list[str | BinaryContent] = []
            for idx, r in enumerate(results, 1):
                formatted.append({
                    "citation": f"[{idx}]",
                    "filename": r.get("filename"),
                    "media_type": r.get("media_type"),
                    "page_number": r.get("page_number"),
                    "text_excerpt": (r.get("text") or "")[:800],
                })
                media_parts.extend(evidence_media_parts(r))
            if media_parts:
                return ToolReturn(return_value=formatted, content=media_parts)
            return formatted
        except Exception as e:
            return [{"error": str(e)}]

    @agent.tool
    async def search_web(ctx: RunContext[AgentDeps], query: str, max_results: int = 4) -> list[dict]:
        """Realiza una búsqueda en internet en tiempo real para obtener información actualizada."""
        try:
            loop = asyncio.get_event_loop()
            def _sync_search():
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=max_results))

            raw_results = await loop.run_in_executor(None, _sync_search)
            return [
                {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
                for r in raw_results
            ]
        except Exception as e:
            return [{"error": f"Fallo en la búsqueda web: {str(e)}"}]

    @agent.tool
    async def add_source_to_knowledge_base(
        ctx: RunContext[AgentDeps],
        source_type: str,
        title_or_url: str,
        content: str = "",
    ) -> dict:
        """Añade e indexa una nueva fuente (URL web o nota de texto) en la base de conocimiento, a nivel del cuaderno actual."""
        from knowledge.models import (
            Document,
            Notebook,
            NotebookDocument,
            Page,
            calculate_content_hash,
        )
        from knowledge.services import get_rag_service

        try:
            rag = get_rag_service()

            raw_text = ""
            filename = ""
            if source_type == "web" or title_or_url.startswith("http://") or title_or_url.startswith("https://"):
                url = title_or_url.strip()
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
                loop = asyncio.get_event_loop()
                def fetch_url():
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        html = resp.read().decode("utf-8", errors="ignore")
                        text_only = re.sub(r"<[^>]+>", " ", html)
                        return re.sub(r"\s+", " ", text_only).strip()

                raw_text = await loop.run_in_executor(None, fetch_url)
                filename = url.replace("https://", "").replace("http://", "").split("/")[0] + "_web_doc.txt"
            else:
                filename = (title_or_url.strip() or "Nota_Agente") + ".txt"
                raw_text = content.strip()

            if not raw_text:
                return {"error": "No se pudo extraer contenido de la fuente."}

            content_hash = calculate_content_hash(filename, raw_text)
            ctx.deps.settings.allowed_upload_dir.mkdir(parents=True, exist_ok=True)
            temp_file = ctx.deps.settings.allowed_upload_dir / f"{content_hash[:16]}-{filename}"
            temp_file.write_text(raw_text, encoding="utf-8")

            report = await rag.ingestor.ingest(temp_file)
            doc_id = report.get("document_id")

            @sync_to_async
            def link_in_django():
                doc, _ = Document.objects.get_or_create(
                    content_hash=content_hash,
                    defaults={
                        "id": doc_id,
                        "original_filename": filename,
                        "media_type": "text",
                        "byte_size": len(raw_text.encode("utf-8")),
                        "source_path": str(temp_file),
                        "status": "indexed",
                    }
                )
                target_nb = None
                if ctx.deps.notebook_id:
                    target_nb = Notebook.objects.filter(id=ctx.deps.notebook_id).first()
                elif ctx.deps.page_id:
                    p = Page.objects.filter(id=ctx.deps.page_id).first()
                    if p:
                        target_nb = p.notebook
                if target_nb:
                    NotebookDocument.objects.get_or_create(notebook=target_nb, document=doc)
                return {
                    "status": "success" if target_nb else "error",
                    "filename": filename,
                    "notebook": target_nb.name if target_nb else None,
                    "document_id": doc.id,
                    **({} if target_nb else {"error": "No hay un cuaderno activo al que asociar la fuente."}),
                }

            res = await link_in_django()
            return res
        except Exception as e:
            return {"error": f"No se pudo guardar la fuente: {str(e)}"}

    @agent.tool
    async def manage_memory(
        ctx: RunContext[AgentDeps],
        action: str,
        content: str = "",
        category: str = "user_preference",
        memory_id: str | None = None,
    ) -> dict:
        """Gestiona hechos duraderos y preferencias en la memoria persistente del agente (action: add, list, remove)."""
        from knowledge.models import AgentMemory

        def _db_op():
            if action == "add":
                if not content.strip():
                    return {"error": "El contenido de la memoria no puede estar vacío."}
                mem = AgentMemory.objects.create(category=category, content=content.strip(), source="agent_auto")
                return {"status": "success", "action": "add", "id": mem.id, "content": mem.content}
            elif action == "remove":
                if memory_id:
                    AgentMemory.objects.filter(id=memory_id).delete()
                    return {"status": "success", "action": "remove", "id": memory_id}
                elif content:
                    AgentMemory.objects.filter(content__icontains=content).delete()
                    return {"status": "success", "action": "remove", "content_matched": content}
                return {"error": "Debes especificar memory_id o content para eliminar."}
            else:  # list
                mems = list(AgentMemory.objects.all().values("id", "category", "content", "updated_at"))
                return {"status": "success", "action": "list", "memories": mems}

        try:
            return await sync_to_async(_db_op)()
        except Exception as e:
            return {"error": f"Error gestionando memoria: {str(e)}"}

    @agent.tool
    async def manage_skill(
        ctx: RunContext[AgentDeps],
        action: str,
        name: str = "",
        description: str = "",
        instructions: str = "",
        category: str = "general",
    ) -> dict:
        """Crea, consulta o actualiza habilidades y procedimientos operativos del agente (action: create, list, view, update, delete)."""
        from knowledge.models import AgentSkill

        def _skill_op():
            if action in {"create", "update"}:
                if not name.strip() or not instructions.strip():
                    return {"error": "name e instructions son requeridos para crear/actualizar una skill."}
                skill, created = AgentSkill.objects.update_or_create(
                    name=name.strip().lower().replace(" ", "-"),
                    defaults={
                        "description": description.strip() or name,
                        "instructions": instructions.strip(),
                        "category": category,
                        "is_active": True,
                    }
                )
                return {"status": "success", "action": "create" if created else "update", "name": skill.name}
            elif action == "view":
                skill = AgentSkill.objects.filter(name=name.strip().lower().replace(" ", "-")).first()
                if not skill:
                    return {"error": f"Skill '{name}' no encontrada."}
                return {"name": skill.name, "description": skill.description, "instructions": skill.instructions}
            elif action == "delete":
                AgentSkill.objects.filter(name=name.strip().lower().replace(" ", "-")).delete()
                return {"status": "success", "action": "delete", "name": name}
            else:  # list
                skills = list(AgentSkill.objects.filter(is_active=True).values("name", "category", "description"))
                return {"status": "success", "action": "list", "skills": skills}

        try:
            return await sync_to_async(_skill_op)()
        except Exception as e:
            return {"error": f"Error gestionando skill: {str(e)}"}

    @agent.tool
    async def create_workspace_page(
        ctx: RunContext[AgentDeps],
        title: str,
        notebook_id: str | None = None,
    ) -> dict:
        """Prepara una página nueva dentro del cuaderno actual o especificado. No recibe el contenido:
        escríbelo como tu siguiente respuesta de texto normal, se transmite en vivo a la página."""
        from knowledge.models import Notebook, Page, Workspace

        def _create():
            target_nb_id = notebook_id or ctx.deps.notebook_id
            if not target_nb_id:
                # Find first available notebook
                ws = Workspace.objects.first()
                nb = Notebook.objects.filter(workspace=ws).first() if ws else Notebook.objects.first()
                if not nb:
                    return {"error": "No hay cuadernos disponibles en el espacio para crear la página."}
                target_nb = nb
            else:
                target_nb = Notebook.objects.filter(id=target_nb_id).first()
                if not target_nb:
                    return {"error": f"Cuaderno con ID {target_nb_id} no encontrado."}

            page_title = title.strip() or "Nueva Página"
            page = Page.objects.create(
                notebook=target_nb,
                title=page_title,
                plain_text="",
                content_json={
                    "type": "doc",
                    "content": [{"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": page_title}]}],
                },
            )
            return {"status": "ready_for_content", "page_id": page.id, "title": page.title, "notebook_id": target_nb.id, "notebook": target_nb.name, "mode": "create"}

        try:
            result = await sync_to_async(_create)()
            if result.get("status") == "ready_for_content":
                # Hands off to chat_stream_view: the model's NEXT text-producing step is what
                # actually contains the page content now (see AgentDeps.page_write_state), not
                # this tool call's own return value.
                ctx.deps.page_write_state = {
                    "page_id": result["page_id"],
                    "notebook_id": result["notebook_id"],
                    "title": result["title"],
                    "mode": "create",
                }
                result["instructions"] = (
                    "Página creada. Escribe ahora el contenido en Markdown como tu respuesta de texto "
                    "normal (sin saludos ni confirmaciones) -- se transmitirá en vivo a esta página."
                )
            return result
        except Exception as e:
            return {"error": f"No se pudo crear la página: {str(e)}"}

    @agent.tool
    async def update_page_notes(
        ctx: RunContext[AgentDeps],
        page_id: str | None = None,
    ) -> dict:
        """Prepara la página actual (o una especificada) para recibir contenido adicional. No recibe el
        contenido: escríbelo como tu siguiente respuesta de texto normal, se transmite en vivo a la página."""
        from knowledge.models import Page

        def _resolve():
            target_page_id = page_id or ctx.deps.page_id
            if not target_page_id:
                return {"error": "No hay página activa seleccionada para actualizar."}
            p = Page.objects.filter(id=target_page_id).first()
            if not p:
                return {"error": f"Página con ID {target_page_id} no encontrada."}
            return {"status": "ready_for_content", "page_id": p.id, "notebook_id": p.notebook_id, "title": p.title, "mode": "append"}

        try:
            result = await sync_to_async(_resolve)()
            if result.get("status") == "ready_for_content":
                ctx.deps.page_write_state = {
                    "page_id": result["page_id"],
                    "notebook_id": result["notebook_id"],
                    "title": result["title"],
                    "mode": "append",
                }
                result["instructions"] = (
                    "Página lista. Escribe ahora el contenido adicional en Markdown como tu respuesta de "
                    "texto normal (sin saludos ni confirmaciones) -- se transmitirá en vivo a esta página."
                )
            return result
        except Exception as e:
            return {"error": f"No se pudo preparar la página: {str(e)}"}

    @agent.tool
    async def get_workspace_structure(ctx: RunContext[AgentDeps]) -> dict:
        """Obtiene la jerarquía completa de cuadernos y páginas disponibles en el espacio de trabajo actual."""
        from knowledge.models import Workspace

        def _tree():
            ws = Workspace.objects.first()
            if ctx.deps.workspace_id:
                ws = Workspace.objects.filter(id=ctx.deps.workspace_id).first() or ws
            if not ws:
                return {"error": "No hay espacios de trabajo configurados."}

            structure = []
            for nb in ws.notebooks.all():
                pages_list = list(nb.pages.all().values("id", "title", "updated_at"))
                structure.append({
                    "notebook_id": nb.id,
                    "notebook_name": nb.name,
                    "pages_count": len(pages_list),
                    "pages": pages_list,
                })
            return {"workspace_id": ws.id, "workspace_name": ws.name, "notebooks": structure}

        try:
            return await sync_to_async(_tree)()
        except Exception as e:
            return {"error": f"Error obteniendo estructura: {str(e)}"}

    return agent
