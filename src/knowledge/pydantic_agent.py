from __future__ import annotations

import asyncio
import html
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal


def extract_clean_text_from_html(raw_html: str) -> str:
    """Extract and sanitize clean readable text from raw HTML, stripping scripts, styles, and tags."""
    if not raw_html:
        return ""
    # Remove script, style, noscript, svg, header, footer, nav, head tags and their contents
    cleaned = re.sub(
        r"<(script|style|noscript|svg|header|footer|nav|head)[\s\S]*?</\1>",
        " ",
        raw_html,
        flags=re.IGNORECASE,
    )
    # Remove HTML comments
    cleaned = re.sub(r"<!--[\s\S]*?-->", " ", cleaned)
    # Replace block level tags with newlines
    cleaned = re.sub(r"</?(p|div|h[1-6]|li|tr|blockquote|section|article)[^>]*>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
    # Remove all remaining HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    # Decode HTML entities (&amp;, &nbsp;, &#39;, etc.)
    cleaned = html.unescape(cleaned)
    # Normalize whitespaces
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


ArtifactType = Literal[
    "podcast",
    "diagram",
    "mindmap",
    "quiz",
    "flashcards",
    "study_guide",
    "summary",
    "infographic",
    "timeline",
]

from asgiref.sync import sync_to_async
from ddgs import DDGS
from openai import AsyncOpenAI
from pydantic_ai import Agent, BinaryContent, RunContext
from pydantic_ai.messages import ToolReturn
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

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
    thread_id: str | None = None
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
    collected_sources: list[dict] = field(default_factory=list)
    executed_tools: list[dict] = field(default_factory=list)

    def record_tool_start(self, tool_name: str, label: str, icon: str = "wrench") -> dict:
        step = {
            "tool": tool_name,
            "label": label,
            "icon": icon,
            "started_at": time.perf_counter(),
            "status": "running",
        }
        self.executed_tools.append(step)
        if self.on_tool_event:
            self.on_tool_event({
                "type": "tool_start",
                "tool": tool_name,
                "label": label,
                "icon": icon,
                "status": "running",
            })
        return step

    def record_tool_end(self, tool_name: str, summary: str | None = None, status: str = "done") -> None:
        for step in reversed(self.executed_tools):
            if step.get("tool") == tool_name and step.get("status") == "running":
                step["status"] = status
                started_at = step.get("started_at", time.perf_counter())
                duration = int((time.perf_counter() - started_at) * 1000)
                step["duration_ms"] = duration
                if summary:
                    step["summary"] = summary
                if self.on_tool_event:
                    self.on_tool_event({
                        "type": "tool_end",
                        "tool": tool_name,
                        "label": step.get("label", tool_name),
                        "summary": summary or step.get("label", tool_name),
                        "duration_ms": duration,
                        "status": status,
                    })
                break


SYSTEM_PROMPT_ES = """Eres el asistente de conocimiento y copiloto operativo RAGPoC potenciado por PydanticAI (inspirado en Hermes Agent).
Cuentas con memoria declarativa persistente (AgentMemory), memoria procedimental/habilidades (AgentSkills) y herramientas de acción en el espacio de trabajo.

══════════════════════════════════════════════
HERRAMIENTAS DISPONIBLES:
══════════════════════════════════════════════
1. `search_knowledge_base`: Busca evidencia en notas, documentos (PDFs, imágenes, videos, texto) del cuaderno o espacio de trabajo.
2. `search_web`: Búsquedas en tiempo real en internet.
3. `fetch_web_page`: Extrae y lee el texto limpio de cualquier página web (URL) para análisis profundo.
4. `add_source_to_knowledge_base`: Guarda e indexa URLs o notas de texto/resúmenes en la base de conocimiento vectorial del cuaderno actual (estilo NotebookLM).
5. `manage_memory`: Guarda, actualiza o elimina hechos duraderos sobre el usuario, preferencias, convenciones y lecciones aprendidas (estilo Hermes memory).
6. `manage_skill`: Crea, consulta o actualiza habilidades y flujos de trabajo reutilizables (estilo Hermes skills).
7. `create_workspace_page`: Prepara una página nueva (con título) dentro del cuaderno activo o especificado. NO recibe el contenido como argumento: después de llamarla, escribe el contenido en Markdown como tu siguiente respuesta de texto normal — se transmite en vivo a la página y se guarda automáticamente al terminar.
8. `update_page_notes`: Prepara la página actual (o una especificada) para recibir contenido adicional. Igual que la anterior: no lleva el contenido como argumento, escríbelo como tu siguiente respuesta de texto normal.
9. `get_workspace_structure`: Explora la jerarquía completa de cuadernos y páginas del espacio de trabajo.
10. `search_past_conversations`: Lista/busca conversaciones de chat anteriores (distintas de la actual), acotadas al cuaderno o espacio de trabajo activo. Úsala cuando el usuario haga referencia a algo hablado antes que no aparece en el contexto actual.
11. `get_conversation_messages`: Recupera el contenido completo de una conversación pasada por su `thread_id` (obtenido con `search_past_conversations`) para revisar el detalle exacto de lo que se dijo.
12. `generate_notebook_artifact`: Genera artefactos multimedia y de conocimiento para el cuaderno actual (podcasts de audio .wav con 2 locutores, diagramas Mermaid, mapas mentales, cuestionarios interactivos, flashcards, guías de estudio, resúmenes ejecutivos, infografías y líneas de tiempo) y los guarda en la galería Studio.

══════════════════════════════════════════════
DIRECTIVAS DE OPERACIÓN (HERMES STYLE):
══════════════════════════════════════════════
- **Búsqueda Vectorial Agéntica (100% Tool-Driven):** Tu contexto inicial no incluye fragmentos pre-cargados de la base de conocimiento. Por lo tanto, DEBES invocar proactivamente `search_knowledge_base` siempre que la pregunta del usuario requiera información, hechos, documentos (PDFs, imágenes, videos, texto) o notas del cuaderno/espacio de trabajo.
- **Investigación e Indexación Agéntica (Flujo Estilo NotebookLM):** Cuando el usuario te pida buscar información externa en internet, investigar un tema o agregar/indexar nuevas fuentes a la base de conocimiento o al cuaderno:
  1. Usa `search_web` para encontrar enlaces y fuentes relevantes sobre el tema.
  2. Si requieres leer el contenido de una URL antes de sintetizar o guardar, usa `fetch_web_page`.
  3. Invoca proactivamente `add_source_to_knowledge_base` (con `source_type='web'` pasando la URL o `source_type='text'` con una nota técnica/resumen estructurado) para registrar e indexar cada fuente en el cuaderno.
  4. Confirma al usuario las fuentes que han sido indexadas exitosamente en el cuaderno para que sepa que ya están integradas en la base de conocimiento.
- **Consultas Semánticas Precisas:** Al invocar `search_knowledge_base`, formula términos de búsqueda y conceptos clave concretos y descriptivos (en lugar de copiar literalmente preguntas conversacionales completas del usuario). Puedes invocar la herramienta múltiples veces si necesitas contrastar información o profundizar en diferentes temas.
- **Generación de Artefactos (Studio Artifacts):** Cuando el usuario te pida crear o generar un podcast, diagrama, mapa mental, quiz, tarjetas de estudio (flashcards), infografía, línea de tiempo o guía, invoca proactivamente `generate_notebook_artifact` con el tipo (`podcast`, `diagram`, `mindmap`, `quiz`, `flashcards`, `study_guide`, `summary`, `infographic`, `timeline`) e instrucciones pertinentes.
- **Memoria Declarativa Proactiva:** Cuando el usuario exprese una preferencia estable o descubras un hecho clave del proyecto, invoca proactivamente `manage_memory(action='add', content=...)`.
- **Memoria Procedimental (Skills):** Si descubres o el usuario te enseña un flujo de trabajo complejo repetible, guárdalo con `manage_skill(action='create', name=..., instructions=...)`.
- **Aislamiento de Sesión Actual vs. Conversaciones Pasadas:**
  - Cuando el usuario pregunte sobre la conversación en curso (ej. "¿de qué estamos hablando?", "resumen de esta conversación", "¿qué te acabo de pedir?"), responde usando EXCLUSIVAMENTE el historial de mensajes de la sesión actual (`message_history`). NUNCA llames a `search_past_conversations` para responder sobre la sesión en curso.
  - Invoca `search_past_conversations` ÚNICAMENTE si el usuario hace referencia explícita a otro hilo o conversación anterior (ej. "en la otra conversación", "lo que te pregunté ayer", "busca en mis chats anteriores").
- **Aislamiento Estricto de Ámbito (Scope Segregation):** Cuando estés en el ámbito de un cuaderno ('Este cuaderno'), NO busques ni mezcles información de otros cuadernos a menos que el usuario lo pida expresamente cambiando el ámbito a 'Espacio de trabajo'.
- Responde siempre en español claro y conciso con formato Markdown (negritas, listas, tablas).
- Cita las fuentes de la base de conocimiento usando [n] cuando utilices información recuperada de `search_knowledge_base`.
- **Después de `create_workspace_page` o `update_page_notes`:** tu siguiente respuesta de texto ES el contenido que se guardará en la página. Escribe ÚNICAMENTE el contenido en Markdown — sin saludos, sin "aquí tienes", sin confirmaciones. Si quieres además comentarle algo al usuario en el chat, hazlo en un mensaje aparte, no mezclado con el contenido de la página.
"""

SYSTEM_PROMPT_EN = """You are the RAGPoC knowledge assistant and operational copilot powered by PydanticAI (inspired by Hermes Agent).
You possess persistent declarative memory (AgentMemory), procedural memory/skills (AgentSkills), and workspace action tools.

══════════════════════════════════════════════
AVAILABLE TOOLS:
══════════════════════════════════════════════
1. `search_knowledge_base`: Search for evidence in notes and documents (PDFs, images, videos, text) across the notebook or workspace.
2. `search_web`: Real-time web search.
3. `fetch_web_page`: Extract and read clean text content from any web URL for deep analysis.
4. `add_source_to_knowledge_base`: Save and index web URLs or text notes/summaries into the vector database at the notebook level (NotebookLM style).
5. `manage_memory`: Save, update, or remove lasting facts about the user, preferences, conventions, and lessons learned (Hermes memory style).
6. `manage_skill`: Create, list, or update reusable skills and workflows (Hermes skills style).
7. `create_workspace_page`: Prepare a new page (with a title) inside the active or specified notebook. Does NOT receive content as an argument: after calling it, write the Markdown content as your NEXT normal text response — it streams live into the page and is automatically saved.
8. `update_page_notes`: Prepare the current (or specified) page to receive additional content. Same as above: write the content as your next normal text response.
9. `get_workspace_structure`: Explore the full hierarchy of notebooks and pages in the workspace.
10. `search_past_conversations`: List/search past chat threads (excluding the active one) scoped to the active notebook or workspace.
11. `get_conversation_messages`: Retrieve full messages from a past conversation by its `thread_id` to inspect what was discussed.
12. `generate_notebook_artifact`: Generate multimedia and knowledge artifacts for the current notebook (2-speaker .wav podcasts, Mermaid diagrams, mind maps, quizzes, flashcards, study guides, summaries, infographics, timelines) and save them to the Studio gallery.

══════════════════════════════════════════════
OPERATIONAL DIRECTIVES (HERMES STYLE):
══════════════════════════════════════════════
- **Agentic Vector Search (100% Tool-Driven):** Your initial context contains no preloaded knowledge chunks. You MUST proactively invoke `search_knowledge_base` whenever the user's question requires facts, documents (PDFs, images, videos, text), or notes.
- **Agentic Research & Indexing (NotebookLM Style):** When asked to search external information, research a topic, or add/index sources into the knowledge base or notebook:
  1. Use `search_web` to discover relevant sources and URLs.
  2. Use `fetch_web_page` if you need to read a page's full content before synthesizing.
  3. Call `add_source_to_knowledge_base` (with `source_type='web'` or `source_type='text'`) to save and index each source in the notebook.
  4. Confirm to the user which sources have been indexed into the notebook's knowledge base.
- **Precise Semantic Queries:** Formulate descriptive, concrete search keywords and concepts when calling `search_knowledge_base`. You can invoke it multiple times to cross-reference or explore different sub-topics.
- **Studio Artifacts Generation:** When asked to create or generate a podcast, diagram, mind map, quiz, flashcards, infographic, timeline, or study guide, proactively call `generate_notebook_artifact`.
- **Proactive Declarative Memory:** When the user expresses a stable preference or a key project fact is uncovered, proactively call `manage_memory(action='add', content=...)`.
- **Procedural Memory (Skills):** If you discover or the user teaches you a repeatable complex workflow, save it with `manage_skill(action='create', name=..., instructions=...)`.
- **Current vs. Past Conversation Isolation:** For questions about the current session (e.g. "what are we talking about?", "summary of this chat"), rely EXCLUSIVELY on your active message history (`message_history`). NEVER call `search_past_conversations` to answer about the active conversation.
- **Strict Scope Segregation:** When scoped to a notebook ('This notebook'), do NOT search or leak discussions or information from other notebooks unless explicitly requested.
- Always respond in clear, concise English with Markdown formatting (bold, lists, tables).
- Cite knowledge base sources using [n] whenever using information retrieved from `search_knowledge_base`.
- **After `create_workspace_page` or `update_page_notes`:** your next text response IS the page content. Output ONLY Markdown — no greetings, no introductory filler, no conversational confirmations.
"""


def get_system_prompt(language: str = "es") -> str:
    lang = (language or "es").strip().lower()
    if lang == "en":
        return SYSTEM_PROMPT_EN
    if lang == "auto":
        return SYSTEM_PROMPT_EN + "\n- **Language Matching Directive:** Detect the language used by the user in their query and respond fluently in that exact same language.\n"
    return SYSTEM_PROMPT_ES


SYSTEM_PROMPT = SYSTEM_PROMPT_ES


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


def create_pydantic_rag_agent(settings: Settings | None = None, language: str | None = None) -> Agent[AgentDeps, str]:
    from knowledge.settings_store import get_current_language

    settings = settings or get_settings()
    active_lang = language or get_current_language()

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key or "sk-dummy",
        http_client=new_async_client(),
    )
    model = OpenRouterModel(
        settings.chat_model,
        provider=OpenRouterProvider(openai_client=client),
    )

    agent: Agent[AgentDeps, str] = Agent(
        model=model,
        deps_type=AgentDeps,
        system_prompt=get_system_prompt(active_lang),
    )

    @agent.tool
    async def search_knowledge_base(
        ctx: RunContext[AgentDeps],
        query: str,
        top_k: int = 5,
    ) -> list[dict] | ToolReturn:
        """Busca en la base de conocimiento local (notas, imágenes, PDFs, videos) del cuaderno actual."""
        ctx.deps.record_tool_start("search_knowledge_base", f"Buscando en la base de conocimiento: '{query[:40]}'…", "search")
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
            refs = []
            media_parts: list[str | BinaryContent] = []
            for idx, r in enumerate(results, 1):
                filename = r.get("filename") or "Documento"
                media_type = r.get("media_type")
                formatted.append({
                    "citation": f"[{idx}]",
                    "filename": filename,
                    "media_type": media_type,
                    "page_number": r.get("page_number"),
                    "text_excerpt": (r.get("text") or "")[:800],
                })
                refs.append({
                    "citation": f"[{idx}]",
                    "label": filename,
                    "filename": filename,
                    "media_type": media_type,
                })
                media_parts.extend(evidence_media_parts(r))

            if refs and ctx.deps.on_tool_event:
                ctx.deps.on_tool_event({"type": "sources", "sources": refs})

            for ref in refs:
                if not any(existing.get("filename") == ref.get("filename") for existing in ctx.deps.collected_sources):
                    ctx.deps.collected_sources.append(ref)

            ctx.deps.record_tool_end("search_knowledge_base", f"{len(refs)} fuentes encontradas")
            if media_parts:
                return ToolReturn(return_value=formatted, content=media_parts)
            return formatted
        except Exception as e:
            ctx.deps.record_tool_end("search_knowledge_base", f"Error: {str(e)[:30]}", status="error")
            return [{"error": str(e)}]

    @agent.tool
    async def search_web(ctx: RunContext[AgentDeps], query: str, max_results: int = 5) -> list[dict]:
        """Realiza una búsqueda en internet en tiempo real para obtener información actualizada."""
        ctx.deps.record_tool_start("search_web", f"Buscando en la web: '{query[:40]}'…", "globe")
        try:
            loop = asyncio.get_event_loop()
            def _sync_search():
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=max_results))

            raw_results = await loop.run_in_executor(None, _sync_search)
            results = [
                {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
                for r in raw_results
            ]
            ctx.deps.record_tool_end("search_web", f"{len(results)} resultados web encontrados")
            return results
        except Exception as e:
            ctx.deps.record_tool_end("search_web", f"Error en búsqueda web: {str(e)[:30]}", status="error")
            return [{"error": f"Fallo en la búsqueda web: {str(e)}"}]

    @agent.tool
    async def fetch_web_page(ctx: RunContext[AgentDeps], url: str) -> dict:
        """Extrae y lee el contenido de texto limpio de cualquier página web (URL) para análisis profundo antes o durante la indexación."""
        ctx.deps.record_tool_start("fetch_web_page", f"Leyendo página web: {url[:45]}…", "link-45deg")
        try:
            target_url = url.strip()
            if not target_url.startswith("http://") and not target_url.startswith("https://"):
                target_url = "https://" + target_url

            loop = asyncio.get_event_loop()
            def _fetch():
                req = urllib.request.Request(
                    target_url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw_html = resp.read().decode("utf-8", errors="ignore")
                    return extract_clean_text_from_html(raw_html)

            content = await loop.run_in_executor(None, _fetch)
            if not content:
                ctx.deps.record_tool_end("fetch_web_page", "Sin contenido legible", status="error")
                return {"error": f"No se pudo extraer texto legible de {target_url}."}

            ctx.deps.record_tool_end("fetch_web_page", f"Página web analizada ({len(content)} car.)")
            return {
                "status": "success",
                "url": target_url,
                "content_preview": content[:15000],
                "char_count": len(content),
            }
        except Exception as e:
            ctx.deps.record_tool_end("fetch_web_page", f"Error: {str(e)[:30]}", status="error")
            return {"error": f"Error al acceder a la página {url}: {str(e)}"}

    @agent.tool
    async def add_source_to_knowledge_base(
        ctx: RunContext[AgentDeps],
        source_type: str = "auto",
        title_or_url: str = "",
        content: str = "",
        notebook_id: str | None = None,
    ) -> dict:
        """Añade e indexa una nueva fuente (URL web o nota de texto/resumen) en la base de conocimiento vectorial, a nivel del cuaderno actual (estilo NotebookLM)."""
        from knowledge.models import (
            Document,
            Notebook,
            NotebookDocument,
            Page,
            Workspace,
            calculate_content_hash,
        )
        from knowledge.services import get_rag_service

        target_str = title_or_url.strip()
        is_url = source_type == "web" or target_str.startswith("http://") or target_str.startswith("https://")
        ctx.deps.record_tool_start("add_source_to_knowledge_base", f"Indexando fuente: '{target_str[:35]}'…", "journal-plus")

        try:
            rag = get_rag_service()
            raw_text = ""
            filename = ""

            if is_url:
                url = target_str if (target_str.startswith("http://") or target_str.startswith("https://")) else f"https://{target_str}"
                loop = asyncio.get_event_loop()
                def fetch_url():
                    req = urllib.request.Request(
                        url,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}
                    )
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        raw_html = resp.read().decode("utf-8", errors="ignore")
                        return extract_clean_text_from_html(raw_html)

                try:
                    raw_text = await loop.run_in_executor(None, fetch_url)
                except Exception as fetch_err:
                    if content.strip():
                        raw_text = content.strip()
                    else:
                        ctx.deps.record_tool_end("add_source_to_knowledge_base", f"Error descargando URL", status="error")
                        return {"error": f"No se pudo descargar la URL {url}: {str(fetch_err)}"}

                domain_part = url.replace("https://", "").replace("http://", "").split("/")[0].replace(":", "_")
                filename = f"{domain_part}_web_doc.txt"
            else:
                filename = (target_str or "Nota_Agente").replace("/", "_").replace("\\", "_")
                if not filename.endswith(".txt"):
                    filename += ".txt"
                raw_text = content.strip() or target_str

            if not raw_text:
                ctx.deps.record_tool_end("add_source_to_knowledge_base", "Sin contenido para la fuente", status="error")
                return {"error": "No se pudo extraer ni proporcionar contenido para la fuente."}

            import hashlib
            digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
            ctx.deps.settings.allowed_upload_dir.mkdir(parents=True, exist_ok=True)
            temp_file = ctx.deps.settings.allowed_upload_dir / f"{digest[:16]}-{filename}"
            temp_file.write_text(raw_text, encoding="utf-8")

            report = await rag.ingestor.ingest(temp_file)
            doc_id = report.get("document_id")

            @sync_to_async
            def link_in_django():
                doc = Document.objects.filter(id=doc_id).first()
                if not doc:
                    doc = Document.objects.filter(source_path=str(temp_file)).first()
                if not doc:
                    doc = Document.objects.filter(content_hash=digest).first()
                if not doc:
                    doc = Document.objects.create(
                        id=doc_id,
                        original_filename=filename,
                        media_type="text",
                        byte_size=len(raw_text.encode("utf-8")),
                        source_path=str(temp_file),
                        content_hash=digest,
                        status=report.get("status", "indexed"),
                    )
                target_nb = None
                target_nb_id = notebook_id or ctx.deps.notebook_id
                if target_nb_id:
                    target_nb = Notebook.objects.filter(id=target_nb_id).first()
                if not target_nb and ctx.deps.page_id:
                    p = Page.objects.filter(id=ctx.deps.page_id).first()
                    if p:
                        target_nb = p.notebook
                if not target_nb and ctx.deps.workspace_id:
                    target_nb = Notebook.objects.filter(workspace_id=ctx.deps.workspace_id).first()
                if not target_nb:
                    ws = Workspace.objects.first()
                    target_nb = Notebook.objects.filter(workspace=ws).first() if ws else Notebook.objects.first()
                if not target_nb:
                    ws = Workspace.objects.first() or Workspace.objects.create(name="Mi Espacio de Trabajo")
                    target_nb = Notebook.objects.create(workspace=ws, name="General")

                if target_nb and doc:
                    NotebookDocument.objects.get_or_create(notebook=target_nb, document=doc)

                return {
                    "status": "success",
                    "filename": filename,
                    "notebook_id": str(target_nb.id),
                    "notebook": target_nb.name,
                    "document_id": str(doc.id),
                    "chunk_count": report.get("chunk_count", 0),
                    "message": f"Fuente '{filename}' indexada con éxito en el cuaderno '{target_nb.name}'.",
                }

            res = await link_in_django()

            if res.get("status") == "success":
                ctx.deps.record_tool_end("add_source_to_knowledge_base", f"Fuente indexada en '{res.get('notebook', 'cuaderno')}'")
                if ctx.deps.on_tool_event:
                    ctx.deps.on_tool_event({
                        "type": "source_added",
                        "notebook_id": res["notebook_id"],
                        "notebook_name": res["notebook"],
                        "filename": res["filename"],
                        "document_id": res["document_id"],
                        "title": res["filename"],
                    })
            else:
                ctx.deps.record_tool_end("add_source_to_knowledge_base", "Fallo al indexar", status="error")

            return res
        except Exception as e:
            ctx.deps.record_tool_end("add_source_to_knowledge_base", f"Error: {str(e)[:30]}", status="error")
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

        ctx.deps.record_tool_start("manage_memory", f"Accediendo a la memoria persistente ({action})…", "brain")

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
            res = await sync_to_async(_db_op)()
            ctx.deps.record_tool_end("manage_memory", f"Memoria ({action}) procesada")
            return res
        except Exception as e:
            ctx.deps.record_tool_end("manage_memory", f"Error: {str(e)[:30]}", status="error")
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

        ctx.deps.record_tool_start("manage_skill", f"Gestionando habilidad: '{name or action}'…", "lightning")

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
            res = await sync_to_async(_skill_op)()
            ctx.deps.record_tool_end("manage_skill", f"Habilidad '{name or action}' procesada")
            return res
        except Exception as e:
            ctx.deps.record_tool_end("manage_skill", f"Error: {str(e)[:30]}", status="error")
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

        page_title = title.strip() or "Nueva Página"
        ctx.deps.record_tool_start("create_workspace_page", f"Creando página de notas: '{page_title[:35]}'…", "file-earmark-plus")

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
                ctx.deps.record_tool_end("create_workspace_page", f"Página '{page_title}' lista para redactar")
            else:
                ctx.deps.record_tool_end("create_workspace_page", "Fallo al crear página", status="error")
            return result
        except Exception as e:
            ctx.deps.record_tool_end("create_workspace_page", f"Error: {str(e)[:30]}", status="error")
            return {"error": f"No se pudo crear la página: {str(e)}"}

    @agent.tool
    async def update_page_notes(
        ctx: RunContext[AgentDeps],
        page_id: str | None = None,
    ) -> dict:
        """Prepara la página actual (o una especificada) para recibir contenido adicional. No recibe el
        contenido: escríbelo como tu siguiente respuesta de texto normal, se transmite en vivo a la página."""
        from knowledge.models import Page

        ctx.deps.record_tool_start("update_page_notes", "Preparando notas de la página…", "pencil-square")

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
                ctx.deps.record_tool_end("update_page_notes", f"Página '{result.get('title', '')}' lista para escribir")
            else:
                ctx.deps.record_tool_end("update_page_notes", "Fallo al preparar página", status="error")
            return result
        except Exception as e:
            ctx.deps.record_tool_end("update_page_notes", f"Error: {str(e)[:30]}", status="error")
            return {"error": f"No se pudo preparar la página: {str(e)}"}

    @agent.tool
    async def get_workspace_structure(ctx: RunContext[AgentDeps]) -> dict:
        """Obtiene la jerarquía completa de cuadernos y páginas disponibles en el espacio de trabajo actual."""
        from knowledge.models import Workspace

        ctx.deps.record_tool_start("get_workspace_structure", "Explorando jerarquía del espacio de trabajo…", "folder2-open")

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
            res = await sync_to_async(_tree)()
            ctx.deps.record_tool_end("get_workspace_structure", "Estructura del espacio cargada")
            return res
        except Exception as e:
            ctx.deps.record_tool_end("get_workspace_structure", f"Error: {str(e)[:30]}", status="error")
            return {"error": f"Error obteniendo estructura: {str(e)}"}

    @agent.tool
    async def search_past_conversations(
        ctx: RunContext[AgentDeps],
        query: str = "",
        scope: str | None = None,
        limit: int = 10,
    ) -> list[dict] | dict:
        """Lista o busca conversaciones de chat anteriores o pasadas (hilos distintos a la conversación en curso).
        
        IMPORTANTE: NUNCA uses esta herramienta para responder qué se ha hablado en la sesión o conversación actual ("¿de qué estamos hablando?", "resumen de lo hablado"). Esa información ya está en tu historial de mensajes inmediato.
        Úsala ÚNICAMENTE cuando el usuario haga referencia explícita a un hilo o conversación pasada/anterior.

        scope: 'notebook' (solo conversaciones del cuaderno activo) o 'workspace' (del espacio de trabajo). Si se omite, se ajusta automáticamente al ámbito activo."""
        from django.db.models import Q

        from knowledge.models import ChatThread, Notebook

        ctx.deps.record_tool_start("search_past_conversations", f"Consultando conversaciones anteriores: '{query[:30]}'…", "chat-square-text")

        def _search():
            effective_scope = scope
            if not effective_scope or effective_scope not in ("notebook", "workspace", "all"):
                effective_scope = "notebook" if ctx.deps.notebook_id else "workspace"

            qs = ChatThread.objects.all()
            if effective_scope == "notebook" or ctx.deps.notebook_id:
                if ctx.deps.notebook_id:
                    qs = qs.filter(notebook_id=ctx.deps.notebook_id)
                else:
                    return {"error": "No hay un cuaderno activo para acotar la búsqueda a 'notebook'."}
            elif effective_scope == "workspace":
                ws_id = ctx.deps.workspace_id
                if not ws_id and ctx.deps.notebook_id:
                    nb = Notebook.objects.filter(id=ctx.deps.notebook_id).first()
                    ws_id = nb.workspace_id if nb else None
                if not ws_id:
                    return {"error": "No hay un espacio de trabajo activo para acotar la búsqueda a 'workspace'."}
                qs = qs.filter(Q(workspace_id=ws_id) | Q(notebook__workspace_id=ws_id))

            if ctx.deps.thread_id:
                qs = qs.exclude(id=ctx.deps.thread_id)  # no listar la conversación en curso

            if query.strip():
                qs = qs.filter(Q(title__icontains=query) | Q(messages__content__icontains=query)).distinct()

            qs = qs.select_related("notebook", "notebook__workspace", "workspace").order_by("-updated_at")
            qs = qs[: max(1, min(limit, 30))]

            results = []
            for t in qs:
                last_msg = t.messages.order_by("-created_at").first()
                ws_name = t.workspace.name if t.workspace_id else (t.notebook.workspace.name if t.notebook_id else None)
                results.append({
                    "thread_id": t.id,
                    "title": t.title,
                    "notebook": t.notebook.name if t.notebook_id else None,
                    "workspace": ws_name,
                    "message_count": t.messages.count(),
                    "updated_at": t.updated_at.isoformat(),
                    "last_message_preview": (last_msg.content[:200] if last_msg else ""),
                })
            return results

        try:
            res = await sync_to_async(_search)()
            count = len(res) if isinstance(res, list) else 0
            ctx.deps.record_tool_end("search_past_conversations", f"{count} conversaciones revisadas")
            return res
        except Exception as e:
            ctx.deps.record_tool_end("search_past_conversations", f"Error: {str(e)[:30]}", status="error")
            return {"error": f"Error buscando conversaciones: {str(e)}"}

    @agent.tool
    async def get_conversation_messages(
        ctx: RunContext[AgentDeps],
        thread_id: str,
        limit: int = 40,
    ) -> dict:
        """Recupera los mensajes completos de una conversación pasada dada su `thread_id` (obtenida con `search_past_conversations`)."""
        from knowledge.models import ChatThread

        ctx.deps.record_tool_start("get_conversation_messages", "Leyendo mensajes de conversación previa…", "chat-left-dots")

        def _load():
            t = ChatThread.objects.select_related("notebook", "notebook__workspace", "workspace").filter(id=thread_id).first()
            if not t:
                return {"error": f"Conversación con ID {thread_id} no encontrada."}
            # Scope validation: enforce notebook segregation
            if ctx.deps.notebook_id and t.notebook_id and t.notebook_id != ctx.deps.notebook_id:
                return {"error": f"Acceso restringido: la conversación '{t.title}' pertenece a otro cuaderno ('{t.notebook.name}'). El ámbito actual está restringido a este cuaderno."}
            msgs = list(t.messages.order_by("created_at").values("role", "content", "created_at")[: max(1, min(limit, 100))])
            for m in msgs:
                m["created_at"] = m["created_at"].isoformat()
            ws_name = t.workspace.name if t.workspace_id else (t.notebook.workspace.name if t.notebook_id else None)
            return {
                "thread_id": t.id,
                "title": t.title,
                "notebook": t.notebook.name if t.notebook_id else None,
                "workspace": ws_name,
                "messages": msgs,
            }

        try:
            res = await sync_to_async(_load)()
            msg_count = len(res.get("messages", [])) if isinstance(res, dict) else 0
            ctx.deps.record_tool_end("get_conversation_messages", f"{msg_count} mensajes recuperados")
            return res
        except Exception as e:
            ctx.deps.record_tool_end("get_conversation_messages", f"Error: {str(e)[:30]}", status="error")
            return {"error": f"Error obteniendo la conversación: {str(e)}"}

    @agent.tool
    async def generate_notebook_artifact(
        ctx: RunContext[AgentDeps],
        artifact_type: ArtifactType,
        instructions: str = "",
        render_mode: str | None = None,
        notebook_id: str | None = None,
    ) -> dict:
        """Genera un artefacto de conocimiento para el cuaderno actual y lo guarda en la galería del Studio.
        Tipos soportados:
        - 'podcast': Genera un podcast de audio (.wav) con guion conversacional entre 2 locutores (Ana y Marco).
        - 'diagram': Genera un diagrama de arquitectura o flujo en formato Mermaid.js.
        - 'mindmap': Genera un mapa mental jerárquico (Mermaid.js o imagen ilustrada si render_mode='image').
        - 'quiz': Genera un cuestionario interactivo de opción múltiple con respuestas y explicaciones.
        - 'flashcards': Genera tarjetas de repaso y memoria activa en JSON.
        - 'study_guide': Genera una guía de estudio estructurada en Markdown.
        - 'summary': Genera un resumen ejecutivo de 1 página con viñetas y tablas.
        - 'infographic': Genera una infografía visual completa basada en IA.
        - 'timeline': Genera una línea de tiempo cronológica visual.
        """
        from knowledge.models import Notebook, NotebookArtifact, Page
        from knowledge.views import get_artifact_system_prompt, get_artifact_title, _collect_notebook_context
        from knowledge.artifact_media import build_media_artifact, describe_artifact_settings
        from knowledge.settings_store import get_current_language

        ctx.deps.record_tool_start("generate_notebook_artifact", f"Generando artefacto Studio ({artifact_type})…", "stars")

        target_nb_id = notebook_id or ctx.deps.notebook_id
        if not target_nb_id and ctx.deps.page_id:
            @sync_to_async
            def _find_nb():
                p = Page.objects.filter(id=ctx.deps.page_id).first()
                return p.notebook_id if p else None
            target_nb_id = await _find_nb()
        elif not target_nb_id and ctx.deps.workspace_id:
            @sync_to_async
            def _find_first_nb():
                nb = Notebook.objects.filter(workspace_id=ctx.deps.workspace_id).first()
                return nb.id if nb else None
            target_nb_id = await _find_first_nb()

        if not target_nb_id:
            ctx.deps.record_tool_end("generate_notebook_artifact", "No se encontró cuaderno", status="error")
            return {"error": "No se pudo determinar el cuaderno activo para generar el artefacto."}

        @sync_to_async
        def _get_nb():
            return Notebook.objects.filter(id=target_nb_id).first()

        notebook = await _get_nb()
        if not notebook:
            ctx.deps.record_tool_end("generate_notebook_artifact", "Cuaderno no encontrado", status="error")
            return {"error": f"Cuaderno con ID {target_nb_id} no encontrado."}

        try:
            language = get_current_language()
            is_en = language == "en"

            @sync_to_async
            def _get_context():
                return _collect_notebook_context(notebook, instructions, ctx.deps.retriever)

            full_context = await _get_context()

            preferences: dict = {}
            if render_mode:
                preferences["render_mode"] = render_mode

            is_media = artifact_type in {"podcast", "infographic", "timeline"} or (artifact_type == "mindmap" and render_mode == "image")

            if is_media:
                def on_progress(msg: str):
                    if ctx.deps.on_tool_event:
                        ctx.deps.on_tool_event({"type": "status", "message": msg})

                title, content, metadata = await build_media_artifact(
                    artifact_type=artifact_type,
                    notebook=notebook,
                    full_context=full_context,
                    custom_instructions=instructions,
                    settings=ctx.deps.settings,
                    on_progress=on_progress,
                    preferences=preferences,
                    language=language,
                )
            else:
                settings_directive = describe_artifact_settings(artifact_type, preferences, language=language)
                default_counts = {"quiz": "4", "flashcards": "5"}
                count = preferences.get("count") or default_counts.get(artifact_type, "")
                chosen_system = get_artifact_system_prompt(artifact_type, count=count, language=language)
                if is_en:
                    user_prompt = f"Notebook content for '{notebook.name}':\n{full_context}\n\nAdditional user instructions: {instructions}"
                    if settings_directive:
                        user_prompt += f"\n\nFormatting preferences: {settings_directive}"
                else:
                    user_prompt = f"Contenido del cuaderno '{notebook.name}':\n{full_context}\n\nInstrucciones adicionales del usuario: {instructions}"
                    if settings_directive:
                        user_prompt += f"\n\nPreferencias de formato: {settings_directive}"

                client = AsyncOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=ctx.deps.settings.openrouter_api_key or "sk-dummy",
                    http_client=new_async_client(),
                )
                completion = await client.chat.completions.create(
                    model=ctx.deps.settings.chat_model,
                    messages=[
                        {"role": "system", "content": chosen_system},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )
                content = completion.choices[0].message.content or ""
                title = get_artifact_title(artifact_type, notebook.name, language=language)
                metadata = {"render_mode": "mermaid"} if artifact_type == "mindmap" else {}

            @sync_to_async
            def _save():
                return NotebookArtifact.objects.create(
                    notebook=notebook,
                    artifact_type=artifact_type,
                    title=title,
                    content=content,
                    metadata_json=metadata,
                )

            saved_artifact = await _save()

            if ctx.deps.on_tool_event:
                ctx.deps.on_tool_event({
                    "type": "artifact_created",
                    "artifact_id": str(saved_artifact.id),
                    "artifact_type": saved_artifact.artifact_type,
                    "title": saved_artifact.title,
                })

            ctx.deps.record_tool_end("generate_notebook_artifact", f"Artefacto '{saved_artifact.title}' guardado")

            return {
                "status": "success",
                "artifact_id": str(saved_artifact.id),
                "artifact_type": saved_artifact.artifact_type,
                "title": saved_artifact.title,
                "message": f"Artefacto '{saved_artifact.title}' generado con éxito y guardado en la galería Studio del cuaderno.",
            }
        except Exception as e:
            ctx.deps.record_tool_end("generate_notebook_artifact", f"Error: {str(e)[:30]}", status="error")
            return {"error": f"Error generando artefacto: {str(e)}"}

    return agent
