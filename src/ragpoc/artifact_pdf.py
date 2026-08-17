from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from pathlib import Path

import pymupdf
from markdown_it import MarkdownIt

_md = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"])

_PAGE_CSS = """
body { font-family: sans-serif; font-size: 11pt; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 18pt; margin: 14pt 0 8pt 0; }
h2 { font-size: 15pt; margin: 14pt 0 7pt 0; }
h3 { font-size: 13pt; margin: 12pt 0 6pt 0; }
p { margin: 0 0 8pt 0; }
ul, ol { margin: 0 0 8pt 0; padding-left: 18pt; }
li { margin-bottom: 4pt; }
code { font-family: monospace; background-color: #f0f0f0; padding: 1pt 3pt; }
pre { font-family: monospace; background-color: #f0f0f0; padding: 8pt; white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%; margin-bottom: 8pt; }
th, td { border: 0.5pt solid #999; padding: 4pt 6pt; text-align: left; }
blockquote { border-left: 2pt solid #999; margin: 0 0 8pt 0; padding-left: 10pt; color: #444; }
.artifact-title { font-size: 20pt; font-weight: bold; margin-bottom: 16pt; }
.qa-block { margin-bottom: 14pt; padding-bottom: 10pt; border-bottom: 0.5pt solid #ddd; }
.qa-question { font-weight: bold; margin-bottom: 6pt; }
.qa-option { margin: 2pt 0 2pt 10pt; }
.qa-option-correct { font-weight: bold; color: #146c2e; }
.qa-explanation { margin-top: 6pt; color: #444; }
.card-front { font-weight: bold; margin-bottom: 4pt; }
.card-back { margin-bottom: 14pt; color: #333; }
.artifact-media-img { max-width: 100%; margin-bottom: 10pt; }
.podcast-turn { margin-bottom: 10pt; }
.podcast-speaker { font-weight: bold; }
.podcast-note { color: #444; font-style: italic; margin-bottom: 14pt; }
"""


def _esc(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _strip_fence(content: str, lang: str) -> str:
    return re.sub(rf"```{lang}", "", content or "", flags=re.IGNORECASE).replace("```", "").strip()


def _render_quiz(content: str, metadata: dict) -> str:
    try:
        parsed = json.loads(_strip_fence(content, "json"))
    except (json.JSONDecodeError, TypeError):
        return f"<pre>{_esc(content)}</pre>"

    questions = parsed.get("questions", parsed) if isinstance(parsed, dict) else parsed
    if not isinstance(questions, list):
        return f"<pre>{_esc(content)}</pre>"

    blocks = []
    for idx, q in enumerate(questions, start=1):
        correct_idx = q.get("correct_index", q.get("answer_index", -1))
        options_html = "".join(
            f'<div class="qa-option{" qa-option-correct" if opt_idx == correct_idx else ""}">'
            f'{"&#10003; " if opt_idx == correct_idx else ""}{_esc(opt)}</div>'
            for opt_idx, opt in enumerate(q.get("options", []))
        )
        explanation = q.get("explanation", "")
        exp_html = f'<div class="qa-explanation">{_esc(explanation)}</div>' if explanation else ""
        blocks.append(
            f'<div class="qa-block"><div class="qa-question">{idx}. {_esc(q.get("question", ""))}</div>'
            f"{options_html}{exp_html}</div>"
        )
    return "".join(blocks)


def _render_flashcards(content: str, metadata: dict) -> str:
    try:
        parsed = json.loads(_strip_fence(content, "json"))
    except (json.JSONDecodeError, TypeError):
        return f"<pre>{_esc(content)}</pre>"

    cards = parsed.get("cards", parsed) if isinstance(parsed, dict) else parsed
    if not isinstance(cards, list):
        return f"<pre>{_esc(content)}</pre>"

    return "".join(
        f'<div class="qa-block"><div class="card-front">{idx}. {_esc(card.get("front", ""))}</div>'
        f'<div class="card-back">{_esc(card.get("back", ""))}</div></div>'
        for idx, card in enumerate(cards, start=1)
    )


def _render_diagram(content: str, metadata: dict) -> str:
    code = _strip_fence(content, "mermaid")
    return (
        "<p><i>Este artefacto es un diagrama Mermaid. A continuación su código fuente:</i></p>"
        f"<pre>{_esc(code)}</pre>"
    )


def _render_mindmap(content: str, metadata: dict) -> str:
    if (metadata or {}).get("render_mode") == "image":
        return _render_image(content, metadata)
    return _render_diagram(content, metadata)


def _image_data_uri(media_path: str, mime_type: str) -> str | None:
    path = Path(media_path)
    if not path.exists() or not path.is_file():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _render_image(content: str, metadata: dict) -> str:
    metadata = metadata or {}
    media_path = metadata.get("media_path")
    mime_type = metadata.get("mime_type", "image/png")
    data_uri = _image_data_uri(media_path, mime_type) if media_path else None
    if not data_uri:
        return "<p><i>Imagen no disponible.</i></p>"
    return f'<img class="artifact-media-img" src="{data_uri}">'


def _render_podcast(content: str, metadata: dict) -> str:
    turns = (metadata or {}).get("script") or []
    note = '<p class="podcast-note">Este artefacto es un episodio de audio. A continuación su transcripción:</p>'
    if not turns:
        return note + f"<pre>{_esc(content)}</pre>"
    blocks = "".join(
        f'<div class="podcast-turn"><span class="podcast-speaker">{_esc(t.get("speaker", "?"))}:</span> '
        f'{_esc(t.get("text", ""))}</div>'
        for t in turns
    )
    return note + blocks


_RENDERERS = {
    "quiz": _render_quiz,
    "flashcards": _render_flashcards,
    "diagram": _render_diagram,
    "mindmap": _render_mindmap,
    "infographic": _render_image,
    "timeline": _render_image,
    "podcast": _render_podcast,
}


def _paginate_html(full_html: str) -> bytes:
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (40, 40, -40, -40)
    buffer = BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    story = pymupdf.Story(html=full_html, user_css=_PAGE_CSS)
    more = 1
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()
    return buffer.getvalue()


def render_artifact_pdf(title: str, artifact_type: str, content: str, metadata_json: dict | None = None) -> bytes:
    """Transforms an artifact's own stored content (markdown / structured JSON / mermaid source /
    generated image / podcast script) directly into a native, paginated, text-based PDF — no DOM
    screenshotting involved, so the full content is always included regardless of how long it is."""
    renderer = _RENDERERS.get(artifact_type, lambda c, m: _md.render(c or ""))
    body_html = renderer(content, metadata_json or {})
    full_html = f'<div class="artifact-title">{_esc(title)}</div>{body_html}'
    return _paginate_html(full_html)


def render_page_pdf(title: str, html: str) -> bytes:
    """Turns a Page's already-rendered HTML (Tiptap's own editor.getHTML() output) into a
    native, paginated, text-based PDF via the same Story pipeline used for artifacts."""
    full_html = f'<div class="artifact-title">{_esc(title)}</div>{html or ""}'
    return _paginate_html(full_html)
