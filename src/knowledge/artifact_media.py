from __future__ import annotations

import json
import wave
from io import BytesIO
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from ragpoc.config import Settings
from ragpoc.openrouter_media import generate_image, synthesize_speech

OnProgress = Callable[[str], None]


def _chat_text(settings: Settings, system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=settings.openrouter_api_key)
    completion = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return completion.choices[0].message.content or ""


def _parse_json_block(text: str) -> dict:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


# Translates the simple, non-technical chip choices the user picks in the Studio settings panel
# (see ARTIFACT_SETTINGS in console.html) into a plain-language directive appended to the
# generation prompt. `render_mode` for mindmap is deliberately absent here — it's a routing
# decision (mermaid vs. image pipeline), not a prompt hint.
_SETTINGS_PHRASES: dict[str, dict[str, dict[str, str]]] = {
    "diagram": {
        "focus": {
            "flujo": "Enfócate en el flujo de un proceso.",
            "arquitectura": "Enfócate en la arquitectura general del sistema.",
            "secuencia": "Muestra la secuencia de pasos en orden, tipo diagrama de secuencia.",
        },
    },
    "mindmap": {
        "detail": {
            "simple": "Estructura simple y directa, máximo 4 ramas principales.",
            "detallado": "Incluye ramas y subramas con buen nivel de detalle.",
        },
    },
    "quiz": {
        "count": {
            "3": "Genera exactamente 3 preguntas.",
            "5": "Genera exactamente 5 preguntas.",
            "8": "Genera exactamente 8 preguntas.",
        },
        "difficulty": {
            "facil": "Las preguntas deben ser de dificultad fácil.",
            "media": "Las preguntas deben ser de dificultad media.",
            "dificil": "Las preguntas deben ser desafiantes y de dificultad alta.",
        },
    },
    "study_guide": {
        "depth": {
            "basica": "Mantén la guía breve y básica, solo lo esencial.",
            "completa": "Genera una guía completa y detallada.",
        },
    },
    "flashcards": {
        "count": {
            "5": "Genera exactamente 5 tarjetas.",
            "10": "Genera exactamente 10 tarjetas.",
            "15": "Genera exactamente 15 tarjetas.",
        },
    },
    "summary": {
        "length": {
            "breve": "Mantén el resumen muy breve, medio página.",
            "estandar": "Resumen de 1 página.",
            "detallado": "Resumen detallado, hasta 2 páginas.",
        },
    },
    "infographic": {
        "style": {
            "minimalista": "Estilo visual minimalista y limpio.",
            "colorido": "Estilo visual colorido y llamativo.",
            "corporativo": "Estilo visual corporativo y serio.",
        },
        "format": {
            "vertical": "Formato vertical, tipo póster para imprimir.",
            "horizontal": "Formato horizontal, pensado para pantalla.",
        },
    },
    "timeline": {
        "detail": {
            "hitos": "Muestra solo los hitos principales, sin descripciones largas.",
            "descripciones": "Incluye una breve descripción junto a cada hito.",
        },
    },
    "podcast": {
        "duration": {
            "corto": "El episodio debe ser corto: entre 5 y 6 turnos de diálogo.",
            "estandar": "El episodio debe tener duración estándar: entre 8 y 10 turnos de diálogo.",
            "largo": "El episodio debe ser largo: entre 12 y 14 turnos de diálogo.",
        },
        "tone": {
            "informal": "Tono informal, cercano y ameno.",
            "formal": "Tono formal y educativo.",
        },
    },
}


def describe_artifact_settings(artifact_type: str, preferences: dict | None) -> str:
    """Builds a one-line, plain-language directive from the user's chip selections, meant to be
    appended to the generation prompt alongside their free-text instructions."""
    phrases = _SETTINGS_PHRASES.get(artifact_type, {})
    lines = []
    for key, value in (preferences or {}).items():
        phrase = phrases.get(key, {}).get(value)
        if phrase:
            lines.append(phrase)
    return " ".join(lines)


def _augment_instructions(custom_instructions: str, artifact_type: str, preferences: dict | None) -> str:
    directive = describe_artifact_settings(artifact_type, preferences)
    if not directive:
        return custom_instructions
    return f"{custom_instructions}\n\nPreferencias de formato: {directive}".strip()


def _save_media(settings: Settings, data: bytes, ext: str) -> Path:
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = settings.artifacts_dir / f"{uuid4().hex}.{ext}"
    path.write_bytes(data)
    return path


async def build_media_artifact(
    *,
    artifact_type: str,
    notebook: Any,
    full_context: str,
    custom_instructions: str,
    settings: Settings,
    on_progress: OnProgress,
    preferences: dict | None = None,
) -> tuple[str, str, dict]:
    """Runs the 2-stage (extract-then-generate) pipeline for a media-based artifact and returns
    (title, content, metadata_json) ready to save on a NotebookArtifact."""
    custom_instructions = _augment_instructions(custom_instructions, artifact_type, preferences)
    if artifact_type == "mindmap":
        return await _build_mindmap_image(notebook, full_context, custom_instructions, settings, on_progress)
    if artifact_type == "infographic":
        return await _build_poster(
            notebook, full_context, custom_instructions, settings, on_progress,
            artifact_label="Infografía",
            extraction_prompt=_INFOGRAPHIC_EXTRACTION_PROMPT,
            image_prompt_builder=_infographic_image_prompt,
            aspect_ratio="3:4",
        )
    if artifact_type == "timeline":
        return await _build_poster(
            notebook, full_context, custom_instructions, settings, on_progress,
            artifact_label="Línea de Tiempo",
            extraction_prompt=_TIMELINE_EXTRACTION_PROMPT,
            image_prompt_builder=_timeline_image_prompt,
            aspect_ratio="16:9",
        )
    if artifact_type == "podcast":
        return await _build_podcast(notebook, full_context, custom_instructions, settings, on_progress)
    raise ValueError(f"Unsupported media artifact_type: {artifact_type}")


# ---------------------------------------------------------------------------
# Mapa mental ilustrado
# ---------------------------------------------------------------------------

_MINDMAP_OUTLINE_PROMPT = (
    "Eres un experto en organización visual del conocimiento. A partir de las notas, extrae una jerarquía de "
    'conceptos en JSON estricto: {"topic": "tema central", "branches": [{"label": "subtema", "children": ["detalle 1", "detalle 2"]}]}. '
    "Máximo 6 ramas, máximo 4 hijos por rama. Responde solo el JSON."
)


async def _build_mindmap_image(notebook, full_context, custom_instructions, settings, on_progress: OnProgress):
    on_progress("Extrayendo jerarquía de conceptos…")
    outline_raw = _chat_text(
        settings, _MINDMAP_OUTLINE_PROMPT,
        f"Notas del cuaderno '{notebook.name}':\n{full_context}\n\nInstrucciones adicionales: {custom_instructions}",
    )
    outline = _parse_json_block(outline_raw)

    on_progress("Ilustrando el mapa mental…")
    image_prompt = (
        f"Póster de mapa mental limpio y profesional en español, tema central '{outline.get('topic', '')}'. "
        f"Estructura jerárquica clara con ramas de colores distintos y tipografía legible, fondo blanco. "
        f"Ramas y subtemas: {json.dumps(outline.get('branches', []), ensure_ascii=False)}. "
        "Estilo infografía educativa minimalista, sin texto ilegible."
    )
    image_bytes = await generate_image(image_prompt, settings=settings, aspect_ratio="1:1")
    media_path = _save_media(settings, image_bytes, "png")

    title = f"Mapa Mental · {notebook.name}"
    metadata = {"render_mode": "image", "media_path": str(media_path), "mime_type": "image/png", "outline": outline}
    on_progress("Listo.")
    return title, json.dumps(outline, ensure_ascii=False), metadata


# ---------------------------------------------------------------------------
# Infografía y línea de tiempo (comparten el mismo pipeline de 2 etapas)
# ---------------------------------------------------------------------------

_INFOGRAPHIC_EXTRACTION_PROMPT = (
    "Eres un diseñador de infografías. A partir de las notas, extrae un brief de diseño en JSON estricto: "
    '{"title": "...", "sections": [{"heading": "...", "points": ["...", "..."]}], "stats": [{"label": "...", "value": "..."}]}. '
    "Máximo 4 secciones, máximo 4 puntos por sección, máximo 4 estadísticas destacadas. Responde solo el JSON."
)

_TIMELINE_EXTRACTION_PROMPT = (
    "Eres un historiador y diseñador de líneas de tiempo. A partir de las notas, extrae los hitos cronológicos en "
    'JSON estricto: {"title": "...", "events": [{"order": "fecha o número de secuencia", "label": "...", "description": "..."}]}. '
    "Ordena los eventos cronológicamente. Máximo 8 eventos. Responde solo el JSON."
)


def _infographic_image_prompt(brief: dict) -> str:
    return (
        f"Infografía vertical profesional en español, título '{brief.get('title', '')}'. "
        f"Secciones con iconos y jerarquía visual clara: {json.dumps(brief.get('sections', []), ensure_ascii=False)}. "
        f"Estadísticas destacadas en tarjetas: {json.dumps(brief.get('stats', []), ensure_ascii=False)}. "
        "Estilo editorial limpio, paleta de 2-3 colores, tipografía legible, sin texto ilegible ni artefactos."
    )


def _timeline_image_prompt(brief: dict) -> str:
    return (
        f"Línea de tiempo horizontal profesional en español, título '{brief.get('title', '')}'. "
        f"Eventos en orden cronológico con marcadores y fechas visibles: {json.dumps(brief.get('events', []), ensure_ascii=False)}. "
        "Estilo infografía editorial limpia, un solo eje horizontal, tipografía legible, sin texto ilegible."
    )


async def _build_poster(
    notebook, full_context, custom_instructions, settings, on_progress: OnProgress, *,
    artifact_label: str, extraction_prompt: str, image_prompt_builder: Callable[[dict], str], aspect_ratio: str,
):
    on_progress(f"Extrayendo contenido para {artifact_label.lower()}…")
    brief_raw = _chat_text(
        settings, extraction_prompt,
        f"Notas del cuaderno '{notebook.name}':\n{full_context}\n\nInstrucciones adicionales: {custom_instructions}",
    )
    brief = _parse_json_block(brief_raw)

    on_progress(f"Generando imagen de {artifact_label.lower()}…")
    image_prompt = image_prompt_builder(brief)
    image_bytes = await generate_image(image_prompt, settings=settings, aspect_ratio=aspect_ratio)
    media_path = _save_media(settings, image_bytes, "png")

    title = f"{artifact_label} · {notebook.name}"
    metadata = {"media_path": str(media_path), "mime_type": "image/png", "design_brief": brief}
    on_progress("Listo.")
    return title, json.dumps(brief, ensure_ascii=False), metadata


# ---------------------------------------------------------------------------
# Podcast
# ---------------------------------------------------------------------------

_PODCAST_SCRIPT_PROMPT = (
    "Eres guionista de un podcast educativo de 2 conductores (Ana y Marco) que conversan de forma natural y "
    "dinámica sobre las notas proporcionadas, explicando los conceptos clave como en un episodio real. "
    "Entre 8 y 14 turnos de diálogo, turnos cortos (2-4 frases). Devuelve JSON estricto: "
    '{"title": "...", "turns": [{"speaker": "A", "text": "..."}, {"speaker": "B", "text": "..."}]}. Responde solo el JSON.'
)


async def _build_podcast(notebook, full_context, custom_instructions, settings, on_progress: OnProgress):
    on_progress("Escribiendo el guion del podcast…")
    script_raw = _chat_text(
        settings, _PODCAST_SCRIPT_PROMPT,
        f"Notas del cuaderno '{notebook.name}':\n{full_context}\n\nInstrucciones adicionales: {custom_instructions}",
    )
    script = _parse_json_block(script_raw)
    turns = [t for t in script.get("turns", []) if (t.get("text") or "").strip()]
    if not turns:
        raise ValueError("El modelo no generó ningún turno de diálogo para el podcast.")

    voice_by_speaker = {"A": settings.tts_voice_a, "B": settings.tts_voice_b}
    clips: list[bytes] = []
    total = len(turns)
    for idx, turn in enumerate(turns, start=1):
        on_progress(f"Sintetizando voz {idx}/{total}…")
        voice = voice_by_speaker.get(turn.get("speaker"), settings.tts_voice_a)
        # Gemini TTS (the configured tts_model) only supports response_format="pcm" — raw
        # headerless 16-bit/24kHz/mono samples — not "mp3"/"wav".
        clip = await synthesize_speech(turn["text"].strip(), settings=settings, voice=voice, audio_format="pcm")
        clips.append(clip)

    on_progress("Uniendo audio…")
    combined = _pcm_clips_to_wav(clips)
    media_path = _save_media(settings, combined, "wav")

    title = f"Podcast · {notebook.name}"
    transcript = "\n\n".join(f"{t.get('speaker', '?')}: {t.get('text', '')}" for t in turns)
    metadata = {
        "media_path": str(media_path),
        "mime_type": "audio/wav",
        "script": turns,
    }
    on_progress("Listo.")
    return title, transcript, metadata


_PCM_SAMPLE_RATE_HZ = 24_000
_PCM_SAMPLE_WIDTH_BYTES = 2
_PCM_CHANNELS = 1


def _pcm_clips_to_wav(clips: list[bytes]) -> bytes:
    """Concatenates raw (headerless) PCM clips and wraps the result in a single WAV container via
    the stdlib `wave` module — no ffmpeg/pydub dependency needed. Since PCM has no per-clip header,
    simple byte concatenation is exact/lossless as long as every clip shares the same sample
    rate/width/channel count, which holds because they all come from the same TTS model/voice
    family for a single podcast generation (confirmed empirically: OpenRouter's Gemini TTS always
    returns `audio/pcm;rate=24000;channels=1`, 16-bit samples)."""
    if not clips:
        raise ValueError("No hay clips de audio para unir.")
    raw = b"".join(clips)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_out:
        wav_out.setnchannels(_PCM_CHANNELS)
        wav_out.setsampwidth(_PCM_SAMPLE_WIDTH_BYTES)
        wav_out.setframerate(_PCM_SAMPLE_RATE_HZ)
        wav_out.writeframes(raw)
    return buffer.getvalue()
