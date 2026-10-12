from __future__ import annotations

from typing import Any

from ragpoc.config import Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

PROVIDER_OPENROUTER = "openrouter"
PROVIDER_GOOGLE = "google"
PROVIDER_LOCAL = "local"

# Presets shown in the Ajustes UI when switching provider (user-editable).
PROVIDER_MODEL_PRESETS: dict[str, dict[str, str]] = {
    PROVIDER_OPENROUTER: {
        "chat_model": "google/gemini-3.7-flash",
        "embedding_model": "google/gemini-embedding-2-preview",
    },
    PROVIDER_GOOGLE: {
        "chat_model": "gemini-3.7-flash",
        "embedding_model": "gemini-embedding-2-preview",
    },
    PROVIDER_LOCAL: {
        "chat_model": "qwen3-vl-2b",
        "embedding_model": "local",
    },
}

_KNOWN_PROVIDERS = (PROVIDER_OPENROUTER, PROVIDER_GOOGLE, PROVIDER_LOCAL)


class LLMConfigError(RuntimeError):
    pass


def active_llm_provider(settings: Settings) -> str:
    provider = (settings.llm_provider or PROVIDER_OPENROUTER).strip().lower()
    if provider not in _KNOWN_PROVIDERS:
        raise LLMConfigError(
            f"Unknown llm_provider {settings.llm_provider!r}; "
            f"expected one of {', '.join(_KNOWN_PROVIDERS)}"
        )
    return provider


def is_local_provider(settings: Settings) -> bool:
    return active_llm_provider(settings) == PROVIDER_LOCAL


def llm_base_url(settings: Settings) -> str:
    if is_local_provider(settings):
        return ""
    return GOOGLE_BASE_URL if active_llm_provider(settings) == PROVIDER_GOOGLE else OPENROUTER_BASE_URL


def llm_api_key(settings: Settings) -> str | None:
    if is_local_provider(settings):
        return None
    if active_llm_provider(settings) == PROVIDER_GOOGLE:
        return settings.google_api_key
    return settings.openrouter_api_key


def llm_provider_label(settings: Settings) -> str:
    provider = active_llm_provider(settings)
    if provider == PROVIDER_LOCAL:
        return "Local (sin API)"
    return "Google AI Studio" if provider == PROVIDER_GOOGLE else "OpenRouter"


def friendly_llm_error(exc: Exception, settings: Settings) -> str:
    """Turn a model-call exception into something the user can act on.

    The chat stream used to render ``str(exc)`` verbatim as the assistant's
    answer, which for the most common failure -- no route to the cloud endpoint
    -- was openai's bare ``"Connection error."``: no provider named, no hint
    about what to do. That is how a user ends up believing local mode is broken
    while the app is silently still on the cloud provider. Keep the technical
    detail visible, but say who failed and how to recover.
    """
    detail = (str(exc) or exc.__class__.__name__).strip()
    try:
        local = is_local_provider(settings)
        label = llm_provider_label(settings)
    except Exception:
        local, label = False, "el proveedor de IA"

    if isinstance(exc, LLMConfigError):
        # Already written for the user ("Configura tu API key…", "esta acción
        # necesita Modo Nube…"): re-wrapping would bury the instruction.
        return detail

    connection_like = (
        detail in {"Connection error.", "Connection error"}
        or "APIConnectionError" in type(exc).__name__
        or "connection" in detail.lower()
    )
    if connection_like and not local:
        return (
            f"No se pudo conectar con {label} ({detail}). "
            "Comprueba tu conexión a internet. Para usar la aplicación sin conexión, "
            "cambia a Modo Local en Ajustes (necesita el modelo descargado)."
        )
    if local and "exceeds max length" in detail:
        # The prompt outgrew the context window; the window is the knob.
        return (
            f"El mensaje es demasiado largo para la ventana de contexto ({detail}). "
            "Súbelo en Ajustes → Modelo local → Ventana de contexto (prueba con 4096) "
            "o adjunta menos contenido a la conversación."
        )
    if local:
        return (
            f"Error del modelo local ({detail}). "
            "Revisa Ajustes → Modelo local: el modelo debe estar descargado y con margen de RAM."
        )
    return f"Error de {label}: {detail}"


def require_api_key(settings: Settings) -> str:
    if is_local_provider(settings):
        # The local runtime needs no credential; the model is a file on disk.
        return ""
    key = llm_api_key(settings)
    if not key:
        if active_llm_provider(settings) == PROVIDER_GOOGLE:
            raise LLMConfigError(
                "Configura tu Google AI Studio API key en Ajustes o en GOOGLE_API_KEY."
            )
        raise LLMConfigError("Configura tu OpenRouter API key en Ajustes o en OPENROUTER_API_KEY.")
    return key


def build_openai_client(settings: Settings, *, timeout: float = 120.0, async_: bool = True):
    """Build an (Async)OpenAI client pointed at the active provider's OpenAI-compatible endpoint.

    A missing key keeps the historical deferred behavior (placeholder credential, failure
    surfaces at request time); callers that want an early friendly error use `require_api_key`.
    """
    if is_local_provider(settings):
        # There is no endpoint behind Modo Local: the model is a file on disk and
        # runs through ragpoc.local_llm. Reaching this client in local mode is a
        # routing bug, and the old behavior -- a dummy key against an empty base
        # URL -- surfaced as a baffling "Connection error." in the UI.
        raise LLMConfigError(
            "Esta acción no está disponible en Modo Local: usa Modo Nube en Ajustes → "
            "Proveedor de IA, o ejecuta la acción con el modelo local desde el chat."
        )
    api_key = llm_api_key(settings) or "sk-dummy"
    base_url = llm_base_url(settings)
    if async_:
        from openai import AsyncOpenAI

        from ragpoc.http import new_async_client

        return AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=new_async_client(timeout=timeout),
        )
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def google_model_name(name: str) -> str:
    """Google AI Studio model IDs don't carry OpenRouter's `google/` vendor prefix."""
    return name[len("google/"):] if name.startswith("google/") else name


def build_chat_model(settings: Settings, client=None):
    """Build the pydantic-ai chat model for the active provider.

    Google uses pydantic-ai's native GoogleModel (google-genai) rather than the OpenAI-compatible
    layer: Gemini 3.x refuses tool-call histories whose functionCall parts lack the
    `thought_signature` of the original response, and only the native integration round-trips
    those signatures (the OpenAI-compat view strips them).

    Local runs the ONNX GenAI engine in-process through a FunctionModel adapter:
    no endpoint, no key, and the agent's tools work unchanged.
    """
    provider = active_llm_provider(settings)

    if provider == PROVIDER_LOCAL:
        from knowledge.local_agent_model import build_local_agent_model

        engine = _shared_local_engine(settings)
        return build_local_agent_model(
            engine,
            name=settings.local_chat_model,
            transcriber=_shared_transcriber(settings),
        ).model

    if provider == PROVIDER_GOOGLE:
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleModel(
            google_model_name(settings.chat_model),
            provider=GoogleProvider(api_key=settings.google_api_key),
        )

    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    client = client or build_openai_client(settings)
    return OpenRouterModel(
        settings.chat_model,
        provider=OpenRouterProvider(openai_client=client),
    )


# One engine per process: loading the ONNX session costs ~13 s and ~1.8 GB, so
# rebuilding it per request would be both slow and a memory leak.
_LOCAL_ENGINES: dict[str, Any] = {}


def active_model_name(settings: Settings) -> str:
    """Model id to log and display: the local checkpoint in local mode, the cloud id otherwise."""
    return settings.local_chat_model if is_local_provider(settings) else settings.chat_model


def _local_messages(system: str, user: str):
    from ragpoc.local_llm import ChatMessage

    messages = []
    if system:
        messages.append(ChatMessage(role="system", content=system))
    messages.append(ChatMessage(role="user", content=user))
    return messages


def local_completion(settings: Settings, system: str, user: str, *, temperature: float = 0.2, max_new_tokens: int = 1024):
    """Single-turn completion on the local engine (no tools, no network).

    The inline AI actions and text artifacts are plain system+user completions;
    this is their Modo Local path, so the app works the same in both modes.
    """
    engine = _shared_local_engine(settings)
    engine.load()
    return engine.generate(_local_messages(system, user), max_new_tokens=max_new_tokens, temperature=temperature)


def local_completion_stream(settings: Settings, system: str, user: str, *, temperature: float = 0.2, max_new_tokens: int = 1024):
    """Streaming variant of local_completion: yields text pieces as generated."""
    engine = _shared_local_engine(settings)
    engine.load()
    return engine.generate_stream(_local_messages(system, user), max_new_tokens=max_new_tokens, temperature=temperature)


def local_last_metrics(settings: Settings):
    """GenerationMetrics of the last generate_stream() on the shared engine.

    ``generate_stream`` leaves them on the engine (``generate`` reads them the
    same way), so streaming callers can log token counts after the fact.
    """
    return getattr(_shared_local_engine(settings), "_last_metrics", None)


def _shared_local_engine(settings: Settings):
    from ragpoc.local_llm import engine_from_settings

    key = f"{settings.local_chat_model}|{settings.local_chat_context_length}|{settings.local_chat_device}"
    engine = _LOCAL_ENGINES.get(key)
    if engine is None:
        engine = engine_from_settings(settings)
        engine.load()
        _LOCAL_ENGINES[key] = engine
    return engine


def _shared_transcriber(settings: Settings):
    """Lazy shared whisper transcriber, loaded only when audio evidence appears."""
    global _SHARED_TRANSCRIBER
    if _SHARED_TRANSCRIBER is not None:
        return _SHARED_TRANSCRIBER
    from pathlib import Path

    from ragpoc.whisper_transcribe import WhisperTranscriber

    root = Path(settings.local_chat_cache_dir) / "whisper-tiny"
    if not (root / "genai_config.json").exists():
        return None
    try:
        transcriber = WhisperTranscriber(root, language="es")
    except Exception:
        return None
    _SHARED_TRANSCRIBER = transcriber
    return transcriber


_SHARED_TRANSCRIBER: Any = None


def reset_local_engine() -> None:
    """Drop the cached engine (used when the model or settings change)."""
    global _SHARED_TRANSCRIBER
    for engine in _LOCAL_ENGINES.values():
        try:
            engine.unload()
        except Exception:
            continue
    _LOCAL_ENGINES.clear()
    if _SHARED_TRANSCRIBER is not None:
        try:
            _SHARED_TRANSCRIBER.unload()
        except Exception:
            pass
        _SHARED_TRANSCRIBER = None
