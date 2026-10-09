from __future__ import annotations

from ragpoc.config import Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

PROVIDER_OPENROUTER = "openrouter"
PROVIDER_GOOGLE = "google"

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
}


class LLMConfigError(RuntimeError):
    pass


def active_llm_provider(settings: Settings) -> str:
    provider = (settings.llm_provider or PROVIDER_OPENROUTER).strip().lower()
    if provider not in (PROVIDER_OPENROUTER, PROVIDER_GOOGLE):
        raise LLMConfigError(
            f"Unknown llm_provider {settings.llm_provider!r}; expected 'openrouter' or 'google'"
        )
    return provider


def llm_base_url(settings: Settings) -> str:
    return GOOGLE_BASE_URL if active_llm_provider(settings) == PROVIDER_GOOGLE else OPENROUTER_BASE_URL


def llm_api_key(settings: Settings) -> str | None:
    if active_llm_provider(settings) == PROVIDER_GOOGLE:
        return settings.google_api_key
    return settings.openrouter_api_key


def llm_provider_label(settings: Settings) -> str:
    return "Google AI Studio" if active_llm_provider(settings) == PROVIDER_GOOGLE else "OpenRouter"


def require_api_key(settings: Settings) -> str:
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


def build_chat_model(settings: Settings, client=None):
    """Build the pydantic-ai chat model for the active provider."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    if active_llm_provider(settings) == PROVIDER_OPENROUTER:
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        client = client or build_openai_client(settings)
        return OpenRouterModel(
            settings.chat_model,
            provider=OpenRouterProvider(openai_client=client),
        )
    client = client or build_openai_client(settings)
    return OpenAIChatModel(
        settings.chat_model,
        provider=OpenAIProvider(openai_client=client),
    )
