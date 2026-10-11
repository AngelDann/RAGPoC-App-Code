from __future__ import annotations

from ragpoc.config import Settings
from ragpoc.config import get_settings as get_env_settings


def _load_app_config():
    try:
        from knowledge.models import AppConfig
        return AppConfig.objects.filter(pk=1).first()
    except Exception:
        # DB not migrated yet / not ready during Django bootstrap.
        return None


def get_effective_settings() -> Settings:
    """Env settings with any BYOK override saved from the Ajustes UI applied on top."""
    settings = get_env_settings()
    config = _load_app_config()
    if config:
        updates = {}
        if config.openrouter_api_key:
            updates["openrouter_api_key"] = config.openrouter_api_key
        if config.google_api_key:
            updates["google_api_key"] = config.google_api_key
        if config.llm_provider:
            updates["llm_provider"] = config.llm_provider
        if config.chat_model:
            updates["chat_model"] = config.chat_model
        if config.embedding_model:
            updates["embedding_model"] = config.embedding_model
        if config.web_search_engine:
            updates["web_search_engine"] = config.web_search_engine
        if config.searxng_url is not None:
            updates["searxng_url"] = config.searxng_url
        if config.web_search_deep_read is not None:
            updates["web_search_deep_read"] = config.web_search_deep_read
        if config.embedding_provider:
            updates["embedding_provider"] = config.embedding_provider
        if config.local_embedding_dtype:
            updates["local_embedding_dtype"] = config.local_embedding_dtype
        if config.local_embedding_device:
            updates["local_embedding_device"] = config.local_embedding_device
        if config.local_chat_model:
            updates["local_chat_model"] = config.local_chat_model
        if config.local_chat_context_length:
            updates["local_chat_context_length"] = config.local_chat_context_length
        if updates:
            settings = settings.model_copy(update=updates)
    return settings


def get_current_language() -> str:
    """Returns the configured app language ('es', 'en', 'auto'). Defaults to 'es'."""
    config = _load_app_config()
    return config.language if config and config.language else "es"


def _key_status(stored_key: str, env_key: str) -> dict:
    if stored_key:
        source, active_key = "byok", stored_key
    elif env_key:
        source, active_key = "env", env_key
    else:
        source, active_key = "none", ""
    return {
        "source": source,
        "configured": bool(active_key),
        "masked_key": _mask(active_key) if active_key else None,
        "has_env_fallback": bool(env_key),
    }


def get_api_key_status() -> dict:
    """Describe where the active API keys come from, without leaking them, plus app settings."""
    config = _load_app_config()
    language = config.language if config and config.language else "es"
    web_search_engine = config.web_search_engine if config and config.web_search_engine else "duckduckgo"
    searxng_url = config.searxng_url if config and config.searxng_url else ""
    web_search_deep_read = config.web_search_deep_read if config and config.web_search_deep_read is not None else True
    embedding_provider = config.embedding_provider if config and config.embedding_provider else ""
    local_embedding_dtype = config.local_embedding_dtype if config and config.local_embedding_dtype else ""
    local_embedding_device = config.local_embedding_device if config and config.local_embedding_device else ""
    llm_provider = (config.llm_provider if config and config.llm_provider else "").strip().lower()
    local_chat_model = config.local_chat_model if config and config.local_chat_model else ""
    local_chat_context_length = config.local_chat_context_length if config else None

    env_settings = get_env_settings()
    effective_provider = embedding_provider or env_settings.embedding_provider
    effective_llm_provider = llm_provider or (env_settings.llm_provider or "openrouter")

    openrouter_status = _key_status(
        config.openrouter_api_key if config else "",
        env_settings.openrouter_api_key or "",
    )
    google_status = _key_status(
        config.google_api_key if config else "",
        env_settings.google_api_key or "",
    )
    active_key = google_status if effective_llm_provider == "google" else openrouter_status

    return {
        # Legacy shape: status of the active provider's key (kept for older JS callers).
        **active_key,
        "llm_provider": effective_llm_provider,
        "chat_model": (config.chat_model if config and config.chat_model else "") or env_settings.chat_model,
        "embedding_model": (config.embedding_model if config and config.embedding_model else "") or env_settings.embedding_model,
        "openrouter_key": openrouter_status,
        "google_key": google_status,
        "language": language,
        "web_search_engine": web_search_engine,
        "searxng_url": searxng_url,
        "web_search_deep_read": web_search_deep_read,
        "embedding_provider": effective_provider,
        "local_embedding_dtype": local_embedding_dtype or env_settings.local_embedding_dtype,
        "local_embedding_device": local_embedding_device or env_settings.local_embedding_device,
        "local_chat_model": local_chat_model or env_settings.local_chat_model,
        "local_chat_context_length": local_chat_context_length or env_settings.local_chat_context_length,
    }


def set_byok_api_key(api_key: str) -> None:
    from knowledge.models import AppConfig
    config = AppConfig.load()
    config.openrouter_api_key = api_key.strip()
    config.save()


def set_app_settings(
    api_key: str | None = None,
    language: str | None = None,
    web_search_engine: str | None = None,
    searxng_url: str | None = None,
    web_search_deep_read: bool | None = None,
    embedding_provider: str | None = None,
    local_embedding_dtype: str | None = None,
    local_embedding_device: str | None = None,
    local_chat_model: str | None = None,
    local_chat_context_length: int | None = None,
    google_api_key: str | None = None,
    llm_provider: str | None = None,
    chat_model: str | None = None,
    embedding_model: str | None = None,
) -> None:
    from knowledge.models import AppConfig
    config = AppConfig.load()
    if api_key is not None:
        config.openrouter_api_key = api_key.strip()
    if google_api_key is not None:
        config.google_api_key = google_api_key.strip()
    if llm_provider is not None:
        config.llm_provider = llm_provider.strip().lower()
    if chat_model is not None:
        config.chat_model = chat_model.strip()
    if embedding_model is not None:
        config.embedding_model = embedding_model.strip()
    if language is not None:
        config.language = language.strip().lower()
    if web_search_engine is not None:
        config.web_search_engine = web_search_engine.strip().lower()
    if searxng_url is not None:
        config.searxng_url = searxng_url.strip()
    if web_search_deep_read is not None:
        config.web_search_deep_read = bool(web_search_deep_read)
    if embedding_provider is not None:
        config.embedding_provider = embedding_provider.strip().lower()
    if local_embedding_dtype is not None:
        config.local_embedding_dtype = local_embedding_dtype.strip().lower()
    if local_embedding_device is not None:
        config.local_embedding_device = local_embedding_device.strip().lower()
    if local_chat_model is not None:
        config.local_chat_model = local_chat_model.strip()
    if local_chat_context_length is not None:
        config.local_chat_context_length = int(local_chat_context_length)
    config.save()


def _mask(key: str) -> str:
    if len(key) <= 8:
        return "•" * len(key)
    return f"{key[:4]}{'•' * (len(key) - 8)}{key[-4:]}"
