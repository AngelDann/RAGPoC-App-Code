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
        if updates:
            settings = settings.model_copy(update=updates)
    return settings


def get_current_language() -> str:
    """Returns the configured app language ('es', 'en', 'auto'). Defaults to 'es'."""
    config = _load_app_config()
    return config.language if config and config.language else "es"


def get_api_key_status() -> dict:
    """Describe where the active OpenRouter API key comes from, without leaking it, plus app settings."""
    config = _load_app_config()
    stored_key = config.openrouter_api_key if config else ""
    env_key = get_env_settings().openrouter_api_key or ""
    language = config.language if config and config.language else "es"
    web_search_engine = config.web_search_engine if config and config.web_search_engine else "duckduckgo"
    searxng_url = config.searxng_url if config and config.searxng_url else ""
    web_search_deep_read = config.web_search_deep_read if config and config.web_search_deep_read is not None else True
    embedding_provider = config.embedding_provider if config and config.embedding_provider else ""
    local_embedding_dtype = config.local_embedding_dtype if config and config.local_embedding_dtype else ""
    local_embedding_device = config.local_embedding_device if config and config.local_embedding_device else ""

    if stored_key:
        source = "byok"
        active_key = stored_key
    elif env_key:
        source = "env"
        active_key = env_key
    else:
        source = "none"
        active_key = ""

    env_settings = get_env_settings()
    effective_provider = embedding_provider or env_settings.embedding_provider

    return {
        "source": source,
        "configured": bool(active_key),
        "masked_key": _mask(active_key) if active_key else None,
        "has_env_fallback": bool(env_key),
        "language": language,
        "web_search_engine": web_search_engine,
        "searxng_url": searxng_url,
        "web_search_deep_read": web_search_deep_read,
        "embedding_provider": effective_provider,
        "local_embedding_dtype": local_embedding_dtype or env_settings.local_embedding_dtype,
        "local_embedding_device": local_embedding_device or env_settings.local_embedding_device,
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
) -> None:
    from knowledge.models import AppConfig
    config = AppConfig.load()
    if api_key is not None:
        config.openrouter_api_key = api_key.strip()
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
    config.save()


def _mask(key: str) -> str:
    if len(key) <= 8:
        return "•" * len(key)
    return f"{key[:4]}{'•' * (len(key) - 8)}{key[-4:]}"
