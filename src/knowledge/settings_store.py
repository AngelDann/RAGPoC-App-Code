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
    if config and config.openrouter_api_key:
        settings = settings.model_copy(update={"openrouter_api_key": config.openrouter_api_key})
    return settings


def get_api_key_status() -> dict:
    """Describe where the active OpenRouter API key comes from, without leaking it, plus app settings."""
    config = _load_app_config()
    stored_key = config.openrouter_api_key if config else ""
    env_key = get_env_settings().openrouter_api_key or ""

    if stored_key:
        source = "byok"
        active_key = stored_key
    elif env_key:
        source = "env"
        active_key = env_key
    else:
        source = "none"
        active_key = ""

    return {
        "source": source,
        "configured": bool(active_key),
        "masked_key": _mask(active_key) if active_key else None,
        "has_env_fallback": bool(env_key),
    }


def set_byok_api_key(api_key: str) -> None:
    from knowledge.models import AppConfig
    config = AppConfig.load()
    config.openrouter_api_key = api_key.strip()
    config.save()


def set_app_settings(api_key: str | None = None) -> None:
    from knowledge.models import AppConfig
    config = AppConfig.load()
    if api_key is not None:
        config.openrouter_api_key = api_key.strip()
    config.save()


def _mask(key: str) -> str:
    if len(key) <= 8:
        return "•" * len(key)
    return f"{key[:4]}{'•' * (len(key) - 8)}{key[-4:]}"
