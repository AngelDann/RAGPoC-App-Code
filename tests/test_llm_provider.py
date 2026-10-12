from pathlib import Path

import pytest

from knowledge.llm import (
    GOOGLE_BASE_URL,
    OPENROUTER_BASE_URL,
    LLMConfigError,
    active_llm_provider,
    build_chat_model,
    build_openai_client,
    llm_api_key,
    llm_base_url,
    llm_provider_label,
    require_api_key,
)
from ragpoc.config import Settings


def make_settings(tmp_path: Path, **overrides) -> Settings:
    defaults = dict(
        _env_file=None,
        data_dir=tmp_path / "data",
        allowed_upload_dir=tmp_path / "data",
        openrouter_api_key="or-key",
        google_api_key="g-key",
    )
    defaults.update(overrides)
    return Settings(**defaults)


# --- provider resolution ---

def test_active_llm_provider_defaults_to_openrouter(tmp_path):
    settings = make_settings(tmp_path)
    settings = settings.model_copy(update={"llm_provider": ""})
    assert active_llm_provider(settings) == "openrouter"
    assert llm_base_url(settings) == OPENROUTER_BASE_URL
    assert llm_api_key(settings) == "or-key"
    assert llm_provider_label(settings) == "OpenRouter"


def test_active_llm_provider_google(tmp_path):
    settings = make_settings(tmp_path, llm_provider="google")
    assert active_llm_provider(settings) == "google"
    assert llm_base_url(settings) == GOOGLE_BASE_URL
    assert llm_api_key(settings) == "g-key"
    assert llm_provider_label(settings) == "Google AI Studio"


def test_active_llm_provider_normalizes_case_and_spaces(tmp_path):
    settings = make_settings(tmp_path, llm_provider="  GOOGLE ")
    assert active_llm_provider(settings) == "google"


def test_active_llm_provider_unknown_raises(tmp_path):
    settings = make_settings(tmp_path, llm_provider="bing")
    with pytest.raises(LLMConfigError, match="bing"):
        active_llm_provider(settings)


# --- key requirements ---

def test_require_api_key_openrouter_missing(tmp_path):
    settings = make_settings(tmp_path, openrouter_api_key=None, llm_provider="openrouter")
    with pytest.raises(LLMConfigError, match="OpenRouter API key"):
        require_api_key(settings)


def test_require_api_key_google_missing(tmp_path):
    settings = make_settings(tmp_path, google_api_key=None, llm_provider="google")
    with pytest.raises(LLMConfigError, match="Google AI Studio API key"):
        require_api_key(settings)


# --- clients ---

def test_build_openai_client_targets_google(tmp_path):
    settings = make_settings(tmp_path, llm_provider="google")
    client = build_openai_client(settings)
    assert str(client.base_url).rstrip("/").endswith("generativelanguage.googleapis.com/v1beta/openai")
    assert client.api_key == "g-key"


def test_build_openai_client_targets_openrouter(tmp_path):
    settings = make_settings(tmp_path, llm_provider="openrouter")
    client = build_openai_client(settings)
    assert str(client.base_url).rstrip("/").endswith("openrouter.ai/api/v1")
    assert client.api_key == "or-key"


def test_build_openai_client_sync(tmp_path):
    settings = make_settings(tmp_path, llm_provider="google")
    client = build_openai_client(settings, async_=False)
    assert "generativelanguage.googleapis.com" in str(client.base_url)


# --- pydantic-ai models ---

def test_build_chat_model_openrouter_uses_openrouter_model(tmp_path):
    from pydantic_ai.models.openrouter import OpenRouterModel

    settings = make_settings(tmp_path, llm_provider="openrouter")
    model = build_chat_model(settings)
    assert isinstance(model, OpenRouterModel)
    assert model.model_name == settings.chat_model


def test_build_chat_model_google_uses_native_google_model(tmp_path):
    from pydantic_ai.models.google import GoogleModel

    settings = make_settings(tmp_path, llm_provider="google")
    model = build_chat_model(settings)
    assert isinstance(model, GoogleModel)
    assert model.model_name == "gemini-3.7-flash"


def test_google_model_name_strips_vendor_prefix(tmp_path):
    from knowledge.llm import google_model_name

    assert google_model_name("google/gemini-3.7-flash") == "gemini-3.7-flash"
    assert google_model_name("gemini-3.7-flash") == "gemini-3.7-flash"


# --- friendly error rendering ---


def test_friendly_error_suggests_local_mode_when_the_cloud_is_unreachable(tmp_path):
    """Regression: the chat rendered openai's bare "Connection error." as the answer."""
    from knowledge.llm import friendly_llm_error

    settings = make_settings(tmp_path, llm_provider="openrouter")
    message = friendly_llm_error(Exception("Connection error."), settings)
    assert "OpenRouter" in message
    assert "Modo Local" in message
    assert "Connection error." in message  # the technical detail stays reachable


def test_friendly_error_names_the_active_cloud_provider(tmp_path):
    from knowledge.llm import friendly_llm_error

    settings = make_settings(tmp_path, llm_provider="google")
    message = friendly_llm_error(Exception("Connection error."), settings)
    assert "Google AI Studio" in message


def test_friendly_error_keeps_unknown_errors_readable(tmp_path):
    from knowledge.llm import friendly_llm_error

    settings = make_settings(tmp_path, llm_provider="openrouter")
    message = friendly_llm_error(Exception("boom"), settings)
    assert "boom" in message


def test_friendly_error_for_the_local_model_points_at_the_model_settings(tmp_path):
    from knowledge.llm import friendly_llm_error

    settings = make_settings(tmp_path, llm_provider="local")
    message = friendly_llm_error(Exception("model directory not found"), settings)
    assert "local" in message.lower()
    assert "model directory not found" in message
