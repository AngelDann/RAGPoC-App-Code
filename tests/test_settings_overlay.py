import json
from base64 import b64encode

import pytest

from knowledge.models import AppConfig
from knowledge.settings_store import get_api_key_status, get_effective_settings, set_app_settings
from ragpoc.config import Settings


@pytest.fixture(autouse=True)
def isolate_env_settings(tmp_path, monkeypatch):
    """Keep tests hermetic: no real .env / env-var keys leaking into the assertions."""
    test_settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        allowed_upload_dir=tmp_path / "data",
        ui_password="test-password",
    )
    import knowledge.settings_store as settings_store
    from ragpoc import config

    monkeypatch.setattr(settings_store, "get_env_settings", lambda: test_settings)
    monkeypatch.setattr(config, "get_settings", lambda: test_settings)
    yield


@pytest.mark.django_db
def test_google_key_byok_overlay():
    set_app_settings(google_api_key="   g-secret   ", llm_provider="Google")
    config = AppConfig.load()
    assert config.google_api_key == "g-secret"
    assert config.llm_provider == "google"

    settings = get_effective_settings()
    assert settings.google_api_key == "g-secret"
    assert settings.llm_provider == "google"


@pytest.mark.django_db
def test_both_keys_stored_independently():
    set_app_settings(api_key="or-1", google_api_key="g-1")
    config = AppConfig.load()
    assert config.openrouter_api_key == "or-1"
    assert config.google_api_key == "g-1"

    set_app_settings(llm_provider="google")
    config = AppConfig.load()
    assert config.openrouter_api_key == "or-1"  # untouched when switching provider
    assert config.google_api_key == "g-1"


@pytest.mark.django_db
def test_api_key_status_reports_per_provider_keys():
    set_app_settings(api_key="or-1234567890", google_api_key="g-1234567890", llm_provider="google")
    status = get_api_key_status()
    assert status["llm_provider"] == "google"
    assert status["google_key"]["source"] == "byok"
    assert status["google_key"]["configured"] is True
    assert status["google_key"]["masked_key"] != "g-1234567890"  # never leak the key
    assert status["openrouter_key"]["source"] == "byok"
    # Legacy shape mirrors the active provider (google).
    assert status["source"] == "byok"
    assert status["masked_key"] == status["google_key"]["masked_key"]


@pytest.mark.django_db
def test_api_key_status_missing_keys():
    set_app_settings(llm_provider="openrouter")
    status = get_api_key_status()
    assert status["openrouter_key"]["source"] == "none"
    assert status["google_key"]["source"] == "none"
    assert status["configured"] is False


@pytest.mark.django_db
def test_models_override_persist():
    set_app_settings(chat_model="gemini-2.5-flash", embedding_model="gemini-embedding-001")
    settings = get_effective_settings()
    assert settings.chat_model == "gemini-2.5-flash"
    assert settings.embedding_model == "gemini-embedding-001"


@pytest.mark.django_db
def test_settings_endpoint_accepts_both_keys_and_provider():
    from django.test import Client

    client = Client()
    auth = {"HTTP_AUTHORIZATION": "Basic " + b64encode(b"operator:test-password").decode()}

    resp = client.post(
        "/api/settings",
        data=json.dumps({
            "llm_provider": "google",
            "openrouter_api_key": "sk-or-test-1234567890",
            "google_api_key": "AIza-test-1234567890",
            "chat_model": "gemini-2.5-flash",
            "embedding_model": "gemini-embedding-001",
            "embedding_provider": "google",
        }),
        content_type="application/json",
        **auth,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["llm_provider"] == "google"
    assert data["google_key"]["source"] == "byok"
    assert data["openrouter_key"]["source"] == "byok"
    assert data["chat_model"] == "gemini-2.5-flash"
    assert data["embedding_provider"] == "google"

    # Switching provider must not require retyping either key.
    resp = client.post(
        "/api/settings",
        data=json.dumps({"llm_provider": "openrouter"}),
        content_type="application/json",
        **auth,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["llm_provider"] == "openrouter"
    assert data["google_key"]["source"] == "byok"
    assert data["openrouter_key"]["source"] == "byok"

    from knowledge.llm import active_llm_provider, llm_api_key, llm_base_url

    settings = get_effective_settings()
    assert active_llm_provider(settings) == "openrouter"
    assert llm_api_key(settings) == "sk-or-test-1234567890"
    assert "openrouter.ai" in llm_base_url(settings)
