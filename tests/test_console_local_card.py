from base64 import b64encode

import pytest
from django.test import Client

from ragpoc.config import Settings

pytestmark = pytest.mark.django_db

MARKERS = [
    "settings-local-agent-card",
    "settings-local-chat-model-select",
    "settings-local-chat-context-select",
    "btn-download-chat-model",
    "btn-cancel-chat-model-download",
    "btn-save-local-agent",
    "local-agent-ready-badge",
    "Local agent (ONNX Runtime GenAI) settings",
]


@pytest.fixture(autouse=True)
def isolate_env_settings(tmp_path, monkeypatch):
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


@pytest.fixture()
def client():
    session = Client()
    session.defaults.update(
        {"HTTP_AUTHORIZATION": "Basic " + b64encode(b"operator:test-password").decode()}
    )
    return session


@pytest.fixture()
def console_html(client) -> str:
    response = client.get("/")
    assert response.status_code == 200
    return response.content.decode("utf-8")


def test_console_renders_with_local_card(console_html):
    assert len(console_html) > 10000


def test_console_has_local_agent_card(console_html):
    for marker in MARKERS:
        assert marker in console_html, f"falta el marcador {marker!r}"


def test_console_provider_select_offers_local(console_html):
    assert 'value="local"' in console_html
    assert "settings-llm-provider-select" in console_html


def test_console_local_model_options(console_html):
    assert "qwen3-vl-2b" in console_html
    assert "qwen3-0.6b" in console_html


def test_console_context_length_options(console_html):
    assert 'value="1024"' in console_html
    assert 'value="2048"' in console_html
    assert 'value="4096"' in console_html


def test_console_has_both_i18n_languages(console_html):
    assert "Local model (no API)" in console_html
    assert "Modelo local (sin API)" in console_html
