import json
from base64 import b64encode

import pytest
from django.test import Client

from knowledge.models import AppConfig
from ragpoc.config import Settings

pytestmark = pytest.mark.django_db

AUTH = {"HTTP_AUTHORIZATION": "Basic " + b64encode(b"operator:test-password").decode()}


@pytest.fixture(autouse=True)
def isolate_env_settings(tmp_path, monkeypatch):
    """Hermetic settings: no real .env keys, and Basic Auth enabled."""
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
    session.defaults.update(AUTH)
    return session


def _post(client, url, payload=None):
    return client.post(
        url, data=json.dumps(payload or {}), content_type="application/json", **AUTH
    )


# --- chat model catalogue ---


def test_chat_models_lists_all_models(client):
    response = client.get("/api/chat-models")
    assert response.status_code == 200
    body = response.json()
    keys = [m["key"] for m in body["models"]]
    assert "qwen3-vl-2b" in keys
    assert "whisper-tiny" in keys
    assert body["active_model"]


def test_chat_models_entry_carries_size_and_modalities(client):
    body = client.get("/api/chat-models").json()
    vl = next(m for m in body["models"] if m["key"] == "qwen3-vl-2b")
    assert vl["size_mb"] == 1373.0
    assert "image" in vl["modalities"]
    assert "installed" in vl
    assert "path" in vl


def test_chat_model_status_unknown_model_is_404(client):
    response = client.get("/api/chat-model/status", {"model": "nope"})
    assert response.status_code == 404


def test_chat_model_status_returns_state(client):
    response = client.get("/api/chat-model/status", {"model": "qwen3-vl-2b"})
    assert response.status_code == 200
    body = response.json()
    assert body["state"] in {"not_downloaded", "downloading", "ready", "error"}
    assert body["model_key"] == "qwen3-vl-2b"


def test_chat_model_download_rejects_unknown_model(client):
    response = _post(client, "/api/chat-model/download", {"model": "nope"})
    assert response.status_code == 404


def test_chat_model_download_rejects_get(client):
    assert client.get("/api/chat-model/download").status_code == 405


def test_chat_model_cancel_rejects_get(client):
    assert client.get("/api/chat-model/cancel").status_code == 405


def test_chat_model_cancel_unknown_model_is_404(client):
    response = _post(client, "/api/chat-model/cancel", {"model": "nope"})
    assert response.status_code == 404


# --- agent status ---


def test_agent_status_is_get_only(client):
    assert client.post("/api/agent/status").status_code == 405


def test_agent_status_reports_memory_budget(client):
    body = client.get("/api/agent/status").json()
    assert "memory" in body
    assert body["memory"]["hard_limit_mb"] == 3072.0
    assert body["memory"]["soft_limit_mb"] == 2048.0
    assert body["memory"]["within_hard_limit"] is True


def test_agent_status_reports_model_and_manifest(client):
    body = client.get("/api/agent/status").json()
    assert body["model_key"] == "qwen3-vl-2b"
    assert body["manifest"]["display_name"]
    assert body["context_length"] >= 512
    assert isinstance(body["would_fit"], bool)


def test_agent_status_estimate_is_positive(client):
    body = client.get("/api/agent/status").json()
    assert body["estimate_mb"] > 1000


# --- settings round trip ---


def test_settings_exposes_local_chat_fields(client):
    body = client.get("/api/settings").json()
    assert "local_chat_model" in body
    assert "local_chat_context_length" in body


def test_settings_persists_local_chat_model(client):
    response = _post(client, "/api/settings", {"local_chat_model": "qwen3-0.6b"})
    assert response.status_code == 200
    assert response.json()["local_chat_model"] == "qwen3-0.6b"
    assert AppConfig.load().local_chat_model == "qwen3-0.6b"


def test_settings_persists_context_length_as_int(client):
    response = _post(client, "/api/settings", {"local_chat_context_length": 1024})
    assert response.status_code == 200
    assert response.json()["local_chat_context_length"] == 1024
    assert AppConfig.load().local_chat_context_length == 1024


def test_settings_accepts_context_length_as_string(client):
    response = _post(client, "/api/settings", {"local_chat_context_length": "512"})
    assert response.status_code == 200
    assert AppConfig.load().local_chat_context_length == 512


def test_settings_ignores_bad_context_length(client):
    config = AppConfig.load()
    config.local_chat_context_length = 2048
    config.save()
    response = _post(client, "/api/settings", {"local_chat_context_length": "not-a-number"})
    assert response.status_code == 200
    assert AppConfig.load().local_chat_context_length == 2048


def test_settings_persists_local_provider(client):
    response = _post(client, "/api/settings", {"llm_provider": "local"})
    assert response.status_code == 200
    assert response.json()["llm_provider"] == "local"


def test_settings_invalid_json_is_400(client):
    response = client.post("/api/settings", data="not json", content_type="application/json")
    assert response.status_code == 400
