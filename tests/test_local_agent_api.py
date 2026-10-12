import asyncio
import json
import time
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
        # Model caches too: without these the download endpoints read (and wrote)
        # the developer's real data/models, and "already installed" short-circuits
        # made the download-path tests pass or fail depending on the machine.
        local_chat_cache_dir=tmp_path / "models",
        local_embedding_cache_dir=tmp_path / "models",
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


def test_chat_model_cancel_reaches_the_running_download(client, monkeypatch):
    """Regression: cancel hit a throwaway downloader, never the running thread."""
    import threading

    from ragpoc.chat_models import ChatModelDownloader

    started, release = threading.Event(), threading.Event()

    def slow_download(self):
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr(ChatModelDownloader, "_download_files", slow_download)
    assert _post(client, "/api/chat-model/download", {"model": "qwen3-0.6b"}).status_code == 200
    assert started.wait(timeout=5)

    assert _post(client, "/api/chat-model/cancel", {"model": "qwen3-0.6b"}).status_code == 200
    release.set()

    deadline = time.monotonic() + 5
    body = {}
    while time.monotonic() < deadline:
        body = client.get("/api/chat-model/status", {"model": "qwen3-0.6b"}).json()
        if body.get("message") == "cancelled":
            break
        time.sleep(0.05)
    assert body["message"] == "cancelled"


def test_chat_model_status_stays_downloading_while_the_thread_runs(client, monkeypatch):
    """Regression: the poll saw ``not_downloaded`` mid-download and the UI quit."""
    import threading

    from ragpoc.chat_models import ChatModelDownloader

    started, release = threading.Event(), threading.Event()

    def slow_download(self):
        started.set()
        release.wait(timeout=5)
        self._set(state="error", error="stop")

    monkeypatch.setattr(ChatModelDownloader, "_download_files", slow_download)
    assert _post(client, "/api/chat-model/download", {"model": "qwen3-0.6b"}).status_code == 200
    assert started.wait(timeout=5)
    body = client.get("/api/chat-model/status", {"model": "qwen3-0.6b"}).json()
    assert body["state"] == "downloading"
    release.set()


def test_model_status_stays_downloading_while_the_thread_runs(client, monkeypatch):
    """Same regression for the embedding model box polled via /api/model/status."""
    import threading

    from ragpoc.model_downloader import ModelDownloader

    started, release = threading.Event(), threading.Event()

    def slow_download(self, dtype):
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr(ModelDownloader, "_download_files", slow_download)
    assert _post(client, "/api/model/download", {"dtype": "q8"}).status_code == 200
    assert started.wait(timeout=5)
    body = client.get("/api/model/status").json()
    assert body["state"] == "downloading"
    release.set()


# --- modo local sin API key ---


def test_inline_ai_action_runs_in_local_mode_without_an_api_key(client, monkeypatch):
    """Regression: local mode answered "API key de Local (sin API) no configurada."."""
    from types import SimpleNamespace

    from knowledge import llm as llm_mod

    assert _post(client, "/api/settings", {"llm_provider": "local"}).status_code == 200

    class StubEngine:
        def load(self):
            pass

        def generate(self, messages, **kwargs):
            return SimpleNamespace(text="borrador local", metrics=SimpleNamespace())

    monkeypatch.setattr(llm_mod, "_shared_local_engine", lambda settings: StubEngine())
    response = _post(client, "/api/ai/inline-action", {"action": "ask", "prompt": "di hola"})
    assert response.status_code == 200, response.content
    body = response.json()
    assert body["status"] == "success"
    assert body["content"] == "borrador local"
    assert "API key" not in json.dumps(body)


@pytest.mark.django_db(transaction=True)
def test_chat_stream_runs_in_local_mode_without_an_api_key(client, monkeypatch):
    """The chat itself: local mode must not stop at an API-key guard."""
    from pydantic_ai.models.test import TestModel

    from knowledge import views
    from knowledge.services import get_rag_service as real_get_rag_service
    from ragpoc.embeddings import FakeEmbeddingProvider

    assert _post(client, "/api/settings", {"llm_provider": "local"}).status_code == 200

    import knowledge.settings_store as settings_store

    rag_service = real_get_rag_service(settings_store.get_env_settings(), FakeEmbeddingProvider())
    monkeypatch.setattr(views, "get_rag_service", lambda: rag_service)
    # The agent is built for real around a TestModel: what is under test is that
    # the request gets past the credential guard and into the model.
    monkeypatch.setattr(
        "knowledge.pydantic_agent.build_chat_model", lambda settings, client=None: TestModel()
    )

    response = _post(client, "/chat/stream", {"question": "Hola", "scope": "workspace"})
    assert response.status_code == 200, response.content

    stream = response.streaming_content
    if hasattr(stream, "__aiter__"):
        # The view streams through an async iterator; the sync test client hands
        # it over as-is.
        async def _collect():
            return [chunk async for chunk in stream]

        chunks = asyncio.run(_collect())
    else:
        chunks = list(stream)
    events = b"".join(chunks).decode("utf-8")
    assert "API key" not in events
    assert '"type": "token"' in events
    assert '"type": "done"' in events


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
