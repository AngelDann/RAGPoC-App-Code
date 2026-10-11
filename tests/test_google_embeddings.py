import asyncio
import json
from pathlib import Path

import httpx
import pytest

from ragpoc.config import Settings
from ragpoc.embeddings import (
    EmbeddingError,
    FakeEmbeddingProvider,
    GoogleEmbeddingProvider,
    OpenRouterEmbeddingProvider,
    get_provider,
)


def make_settings(tmp_path: Path, **overrides) -> Settings:
    defaults = dict(_env_file=None, data_dir=tmp_path / "data", allowed_upload_dir=tmp_path / "data", google_api_key="test-google-key")
    defaults.update(overrides)
    return Settings(**defaults)


# --- factory ---

def test_get_provider_google_returns_google_provider(tmp_path):
    settings = make_settings(tmp_path, embedding_provider="google")
    provider = get_provider(settings)
    assert isinstance(provider, GoogleEmbeddingProvider)
    assert isinstance(provider, object)
    assert provider.__class__.__name__ == "GoogleEmbeddingProvider"


def test_get_provider_google_is_embedding_provider(tmp_path):
    from ragpoc.embeddings import EmbeddingProvider

    settings = make_settings(tmp_path, embedding_provider="google")
    assert isinstance(get_provider(settings), EmbeddingProvider)


def test_get_provider_unknown_provider_message_mentions_google(tmp_path):
    settings = make_settings(tmp_path, embedding_provider="bogus")
    with pytest.raises(ValueError, match="google"):
        get_provider(settings)


# --- text embeddings via OpenAI-compatible endpoint ---

def _mock_client(handler) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


def test_embed_texts_posts_to_openai_compatible_endpoint(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ],
                "usage": {"prompt_tokens": 7, "total_tokens": 7},
            },
        )

    settings = make_settings(tmp_path, embedding_provider="google", google_api_key="k-123")
    provider = GoogleEmbeddingProvider(settings, client=_mock_client(handler))
    vectors = asyncio.run(provider.embed_texts(["a", "b"]))

    assert captured["url"].endswith("/v1beta/openai/embeddings")
    # The OpenAI-compatible layer only accepts Bearer auth (x-goog-api-key 404s there).
    assert captured["auth"] == "Bearer k-123"
    assert captured["body"]["model"] == "gemini-embedding-2-preview"
    assert captured["body"]["input"] == ["a", "b"]
    # Response items come back sorted by index: 0 first even if listed second.
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]


def test_embed_texts_strips_openrouter_vendor_prefix(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    settings = make_settings(tmp_path, embedding_model="google/gemini-embedding-2-preview")
    provider = GoogleEmbeddingProvider(settings, client=_mock_client(handler))
    asyncio.run(provider.embed_texts(["x"]))
    assert captured["body"]["model"] == "gemini-embedding-2-preview"


def test_embed_query_returns_single_vector(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["input"] == ["what is rag"]
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.5, 0.5]}]})

    settings = make_settings(tmp_path)
    provider = GoogleEmbeddingProvider(settings, client=_mock_client(handler))
    vector = asyncio.run(provider.embed_query("what is rag"))
    assert vector == [0.5, 0.5]


def test_embed_texts_sends_dimensions_when_configured(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    settings = make_settings(tmp_path, embedding_dimension=256)
    provider = GoogleEmbeddingProvider(settings, client=_mock_client(handler))
    asyncio.run(provider.embed_texts(["x"]))
    assert captured["body"]["dimensions"] == 256


# --- errors ---

def test_missing_google_key_raises_with_google_message(tmp_path):
    settings = make_settings(tmp_path, google_api_key=None)
    provider = GoogleEmbeddingProvider(settings)
    with pytest.raises(EmbeddingError, match="Google AI Studio"):
        asyncio.run(provider.embed_texts(["x"]))


def test_error_status_raises_embedding_error(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="API key not valid")

    settings = make_settings(tmp_path)
    provider = GoogleEmbeddingProvider(settings, client=_mock_client(handler))
    with pytest.raises(EmbeddingError, match="401"):
        asyncio.run(provider.embed_texts(["x"]))


def test_retry_on_429_then_success(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="rate limited")
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    settings = make_settings(tmp_path)
    provider = GoogleEmbeddingProvider(settings, client=_mock_client(handler))
    vectors = asyncio.run(provider.embed_texts(["x"]))
    assert vectors == [[1.0]]
    assert calls["n"] == 2


# --- multimodal via native :embedContent ---

def test_embed_image_uses_embed_content_with_inline_data(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("x-goog-api-key")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embedding": {"values": [0.1, 0.2]}})

    image = tmp_path / "pic.png"
    image.write_bytes(b"\x89PNG-fake")

    settings = make_settings(tmp_path, google_api_key="k-9")
    provider = GoogleEmbeddingProvider(settings, client=_mock_client(handler))
    vector = asyncio.run(provider.embed_image(image))

    assert "/v1beta/models/gemini-embedding-2-preview:embedContent" in captured["url"]
    assert captured["auth"] == "k-9"
    part = captured["body"]["content"]["parts"][0]["inline_data"]
    assert part["mime_type"] == "image/png"
    assert vector == [0.1, 0.2]


def test_embed_video_and_audio_use_embed_content(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embedding": {"values": [1.0]}})

    settings = make_settings(tmp_path)
    provider = GoogleEmbeddingProvider(settings, client=_mock_client(handler))

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-mp4")
    vector = asyncio.run(provider.embed_video(video))
    assert vector == [1.0]
    assert captured["body"]["content"]["parts"][0]["inline_data"]["mime_type"] == "video/mp4"

    audio = tmp_path / "voice.mp3"
    audio.write_bytes(b"fake-mp3")
    vector = asyncio.run(provider.embed_audio(audio))
    assert vector == [1.0]
    assert captured["body"]["content"]["parts"][0]["inline_data"]["mime_type"] == "audio/mpeg"


def test_embed_content_dimension_when_configured(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embedding": {"values": [1.0]}})

    settings = make_settings(tmp_path, embedding_dimension=128)
    provider = GoogleEmbeddingProvider(settings, client=_mock_client(handler))
    image = tmp_path / "pic.jpg"
    image.write_bytes(b"fake-jpeg")
    asyncio.run(provider.embed_image(image))
    assert captured["body"]["outputDimensionality"] == 128


# --- usage hook ---

def test_usage_hook_reports_google_usage(tmp_path):
    reported = []

    async def hook(payload):
        reported.append(payload)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0]}], "usage": {"prompt_tokens": 3, "total_tokens": 3}},
        )

    settings = make_settings(tmp_path)
    provider = GoogleEmbeddingProvider(settings, client=_mock_client(handler), usage_hook=hook)
    asyncio.run(provider.embed_texts(["x"]))
    assert reported and reported[0]["kind"] == "text"
    assert reported[0]["input_tokens"] == 3


# --- provider family sanity (both cloud providers satisfy the protocol) ---

def test_openrouter_still_default(tmp_path):
    settings = make_settings(tmp_path, google_api_key=None)
    assert isinstance(get_provider(settings), OpenRouterEmbeddingProvider)


def test_fake_provider_is_fallback_for_tests():
    assert asyncio.run(FakeEmbeddingProvider().embed_query("x"))
