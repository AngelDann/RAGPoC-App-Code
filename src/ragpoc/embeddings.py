from __future__ import annotations

import asyncio
import base64
import random
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

import httpx

from ragpoc.config import Settings

UsageHook = Callable[[dict], Awaitable[None]]


class EmbeddingError(RuntimeError):
    pass


AUDIO_MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


class EmbeddingProvider(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_image(self, path: Path) -> list[float]: ...

    async def embed_video(self, path: Path) -> list[float]: ...

    async def embed_audio(self, path: Path) -> list[float]: ...


class OpenRouterEmbeddingProvider:
    endpoint = "https://openrouter.ai/api/v1/embeddings"

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        usage_hook: UsageHook | None = None,
    ):
        self.settings, self.client = settings, client
        self._semaphore = asyncio.Semaphore(settings.embedding_concurrency)
        self.usage_hook = usage_hook

    async def _post(self, payload: dict, *, kind: str, count: int = 1) -> list[list[float]]:
        if not self.settings.openrouter_api_key:
            raise EmbeddingError("Configura tu OpenRouter API key en Ajustes o en OPENROUTER_API_KEY.")
        headers = {"Authorization": f"Bearer {self.settings.openrouter_api_key}"}
        own_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=90)
        started = time.perf_counter()
        try:
            for attempt in range(4):
                async with self._semaphore:
                    response = await client.post(self.endpoint, headers=headers, json=payload)
                if response.status_code in {429, 500, 502, 503, 529} and attempt < 3:
                    delay = float(response.headers.get("Retry-After", 2**attempt)) + random.random()
                    await asyncio.sleep(delay)
                    continue
                if response.is_error:
                    await self._report_usage(kind, count, started, status="error", error_message=response.text[:500])
                    raise EmbeddingError(f"OpenRouter embedding request failed ({response.status_code}): {response.text[:500]}")
                body = response.json()
                data = body.get("data", [])
                vectors = [item["embedding"] for item in sorted(data, key=lambda item: item.get("index", 0))]
                if not vectors:
                    await self._report_usage(kind, count, started, status="error", error_message="OpenRouter returned no embeddings.")
                    raise EmbeddingError("OpenRouter returned no embeddings.")
                await self._report_usage(kind, count, started, usage=body.get("usage") or {})
                return vectors
        finally:
            if own_client:
                await client.aclose()
        raise EmbeddingError("OpenRouter retry limit reached.")

    async def _report_usage(
        self,
        kind: str,
        count: int,
        started: float,
        usage: dict | None = None,
        status: str = "success",
        error_message: str = "",
    ) -> None:
        if not self.usage_hook:
            return
        usage = usage or {}
        try:
            await self.usage_hook({
                "model": self.settings.embedding_model,
                "kind": kind,
                "count": count,
                "input_tokens": usage.get("prompt_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "status": status,
                "error_message": error_message,
            })
        except Exception:
            pass

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self._post(
            {"model": self.settings.embedding_model, "input": texts}, kind="text", count=len(texts)
        )

    async def embed_image(self, path: Path) -> list[float]:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        payload = {
            "model": self.settings.embedding_model,
            "input": [{"content": [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}]}],
        }
        return (await self._post(payload, kind="image"))[0]

    async def embed_video(self, path: Path) -> list[float]:
        """Embed a short local MP4 clip through OpenRouter's video-capable endpoint."""
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        data_url = f"data:video/mp4;base64,{encoded}"
        return (await self._post({"model": self.settings.embedding_model, "input": data_url}, kind="video"))[0]

    async def embed_audio(self, path: Path) -> list[float]:
        """Embed a local audio file through OpenRouter's audio-capable endpoint."""
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = AUDIO_MIME_TYPES.get(path.suffix.lower(), "audio/mpeg")
        data_url = f"data:{mime};base64,{encoded}"
        return (await self._post({"model": self.settings.embedding_model, "input": data_url}, kind="audio"))[0]


class FakeEmbeddingProvider:
    """Deterministic offline provider for tests and local structural validation."""

    dimension = 8

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(item.encode()) for item in texts]

    async def embed_image(self, path: Path) -> list[float]:
        return self._vector(path.read_bytes())

    async def embed_video(self, path: Path) -> list[float]:
        return self._vector(path.read_bytes())

    async def embed_audio(self, path: Path) -> list[float]:
        return self._vector(path.read_bytes())

    def _vector(self, raw: bytes) -> list[float]:
        values = [0.0] * self.dimension
        for index, byte in enumerate(raw):
            values[index % self.dimension] += byte / 255.0
        norm = sum(v * v for v in values) ** 0.5 or 1.0
        return [v / norm for v in values]


def get_provider(settings: Settings, usage_hook: UsageHook | None = None) -> EmbeddingProvider:
    return OpenRouterEmbeddingProvider(settings, usage_hook=usage_hook)
