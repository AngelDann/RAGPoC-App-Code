from __future__ import annotations

import asyncio
import base64
import random

import httpx

from ragpoc.config import Settings


class MediaGenerationError(RuntimeError):
    pass


_IMAGES_ENDPOINT = "https://openrouter.ai/api/v1/images"
_SPEECH_ENDPOINT = "https://openrouter.ai/api/v1/audio/speech"


def _auth_headers(settings: Settings) -> dict[str, str]:
    if not settings.openrouter_api_key:
        raise MediaGenerationError("Configura tu OpenRouter API key en Ajustes o en OPENROUTER_API_KEY.")
    return {"Authorization": f"Bearer {settings.openrouter_api_key}"}


async def _post_with_retry(client: httpx.AsyncClient, url: str, *, headers: dict, json_body: dict) -> httpx.Response:
    response: httpx.Response
    for attempt in range(4):
        response = await client.post(url, headers=headers, json=json_body)
        if response.status_code in {429, 500, 502, 503, 529} and attempt < 3:
            delay = float(response.headers.get("Retry-After", 2**attempt)) + random.random()
            await asyncio.sleep(delay)
            continue
        break
    return response


async def generate_image(
    prompt: str,
    *,
    settings: Settings,
    aspect_ratio: str = "1:1",
) -> bytes:
    """Generates a single image via OpenRouter's image API and returns raw image bytes (PNG/JPEG)."""
    headers = _auth_headers(settings)
    payload = {
        "model": settings.image_model,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        response = await _post_with_retry(client, _IMAGES_ENDPOINT, headers=headers, json_body=payload)
        if response.is_error:
            raise MediaGenerationError(f"OpenRouter image request failed ({response.status_code}): {response.text[:500]}")
        data = response.json()
        items = data.get("data") or []
        if not items:
            raise MediaGenerationError("OpenRouter returned no image data.")
        item = items[0]
        b64 = item.get("b64_json")
        if b64:
            return base64.b64decode(b64)
        url = item.get("url")
        if url:
            img_response = await client.get(url)
            if img_response.is_error:
                raise MediaGenerationError(f"Could not download generated image ({img_response.status_code}).")
            return img_response.content
        raise MediaGenerationError("OpenRouter image response had neither b64_json nor url.")


async def synthesize_speech(
    text: str,
    *,
    settings: Settings,
    voice: str,
    audio_format: str = "mp3",
) -> bytes:
    """Synthesizes speech for a single line of text via OpenRouter's TTS API. Returns raw audio bytes.
    OpenRouter's /audio/speech endpoint only accepts "mp3" or "pcm" for response_format (no "wav")."""
    headers = _auth_headers(settings)
    payload = {
        "model": settings.tts_model,
        "input": text,
        "voice": voice,
        "response_format": audio_format,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        response = await _post_with_retry(client, _SPEECH_ENDPOINT, headers=headers, json_body=payload)
        if response.is_error:
            raise MediaGenerationError(f"OpenRouter TTS request failed ({response.status_code}): {response.text[:500]}")
        return response.content
