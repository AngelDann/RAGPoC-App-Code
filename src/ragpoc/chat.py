from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol


class ChatProvider(Protocol):
    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]: ...


class UnavailableChatProvider:
    def __init__(self, message: str):
        self.message = message

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        raise RuntimeError(self.message)
        yield ""  # pragma: no cover


class OpenRouterChatProvider:
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        import json

        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "messages": messages, "stream": True, "temperature": 0.2}
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", self.endpoint, headers=headers, json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    text = event.get("choices", [{}])[0].get("delta", {}).get("content")
                    if text:
                        yield text
