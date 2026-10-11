"""pydantic-ai adapter for the local ONNX GenAI engine.

``FunctionModel`` lets a plain callable stand in for a hosted model. This module
implements that callable on top of ``ragpoc.local_llm``, so the existing agent
(``knowledge.pydantic_agent``) runs unchanged against a local model with its
current tools.

Three responsibilities live here and nowhere else:

* turn pydantic-ai message history into engine ``ChatMessage`` objects, keeping
  ``BinaryContent`` evidence as ``MediaInput``;
* expose the agent's tool schemas to the model's chat template;
* parse ``<tool_call>`` blocks out of the generated text back into ``ToolCallPart``.

The engine is text-first: it never invents tool calls itself, so a small model
that fails to produce a well-formed call simply answers in prose instead of
crashing the loop.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaToolCall,
    FunctionModel,
    ToolDefinition,
)

from ragpoc.local_llm import ChatMessage, LocalLlmEngine, MediaInput

TOOL_CALL_OPEN = "<" + "tool_call" + ">"
TOOL_CALL_CLOSE = "</" + "tool_call" + ">"

_TOOL_BLOCK = re.compile(re.escape(TOOL_CALL_OPEN) + r"(.*?)" + re.escape(TOOL_CALL_CLOSE), re.DOTALL)


class LocalAgentModelError(RuntimeError):
    pass


@dataclass
class ParsedTurn:
    """What a single generated turn produced: prose plus zero or more calls."""

    text: str
    tool_calls: list[ToolCallPart]
    media: list[MediaInput]


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Return ``(visible_text, raw_calls)`` pulled out of a generated turn.

    Tolerates a missing closing tag and an arguments payload that is either a
    JSON object or a JSON-encoded string, because a small model does both.
    """
    visible: list[str] = []
    calls: list[dict[str, Any]] = []
    cursor = 0
    for match in _TOOL_BLOCK.finditer(text):
        visible.append(text[cursor : match.start()])
        cursor = match.end()
        raw = match.group(1).strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = _recover_json(raw)
            if payload is None:
                continue
        if isinstance(payload, dict):
            calls.append(payload)
    visible.append(text[cursor:])
    joined = "".join(visible).strip()
    if not calls:
        bare = _parse_bare_call(joined)
        if bare is not None:
            return "", [bare]
    return joined, calls


def _looks_like_bare_call(text: str) -> bool:
    """True while the turn might still turn out to be a bare JSON call."""
    stripped = text.lstrip()
    return stripped.startswith("{")


def _parse_bare_call(text: str) -> dict[str, Any] | None:
    """Recognise a tool call emitted without the `` wrapper.

    The template asks for ``{"name": ..., "arguments": ...}`` inside the tags,
    but a small model often returns just the JSON object. Dropping the call
    would silently turn a tool request into an answer, so it is accepted.
    """
    if not text or TOOL_CALL_OPEN in text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    fragment = text[start:].strip()
    payload = _recover_json(fragment)
    if not isinstance(payload, dict):
        return None
    has_name = "name" in payload or "tool_name" in payload or "function" in payload
    has_args = "arguments" in payload or "args" in payload or "function" in payload
    return payload if has_name and has_args else None


def _recover_json(raw: str) -> dict[str, Any] | None:
    """Best effort salvage of a truncated or lightly malformed call body."""
    start = raw.find("{")
    if start < 0:
        return None
    fragment = raw[start:]
    while fragment:
        try:
            return json.loads(fragment)
        except json.JSONDecodeError:
            fragment = fragment[:-1]
    return None


def _coerce_args(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"input": arguments}
        return parsed if isinstance(parsed, dict) else {"input": parsed}
    return {"input": arguments}


def to_tool_parts(raw_calls: list[dict[str, Any]], visible: str) -> list[ToolCallPart]:
    parts: list[ToolCallPart] = []
    for index, call in enumerate(raw_calls):
        name = call.get("name") or call.get("function", {}).get("name") or call.get("tool_name")
        if not name:
            continue
        arguments = call.get("arguments")
        if arguments is None and isinstance(call.get("function"), dict):
            arguments = call["function"].get("arguments")
        parts.append(
            ToolCallPart(
                tool_name=str(name),
                args=_coerce_args(arguments),
                tool_call_id=str(call.get("id") or call.get("tool_call_id") or f"local-call-{index}"),
            )
        )
    return parts


def _detect_language(text: str) -> str:
    """Best-effort language tag for choosing the right nudge wording."""
    try:
        from ragpoc.eval_quality import detect_language

        return detect_language(text) or "es"
    except Exception:
        return "es"


def apply_guard(
    engine: LocalLlmEngine,
    messages: list[ChatMessage],
    text: str,
    *,
    max_new_tokens: int | None = None,
) -> str:
    """Guard an already-generated answer, retrying once if it fabricates.

    Split from ``guarded_answer`` so callers that need the raw text (the
    adapter parses tool calls out of it) generate only once.
    """
    from ragpoc.guardrails import (
        ABSTENTION_NUDGE_EN,
        ABSTENTION_NUDGE_ES,
        needs_guard,
        wrap_unverified,
    )

    text = text.strip()
    if not text:
        return text

    had_evidence = any(message.media for message in messages)
    had_tool_calls = any(
        message.role == "tool" or TOOL_CALL_OPEN in str(message.content or "") for message in messages
    )
    if not needs_guard(text, had_evidence=had_evidence, had_tool_calls=had_tool_calls):
        return text

    language = _detect_language(text)
    nudge = ABSTENTION_NUDGE_EN if language == "en" else ABSTENTION_NUDGE_ES
    retry = engine.generate(
        [*messages, ChatMessage(role="system", content=nudge)],
        max_new_tokens=max_new_tokens or engine.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
    )
    retried = retry.text.strip()
    if retried and not needs_guard(retried, had_evidence=had_evidence, had_tool_calls=had_tool_calls):
        return retried
    return wrap_unverified(retried or text, language=language)


def guarded_answer(
    engine: LocalLlmEngine,
    messages: list[ChatMessage],
    *,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
) -> str:
    """Generate an answer, then apply the anti-fabrication guard.

    Used by the benchmark so what gets measured is exactly what ships.
    """
    result = engine.generate(
        messages,
        max_new_tokens=max_new_tokens or engine.max_new_tokens,
        temperature=engine.temperature if temperature is None else temperature,
        top_p=engine.top_p,
        top_k=engine.top_k,
    )
    return apply_guard(engine, messages, result.text, max_new_tokens=max_new_tokens)


def _content_to_chat(part: Any) -> tuple[str, list[MediaInput]]:
    """Flatten one message part into ``(text, media)``."""
    content = getattr(part, "content", "")
    if isinstance(content, str):
        return content, []
    if not isinstance(content, (list, tuple)):
        return str(content), []
    text_bits: list[str] = []
    media: list[MediaInput] = []
    for item in content:
        if isinstance(item, str):
            text_bits.append(item)
        elif isinstance(item, BinaryContent):
            media.append(MediaInput(data=bytes(item.data), media_type=item.media_type))
        else:
            for attr in ("content", "text", "url"):
                value = getattr(item, attr, None)
                if isinstance(value, str):
                    text_bits.append(value)
                    break
    return "\n".join(b for b in text_bits if b), media


def history_to_messages(history: list[Any]) -> list[ChatMessage]:
    """Convert pydantic-ai history into engine chat messages, in order."""
    messages: list[ChatMessage] = []
    for entry in history:
        if isinstance(entry, ModelRequest):
            for part in entry.parts:
                if isinstance(part, SystemPromptPart):
                    text, media = _content_to_chat(part)
                    messages.append(ChatMessage(role="system", content=text, media=media))
                elif isinstance(part, UserPromptPart):
                    text, media = _content_to_chat(part)
                    messages.append(ChatMessage(role="user", content=text, media=media))
                elif isinstance(part, ToolReturnPart):
                    text, _ = _content_to_chat(part)
                    messages.append(
                        ChatMessage(
                            role="tool",
                            content=_tool_return_text(part, text),
                            tool_call_id=part.tool_call_id,
                            name=part.tool_name,
                        )
                    )
                elif isinstance(part, RetryPromptPart):
                    text, _ = _content_to_chat(part)
                    messages.append(
                        ChatMessage(
                            role="tool",
                            content=f"Tool call failed: {text}",
                            tool_call_id=part.tool_call_id,
                            name=part.tool_name,
                        )
                    )
        elif isinstance(entry, ModelResponse):
            for part in entry.parts:
                if isinstance(part, TextPart):
                    messages.append(ChatMessage(role="assistant", content=part.content))
                elif isinstance(part, ToolCallPart):
                    payload = json.dumps(
                        {
                            "name": part.tool_name,
                            "arguments": part.args if part.args is not None else {},
                        },
                        ensure_ascii=False,
                    )
                    messages.append(ChatMessage(role="assistant", content=TOOL_CALL_OPEN + payload + TOOL_CALL_CLOSE))
    return messages


def _tool_return_text(part: ToolReturnPart, text: str) -> str:
    if isinstance(part.content, str):
        return f"<{part.tool_name}>{part.content}</{part.tool_name}>"
    return f"<{part.tool_name}>{text}</{part.tool_name}>"


def tools_to_payload(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Render pydantic-ai tool definitions in the schema the template expects."""
    payload: list[dict[str, Any]] = []
    for tool in tools:
        if getattr(tool, "kind", "function") != "function":
            continue
        payload.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.parameters_json_schema or {"type": "object", "properties": {}},
                },
            }
        )
    return payload


def _collect_media(history: list[Any]) -> list[MediaInput]:
    media: list[MediaInput] = []
    for message in history_to_messages(history):
        media.extend(message.media)
    return media


class LocalAgentModel:
    """A pydantic-ai ``Model`` backed by a local ONNX GenAI engine."""

    def __init__(
        self,
        engine: LocalLlmEngine,
        *,
        name: str = "local",
        transcriber: Any = None,
        adapt_media: bool = True,
    ):
        self.engine = engine
        self.name = name
        self.transcriber = transcriber
        self.adapt_media = adapt_media
        self._model = FunctionModel(
            function=self._respond,
            stream_function=self._stream,
            model_name=name,
        )

    # -- pydantic-ai surface --

    def __getattr__(self, item: str) -> Any:
        return getattr(self._model, item)

    @property
    def model(self) -> FunctionModel:
        return self._model

    # -- generation --

    def _prepare(self, history: list[Any], info: AgentInfo) -> tuple[list[ChatMessage], list[dict[str, Any]]]:
        messages = history_to_messages(history)
        if self.adapt_media:
            messages = self._reduce_media(messages)
        if not messages:
            raise LocalAgentModelError("empty history: nothing to send to the local model")
        tools = tools_to_payload(list(info.function_tools or []))
        if info.allow_text_output is False and not tools:
            raise LocalAgentModelError("no tools and no text output allowed")
        return messages, tools

    def _reduce_media(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Turn raw evidence into what the vision encoder can ingest.

        Video arrives as MP4 clips and audio as whole files; neither is accepted
        by the ONNX vision path, so they become frames and transcripts. Notes
        are appended to the message text so the model knows what was dropped.

        Whisper is resident only for as long as this takes: see
        ``WhisperTranscriber.session``.
        """
        if self.transcriber is not None and self._has_audio(messages):
            with self.transcriber.session():
                return self._adapt_messages(messages)
        return self._adapt_messages(messages)

    @staticmethod
    def _has_audio(messages: list[ChatMessage]) -> bool:
        return any(item.is_audio for message in messages for item in message.media)

    def _adapt_messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        from ragpoc.local_media import adapt_media as _adapt

        reduced: list[ChatMessage] = []
        for message in messages:
            if not message.media:
                reduced.append(message)
                continue
            adapted = _adapt(message.media, transcriber=self.transcriber)
            text = str(message.content or "")
            if adapted.text:
                text = (text + "\n" + adapted.text).strip()
            if adapted.notes:
                text = (text + "\n" + "\n".join(adapted.notes)).strip()
            reduced.append(
                ChatMessage(
                    role=message.role,
                    content=text,
                    media=adapted.images,
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                )
            )
        return reduced

    async def _respond(self, history: list[Any], info: AgentInfo) -> ModelResponse:
        messages, tools = self._prepare(history, info)
        result = self.engine.generate(
            messages,
            tools=tools or None,
            max_new_tokens=self.engine.max_new_tokens,
            temperature=self.engine.temperature,
            top_p=self.engine.top_p,
            top_k=self.engine.top_k,
        )
        visible, raw_calls = parse_tool_calls(result.text)
        parts: list[Any] = to_tool_parts(raw_calls, visible)
        if not parts:
            if not visible:
                visible = result.text.strip()
            if not visible:
                raise LocalAgentModelError("local model produced no text and no tool call")
            visible = apply_guard(self.engine, messages, visible)
            parts = [TextPart(content=visible)]
        return ModelResponse(parts=parts, model_name=self.name)

    async def _stream(self, history: list[Any], info: AgentInfo) -> AsyncIterator[Any]:
        messages, tools = self._prepare(history, info)
        buffer: list[str] = []
        for chunk in self.engine.generate_stream(
            messages,
            tools=tools or None,
            max_new_tokens=self.engine.max_new_tokens,
            temperature=self.engine.temperature,
            top_p=self.engine.top_p,
            top_k=self.engine.top_k,
        ):
            buffer.append(chunk)
            # Hold back anything that is, or might still become, a tool call so
            # its JSON payload never reaches the user-visible answer.
            if TOOL_CALL_OPEN in "".join(buffer) or _looks_like_bare_call("".join(buffer)):
                continue
            yield chunk
        full = "".join(buffer)
        visible, raw_calls = parse_tool_calls(full)
        if raw_calls:
            for position, call in enumerate(raw_calls):
                name = call.get("name") or call.get("function", {}).get("name")
                arguments = call.get("arguments")
                if arguments is None and isinstance(call.get("function"), dict):
                    arguments = call["function"].get("arguments")
                yield {
                    position: DeltaToolCall(
                        name=str(name) if name else "",
                        args=json.dumps(_coerce_args(arguments), ensure_ascii=False),
                    )
                }
            return
        if visible and TOOL_CALL_OPEN in full:
            # A call block that failed to parse still surfaces its prose.
            yield visible


def build_local_agent_model(
    engine: LocalLlmEngine, *, name: str | None = None, transcriber: Any = None
) -> LocalAgentModel:
    return LocalAgentModel(engine, name=name or "local", transcriber=transcriber)


__all__ = [
    "LocalAgentModel",
    "LocalAgentModelError",
    "ParsedTurn",
    "build_local_agent_model",
    "history_to_messages",
    "parse_tool_calls",
    "tools_to_payload",
    "to_tool_parts",
]
