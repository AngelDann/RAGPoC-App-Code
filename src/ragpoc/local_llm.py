"""Local chat LLM runtime on ONNX Runtime GenAI.

Runs the quantized ONNX export of a small multimodal model entirely on the
user's machine: no API key, no network at inference time. Text, image and audio
inputs are all supported; video arrives as extracted frames plus an audio track.

The engine is deliberately dumb about prompts and tool schemas -- it accepts
already-formatted messages and returns text. Prompt construction, tool-call
parsing and the agent loop live one layer up, so this module can be exercised
and benchmarked on its own.

Version note: ``onnxruntime-genai`` is pinned to 0.14.1, the last release whose
ORT API version is satisfied by ``onnxruntime-directml`` 1.24.4. Newer genai
wheels require ORT API 26 and fail to import against this runtime.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragpoc.budget import MemoryBudget
from ragpoc.metrics import MB, GenerationMetrics

SUPPORTED_DTYPES = ("q4f16", "q4", "quantized", "fp16")

# onnx-community publishes two incompatible layouts in the same repo:
#   * ``onnx/<name>_<dtype>.onnx`` -- transformers.js export, no genai_config.json
#   * ``onnxruntime/<ep>/<quant>/`` -- ONNX GenAI export, self-contained, with
#     genai_config.json + tokenizer + per-component .onnx files
# Only the second loads in onnxruntime-genai, so it is what we resolve to.
_GENAI_DIR_PREFERENCE = (
    "onnxruntime/cpu_and_mobile/cpu-int4-rtn-block-32",
    "onnxruntime/cpu_and_mobile/cpu-int4-kld-block-128",
    "onnxruntime/cpu_and_mobile/cpu-int4-kld-block-64",
    "onnxruntime/cpu_and_mobile/cpu-int4-rtn-block-64",
    "onnxruntime/cuda/cuda-int4-rtn-block-32",
    "onnxruntime/cuda/cuda-int4-kld-block-128",
)


class LocalLlmError(RuntimeError):
    pass


class LocalLlmNotLoaded(LocalLlmError):
    pass


class LocalLlmBudgetExceeded(LocalLlmError):
    pass


@dataclass
class MediaInput:
    """Binary evidence attached to a prompt.

    ``media_type`` follows MIME conventions; ``data`` is the raw file content.
    Video is never passed whole -- callers split it into frames first.
    """

    data: bytes
    media_type: str

    @property
    def is_image(self) -> bool:
        return self.media_type.startswith("image/")

    @property
    def is_audio(self) -> bool:
        return self.media_type.startswith("audio/")

    @property
    def extension(self) -> str:
        return {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "audio/wav": ".wav",
            "audio/mpeg": ".mp3",
            "audio/x-m4a": ".m4a",
            "audio/ogg": ".ogg",
        }.get(self.media_type, "")


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | list[Any] = ""
    media: list[MediaInput] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class GenerationResult:
    text: str
    metrics: GenerationMetrics
    prompt_tokens: int = 0
    finish_reason: str = "stop"

    def to_dict(self) -> dict:
        return {"text": self.text, "finish_reason": self.finish_reason, **self.metrics.summary()}


def _import_genai():
    try:
        import onnxruntime_genai as og
    except Exception as exc:  # pragma: no cover - exercised only on broken installs
        raise LocalLlmError(
            "onnxruntime-genai is not usable in this environment. It must be pinned to "
            "0.14.x to match onnxruntime-directml 1.24.x; 0.15+ requires ORT API 26."
        ) from exc
    return og


def _rss() -> int:
    from ragpoc.metrics import process_rss_bytes

    return process_rss_bytes()


def _create_processor(model: Any) -> Any:
    """Build the multimodal processor across genai API variants.

    0.14.x exposes ``create_multimodal_processor(self)``; earlier builds took
    the tokenizer as an argument.
    """
    try:
        return model.create_multimodal_processor()
    except TypeError:
        return model.create_multimodal_processor(None)


# Qwen3 emits reasoning inside these tags. The agent surfaces them separately
# (or not at all), so they are stripped from the visible answer.
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def split_thinking(text: str) -> tuple[str, str]:
    """Split a model turn into ``(visible, reasoning)``.

    An unterminated `` block is treated as reasoning rather than being
    discarded: a small model routinely runs out of budget mid-thought, and
    dropping it would silently empty the answer.
    """
    if _THINK_OPEN not in text:
        return text.strip(), ""
    visible: list[str] = []
    reasoning: list[str] = []
    depth = 0
    for part in text.split(_THINK_OPEN):
        if depth:
            if _THINK_CLOSE in part:
                thought, _, rest = part.partition(_THINK_CLOSE)
                reasoning.append(thought)
                visible.append(rest)
                depth = 0
            else:
                reasoning.append(part)
            continue
        depth = 1
        visible.append(part)
    return "".join(visible).strip(), "".join(reasoning).strip()


def strip_thinking(text: str) -> str:
    """Visible answer only; falls back to reasoning when nothing else survived."""
    visible, reasoning = split_thinking(text)
    return visible or reasoning


def close_thinking(rendered: str, template_thinks: bool = False) -> str:
    """Inject an empty, already-closed think block into a rendered prompt.

    Qwen3's template opens reasoning unconditionally at generation time unless
    ``enable_thinking=false`` was visible at render time -- and genai's
    ``apply_chat_template`` gives no way to pass that variable, so the model
    always starts thinking and a small one burns the whole token budget before
    reaching the answer. Appending an empty ``...`` block after the
    generation prompt is byte-for-byte what the template itself emits for
    ``enable_thinking=false``, so the budget goes straight to the visible turn.

    Only applied when the model's template actually has think-block handling;
    injecting the tags into a template without them would corrupt the prompt.
    """
    if not template_thinks or _THINK_CLOSE in rendered:
        return rendered
    stripped = rendered.rstrip()
    if not stripped.endswith("assistant"):
        return rendered
    return stripped + "\n" + _THINK_OPEN + "\n" + _THINK_CLOSE + "\n\n"


def resolve_model_path(model_dir: Path, dtype: str = "q4f16") -> Path:
    """Return the directory to hand to ``og.Model``.

    Prefers the ONNX GenAI export tree. ``dtype`` is kept in the signature so
    callers can pass the Settings value unchanged; the GenAI trees are already
    quantized and their dtype is encoded in the directory name.
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise LocalLlmError(f"model directory not found: {model_dir}")
    for candidate in _GENAI_DIR_PREFERENCE:
        path = model_dir / candidate
        if (path / "genai_config.json").exists():
            return path
    for path in sorted(model_dir.rglob("genai_config.json")):
        return path.parent
    raise LocalLlmError(
        f"no genai_config.json found under {model_dir}. The transformers.js "
        f"layout at onnx/*.onnx cannot be loaded by onnxruntime-genai; download "
        f"the onnxruntime/<ep>/<quant>/ subtree instead (dtype={dtype!r})."
    )


def read_context_length(model_dir: Path) -> int | None:
    path = Path(model_dir) / "genai_config.json"
    if not path.exists():
        return None
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return config.get("model", {}).get("context_length")


class LocalLlmEngine:
    """Loaded ONNX GenAI model with lazy tokenizer and multimodal processor."""

    def __init__(
        self,
        model_dir: Path,
        *,
        dtype: str = "q4f16",
        device: str = "auto",
        context_length: int = 2048,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 40,
        budget: MemoryBudget | None = None,
    ):
        if dtype not in SUPPORTED_DTYPES:
            raise LocalLlmError(f"unsupported dtype {dtype!r}; expected one of {SUPPORTED_DTYPES}")
        self.model_dir = Path(model_dir)
        self.dtype = dtype
        self.device = device
        self.context_length = context_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.budget = budget or MemoryBudget()
        self.load_seconds = 0.0
        self.model: Any = None
        self.tokenizer: Any = None
        self._processor: Any = None
        self._model_root: Path | None = None
        self._thinks: bool | None = None

    # -- lifecycle --

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self, budget_key: str = "qwen3-vl-2b-q4f16") -> None:
        if self.is_loaded:
            return
        og = _import_genai()
        decision = self.budget.check(
            budget_key, context_tokens=self.context_length, with_vision=True
        )
        if not decision.allowed:
            raise LocalLlmBudgetExceeded(f"{decision.reason} :: {decision.to_dict()}", decision)
        root = resolve_model_path(self.model_dir, self.dtype)
        started = time.perf_counter()
        rss_before = _rss()
        try:
            self.model = self._build_model(og, root)
            self.tokenizer = og.Tokenizer(self.model)
        except Exception as exc:
            self.model = None
            self.tokenizer = None
            raise LocalLlmError(f"failed to load local model from {root}: {exc}") from exc
        self._model_root = root
        self.load_seconds = time.perf_counter() - started
        self.budget.admit(budget_key, actual_bytes=max(0, _rss() - rss_before))

    def _build_model(self, og: Any, root: Path) -> Any:
        device = self._resolved_device(og)
        try:
            return og.Model(str(root), device) if device else og.Model(str(root))
        except TypeError:
            # Older wheels only accept a path; device selection is done via
            # genai_config.json provider_options instead.
            return og.Model(str(root))

    def _resolved_device(self, og: Any) -> str | None:
        if self.device == "cpu":
            return None
        if self.device == "dml":
            return "dml" if getattr(og, "is_dml_available", lambda: False)() else None
        if getattr(og, "is_dml_available", lambda: False)():
            return "dml"
        return None

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self._processor = None
        self._model_root = None
        self._thinks = None
        self.budget.release_all()

    @property
    def processor(self):
        if not self.is_loaded:
            raise LocalLlmNotLoaded("model is not loaded")
        if self._processor is None:
            self._processor = _create_processor(self.model)
        return self._processor

    # -- prompt assembly --

    def format_chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        *,
        enable_thinking: bool = False,
    ) -> str:
        """Render messages through the model's own chat template.

        onnxruntime-genai 0.14.x accepts the conversation as a JSON string and
        returns a rendered string -- it does not take Python lists or return
        token ids. Tool schemas are passed the same way, as a JSON string.

        ``enable_thinking`` is off by default. The shipped template honours that
        variable only when it is already defined in the render context, and the
        genai API does not expose a way to pass it, so the think block is closed
        explicitly when the template left it open.
        """
        if not self.is_loaded:
            raise LocalLlmNotLoaded("model is not loaded")
        payload = json.dumps(_chat_payload(messages), ensure_ascii=False)
        kwargs: dict[str, Any] = {"add_generation_prompt": True}
        if tools:
            kwargs["tools"] = json.dumps(tools, ensure_ascii=False)
        try:
            rendered = self.tokenizer.apply_chat_template(payload, **kwargs)
        except TypeError:
            # Some templates reject the tools slot; retry without it.
            kwargs.pop("tools", None)
            try:
                rendered = self.tokenizer.apply_chat_template(payload, **kwargs)
            except TypeError:
                rendered = self.tokenizer.apply_chat_template(payload)
        if enable_thinking:
            return rendered
        return close_thinking(rendered, self._template_has_thinking())

    def _template_has_thinking(self) -> bool:
        """True when the model's chat template handles `` blocks."""
        if self._thinks is None:
            self._thinks = False
            for root in (self._model_root, Path(self.model_dir)):
                if root is None:
                    continue
                path = Path(root) / "chat_template.jinja"
                if path.exists():
                    try:
                        self._thinks = _THINK_OPEN in path.read_text(encoding="utf-8")
                    except OSError:
                        self._thinks = False
                    break
        return self._thinks

    def count_tokens(self, text: str) -> int:
        if not self.is_loaded:
            raise LocalLlmNotLoaded("model is not loaded")
        try:
            encoded = self.tokenizer.encode(text)
            return len(encoded)
        except Exception:
            return max(1, len(text.split()))

    # -- generation --

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict] | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 40,
    ) -> GenerationResult:
        chunks: list[str] = []
        result = self.generate_stream(
            messages,
            tools=tools,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            on_first_token=None,
        )
        for piece in result:
            chunks.append(piece)
        return self._finalize(messages, chunks, getattr(self, "_last_metrics", GenerationMetrics()))

    def generate_stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict] | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 40,
        on_first_token=None,
    ) -> Iterator[str]:
        if not self.is_loaded:
            raise LocalLlmNotLoaded("model is not loaded")
        og = _import_genai()
        prompt = self.format_chat(messages, tools)
        media = [m for msg in messages for m in msg.media]
        params = self._make_params(og, max_new_tokens, temperature, top_p, top_k)
        generator = self._make_generator(og, params, prompt, media)

        started = time.perf_counter()
        first_token_at: float | None = None
        output_tokens = 0
        stream = self.tokenizer.create_stream()
        while not generator.is_done():
            generator.generate_next_token()
            token_id = int(generator.get_next_tokens()[-1])
            if first_token_at is None:
                first_token_at = time.perf_counter()
                if on_first_token is not None:
                    on_first_token(first_token_at - started)
            output_tokens += 1
            piece = stream.decode(token_id)
            if piece:
                yield piece
            if output_tokens >= max_new_tokens:
                break
        elapsed = time.perf_counter() - started
        prompt_tokens = self.count_tokens(_flatten_text(messages))
        self._last_metrics = GenerationMetrics(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            elapsed_s=elapsed,
            time_to_first_token_s=(first_token_at - started) if first_token_at else None,
            load_s=getattr(self, "load_seconds", 0.0),
        )

    def _finalize(
        self, messages: list[ChatMessage], chunks: list[str], metrics: GenerationMetrics
    ) -> GenerationResult:
        text = strip_thinking("".join(chunks))
        return GenerationResult(
            text=text,
            metrics=metrics,
            prompt_tokens=metrics.prompt_tokens,
            finish_reason="length" if metrics.output_tokens >= 1 and not text.endswith("\n") else "stop",
        )

    def _make_params(self, og: Any, max_new_tokens: int, temperature: float, top_p: float, top_k: int):
        params = og.GeneratorParams(self.model)
        # The exported genai_config.json advertises context_length 262144 and
        # past_present_share_buffer true. Honouring that would pre-allocate a KV
        # cache far past the 3 GB budget, so max_length is always clamped to the
        # engine's configured window.
        options = {
            "max_length": self.context_length,
            "max_new_tokens": max_new_tokens,
            "temperature": max(temperature, 0.0),
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": temperature > 0.0,
        }
        for key, value in options.items():
            applied = self._set_search_option(params, key, value)
            if not applied and key == "max_length":
                raise LocalLlmError("unable to clamp max_length; refusing to run unbounded")
        return params

    @staticmethod
    def _set_search_option(params: Any, key: str, value: Any) -> bool:
        """Apply one search option across the genai API variants."""
        try:
            params.set_search_options(**{key: value})
            return True
        except TypeError:
            pass
        except Exception:
            return False
        try:
            params.set_search_options({key: value})
            return True
        except Exception:
            return False

    def _make_generator(self, og: Any, params: Any, prompt: Any, media: list[MediaInput]):
        if media:
            return self._make_multimodal_generator(og, params, prompt, media)
        generator = og.Generator(self.model, params)
        tokens = self._encode_prompt(prompt)
        generator.append_tokens(tokens)
        return generator

    def _encode_prompt(self, prompt: Any) -> list[int]:
        if isinstance(prompt, str):
            return list(self.tokenizer.encode(prompt))
        return [int(t) for t in prompt]

    def _make_multimodal_generator(self, og: Any, params: Any, prompt: Any, media: list[MediaInput]):
        # `Images`/`Audios` have no constructor and `open_bytes` is a static
        # factory returning a fresh object; there is no way to accumulate more
        # than one, so callers tile multiple visuals into a single image first.
        image = next((m for m in media if m.is_image), None)
        audio = next((m for m in media if m.is_audio), None)
        text = prompt if isinstance(prompt, str) else self.tokenizer.decode(prompt)
        kwargs: dict[str, Any] = {"prompt": text}
        if image is not None:
            kwargs["images"] = og.Images.open_bytes(image.data)
        if audio is not None:
            kwargs["audios"] = og.Audios.open_bytes(audio.data)
        inputs = self.processor(**kwargs)
        generator = og.Generator(self.model, params)
        generator.set_inputs(inputs)
        return generator


def _chat_payload(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Render messages as the JSON structure the chat template expects.

    Visual evidence has to travel as ``{"type": "image"}`` items inside a list
    content: the template is what emits the `` ******` placeholders,
    and the processor refuses to run unless the placeholder count matches the
    number of images handed over.
    """
    payload: list[dict[str, Any]] = []
    for message in messages:
        text = message.content if isinstance(message.content, str) else json.dumps(
            message.content, ensure_ascii=False
        )
        if not message.media:
            payload.append({"role": message.role, "content": text})
            continue
        content: list[dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": text})
        for item in message.media:
            kind = "video" if item.media_type.startswith("video/") else "image"
            content.append({"type": kind})
        entry: dict[str, Any] = {"role": message.role, "content": content}
        if message.tool_call_id:
            entry["tool_call_id"] = message.tool_call_id
        if message.name:
            entry["name"] = message.name
        payload.append(entry)
    return payload


def _flatten_text(messages: list[ChatMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message.content, str):
            parts.append(message.content)
        else:
            parts.append(json.dumps(message.content, ensure_ascii=False))
    return "\n".join(parts)


def engine_from_settings(settings: Any, budget: MemoryBudget | None = None) -> LocalLlmEngine:
    """Build an engine from a ``ragpoc.config.Settings`` instance."""
    from pathlib import Path as _Path

    from ragpoc.chat_models import CHAT_MODELS

    key = getattr(settings, "local_chat_model", "")
    manifest = CHAT_MODELS.get(key)
    if manifest is not None:
        model_dir = _Path(settings.local_chat_cache_dir) / manifest.key
    else:
        model_dir = _Path(settings.local_chat_model_dir)
    return LocalLlmEngine(
        model_dir,
        dtype=getattr(settings, "local_chat_dtype", "q4f16"),
        device=getattr(settings, "local_chat_device", "auto"),
        context_length=int(getattr(settings, "local_chat_context_length", 2048)),
        max_new_tokens=int(getattr(settings, "local_chat_max_new_tokens", 512)),
        temperature=float(getattr(settings, "local_chat_temperature", 0.2)),
        top_p=float(getattr(settings, "local_chat_top_p", 0.9)),
        top_k=int(getattr(settings, "local_chat_top_k", 40)),
        budget=budget,
    )


__all__ = [
    "ChatMessage",
    "GenerationResult",
    "LocalLlmEngine",
    "LocalLlmError",
    "LocalLlmNotLoaded",
    "LocalLlmBudgetExceeded",
    "MediaInput",
    "engine_from_settings",
    "read_context_length",
    "resolve_model_path",
    "strip_thinking",
    "MB",
]
