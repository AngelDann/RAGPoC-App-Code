"""Audio transcription with an ONNX GenAI Whisper export.

Whisper's encoder/decoder pair is what turns audio evidence into text the agent
can reason over. The chat model used for answers is vision-only, so without this
step audio chunks are search-only: the retriever can match them but the model
cannot say what was said.

Loaded lazily and separately from the chat engine, because it is only needed
when audio evidence actually reaches the prompt and holding both resident at
once competes for the memory budget.
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragpoc.budget import MemoryBudget
from ragpoc.local_llm import LocalLlmError, MediaInput

# Any whisper control token: task/language markers and timestamps alike.
_MARKER_RE = re.compile(r"<\|[^|]*\|>")

# Whisper task/language markers. `<|notimestamps|>` keeps the transcript clean;
# without it the output is interleaved with cue markup the agent cannot use.
_SOT = 50258
_EOT = 50257
_NOTIMESTAMPS = 50363
_TRANSCRIBE = 50359
_LANGUAGE_ES = 50262
_LANGUAGE_EN = 50259

DEFAULT_MAX_TOKENS = 224


class WhisperError(LocalLlmError):
    pass


@dataclass
class TranscriptionResult:
    text: str
    elapsed_s: float
    output_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "elapsed_s": round(self.elapsed_s, 3),
            "output_tokens": self.output_tokens,
        }


def _language_token(language: str) -> int | None:
    return {
        "es": _LANGUAGE_ES,
        "en": _LANGUAGE_EN,
    }.get((language or "").lower()[:2])


class WhisperTranscriber:
    """Transcribes ``MediaInput`` audio into plain text."""

    def __init__(
        self,
        model_dir: Path,
        *,
        device: str = "auto",
        language: str = "es",
        budget: MemoryBudget | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.device = device
        self.language = language
        self.budget = budget or MemoryBudget()
        self.model: Any = None
        self.tokenizer: Any = None
        self._processor: Any = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.is_loaded:
            return
        from ragpoc.local_llm import _import_genai

        og = _import_genai()
        started = time.perf_counter()
        try:
            from ragpoc.local_llm import _create_processor

            self.model = og.Model(str(self.model_dir))
            self.tokenizer = og.Tokenizer(self.model)
            self._processor = _create_processor(self.model)
        except Exception as exc:
            self.model = None
            self.tokenizer = None
            self._processor = None
            raise WhisperError(f"failed to load whisper model from {self.model_dir}: {exc}") from exc
        self.load_seconds = time.perf_counter() - started
        self.budget.admit("whisper-tiny-int8")

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self._processor = None
        self.budget.release("whisper-tiny-int8")

    @contextmanager
    def session(self):
        """Resident only while audio is actually being transcribed.

        Whisper and the chat model together peak at ~2.8 GB, which is most of
        the 3 GB ceiling. Keeping whisper loaded for the whole conversation
        spends that even when no audio evidence ever appears; loading it around
        the transcription makes the common text-and-image path peak near 2.3 GB
        and pays a ~2 s reload only on turns that carry audio.
        """
        was_loaded = self.is_loaded
        self.load()
        try:
            yield self
        finally:
            if not was_loaded:
                self.unload()

    def _prompt(self) -> str:
        # Whisper expects the task/language markers as a prompt of special ids,
        # which the tokenizer maps from these control strings.
        parts = ["<|startoftranscript|>"]
        token = _language_token(self.language)
        if token is not None:
            parts.append(f"<|{self.language}|>")
        parts.append("<|transcribe|>")
        parts.append("<|notimestamps|>")
        return "".join(parts)

    def transcribe(self, media: MediaInput) -> str:
        return self.transcribe_with_metrics(media).text

    def transcribe_with_metrics(self, media: MediaInput) -> TranscriptionResult:
        if not self.is_loaded:
            self.load()
        if not media.data:
            return TranscriptionResult(text="", elapsed_s=0.0)
        og = self._require_genai()
        started = time.perf_counter()
        try:
            # `Audios` has no constructor; `open_bytes` is a static factory.
            audios = og.Audios.open_bytes(media.data)
            inputs = self._processor(prompt=self._prompt(), audios=audios)
        except Exception as exc:
            raise WhisperError(f"whisper could not process the audio: {exc}") from exc

        params = og.GeneratorParams(self.model)
        self._set_option(params, "max_length", DEFAULT_MAX_TOKENS)
        self._set_option(params, "do_sample", False)
        generator = og.Generator(self.model, params)
        generator.set_inputs(inputs)

        pieces: list[str] = []
        stream = self.tokenizer.create_stream()
        count = 0
        try:
            while not generator.is_done():
                generator.generate_next_token()
                token_id = int(generator.get_next_tokens()[-1])
                count += 1
                piece = stream.decode(token_id)
                if piece:
                    pieces.append(piece)
                if count >= DEFAULT_MAX_TOKENS:
                    break
        except Exception as exc:
            raise WhisperError(f"whisper generation failed: {exc}") from exc

        elapsed = time.perf_counter() - started
        return TranscriptionResult(
            text=_clean_transcript("".join(pieces)),
            elapsed_s=elapsed,
            output_tokens=count,
        )

    @staticmethod
    def _set_option(params: Any, key: str, value: Any) -> bool:
        try:
            params.set_search_options(**{key: value})
            return True
        except TypeError:
            try:
                params.set_search_options({key: value})
                return True
            except Exception:
                return False
        except Exception:
            return False

    @staticmethod
    def _require_genai():
        from ragpoc.local_llm import _import_genai

        return _import_genai()


def _clean_transcript(text: str) -> str:
    """Strip whisper control markers that survive decoding.

    Markers are removed rather than truncated at, because whisper interleaves
    timestamp tokens like ``<|0.96|>`` with real text: cutting at the first one
    would throw away most of the transcript.
    """
    out = _MARKER_RE.sub(" ", text)
    return " ".join(out.split()).strip()


def transcriber_from_settings(settings: Any, budget: MemoryBudget | None = None) -> WhisperTranscriber:
    from ragpoc.chat_models import CHAT_MODELS

    key = "whisper-tiny"
    root = Path(settings.local_chat_cache_dir) / key
    manifest = CHAT_MODELS.get(key)
    if manifest is not None and manifest.subdir:
        root = root / manifest.subdir
    return WhisperTranscriber(
        root,
        device=getattr(settings, "local_chat_device", "auto"),
        language="es",
        budget=budget,
    )


__all__ = [
    "TranscriptionResult",
    "WhisperError",
    "WhisperTranscriber",
    "transcriber_from_settings",
]
