"""Chat-model manifests and download for the local agent.

Separate from ``ragpoc.model_downloader`` (which owns the EmbeddingGemma export)
because the two repos publish different directory layouts. onnx-community ships
ONNX GenAI models under ``onnxruntime/<ep>/<quant>/``; only that subtree is
loadable by onnxruntime-genai, so only that subtree is downloaded.

Sizes below were read from the Hugging Face file tree and are the on-disk
footprint of a single quantization, which is also the working set at inference.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ragpoc.config import Settings

MB = 1024 * 1024


@dataclass(frozen=True)
class ChatModelManifest:
    key: str
    repo_id: str
    subdir: str
    display_name: str
    description: str
    size_bytes: int
    modalities: tuple[str, ...]
    budget_key: str
    # Files that must exist for the model to be considered installed.
    required: tuple[str, ...] = ("genai_config.json", "tokenizer.json")


CHAT_MODELS: dict[str, ChatModelManifest] = {
    "qwen3-vl-2b": ChatModelManifest(
        key="qwen3-vl-2b",
        repo_id="onnx-community/Qwen3-VL-2B-Instruct-ONNX",
        subdir="onnxruntime/cpu_and_mobile/cpu-int4-rtn-block-32",
        display_name="Qwen3-VL 2B (int4)",
        description=(
            "Texto, imagen y video. 28 capas, 8 KV heads, head_size 128. "
            "No procesa audio: se resuelve con whisper-tiny."
        ),
        size_bytes=1373 * MB,
        modalities=("text", "image", "video"),
        budget_key="qwen3-vl-2b-q4f16",
    ),
    "qwen3-0.6b": ChatModelManifest(
        key="qwen3-0.6b",
        repo_id="onnx-community/Qwen3-0.6B-ONNX",
        subdir="onnxruntime/cpu_and_mobile/cpu-int4-kld-block-128",
        display_name="Qwen3 0.6B (int4)",
        description="Solo texto. Sirve como modelo de humo y para hardware muy ajustado.",
        size_bytes=511 * MB,
        modalities=("text",),
        budget_key="qwen3-vl-2b-q4f16",
    ),
    "whisper-tiny": ChatModelManifest(
        key="whisper-tiny",
        # onnx-community/whisper-tiny is a transformers.js export with no
        # genai_config.json, so onnxruntime-genai cannot load it. This repo is a
        # real GenAI export (type "whisper", encoder + decoder).
        repo_id="tonythethompson/Whisper-Tiny-GenAI-ONNX",
        subdir="",
        display_name="Whisper tiny (GenAI)",
        description=(
            "Transcripcion de audio. Complementa a Qwen3-VL, que es ciego al sonido. "
            "Repositorio GenAI con encoder y decoder; el export de onnx-community "
            "no tiene genai_config y no es cargable."
        ),
        size_bytes=224 * MB,
        modalities=("audio",),
        budget_key="whisper-tiny-int8",
        required=("genai_config.json", "encoder.onnx", "decoder.onnx", "tokenizer.json"),
    ),
}

DEFAULT_CHAT_MODEL = "qwen3-vl-2b"


@dataclass
class ChatModelStatus:
    state: str = "not_downloaded"  # not_downloaded | downloading | ready | error
    downloaded_bytes: int = 0
    total_bytes: int = 0
    model_key: str = ""
    message: str = ""
    error: str = ""

    @property
    def percent(self) -> float:
        if self.total_bytes <= 0:
            return 0.0
        return min(100.0, self.downloaded_bytes / self.total_bytes * 100)

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "model_key": self.model_key,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "percent": round(self.percent, 2),
            "message": self.message,
            "error": self.error,
        }


ProgressListener = Callable[[dict], None]


class ChatModelDownloader:
    """Downloads one ONNX GenAI model subtree with progress and cancellation."""

    def __init__(self, settings: Settings, model_key: str = DEFAULT_CHAT_MODEL):
        if model_key not in CHAT_MODELS:
            raise ValueError(f"unknown chat model {model_key!r}; expected one of {sorted(CHAT_MODELS)}")
        self.settings = settings
        self.manifest = CHAT_MODELS[model_key]
        self._status = ChatModelStatus(model_key=model_key, total_bytes=self.manifest.size_bytes)
        self._listeners: list[ProgressListener] = []
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    @property
    def model_dir(self) -> Path:
        """Root folder for this specific model, not the shared cache."""
        return Path(self.settings.local_chat_cache_dir) / self.manifest.key

    @property
    def install_dir(self) -> Path:
        """Where the loadable files end up; the repo root when subdir is empty."""
        if not self.manifest.subdir:
            return self.model_dir
        return self.model_dir / self.manifest.subdir

    def add_progress_listener(self, listener: ProgressListener) -> None:
        self._listeners.append(listener)

    def remove_progress_listener(self, listener: ProgressListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _emit(self, **kwargs) -> None:
        payload = {**self._status.to_dict(), **kwargs}
        for listener in list(self._listeners):
            try:
                listener(payload)
            except Exception:
                continue

    def cancel(self) -> None:
        self._cancel.set()

    def reset_cancel(self) -> None:
        self._cancel.clear()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def get_status(self) -> ChatModelStatus:
        """Current status, reconciled against what is actually on disk.

        The in-memory status alone is misleading: a model downloaded by a
        previous run would still read as ``not_downloaded`` because nothing
        checked the install directory.
        """
        with self._lock:
            status = self._status
            if status.state not in {"downloading", "error"} and self.is_installed():
                status.state = "ready"
                status.downloaded_bytes = status.total_bytes or self.manifest.size_bytes
                status.total_bytes = status.total_bytes or self.manifest.size_bytes
                status.message = status.message or "installed"
            return status

    def is_installed(self) -> bool:
        root = self.install_dir
        return root.exists() and all((root / name).exists() for name in self.manifest.required)

    def _set(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self._status, key, value)
        self._emit(**kwargs)

    def download_blocking(self) -> ChatModelStatus:
        # A cancel is scoped to one attempt: a previous cancellation must not
        # silently abort every later download of the same model.
        self.reset_cancel()
        if self.is_installed():
            self._set(state="ready", downloaded_bytes=self.manifest.size_bytes, message="already installed")
            return self.get_status()
        try:
            self._download_files()
        except Exception as exc:
            self._set(state="error", error=str(exc), message="download failed")
            return self.get_status()
        if self.is_cancelled:
            self._set(state="not_downloaded", message="cancelled")
            return self.get_status()
        # Never report ready on trust: a subtree that matches nothing
        # downloads zero bytes and would otherwise look like success.
        if not self.is_installed():
            missing = [name for name in self.manifest.required if not (self.install_dir / name).exists()]
            self._set(
                state="error",
                message="download completed but files are missing",
                error=f"missing {missing} under {self.install_dir}; repo layout may have changed",
            )
            return self.get_status()
        self._set(state="ready", downloaded_bytes=self.manifest.size_bytes, message="installed")
        return self.get_status()

    def download(self) -> ChatModelStatus:
        # Publish "downloading" before the thread starts so the first status
        # poll after POST /api/chat-model/download never reads stale.
        self._set(state="downloading", message=f"downloading {self.manifest.repo_id}")
        thread = threading.Thread(target=self.download_blocking, name="chat-model-download", daemon=True)
        thread.start()
        return self.get_status()

    def _download_files(self) -> None:
        from huggingface_hub import HfApi, hf_hub_download

        self._set(state="downloading", message=f"downloading {self.manifest.repo_id}")
        prefix = f"{self.manifest.subdir}/" if self.manifest.subdir else ""
        files = [
            path
            for path in HfApi().list_repo_files(repo_id=self.manifest.repo_id)
            if path.startswith(prefix) and not path.endswith("/")
        ]
        if not files:
            # Surfaced as a proper error by download_blocking's verification.
            return

        # File by file rather than one snapshot_download() call: the progress
        # bar can only move when bytes are accounted for incrementally, and
        # cancellation needs a place to take effect between files.
        for rel_path in files:
            if self.is_cancelled:
                return
            hf_hub_download(
                repo_id=self.manifest.repo_id,
                filename=rel_path,
                local_dir=str(self.model_dir),
            )
            done = sum(p.stat().st_size for p in self.model_dir.rglob("*") if p.is_file())
            self._set(downloaded_bytes=done)
        if self.is_cancelled:
            return
        self._set(downloaded_bytes=self.manifest.size_bytes)


_downloaders: dict[tuple[str, str], ChatModelDownloader] = {}


def get_chat_downloader(settings: Settings, model_key: str | None = None) -> ChatModelDownloader:
    """The one downloader per (cache dir, model).

    Must be shared across requests: the download runs on a background thread
    attached to one instance, so if every request built its own downloader the
    status endpoint would report ``not_downloaded`` for a model that is being
    downloaded right now (and cancel would never reach the running thread).
    That is exactly what made the packaged app look like downloads failed the
    moment they started.
    """
    key = model_key or settings.local_chat_model
    cache_key = (str(settings.local_chat_cache_dir), key)
    if cache_key not in _downloaders:
        _downloaders[cache_key] = ChatModelDownloader(settings, key)
    return _downloaders[cache_key]


def list_chat_models() -> list[dict]:
    return [
        {
            "key": m.key,
            "display_name": m.display_name,
            "description": m.description,
            "size_mb": round(m.size_bytes / MB, 1),
            "modalities": list(m.modalities),
        }
        for m in CHAT_MODELS.values()
    ]
