from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from ragpoc.config import Settings

_REPO_ID = "onnx-community/embeddinggemma-2-ONNX"

# Files needed per dtype suffix.
_DTYPE_FILES: dict[str, list[str]] = {
    "fp32": ["", ""],
    "fp16": ["_fp16", "_fp16"],
    "q4": ["_q4", "_q4"],
    "q4f16": ["_q4f16", "_q4f16"],
    "q8": ["_quantized", "_quantized"],
}

_BASE_FILES = [
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "config_sentence_transformers.json",
]

_COMPONENTS = ["model", "vision_encoder", "audio_encoder"]

# Approximate total download sizes per dtype (bytes).
_DTYPE_SIZES: dict[str, int] = {
    "fp32": 2_929_000_000,
    "fp16": 1_465_000_000,
    "q4": 473_000_000,
    "q4f16": 426_000_000,
    "q8": 850_000_000,
}


@dataclass
class ModelStatus:
    state: str = "not_downloaded"  # "not_downloaded" | "downloading" | "ready" | "error"
    downloaded_bytes: int = 0
    total_bytes: int = 0
    dtype: str = "q8"
    message: str = ""
    error: str = ""

    @property
    def percent(self) -> float:
        if self.total_bytes <= 0:
            return 0.0
        return min(100.0, self.downloaded_bytes / self.total_bytes * 100)


ProgressListener = Callable[[dict], None]


@dataclass
class ModelDownloader:
    settings: Settings
    _listeners: list[ProgressListener] = field(default_factory=list)
    _cancel_flag: bool = False
    _status: ModelStatus = field(default_factory=ModelStatus)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # -- properties --

    @property
    def model_dir(self) -> Path:
        return self.settings.local_model_dir

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_flag

    # -- status --

    def get_status(self) -> ModelStatus:
        """Current status: live while a download runs, disk-truth otherwise.

        Rebuilding the status from disk alone was a silent lie that made the
        packaged app look broken: while a download is in flight the required
        files are not complete yet, so a disk scan reads ``not_downloaded`` and
        the console's 1.5s poll hides the progress bar and reports
        "Modelo no descargado" while the download keeps running in the
        background. The in-memory status is the only place that knows about an
        in-flight or failed download, so it wins for those two states; every
        other state is still reconciled against what is actually on disk (a
        model downloaded by a previous run must read as ``ready``).
        """
        with self._lock:
            if self._status.state in {"downloading", "error"}:
                # Copy: callers (and JsonResponse serializers) must not be able
                # to corrupt the downloader's own bookkeeping.
                return replace(self._status)
        return self._scan_disk()

    def _required_files(self, dtype: str) -> list[str]:
        suffix = _DTYPE_SUFFIX.get(dtype, "_quantized")
        required = list(_BASE_FILES)
        for comp in _COMPONENTS:
            required.append(f"onnx/{comp}{suffix}.onnx")
            required.append(f"onnx/{comp}{suffix}.onnx_data")
        return required

    def _scan_disk(self) -> ModelStatus:
        model_dir = self.model_dir
        dtype = self.settings.local_embedding_dtype

        if not model_dir.exists():
            return ModelStatus(state="not_downloaded", dtype=dtype, total_bytes=_DTYPE_SIZES.get(dtype, 0))

        required = self._required_files(dtype)
        downloaded = 0
        all_present = True
        for fname in required:
            fpath = model_dir / fname
            if fpath.exists():
                downloaded += fpath.stat().st_size
            else:
                all_present = False

        if all_present:
            return ModelStatus(
                state="ready",
                downloaded_bytes=downloaded,
                total_bytes=downloaded,
                dtype=dtype,
            )
        return ModelStatus(
            state="not_downloaded",
            downloaded_bytes=downloaded,
            total_bytes=_DTYPE_SIZES.get(dtype, 0),
            dtype=dtype,
        )

    # -- progress --

    def add_progress_listener(self, listener: ProgressListener) -> None:
        self._listeners.append(listener)

    def remove_progress_listener(self, listener: ProgressListener) -> None:
        self._listeners = [item for item in self._listeners if item is not listener]

    def _emit_progress(self, **kwargs) -> None:
        with self._lock:
            for key, val in kwargs.items():
                if hasattr(self._status, key):
                    setattr(self._status, key, val)
            payload = {
                "state": self._status.state,
                "downloaded_bytes": self._status.downloaded_bytes,
                "total_bytes": self._status.total_bytes,
                "percent": self._status.percent,
                "dtype": self._status.dtype,
                "message": self._status.message,
                "error": self._status.error,
            }
        for listener in self._listeners:
            try:
                listener(payload)
            except Exception:
                pass

    # -- download --

    def cancel(self) -> None:
        self._cancel_flag = True

    def reset_cancel(self) -> None:
        self._cancel_flag = False

    def download(self, dtype: str | None = None) -> ModelStatus:
        """Download the ONNX model weights for *dtype*.  Blocking; run in a thread."""
        dtype = dtype or self.settings.local_embedding_dtype
        self._cancel_flag = False
        self._status = ModelStatus(state="downloading", dtype=dtype, total_bytes=_DTYPE_SIZES.get(dtype, 0))
        self._emit_progress(state="downloading", message="Starting download…")

        try:
            self._download_files(dtype)
        except Exception as exc:
            if self._cancel_flag:
                self._emit_progress(state="not_downloaded", message="Download cancelled")
            else:
                self._emit_progress(state="error", error=str(exc)[:500], message="download failed")
            return self.get_status()

        if self._cancel_flag:
            self._emit_progress(state="not_downloaded", message="Download cancelled")
            return self.get_status()

        # Never report ready on trust: a dtype suffix that matches nothing
        # downloads zero bytes and would otherwise look like success.
        missing = [name for name in self._required_files(dtype) if not (self.model_dir / name).exists()]
        if missing:
            self._emit_progress(
                state="error",
                message="download completed but files are missing",
                error=f"missing {missing} under {self.model_dir}",
            )
            return self.get_status()

        self._emit_progress(state="ready", message="Model ready")
        return self.get_status()

    def _download_files(self, dtype: str) -> None:
        from huggingface_hub import hf_hub_download

        suffix = _DTYPE_SUFFIX.get(dtype, "_quantized")
        patterns = list(_BASE_FILES)
        for comp in _COMPONENTS:
            patterns.append(f"onnx/{comp}{suffix}.onnx")
            patterns.append(f"onnx/{comp}{suffix}.onnx_data")

        self.model_dir.mkdir(parents=True, exist_ok=True)
        total = _DTYPE_SIZES.get(dtype, 0)
        done = 0

        for fname in patterns:
            if self._cancel_flag:
                return
            self._emit_progress(message=f"Downloading {fname}…")
            hf_hub_download(
                _REPO_ID,
                fname,
                repo_type="model",
                local_dir=str(self.model_dir),
            )
            fpath = self.model_dir / fname
            if fpath.exists():
                done += fpath.stat().st_size
            self._emit_progress(downloaded_bytes=done, total_bytes=total)

    def download_blocking(self, dtype: str | None = None) -> ModelStatus:
        """Alias kept for clarity at call sites."""
        return self.download(dtype)


_DTYPE_SUFFIX = {
    "fp32": "",
    "fp16": "_fp16",
    "q4": "_q4",
    "q4f16": "_q4f16",
    "q8": "_quantized",
}

_downloaders: dict[str, ModelDownloader] = {}


def get_downloader(settings: Settings) -> ModelDownloader:
    key = str(settings.local_model_dir)
    if key not in _downloaders:
        _downloaders[key] = ModelDownloader(settings=settings)
    return _downloaders[key]
