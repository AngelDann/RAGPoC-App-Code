"""Instrumentation primitives for memory, latency and generation metrics.

Everything the evaluation suite reports about resource usage and inference time
goes through this module, so the numbers in the final report are reproducible
and comparable across runs.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

# Windows working-set figures are reported in bytes; keep one unit everywhere.
MB = 1024 * 1024


def _psutil():
    import psutil

    return psutil


def process_rss_bytes(include_children: bool = True) -> int:
    """Resident set size of this process, optionally plus its children.

    ffmpeg is spawned as a child during video ingestion, and uvicorn may hold
    worker children, so a whole-tree RSS is the number the user actually feels.
    """
    psutil = _psutil()
    proc = psutil.Process()
    total = proc.memory_info().rss
    if include_children:
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    return total


def process_vms_bytes() -> int:
    return _psutil().Process().memory_info().vms


def system_available_bytes() -> int:
    return _psutil().virtual_memory().available


@dataclass(frozen=True)
class MemorySnapshot:
    rss_bytes: int
    vms_bytes: int
    system_available_bytes: int
    timestamp: float

    @property
    def rss_mb(self) -> float:
        return self.rss_bytes / MB


def snapshot_memory() -> MemorySnapshot:
    psutil = _psutil()
    info = psutil.Process().memory_info()
    return MemorySnapshot(
        rss_bytes=info.rss,
        vms_bytes=info.vms,
        system_available_bytes=psutil.virtual_memory().available,
        timestamp=time.perf_counter(),
    )


class PeakMemoryTracker:
    """Samples process RSS on a background thread and records the peak.

    Sampling (rather than reading on demand) is what catches transient spikes:
    ONNX session creation, KV-cache growth, ViT forward passes and ffmpeg all
    allocate outside the window a caller would think to measure.
    """

    def __init__(self, interval: float = 0.05, include_children: bool = True):
        self.interval = interval
        self.include_children = include_children
        self.samples: list[MemorySnapshot] = []
        self.baseline_bytes: int = 0
        self.peak_bytes: int = 0
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> PeakMemoryTracker:
        self.baseline_bytes = process_rss_bytes(self.include_children)
        self.peak_bytes = self.baseline_bytes
        self.samples = [snapshot_memory()]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 4)
        self._record()

    def _run(self) -> None:
        assert self._stop is not None
        while not self._stop.wait(self.interval):
            self._record()

    def _record(self) -> None:
        snap = snapshot_memory()
        self.samples.append(snap)
        if snap.rss_bytes > self.peak_bytes:
            self.peak_bytes = snap.rss_bytes

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / MB

    @property
    def delta_mb(self) -> float:
        """Peak RSS above the baseline captured at context entry."""
        return max(0, self.peak_bytes - self.baseline_bytes) / MB

    @property
    def baseline_mb(self) -> float:
        return self.baseline_bytes / MB

    def summary(self) -> dict:
        return {
            "baseline_mb": round(self.baseline_mb, 1),
            "peak_mb": round(self.peak_mb, 1),
            "delta_mb": round(self.delta_mb, 1),
            "samples": len(self.samples),
            "interval_s": self.interval,
        }


@dataclass
class LatencyStats:
    """Wall-clock timing for a single unit of work."""

    label: str
    elapsed_s: float
    started_at: float = 0.0

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_s * 1000


@contextmanager
def timed(label: str = "op") -> Iterator[LatencyStats]:
    stats = LatencyStats(label=label, elapsed_s=0.0, started_at=time.perf_counter())
    try:
        yield stats
    finally:
        stats.elapsed_s = time.perf_counter() - stats.started_at


@dataclass
class GenerationMetrics:
    """Inference metrics for one generated response."""

    prompt_tokens: int = 0
    output_tokens: int = 0
    elapsed_s: float = 0.0
    time_to_first_token_s: float | None = None
    load_s: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return self.output_tokens / self.elapsed_s

    def summary(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "elapsed_s": round(self.elapsed_s, 3),
            "time_to_first_token_s": (
                round(self.time_to_first_token_s, 3) if self.time_to_first_token_s is not None else None
            ),
            "load_s": round(self.load_s, 3),
            "tokens_per_second": round(self.tokens_per_second, 2),
        }


class TokenCounter:
    """Counts tokens emitted by a streaming generator without a tokenizer.

    A tokenizer call per chunk is expensive and would pollute the very latency
    being measured, so we count whitespace-separated pieces and let callers that
    have a real tokenizer overwrite the number afterwards.
    """

    def __init__(self) -> None:
        self.count = 0
        self._pending = ""

    def feed(self, text: str) -> None:
        self._pending += text
        pieces = self._pending.split()
        if self._pending.endswith((" ", "\n", "\t")):
            self.count += len(pieces)
            self._pending = ""
        else:
            if pieces:
                self.count += len(pieces) - 1
                self._pending = pieces[-1]

    def finalize(self) -> int:
        self.count += len(self._pending.split())
        self._pending = ""
        return self.count


@dataclass
class RunRecord:
    """One benchmark row, serializable to the evaluation JSON output."""

    name: str
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, **self.metrics}
