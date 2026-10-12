"""Instrumentation primitives for memory, latency and generation metrics.

Everything the evaluation suite reports about resource usage and inference time
goes through this module, so the numbers in the final report are reproducible
and comparable across runs.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

# Windows working-set figures are reported in bytes; keep one unit everywhere.
MB = 1024 * 1024


def _psutil():
    """psutil when installed, ``None`` otherwise.

    psutil is an optional dependency and is absent from the packaged app
    (and from any environment installed from requirements.txt), where its
    absence used to surface as a hard ``ModuleNotFoundError`` -- a 500 on
    /api/agent/status, and a local agent whose RAM budget always raised.
    Memory accounting must degrade, never crash: the ctypes fallbacks below
    read the same OS counters for the current process.
    """
    try:
        import psutil
    except Exception:
        return None
    return psutil


def _windows_process_memory() -> tuple[int, int]:
    """(RSS, VMS) for this process via GetProcessMemoryInfo; no psutil needed.

    The prototypes have to be declared: GetCurrentProcess() returns a HANDLE,
    and calling it through ctypes without a restype truncates the pseudo-handle
    to 32 bits, which makes GetProcessMemoryInfo fail silently and report 0.
    """
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError(f"GetProcessMemoryInfo failed: {ctypes.get_last_error()}")
    return int(counters.WorkingSetSize), int(counters.PagefileUsage)


def _posix_process_memory() -> tuple[int, int]:
    """(RSS, VMS) from /proc on Linux, peak RSS from getrusage elsewhere."""
    try:
        statm = Path("/proc/self/statm").read_text().split()
        page = os.sysconf("SC_PAGE_SIZE")
        return int(statm[1]) * page, int(statm[2]) * page
    except Exception:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # ru_maxrss is KiB on Linux and bytes on macOS.
        return (peak if sys.platform == "darwin" else peak * 1024), 0


def _fallback_process_memory() -> tuple[int, int]:
    return _windows_process_memory() if os.name == "nt" else _posix_process_memory()


def _process_memory() -> tuple[int, int]:
    """(RSS, VMS) for this process: psutil when available, OS counters otherwise."""
    psutil = _psutil()
    if psutil is None:
        return _fallback_process_memory()
    info = psutil.Process().memory_info()
    return info.rss, info.vms


def _fallback_available_bytes() -> int:
    """Physical memory available to new processes, without psutil."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
        kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            raise OSError(f"GlobalMemoryStatusEx failed: {ctypes.get_last_error()}")
        return int(stat.ullAvailPhys)
    try:
        return int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) * 1024
    except Exception:
        return 0


def process_rss_bytes(include_children: bool = True) -> int:
    """Resident set size of this process, optionally plus its children.

    ffmpeg is spawned as a child during video ingestion, and uvicorn may hold
    worker children, so a whole-tree RSS is the number the user actually feels.
    Children are only counted when psutil is installed; without it the current
    process alone is still a usable budget input.
    """
    psutil = _psutil()
    if psutil is None:
        return _process_memory()[0]
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
    return _process_memory()[1]


def system_available_bytes() -> int:
    psutil = _psutil()
    if psutil is None:
        return _fallback_available_bytes()
    return psutil.virtual_memory().available


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
    rss_bytes, vms_bytes = _process_memory()
    return MemorySnapshot(
        rss_bytes=rss_bytes,
        vms_bytes=vms_bytes,
        system_available_bytes=system_available_bytes(),
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
