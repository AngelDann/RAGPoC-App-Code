"""Hard memory budget for the local inference stack.

The product constraint is explicit: the app should stay near 2 GB in normal use
and must not exceed 3 GB total. This module turns that into an enforced,
measurable limit rather than a hope.

Two things live here:

* ``MemoryBudget`` -- tracks the current resident footprint and refuses to load
  a model that would push past the ceiling, so the failure is a clean error at
  load time instead of the OS swapping the whole desktop app.
* ``MODEL_MEMORY_ESTIMATES`` -- per-model weight footprints derived from the
  published ONNX export sizes, used for the admission decision before any
  allocation happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ragpoc.metrics import MB, process_rss_bytes

# Published on-disk size of the quantized ONNX exports actually shipped.
# Verified against the Hugging Face tree for each repo; keep in sync when the
# dtype selection in Settings changes.
MODEL_MEMORY_ESTIMATES: dict[str, int] = {
    # onnx-community/embeddinggemma-2-ONNX
    "embeddinggemma2-q4f16": 406 * MB,
    "embeddinggemma2-q4": 451 * MB,
    "embeddinggemma2-q8": 811 * MB,
    # text encoder only, for query embedding when no media is being indexed
    "embeddinggemma2-q4f16-text": 150 * MB,
    "embeddinggemma2-q8-text": 299 * MB,
    # onnx-community/Qwen3-VL-2B-Instruct-ONNX
    "qwen3-vl-2b-q4f16": 1310 * MB,
    "qwen3-vl-2b-q4": 1450 * MB,
    # onnx-community/whisper-tiny
    "whisper-tiny-int8": 96 * MB,
}

# Runtime overhead on top of the weight footprint: ONNX memory arena,
# activations, and the KV cache. The KV component scales with context length at
# 112 KB per token for Qwen3-VL-2B (28 layers x 8 KV heads x 128 head_dim x 2
# tensors x 2 bytes).
KV_BYTES_PER_TOKEN = 112 * 1024
RUNTIME_OVERHEAD_BYTES = 300 * MB
VISION_FORWARD_BYTES = 250 * MB

# Product ceiling. Nothing may be admitted that puts the process past this.
HARD_LIMIT_BYTES = 3 * 1024 * MB
# Comfortable working target; the budget warns above this.
SOFT_LIMIT_BYTES = 2 * 1024 * MB


@dataclass
class BudgetDecision:
    allowed: bool
    projected_mb: float
    current_mb: float
    required_mb: float
    limit_mb: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "projected_mb": round(self.projected_mb, 1),
            "current_mb": round(self.current_mb, 1),
            "required_mb": round(self.required_mb, 1),
            "limit_mb": round(self.limit_mb, 1),
            "reason": self.reason,
        }


@dataclass
class MemoryBudget:
    """Admission control for model loads against a hard RSS ceiling."""

    hard_limit_bytes: int = HARD_LIMIT_BYTES
    soft_limit_bytes: int = SOFT_LIMIT_BYTES
    loaded: dict[str, int] = field(default_factory=dict)

    @property
    def current_bytes(self) -> int:
        return process_rss_bytes()

    @property
    def tracked_bytes(self) -> int:
        return sum(self.loaded.values())

    @property
    def headroom_bytes(self) -> int:
        return max(0, self.hard_limit_bytes - self.current_bytes)

    def estimate(
        self,
        model_key: str,
        *,
        context_tokens: int = 2048,
        with_vision: bool = False,
    ) -> int:
        weights = MODEL_MEMORY_ESTIMATES[model_key]
        total = weights + RUNTIME_OVERHEAD_BYTES + KV_BYTES_PER_TOKEN * context_tokens
        if with_vision:
            total += VISION_FORWARD_BYTES
        return total

    def check(
        self,
        model_key: str,
        *,
        context_tokens: int = 2048,
        with_vision: bool = False,
    ) -> BudgetDecision:
        if model_key not in MODEL_MEMORY_ESTIMATES:
            return BudgetDecision(
                allowed=False,
                projected_mb=0.0,
                current_mb=self.current_bytes / MB,
                required_mb=0.0,
                limit_mb=self.hard_limit_bytes / MB,
                reason=f"unknown model key {model_key!r}",
            )
        required = self.estimate(
            model_key, context_tokens=context_tokens, with_vision=with_vision
        )
        current = self.current_bytes
        projected = current + required - self.loaded.get(model_key, 0)
        if projected > self.hard_limit_bytes:
            return BudgetDecision(
                allowed=False,
                projected_mb=projected / MB,
                current_mb=current / MB,
                required_mb=required / MB,
                limit_mb=self.hard_limit_bytes / MB,
                reason=(
                    f"loading {model_key} would reach {projected / MB:.0f} MB, "
                    f"over the {self.hard_limit_bytes / MB:.0f} MB ceiling"
                ),
            )
        warning = projected > self.soft_limit_bytes
        return BudgetDecision(
            allowed=True,
            projected_mb=projected / MB,
            current_mb=current / MB,
            required_mb=required / MB,
            limit_mb=self.hard_limit_bytes / MB,
            reason=("above the soft target but within the ceiling" if warning else "within budget"),
        )

    def admit(self, model_key: str, actual_bytes: int | None = None) -> BudgetDecision:
        """Record a model as resident after it has been loaded.

        The pre-flight gate is ``check``, which projects an estimate on top of
        free memory. By the time ``admit`` runs the allocation has already
        happened and is reflected in the process RSS, so re-projecting the
        estimate would count the same bytes twice. Only the measured footprint
        is recorded here; the ceiling is enforced against the real RSS.
        """
        cost = actual_bytes if actual_bytes is not None else MODEL_MEMORY_ESTIMATES.get(model_key, 0)
        current = self.current_bytes
        if current > self.hard_limit_bytes:
            decision = BudgetDecision(
                allowed=False,
                projected_mb=current / MB,
                current_mb=current / MB,
                required_mb=cost / MB,
                limit_mb=self.hard_limit_bytes / MB,
                reason=(
                    f"process is at {current / MB:.0f} MB after loading {model_key}, "
                    f"over the {self.hard_limit_bytes / MB:.0f} MB ceiling"
                ),
            )
            raise MemoryBudgetExceeded(decision.reason, decision)
        self.loaded[model_key] = cost
        return BudgetDecision(
            allowed=True,
            projected_mb=current / MB,
            current_mb=current / MB,
            required_mb=cost / MB,
            limit_mb=self.hard_limit_bytes / MB,
            reason="recorded",
        )

    def release(self, model_key: str) -> None:
        self.loaded.pop(model_key, None)

    def release_all(self) -> None:
        self.loaded.clear()

    def report(self) -> dict:
        return {
            "current_mb": round(self.current_bytes / MB, 1),
            "tracked_mb": round(self.tracked_bytes / MB, 1),
            "headroom_mb": round(self.headroom_bytes / MB, 1),
            "hard_limit_mb": round(self.hard_limit_bytes / MB, 1),
            "soft_limit_mb": round(self.soft_limit_bytes / MB, 1),
            "within_hard_limit": self.current_bytes <= self.hard_limit_bytes,
            "loaded": dict(self.loaded),
        }


class MemoryBudgetExceeded(RuntimeError):
    def __init__(self, message: str, decision: BudgetDecision):
        super().__init__(message)
        self.decision = decision
