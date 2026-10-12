import time

from ragpoc.budget import (
    MODEL_MEMORY_ESTIMATES,
    MemoryBudget,
    MemoryBudgetExceeded,
)
from ragpoc.eval_retrieval import (
    RetrievalCase,
    RetrievalResult,
    aggregate,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_case,
)
from ragpoc.metrics import (
    MB,
    GenerationMetrics,
    LatencyStats,
    PeakMemoryTracker,
    TokenCounter,
    process_rss_bytes,
    snapshot_memory,
    timed,
)

# --- metrics ---


def test_process_rss_bytes_is_positive():
    assert process_rss_bytes() > 0


def test_memory_metrics_degrade_without_psutil(monkeypatch):
    """The packaged app ships without psutil; metrics must not raise there.

    A missing psutil used to surface as ModuleNotFoundError: a 500 on
    /api/agent/status and a local agent whose RAM budget always raised.
    """
    from ragpoc import metrics

    monkeypatch.setattr(metrics, "_psutil", lambda: None)
    assert metrics.process_rss_bytes() > 0
    assert metrics.process_vms_bytes() >= 0
    assert metrics.system_available_bytes() > 0
    snap = metrics.snapshot_memory()
    assert snap.rss_bytes > 0
    with metrics.PeakMemoryTracker(interval=0.01) as tracker:
        pass
    assert tracker.peak_mb > 0


def test_snapshot_memory_reports_mb():
    snap = snapshot_memory()
    assert snap.rss_mb > 0
    assert abs(snap.rss_bytes / MB - snap.rss_mb) < 1e-6


def test_peak_memory_tracker_records_baseline_and_peak():
    with PeakMemoryTracker(interval=0.01) as tracker:
        _ = bytearray(8 * MB)
        time.sleep(0.05)
    assert tracker.baseline_mb > 0
    assert tracker.peak_mb >= tracker.baseline_mb
    assert len(tracker.samples) >= 1
    summary = tracker.summary()
    assert set(summary) == {"baseline_mb", "peak_mb", "delta_mb", "samples", "interval_s"}


def test_peak_memory_tracker_delta_is_non_negative():
    tracker = PeakMemoryTracker(interval=0.01)
    with tracker:
        pass
    assert tracker.delta_mb >= 0


def test_timed_measures_elapsed():
    with timed("sleep") as stats:
        time.sleep(0.02)
    assert isinstance(stats, LatencyStats)
    assert stats.elapsed_s >= 0.015
    assert stats.elapsed_ms == stats.elapsed_s * 1000


def test_generation_metrics_tokens_per_second():
    metrics = GenerationMetrics(prompt_tokens=10, output_tokens=20, elapsed_s=2.0)
    assert metrics.tokens_per_second == 10.0
    summary = metrics.summary()
    assert summary["tokens_per_second"] == 10.0
    assert summary["output_tokens"] == 20


def test_generation_metrics_zero_elapsed_is_safe():
    assert GenerationMetrics().tokens_per_second == 0.0


def test_token_counter_counts_words_across_chunks():
    counter = TokenCounter()
    counter.feed("hola ")
    counter.feed("mundo esto es")
    counter.feed(" una prueba")
    assert counter.finalize() == 6


# --- budget ---


def test_budget_starts_within_hard_limit():
    budget = MemoryBudget()
    report = budget.report()
    assert report["within_hard_limit"] is True
    assert report["tracked_mb"] == 0.0


def test_budget_estimate_includes_weights_and_kv():
    budget = MemoryBudget()
    small = budget.estimate("qwen3-vl-2b-q4f16", context_tokens=1024)
    large = budget.estimate("qwen3-vl-2b-q4f16", context_tokens=4096)
    assert large > small
    kv_delta = large - small
    # 3072 extra tokens x 112 KB per token
    assert abs(kv_delta - 3072 * 112 * 1024) < MB


def test_budget_estimate_vision_costs_extra():
    budget = MemoryBudget()
    without = budget.estimate("qwen3-vl-2b-q4f16", with_vision=False)
    with_vision = budget.estimate("qwen3-vl-2b-q4f16", with_vision=True)
    assert with_vision > without


def test_budget_rejects_unknown_model():
    decision = MemoryBudget().check("does-not-exist")
    assert decision.allowed is False
    assert "unknown model key" in decision.reason


def test_budget_rejects_load_past_ceiling(tmp_path):
    budget = MemoryBudget(hard_limit_bytes=1 * MB, soft_limit_bytes=1)
    decision = budget.check("qwen3-vl-2b-q4f16")
    assert decision.allowed is False
    assert "over the" in decision.reason
    assert decision.limit_mb <= 1.0


def test_budget_admit_raises_when_over(tmp_path):
    budget = MemoryBudget(hard_limit_bytes=1 * MB)
    try:
        budget.admit("qwen3-vl-2b-q4f16")
    except MemoryBudgetExceeded as exc:
        assert exc.decision.allowed is False
    else:
        raise AssertionError("expected MemoryBudgetExceeded")


def test_budget_admit_and_release_tracks_footprint():
    budget = MemoryBudget(hard_limit_bytes=10 * 1024 * MB)
    budget.admit("whisper-tiny-int8", actual_bytes=96 * MB)
    assert budget.tracked_bytes == 96 * MB
    assert "whisper-tiny-int8" in budget.loaded
    budget.release("whisper-tiny-int8")
    assert budget.tracked_bytes == 0


def test_budget_release_all_clears():
    budget = MemoryBudget(hard_limit_bytes=10 * 1024 * MB)
    budget.admit("whisper-tiny-int8", actual_bytes=1)
    budget.admit("embeddinggemma2-q4f16-text", actual_bytes=2)
    budget.release_all()
    assert budget.loaded == {}


def test_budget_model_estimates_cover_the_shipped_stack():
    for key in ("embeddinggemma2-q4f16", "qwen3-vl-2b-q4f16", "whisper-tiny-int8"):
        assert key in MODEL_MEMORY_ESTIMATES
        assert MODEL_MEMORY_ESTIMATES[key] > 0


# --- retrieval metrics ---


def _result(retrieved, relevant, documents=None):
    return RetrievalResult(
        case_id="c",
        query="q",
        retrieved=retrieved,
        scores=[0.9] * len(retrieved),
        latency_s=0.01,
        relevant=set(relevant),
        relevant_documents=set(documents or []),
        retrieved_document_ids=[f"d{i}" for i in range(len(retrieved))],
    )


def test_recall_at_k_counts_gold_found():
    result = _result(["a", "b", "c"], {"a", "b", "z"})
    assert recall_at_k(result, 3) == 2 / 3
    assert recall_at_k(result, 1) == 1 / 3


def test_recall_at_k_is_zero_without_gold():
    assert recall_at_k(_result(["a"], set()), 3) == 0.0


def test_precision_at_k():
    result = _result(["a", "b", "x"], {"a", "b"})
    assert precision_at_k(result, 2) == 1.0
    assert precision_at_k(result, 3) == 2 / 3


def test_hit_rate_is_binary():
    assert hit_rate_at_k(_result(["a", "b"], {"b"}), 2) == 1.0
    assert hit_rate_at_k(_result(["a", "b"], {"z"}), 2) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(_result(["a", "b", "c"], {"c"})) == 1 / 3
    assert reciprocal_rank(_result(["a", "b", "c"], {"z"})) == 0.0


def test_ndcg_is_one_for_perfect_ranking():
    result = _result(["a", "b"], {"a", "b"})
    assert abs(ndcg_at_k(result, 2) - 1.0) < 1e-9


def test_ndcg_penalises_bad_ordering():
    perfect = ndcg_at_k(_result(["a", "b", "x"], {"a", "b"}), 3)
    worse = ndcg_at_k(_result(["x", "a", "b"], {"a", "b"}), 3)
    assert abs(perfect - 1.0) < 1e-9
    assert worse < perfect


def test_ndcg_discountes_by_position():
    early = ndcg_at_k(_result(["a", "x", "y"], {"a"}), 3)
    late = ndcg_at_k(_result(["x", "y", "a"], {"a"}), 3)
    assert early > late


def test_score_case_covers_all_k():
    metrics = score_case(_result(["a"], {"a"}), ks=(1, 5))
    assert "recall@1" in metrics and "recall@5" in metrics
    assert "mrr" in metrics and "latency_s" in metrics


def test_aggregate_averages_and_reports_latency_percentiles():
    results = [
        _result(["a"], {"a"}),
        _result(["z"], {"a"}),
    ]
    results[0].latency_s = 0.01
    results[1].latency_s = 0.03
    summary = aggregate(results, ks=(1,))
    assert summary["cases"] == 2
    assert abs(summary["recall@1"] - 0.5) < 1e-9
    assert summary["latency_p50_s"] >= 0.01
    assert summary["latency_max_s"] == 0.03


def test_aggregate_empty_is_safe():
    assert aggregate([], ks=(1,))["cases"] == 0


def test_document_mode_uses_document_ids():
    result = RetrievalResult(
        case_id="c",
        query="q",
        retrieved=["c1", "c2"],
        scores=[1.0, 0.5],
        latency_s=0.0,
        relevant=set(),
        relevant_documents={"d9"},
        retrieved_document_ids=["d1", "d9"],
    )
    assert recall_at_k(result, 2, mode="document") == 1.0
    assert reciprocal_rank(result, mode="document") == 0.5


def test_retrieval_case_defaults():
    case = RetrievalCase(query="hola")
    assert case.relevant == set()
    assert case.media_type is None
