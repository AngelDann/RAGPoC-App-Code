"""Tests for the offline answer-quality scorer.

The "$18.50" regression is the reason the module exists: a fabricated price on
an unanswerable question must be caught without running the model.
"""

from __future__ import annotations

from ragpoc.eval_quality import (
    DEFAULT_WEIGHTS,
    HEDGE_PHRASES,
    MAX_ANSWER_CHARS,
    REFUSAL_PHRASES,
    AnswerScore,
    aggregate,
    score_answer,
)

# --- abstention correctness ---


def test_spanish_refusal_scores_high():
    report = score_answer(
        "No lo sé; esa información no está disponible en la base de conocimiento.",
        case_kind="unanswerable",
        expected_language="es",
    )
    assert report.abstained is True
    assert report.abstention_score == 1.0
    assert report.fabrication_risk == 0.0


def test_english_refusal_scores_high():
    report = score_answer(
        "I don't know; that figure is not in the knowledge base.",
        case_kind="unanswerable",
        expected_language="en",
    )
    assert report.abstained is True
    assert report.abstention_score == 1.0
    assert report.fabrication_risk == 0.0


def test_fabricated_price_on_unanswerable_case_is_caught():
    report = score_answer("$18.50", case_kind="unanswerable", expected_language="es")
    assert report.abstained is False
    assert report.abstention_score == 0.0
    assert report.fabrication_risk >= 0.8
    assert report.overall < 0.3


def test_hedged_answer_is_not_flagged_as_fabrication():
    report = score_answer(
        "No puedo determinar el precio de las acciones de ACME.",
        case_kind="unanswerable",
        expected_language="es",
    )
    assert report.abstained is True
    assert report.fabrication_risk == 0.0
    assert report.abstention_score == 1.0


def test_value_near_hedge_downweights_fabrication_risk():
    bare = score_answer("El precio de las acciones era $18.50.", case_kind="unanswerable")
    hedged = score_answer(
        "Según el informe, el precio era aproximadamente $18.50.",
        case_kind="unanswerable",
    )
    assert bare.fabrication_risk >= 0.8
    assert hedged.fabrication_risk < 0.5


def test_answerable_case_has_no_abstention_score():
    report = score_answer(
        "La similitud coseno mide el ángulo entre dos vectores.",
        case_kind="knowledge",
    )
    assert report.abstention_score is None


def test_refusal_lexicons_are_extensible():
    assert score_answer("Sin datos por ahora.", case_kind="unanswerable").abstained is False
    extended = tuple(REFUSAL_PHRASES) + ("sin datos",)
    report = score_answer(
        "Sin datos por ahora.",
        case_kind="unanswerable",
        refusal_phrases=extended,
        hedge_phrases=tuple(HEDGE_PHRASES) + ("sin datos",),
    )
    assert report.abstained is True


# --- groundedness ---

_EVIDENCE = [
    "La similitud coseno mide el angulo entre dos vectores en un espacio vectorial.",
    "La recuperacion ordena los fragmentos por similitud coseno sobre los vectores.",
]


def test_groundedness_is_high_when_answer_reuses_evidence():
    report = score_answer(
        "La similitud coseno mide el ángulo entre dos vectores.",
        evidence=_EVIDENCE,
    )
    assert report.groundedness is not None
    assert report.groundedness >= 0.9


def test_groundedness_is_low_without_evidence_terms():
    report = score_answer(
        "El protocolo de transferencia cuántica bloquea la latencia del clúster.",
        evidence=_EVIDENCE,
    )
    assert report.groundedness is not None
    assert report.groundedness <= 0.1


def test_groundedness_is_between_for_partial_overlap():
    report = score_answer(
        "La similitud coseno mide la latencia del clúster.",
        evidence=_EVIDENCE,
    )
    assert report.groundedness is not None
    assert 0.2 < report.groundedness < 0.8


def test_groundedness_is_none_without_evidence():
    assert score_answer("La similitud coseno mide el ángulo.").groundedness is None
    assert score_answer("La similitud coseno mide el ángulo.", evidence=[]).groundedness is None


# --- citation discipline ---


def test_bracket_and_source_citations_are_detected():
    report = score_answer("El concepto aparece en [1] y se detalla en (fuente 2).")
    assert report.cited is True
    assert report.citation_score == 1.0


def test_chunk_id_is_detected_as_citation():
    report = score_answer("Los vectores indexados viven en chunk-7f3a desde la última ingesta.")
    assert report.cited is True
    assert report.citation_score == 1.0


def test_quoted_filename_is_detected_as_citation():
    report = score_answer('Los datos provienen del archivo "informe_anual.pdf".')
    assert report.cited is True
    assert report.citation_score == 1.0


def test_weak_source_reference_gets_partial_credit():
    report = score_answer("Según el documento, la versión anterior quedó obsoleta.")
    assert report.cited is True
    assert 0.0 < report.citation_score < 1.0


def test_uncited_answer_scores_zero():
    report = score_answer("La similitud coseno mide el ángulo entre dos vectores.")
    assert report.cited is False
    assert report.citation_score == 0.0


# --- shape ---


def test_empty_answer_is_flagged_and_scores_zero():
    report = score_answer("   ")
    assert report.empty is True
    assert report.shape_score == 0.0


def test_too_long_answer_is_penalised():
    report = score_answer("palabra " * 40, max_answer_chars=100)
    assert report.too_long is True
    assert 0.0 < report.shape_score < 1.0


def test_default_max_length_matches_bench():
    assert MAX_ANSWER_CHARS == 1200
    report = score_answer("x" * 1201)
    assert report.too_long is True


def test_leaked_tool_call_payload_is_flagged():
    report = score_answer(
        'El precio es {"name": "search_knowledge_base",'
        ' "arguments": {"query": "precio ACME"}}.'
    )
    assert report.leaked_reasoning is True
    assert report.shape_score <= 0.25


def test_unterminated_code_fence_is_flagged():
    report = score_answer('Resultado:\n```json\n{"a": 1}')
    assert report.leaked_reasoning is True


def test_clean_short_answer_scores_one():
    report = score_answer("Un vector database almacena embeddings.")
    assert report.shape_score == 1.0
    assert (report.empty, report.too_long, report.leaked_reasoning) == (False, False, False)


# --- language fidelity ---


def test_spanish_answer_detected_for_spanish_question():
    report = score_answer(
        "La similitud coseno mide el ángulo entre dos vectores.",
        expected_language="es",
    )
    assert report.detected_language == "es"
    assert report.language_score == 1.0


def test_english_answer_detected_for_english_question():
    report = score_answer(
        "Vector similarity measures the angle between two vectors.",
        expected_language="en",
    )
    assert report.detected_language == "en"
    assert report.language_score == 1.0


def test_spanish_answer_to_english_question_scores_zero():
    report = score_answer(
        "No lo sé, no tengo esa información.",
        expected_language="en",
    )
    assert report.detected_language == "es"
    assert report.language_score == 0.0


def test_languageless_answer_gets_benefit_of_the_doubt():
    report = score_answer("$18.50", expected_language="es")
    assert report.detected_language is None
    assert report.language_score == 0.5


def test_language_score_is_none_without_expectation():
    assert score_answer("La similitud coseno mide el ángulo.").language_score is None


# --- keyword recall (legacy metric) ---


def test_keyword_recall_partial_match():
    report = score_answer(
        "La similitud coseno se usa en recuperación.",
        expected_terms=["coseno", "vector"],
    )
    assert report.keyword_recall == 0.5
    assert report.matched_terms == ["coseno"]


def test_keyword_recall_without_terms_matches_bench_convention():
    assert score_answer("ok").keyword_recall == 1.0


# --- overall and aggregation ---


def test_report_is_answer_score_and_exposes_every_dimension():
    report = score_answer("$18.50", case_kind="unanswerable", expected_language="es")
    assert isinstance(report, AnswerScore)
    dimensions = report.dimension_scores()
    assert set(dimensions) == {
        "abstention",
        "groundedness",
        "citation",
        "shape",
        "language",
        "keyword_recall",
        "fabrication_risk",
    }


def test_overall_is_weighted_mean_over_active_dimensions():
    report = score_answer(
        "La similitud coseno mide el ángulo entre dos vectores.",
        expected_terms=["coseno"],
        expected_language="es",
    )
    active = {"keyword_recall": 1.0, "shape": 1.0, "language": 1.0}
    total = sum(DEFAULT_WEIGHTS[name] for name in active)
    expected = sum(DEFAULT_WEIGHTS[name] * value for name, value in active.items()) / total
    assert abs(report.overall - expected) < 1e-9


def test_aggregate_averages_dimensions_and_counts_flags():
    good = score_answer(
        "La similitud coseno mide el ángulo entre dos vectores.",
        expected_terms=["coseno"],
        expected_language="es",
    )
    empty = score_answer("", expected_language="es")
    summary = aggregate([good, empty])
    assert summary["cases"] == 2
    assert summary["empty_answers"] == 1
    assert abs(summary["shape_score"] - 0.5) < 1e-9
    assert 0.0 < summary["overall"] < 1.0
    assert summary["language_cases"] == 2


def test_aggregate_custom_weights_isolate_a_dimension():
    filled = score_answer("hola mundo")
    empty = score_answer("")
    summary = aggregate([filled, empty], weights={"shape": 1.0})
    assert abs(summary["overall"] - 0.5) < 1e-9
    assert summary["weights"] == {"shape": 1.0}


def test_aggregate_counts_fabrication_incidents():
    summary = aggregate(
        [
            score_answer("$18.50", case_kind="unanswerable"),
            score_answer("No lo sé.", case_kind="unanswerable"),
        ]
    )
    assert summary["abstention_cases"] == 2
    assert summary["fabrication_incidents"] == 1
    assert summary["abstention"] == 0.5


def test_aggregate_empty_is_safe():
    assert aggregate([])["cases"] == 0
