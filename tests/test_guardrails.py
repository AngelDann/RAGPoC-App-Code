"""Guards against the measured hallucination defect.

The motivating case: asked for a fact absent from the knowledge base, the model
answered "$18.50". These tests pin the behaviour that catches it, and equally
important, pin the cases that must NOT be caught.
"""

from ragpoc.guardrails import (
    ABSTENTION_NUDGE_EN,
    ABSTENTION_NUDGE_ES,
    assess,
    find_values,
    is_hedged,
    needs_guard,
    wrap_unverified,
)

PRICE = "$18.50"
DATE = "15 de mayo de 2023."
IDENTIFIER = "1234567890"
QUANTITY = "5000 unidades"
PERCENT = "el 37% del total"
REVENUE = "$2.1 billion"
HEDGED = "No puedo determinar el precio; ese dato no esta en la base de conocimiento."
CITED = "Segun el documento [1], el importe fue de 1200 euros."
PLAIN = "La similitud coseno compara dos vectores."


# --- value detection ---


def test_detects_prices():
    assert find_values(PRICE)


def test_detects_dates():
    assert find_values(DATE)


def test_detects_long_identifiers():
    assert find_values(IDENTIFIER)


def test_detects_percentages():
    assert find_values(PERCENT)


def test_detects_money_in_words():
    assert find_values(REVENUE)


def test_plain_prose_has_no_values():
    assert find_values(PLAIN) == []


# --- hedging ---


def test_hedge_detected_in_spanish():
    assert is_hedged(HEDGED) is True


def test_hedge_detected_in_english():
    assert is_hedged("I cannot determine the price from the available sources.") is True


def test_unhedged_assertion_is_not_hedged():
    assert is_hedged(PRICE) is False


# --- assess ---


def test_fabrication_is_high_risk():
    verdict = assess(PRICE)
    assert verdict.needs_guard is True
    assert verdict.risk >= 0.5
    assert verdict.values_found


def test_hedged_value_is_not_flagged():
    verdict = assess(HEDGED)
    assert verdict.needs_guard is False
    assert verdict.hedged is True


def test_plain_prose_is_not_flagged():
    verdict = assess(PLAIN)
    assert verdict.needs_guard is False
    assert verdict.risk == 0.0


# --- needs_guard: the support conditions ---


def test_fabrication_without_support_is_guarded():
    assert needs_guard(PRICE) is True


def test_evidence_supports_the_answer():
    assert needs_guard(PRICE, had_evidence=True) is False


def test_tool_use_supports_the_answer():
    assert needs_guard(PRICE, had_tool_calls=True) is False


def test_citation_supports_the_answer():
    assert needs_guard(CITED, cited=True) is False


def test_plain_prose_needs_no_guard():
    assert needs_guard(PLAIN) is False


# --- the five measured fabrication cases ---

def test_all_five_measured_cases_are_caught():
    for answer in (PRICE, DATE, IDENTIFIER, QUANTITY, REVENUE):
        assert needs_guard(answer) is True, f"no detectado: {answer!r}"


def test_bare_integer_answer_is_caught():
    """"5000" as the whole answer escaped the value patterns."""
    assert needs_guard("5000") is True
    assert assess("5000").reason == "answer is essentially a bare value"


def test_bare_price_answer_is_caught():
    assert needs_guard("$12.45") is True


def test_bare_value_with_punctuation_is_caught():
    assert needs_guard("15 de abril de 2023.") is True


def test_full_sentence_with_a_number_is_not_a_bare_value():
    verdict = assess("El informe tiene 12 paginas y describe el flujo de indexacion.")
    assert verdict.needs_guard is False or verdict.hedged is False


def test_bare_value_ignored_when_supported():
    assert needs_guard("5000", had_tool_calls=True) is False
    assert needs_guard("$12.45", had_evidence=True) is False


# --- wrap_unverified ---


def test_wrap_marks_the_answer_in_spanish():
    wrapped = wrap_unverified(PRICE, language="es")
    assert wrapped.startswith("\u26a0")
    assert "Sin verificar" in wrapped
    assert PRICE in wrapped


def test_wrap_marks_the_answer_in_english():
    wrapped = wrap_unverified(PRICE, language="en")
    assert "Unverified" in wrapped
    assert PRICE in wrapped


def test_wrap_preserves_the_original_text():
    assert "5000 unidades" in wrap_unverified("5000 unidades", language="es")


# --- nudges ---


def test_nudges_exist_in_both_languages():
    assert "NO inventes" in ABSTENTION_NUDGE_ES
    assert "Do NOT invent" in ABSTENTION_NUDGE_EN


def test_nudges_forbid_the_specific_classes_of_value():
    for token in ("cifras", "fechas", "cantidades", "nombres"):
        assert token in ABSTENTION_NUDGE_ES
    for token in ("figures", "dates", "quantities", "names"):
        assert token in ABSTENTION_NUDGE_EN
