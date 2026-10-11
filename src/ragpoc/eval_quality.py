"""Deterministic answer-quality scoring for the local agent benchmark.

Term recall alone rewards keyword stuffing and cannot see hallucination: the
agent once answered "$18.50" to a question whose answer was not in the knowledge
base at all. Everything here is an offline heuristic (no model, no network, no
downloads) so it can run inside the benchmark as a cheap regression guard over
five dimensions: abstention correctness, groundedness against the retrieved
evidence, citation discipline, answer shape, and language fidelity. Keyword
recall is kept so benchmark numbers stay comparable with older runs.

The per-dimension values are the point; ``overall`` is only a weighted summary
for sorting runs, and hallucinating a specific value on an unanswerable case
multiplies it down, because that is the failure mode that motivated this module.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MAX_ANSWER_CHARS = 1200

# A fabricated specific value on an unanswerable case scales ``overall`` by
# (1 - FABRICATION_PENALTY * fabrication_risk); 1.0 makes a full hallucination a
# zero for that case.
FABRICATION_PENALTY = 1.0

# Relative weights over the active dimensions of a case; absent dimensions
# (no evidence, no expected language) drop out and the rest renormalise.
DEFAULT_WEIGHTS: dict[str, float] = {
    "abstention": 0.30,
    "groundedness": 0.25,
    "keyword_recall": 0.20,
    "citation": 0.10,
    "shape": 0.10,
    "language": 0.05,
}

ABSTENTION_CASE_KINDS = ("unanswerable", "refuse")

LEAKED_SHAPE_PENALTY = 0.25

# How close a hedge or citation must sit to a concrete value to count as
# covering it ("approximately $18.50" is a claim; "$18.50" alone is a fabrication).
COVERAGE_WINDOW_CHARS = 80

# Phrases are matched against lowercase/accent-folded text, so write them that way.
REFUSAL_PHRASES_ES = (
    "no lo se",
    "no lo sabria",
    "no esta disponible",
    "no esta en la base",
    "no esta en la base de conocimiento",
    "no puedo determinar",
    "no puedo saber",
    "no puedo responder",
    "no puedo confirmar",
    "no puedo proporcionar",
    "no puedo asegurar",
    "no tengo informacion",
    "no tengo datos",
    "no tengo acceso",
    "no dispongo de informacion",
    "no hay informacion",
    "no hay datos",
    "no consta",
    "no aparece",
    "no se encuentra",
    "no esta documentado",
    "no es posible determinar",
    "no existe informacion",
    "desconozco",
    "no estoy seguro",
)

REFUSAL_PHRASES_EN = (
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "cannot determine",
    "can't determine",
    "cannot be determined",
    "unable to determine",
    "unable to find",
    "not available",
    "is not available",
    "isn't available",
    "no information",
    "there is no information",
    "no data available",
    "not in the knowledge base",
    "not in the database",
    "not in my knowledge",
    "not mentioned",
    "i don't have",
    "i do not have",
    "i cannot provide",
    "i can't provide",
    "couldn't find",
    "could not find",
    "the knowledge base does not contain",
)

# Labels emitted by ``ragpoc.guardrails`` when an answer is shown but flagged.
# These are abstentions in effect: the user is told the value is not supported.
GUARD_LABEL_PHRASES = (
    "sin verificar",
    "unverified",
    "no hay evidencia en la base de conocimiento",
    "there is no evidence in the knowledge base",
)

REFUSAL_PHRASES = REFUSAL_PHRASES_ES + REFUSAL_PHRASES_EN + GUARD_LABEL_PHRASES

# Epistemic hedges that license a concrete value without asserting it as fact.
HEDGE_PHRASES = REFUSAL_PHRASES + (
    "aproximadamente",
    "aproximado",
    "aprox",
    "cerca de",
    "alrededor de",
    "rondaba",
    "estimado",
    "estimada",
    "podria ser",
    "posiblemente",
    "mas o menos",
    "sin verificar",
    "approximately",
    "approx",
    "around",
    "roughly",
    "about",
    "estimated",
    "or so",
    "up to",
    "more or less",
    "unverified",
)

# (category, pattern, base weight). Matched on folded text; overlapping matches
# keep only the heaviest so "3 de marzo de 2019" is not counted twice.
VALUE_PATTERNS: tuple[tuple[str, re.Pattern[str], float], ...] = (
    (
        "price",
        re.compile(
            r"[$€£]\s?\d[\d.,]*"
            r"|\b\d+[.,]\d{2}\s?(?:usd|eur|euros?|dolares?|pesos?|libras?|gbp|mxn|cop|clp|ars)\b"
        ),
        1.0,
    ),
    ("percentage", re.compile(r"\b\d+(?:[.,]\d+)?\s?%"), 0.7),
    (
        "date",
        re.compile(
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
            r"|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"
            r"|\b\d{1,2}\s+de\s+[a-z]+\s+de\s+\d{4}\b"
            r"|\b(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre"
            r"|octubre|noviembre|diciembre|january|february|march|april|may|june|july|august"
            r"|september|october|november|december)\s+\d{1,2}(?:\s*(?:,|de)\s*|\s+)\d{4}\b"
            r"|\b(?:19|20)\d{2}\b"
        ),
        0.7,
    ),
    (
        "quantity",
        re.compile(
            r"\b\d+(?:[.,]\d+)?\s?(?:unidades?|kg|km|cm|mg|ml|gramos?|kilogramos?|metros?"
            r"|litros?|segundos?|minutos?|horas?|dias?|semanas?|meses?|anos?|personas?"
            r"|veces?|usuarios?|documentos?|fragmentos?|paginas?|capitulos?|tokens?"
            r"|items?|units?|seconds?|minutes?|hours?|days?|weeks?|months?|years?"
            r"|pages?|chapters?)\b"
        ),
        0.5,
    ),
    (
        "identifier",
        re.compile(
            r"\b[a-z]{2,}[-_]\d[\w-]*\b"
            r"|#\w{3,}\b"
            r"|\b(?:id|ref|codigo|code|sku|lote)\s*[:=#/-]\s*[\w-]{2,}\b"
        ),
        0.5,
    ),
)

_CITATION_STRONG = re.compile(
    r"\[\d{1,2}\]"
    r"|\(\s*(?:fuente|source|ref|referencia|reference)\s*\d{1,2}\s*\)"
    r"|\b(?:fuente|source|ref(?:erencia)?|reference)\s*\d{1,2}\b"
    r"|\bchunks?[_ -][\w-]{2,}\b"
    r"|\bchunks?\s*[:=#]?\s*\d+\b"
    r"|\b(?:documento|document|archivo|file)\s*\"[^\"]+\""
    r"|\"[\w .-]+\.(?:pdf|txt|md|docx?|xlsx?|csv|pptx?|html?|json)\""
)

_CITATION_WEAK = re.compile(
    r"segun (?:la |el |los |las )?(?:fuente|documento|fragmento|chunk|base|informe|reporte)"
    r"|according to (?:the )?(?:source|document|chunk|report|knowledge base)"
    r"|(?:el|la|los|las) (?:documentos?|informes?|fuentes?|fragmentos?) "
    r"(?:indica|senala|cita|muestra|establece|describe)"
    r"|(?:the )?(?:document|report|source|study) "
    r"(?:states|shows|indicates|says|mentions|describes)"
)

_LEAK_PATTERNS = (
    re.compile(r"\"(?:tool_calls?|tool_call_id|recipient_name|function|arguments|name)\"\s*:"),
    re.compile(r"\bto=functions\.|<function_calls>|<invoke\b|antml:"),
    re.compile(r"<(?:think|thinking|reasoning|analysis)>"),
)

_WORD_RE = re.compile(r"[a-z0-9]+")

ES_STOPWORDS = frozenset(
    """
    a al algo algunas algunos ante antes como con contra cual cuando de del desde donde dos
    e el ella ellas ello ellos en entre esa esas ese eso esos esta estaba estas esto estos fue
    ha hay la las le les lo los mas me mi muy ni no nos o otras otro para pero poco por porque
    que quien se ser si sin sobre son su sus tambien tan tenia tiene todo todos tras tu un una
    uno unos y ya yo esta estan este estos
    """.split()
)

EN_STOPWORDS = frozenset(
    """
    a about after again all am an and any are as at be because been before being between both
    but by can cannot could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just me
    more most my myself no nor not of off on once only or other our ours ourselves out over own
    same she should so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was we were what when where which while
    who whom why will with you your yours yourself
    """.split()
)

_STOPWORDS = ES_STOPWORDS | EN_STOPWORDS

_ACCENT_TABLE = str.maketrans("áéíóúüñ", "aeiouun")
_QUOTE_TABLE = str.maketrans("’‘´`", "''''")


def _fold(text: str) -> str:
    """Lowercase and drop diacritics 1:1 so match offsets stay valid on the source."""
    return text.lower().translate(_ACCENT_TABLE).translate(_QUOTE_TABLE)


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(_fold(text)) if t not in _STOPWORDS]


def detect_refusal(answer: str, phrases: Sequence[str] = REFUSAL_PHRASES) -> bool:
    """True when the answer abstains instead of asserting a fact."""
    folded = _fold(answer)
    return any(_fold(phrase) in folded for phrase in phrases)


def detect_language(text: str) -> str | None:
    """Stopword heuristic for ES vs EN; ``None`` when no signal or a dead tie."""
    words = _WORD_RE.findall(_fold(text))
    es = sum(1 for w in words if w in ES_STOPWORDS)
    en = sum(1 for w in words if w in EN_STOPWORDS)
    if es > en:
        return "es"
    if en > es:
        return "en"
    return None


def _citation_detection(folded: str) -> tuple[bool, float]:
    if _CITATION_STRONG.search(folded):
        return True, 1.0
    if _CITATION_WEAK.search(folded):
        return True, 0.5
    return False, 0.0


def _covered_spans(folded: str, hedge_phrases: Sequence[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for phrase in hedge_phrases:
        needle = _fold(phrase)
        start = folded.find(needle)
        while start != -1:
            spans.append((start, start + len(needle)))
            start = folded.find(needle, start + 1)
    for pattern in (_CITATION_STRONG, _CITATION_WEAK):
        spans.extend((m.start(), m.end()) for m in pattern.finditer(folded))
    return spans


def _fabrication_risk(folded: str, hedge_phrases: Sequence[str]) -> float:
    """0..1: concrete specific values asserted without hedge or citation nearby."""
    candidates: list[tuple[int, int, float]] = []
    for _category, pattern, weight in VALUE_PATTERNS:
        candidates.extend((m.start(), m.end(), weight) for m in pattern.finditer(folded))
    accepted: list[tuple[int, int, float]] = []
    for start, end, weight in sorted(candidates, key=lambda span: -span[2]):
        if any(start < a_end and end > a_start for a_start, a_end, _ in accepted):
            continue
        accepted.append((start, end, weight))
    if not accepted:
        return 0.0
    cover = _covered_spans(folded, hedge_phrases)
    risk = 0.0
    for start, end, weight in accepted:
        near = any(
            start - COVERAGE_WINDOW_CHARS <= c_end and end + COVERAGE_WINDOW_CHARS >= c_start
            for c_start, c_end in cover
        )
        risk += weight * (0.2 if near else 1.0)
    return min(1.0, risk)


def _groundedness(answer: str, evidence: Sequence[str] | None) -> float | None:
    """IDF-weighted share of the answer's informative tokens present in the evidence.

    Tokens absent from the evidence keep their (maximal) IDF so unsupported
    claims cost more than boilerplate shared with every chunk.
    """
    if not evidence:
        return None
    docs = [_tokens(chunk) for chunk in evidence]
    n_docs = len(docs)
    df: dict[str, int] = {}
    for doc in docs:
        for token in set(doc):
            df[token] = df.get(token, 0) + 1

    def idf(token: str) -> float:
        return math.log((n_docs + 1) / (df.get(token, 0) + 1)) + 1

    answer_tokens = set(_tokens(answer))
    if not answer_tokens:
        return 0.0
    total = sum(idf(token) for token in answer_tokens)
    supported = sum(idf(token) for token in answer_tokens if token in df)
    return supported / total


@dataclass
class QualityReport:
    """Per-answer report: every dimension is exposed, ``overall`` only summarises.

    ``None`` marks a dimension that does not apply (no evidence, no expected
    language, or a case where abstention is not expected) instead of guessing.
    """

    case_id: str
    case_kind: str
    answer_chars: int
    abstention_score: float | None
    abstained: bool
    fabrication_risk: float
    groundedness: float | None
    citation_score: float
    cited: bool
    require_citation: bool
    shape_score: float
    empty: bool
    too_long: bool
    leaked_reasoning: bool
    language_score: float | None
    detected_language: str | None
    keyword_recall: float
    matched_terms: list[str]
    overall: float

    def dimension_scores(self) -> dict[str, float | None]:
        return {
            "abstention": self.abstention_score,
            "groundedness": self.groundedness,
            "citation": self.citation_score,
            "shape": self.shape_score,
            "language": self.language_score,
            "keyword_recall": self.keyword_recall,
            "fabrication_risk": self.fabrication_risk,
        }

    def overall_for(self, weights: Mapping[str, float] | None = None) -> float:
        values: dict[str, float | None] = {
            "abstention": self.abstention_score,
            "groundedness": self.groundedness,
            "citation": self.citation_score if self.require_citation else None,
            "shape": self.shape_score,
            "language": self.language_score,
            "keyword_recall": self.keyword_recall,
        }
        return _weighted_overall(
            values,
            fabrication_risk=self.fabrication_risk,
            penalise_fabrication=self.case_kind in ABSTENTION_CASE_KINDS,
            weights=weights,
        )


AnswerScore = QualityReport


def _weighted_overall(
    values: Mapping[str, float | None],
    *,
    fabrication_risk: float,
    penalise_fabrication: bool,
    weights: Mapping[str, float] | None,
) -> float:
    active = DEFAULT_WEIGHTS if weights is None else weights
    numerator = 0.0
    denominator = 0.0
    for name, value in values.items():
        weight = active.get(name, 0.0)
        if value is None or weight <= 0:
            continue
        numerator += weight * value
        denominator += weight
    base = numerator / denominator if denominator else 0.0
    if penalise_fabrication:
        base *= max(0.0, 1.0 - FABRICATION_PENALTY * fabrication_risk)
    return base


def score_answer(
    answer: str,
    *,
    expected_terms: Sequence[str] = (),
    case_kind: str = "knowledge",
    expected_language: str | None = None,
    evidence: Sequence[str] | None = None,
    require_citation: bool = False,
    case_id: str = "",
    max_answer_chars: int = MAX_ANSWER_CHARS,
    refusal_phrases: Sequence[str] = REFUSAL_PHRASES,
    hedge_phrases: Sequence[str] = HEDGE_PHRASES,
    weights: Mapping[str, float] | None = None,
) -> AnswerScore:
    """Score one answer across every quality dimension.

    ``expected_terms`` uses the legacy substring matching from the benchmark so
    keyword numbers stay comparable with older runs.
    """
    folded = _fold(answer)
    answer_chars = len(answer.strip())
    abstained = detect_refusal(answer, refusal_phrases)
    risk = _fabrication_risk(folded, hedge_phrases)
    if case_kind in ABSTENTION_CASE_KINDS:
        abstention_score = max(0.0, 1.0 - risk) if abstained else 0.0
    else:
        abstention_score = None

    cited, citation_score = _citation_detection(folded)

    empty = answer_chars == 0
    too_long = answer_chars > max_answer_chars
    leaked_patterns = any(pattern.search(folded) for pattern in _LEAK_PATTERNS)
    leaked = leaked_patterns or answer.count("```") % 2 == 1
    if empty:
        shape_score = 0.0
    else:
        shape_score = 1.0
        if too_long:
            shape_score *= min(1.0, max_answer_chars / answer_chars)
        if leaked:
            shape_score = min(shape_score, LEAKED_SHAPE_PENALTY)

    detected_language = detect_language(answer)
    if expected_language is None:
        language_score = None
    elif detected_language == expected_language:
        language_score = 1.0
    elif detected_language is None:
        language_score = 0.5
    else:
        language_score = 0.0

    lowered = answer.lower()
    matched_terms = [term for term in expected_terms if term.lower() in lowered]
    keyword_recall = len(matched_terms) / len(expected_terms) if expected_terms else 1.0

    report = AnswerScore(
        case_id=case_id,
        case_kind=case_kind,
        answer_chars=answer_chars,
        abstention_score=abstention_score,
        abstained=abstained,
        fabrication_risk=risk,
        groundedness=_groundedness(answer, evidence),
        citation_score=citation_score,
        cited=cited,
        require_citation=require_citation,
        shape_score=shape_score,
        empty=empty,
        too_long=too_long,
        leaked_reasoning=leaked,
        language_score=language_score,
        detected_language=detected_language,
        keyword_recall=keyword_recall,
        matched_terms=matched_terms,
        overall=0.0,
    )
    report.overall = report.overall_for(weights)
    return report


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(
    scores: Sequence[AnswerScore],
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Mean of every dimension plus hallucination/shape incident counts.

    Dimensions that do not apply to a case are excluded from their mean, so a
    corpus without evidence does not drag groundedness to zero.
    """
    active_weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if not scores:
        return {"cases": 0, "weights": active_weights}

    def mean_of(values: Sequence[float | None]) -> float | None:
        present = [v for v in values if v is not None]
        return _mean(present) if present else None

    grounded = mean_of([s.groundedness for s in scores])
    languages = mean_of([s.language_score for s in scores])
    return {
        "cases": len(scores),
        "overall": _mean([s.overall_for(weights) for s in scores]),
        "abstention": mean_of([s.abstention_score for s in scores]),
        "abstention_cases": sum(1 for s in scores if s.abstention_score is not None),
        "fabrication_risk": _mean([s.fabrication_risk for s in scores]),
        "fabrication_incidents": sum(1 for s in scores if s.fabrication_risk >= 0.5),
        "groundedness": grounded,
        "evidence_cases": sum(1 for s in scores if s.groundedness is not None),
        "citation_score": _mean([s.citation_score for s in scores]),
        "cited_rate": _mean([1.0 if s.cited else 0.0 for s in scores]),
        "shape_score": _mean([s.shape_score for s in scores]),
        "empty_answers": sum(1 for s in scores if s.empty),
        "too_long_answers": sum(1 for s in scores if s.too_long),
        "leaked_answers": sum(1 for s in scores if s.leaked_reasoning),
        "language_score": languages,
        "language_cases": sum(1 for s in scores if s.language_score is not None),
        "keyword_recall": _mean([s.keyword_recall for s in scores]),
        "weights": active_weights,
    }
