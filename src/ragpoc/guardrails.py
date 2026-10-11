"""Structural guard against fabricated specifics in agent answers.

The measured defect is not subtle: asked five questions whose answer is not in
the knowledge base, a 2B local model returned a concrete fabricated value every
single time -- a price, a date, an ID, a quantity and a revenue. Adding an
explicit "do not invent values" directive to the system prompt moved the
abstention rate from 0.0 to 0.0. Prompting alone does not fix this at this
model size, so the answer is checked after the fact.

The guard is deliberately conservative and purely local: it fires only when an
answer asserts a specific value AND nothing supports it -- no evidence was
attached to the turn and no tool was called. A model that retrieved documents
or cited a source is left alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Concrete values whose invention is actively harmful: prices, dates, quantities,
# identifiers, percentages. A bare number in prose is the tell.
_VALUE_PATTERNS = (
    re.compile(r"[$\u20ac\u00a3]\s?\d"),
    re.compile(r"\b\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?\b"),
    re.compile(r"\b\d+[.,]\d+\s?(?:%|euros?|dolares?|usd|eur)\b", re.IGNORECASE),
    re.compile(r"\b\d+\s?%"),
    re.compile(
        r"\b\d[\d.,]*\s?(?:unidades?|piezas?|personas?|usuarios?|clientes?|a[ñn]os?|meses?|d[íi]as?|"
        r"veces?|horas?|minutos?|segundos?|metros?|kil[oó]metros?|kilos?|gramos?|litros?|"
        r"euros?|d[oó]lares?|usd|eur|kg|km|mb|gb|tb)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\b", re.IGNORECASE),
    re.compile(
        r"\b(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)"
        r"\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{6,}\b"),
)

# An answer whose entire content is a value. Reported separately because a bare
# number carries no evidence of where it came from.
_BARE_VALUE = re.compile(
    r"^[\s\"'`(\[]*[$\u20ac\u00a3]?\s?\d[\d.,\s]*\s?(?:%|euros?|d[oó]lares?|usd|eur|unidades?|"
    r"personas?|a[ñn]os?|meses?|d[íi]as?|billones?|millones?|mil)?[\s.,;:!?)\]\"'`]*$",
    re.IGNORECASE,
)

# Hedges that mean the model is not asserting the value as fact.
_HEDGE_PATTERNS = (
    re.compile(r"no\s+(?:lo\s+)?s[eé]", re.IGNORECASE),
    re.compile(r"no\s+(?:puedo|podr(?:\u00eda|i\u00e1))\s+(?:determinar|confirmar|asegurar|verificar)", re.IGNORECASE),
    re.compile(r"no\s+(?:est[aá]|aparece|consta|figura)", re.IGNORECASE),
    re.compile(r"no\s+(?:hay|existe)\s+(?:informaci[o\u00f3]n|dato|registro)", re.IGNORECASE),
    re.compile(r"no\s+(?:est[aá])\s+en\s+la\s+base", re.IGNORECASE),
    re.compile(r"no\s+disponible", re.IGNORECASE),
    re.compile(r"sin\s+(?:datos?|informaci[o\u00f3]n)", re.IGNORECASE),
    re.compile(r"i\s+(?:do\s+not|don't|cannot|can't)", re.IGNORECASE),
    re.compile(r"not\s+(?:available|in\s+the\s+knowledge\s+base|mentioned|found)", re.IGNORECASE),
    re.compile(r"cannot\s+(?:determine|confirm|verify|provide)", re.IGNORECASE),
    re.compile(r"unable\s+to\s+(?:determine|confirm|verify)", re.IGNORECASE),
    re.compile(r"unverified|no\s+verificado", re.IGNORECASE),
)

# Nudge appended for the retry when the guard fires.
ABSTENTION_NUDGE_ES = (
    "ATENCI\u00d3N: no tienes evidencia de ning\u00fan documento para esta pregunta. "
    "NO inventes cifras, fechas, cantidades ni nombres propios. "
    "Si el dato no est\u00e1 en la base de conocimiento, di exactamente eso y nada m\u00e1s."
)
ABSTENTION_NUDGE_EN = (
    "ATTENTION: you have no documentary evidence for this question. "
    "Do NOT invent figures, dates, quantities or names. "
    "If the value is not in the knowledge base, say exactly that and nothing more."
)

UNVERIFIED_PREFIX_ES = "\u26a0\ufe0f **Sin verificar** \u2014 no hay evidencia en la base de conocimiento para este dato:\n\n"
UNVERIFIED_PREFIX_EN = "\u26a0\ufe0f **Unverified** \u2014 there is no evidence in the knowledge base for this:\n\n"


@dataclass
class GuardVerdict:
    """Whether an answer may be shown as-is, and what to do about it."""

    needs_guard: bool
    risk: float
    values_found: list[str]
    hedged: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "needs_guard": self.needs_guard,
            "risk": round(self.risk, 3),
            "values_found": self.values_found,
            "hedged": self.hedged,
            "reason": self.reason,
        }


def _fold(text: str) -> str:
    return text.lower()


def find_values(text: str) -> list[str]:
    """Concrete values asserted in the text, as matched spans."""
    found: list[str] = []
    for pattern in _VALUE_PATTERNS:
        found.extend(match.group(0) for match in pattern.finditer(text))
    return found


def is_hedged(text: str) -> bool:
    folded = _fold(text)
    return any(pattern.search(folded) for pattern in _HEDGE_PATTERNS)


def assess(text: str) -> GuardVerdict:
    """Risk of fabrication, ignoring whether anything supports the claim."""
    stripped = text.strip()
    if not stripped:
        return GuardVerdict(False, 0.0, [], False, "empty answer")

    # An answer that is essentially just a value is the clearest possible tell:
    # "5000", "$12.45", "15 de mayo de 2023." Nothing about that is a report.
    if _BARE_VALUE.fullmatch(stripped):
        return GuardVerdict(True, 0.95, [stripped], False, "answer is essentially a bare value")

    values = find_values(stripped)
    hedged = is_hedged(stripped)
    if not values:
        return GuardVerdict(False, 0.0, [], hedged, "no specific value asserted")
    if hedged:
        return GuardVerdict(False, 0.0, values, True, "value is hedged, not asserted")
    # Values dominating the whole answer are an assertion, not a report.
    coverage = sum(len(v) for v in values) / max(len(stripped), 1)
    risk = min(1.0, 0.5 + 0.1 * len(values) + (0.3 if coverage > 0.5 else 0.0))
    return GuardVerdict(True, risk, values, False, f"{len(values)} unverified value(s) asserted")


def needs_guard(
    text: str,
    *,
    had_evidence: bool = False,
    had_tool_calls: bool = False,
    cited: bool = False,
) -> bool:
    """True when an unsupported specific value is asserted as fact.

    Evidence, tool use or an explicit citation all count as support: the model
    may well be reporting something real, and second-guessing it would break
    the cases that work today.
    """
    if had_evidence or had_tool_calls or cited:
        return False
    return assess(text).needs_guard


def wrap_unverified(text: str, *, language: str = "es") -> str:
    """Label an unsupported answer instead of silently trusting it."""
    prefix = UNVERIFIED_PREFIX_EN if _fold(language).startswith("en") else UNVERIFIED_PREFIX_ES
    body = text.strip()
    return f"{prefix}{body}"


__all__ = [
    "ABSTENTION_NUDGE_EN",
    "ABSTENTION_NUDGE_ES",
    "GuardVerdict",
    "assess",
    "find_values",
    "is_hedged",
    "needs_guard",
    "wrap_unverified",
]
