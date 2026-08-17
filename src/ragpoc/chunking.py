from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class ApproximateTokenCounter:
    """Conservative tokenizer approximation for chunk sizing; persisted as metadata."""

    name = "regex-approx-v1"

    def count(self, text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE))


@dataclass(frozen=True)
class TextChunk:
    ordinal: int
    content: str
    token_count: int


def chunk_text(
    text: str, counter: TokenCounter, target_tokens: int = 1000, overlap_tokens: int = 150
) -> list[TextChunk]:
    text = text.strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if counter.count(paragraph) <= target_tokens:
            units.append(paragraph)
        else:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip())
    result: list[TextChunk] = []
    current: list[str] = []
    for unit in units:
        proposed = "\n\n".join([*current, unit])
        if current and counter.count(proposed) > target_tokens:
            content = "\n\n".join(current)
            result.append(TextChunk(len(result), content, counter.count(content)))
            overlap: list[str] = []
            for old in reversed(current):
                overlap.insert(0, old)
                if counter.count("\n\n".join(overlap)) >= overlap_tokens:
                    break
            current = overlap
        current.append(unit)
    if current:
        content = "\n\n".join(current)
        result.append(TextChunk(len(result), content, counter.count(content)))
    return result
