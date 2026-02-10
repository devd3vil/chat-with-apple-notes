"""Context packing for reranked candidates."""

from __future__ import annotations

import re
from typing import Any

from src.constraint_extractor import Constraints
from src.reranker import Candidate, ScoredCandidate

_TIME_REGEX = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s?(?:am|pm)\b", re.IGNORECASE)

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "with",
    "at",
    "by",
    "it",
    "this",
    "that",
    "these",
    "those",
    "as",
}


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split())) if text else 0


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


def _keywords(query: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_-]+", query.lower())
    return [t for t in tokens if t and t not in _STOPWORDS]


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    lines = text.splitlines()
    sentences: list[str] = []
    for line in lines:
        parts = re.split(r"(?<=[.!?])\s+", line.strip())
        for part in parts:
            if part:
                sentences.append(part)
    return sentences if sentences else [text]


def _is_relevant_sentence(sentence: str, keywords: list[str], constraints: Constraints) -> bool:
    lower = sentence.lower()
    if any(k in lower for k in keywords):
        return True
    if constraints.entities:
        if any(ent.lower() in lower for ent in constraints.entities):
            return True
    if constraints.time_of_day and constraints.time_of_day in lower:
        return True
    if constraints.relative_time and constraints.relative_time in lower:
        return True
    if constraints.explicit_datetime and constraints.explicit_datetime.lower() in lower:
        return True
    if _TIME_REGEX.search(sentence):
        return True
    if re.search(r"\b(option|either|or)\b", lower):
        return True
    if sentence.strip().startswith(("-", "*", "•")):
        return True
    return False


def _trim_text(text: str, query: str, constraints: Constraints) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return text
    keywords = _keywords(query)

    selected = [s for s in sentences if _is_relevant_sentence(s, keywords, constraints)]
    if not selected:
        selected = sentences[:3]
    selected = selected[:10]
    return " ".join(selected)


def _has_time_evidence(text: str, metadata: dict[str, Any]) -> bool:
    if _TIME_REGEX.search(text or ""):
        return True
    for key in ("time", "datetime", "created_at", "modified_at"):
        value = metadata.get(key) if metadata else None
        if value and _TIME_REGEX.search(str(value)):
            return True
    return False


def pack_context(
    query: str,
    constraints: Constraints,
    scored_candidates: list[ScoredCandidate],
    final_k: int,
    max_tokens: int,
    require_time_evidence: bool = False,
) -> tuple[list[Candidate], dict[str, Any]]:
    if not scored_candidates or final_k <= 0 or max_tokens <= 0:
        return [], {"total_tokens": 0, "per_chunk_tokens": []}

    ordered = scored_candidates[:]
    if require_time_evidence:
        for item in scored_candidates:
            if _has_time_evidence(item.chunk.text, item.chunk.metadata):
                ordered = [item] + [c for c in scored_candidates if c.chunk.chunk_id != item.chunk.chunk_id]
                break

    selected: list[Candidate] = []
    per_chunk_tokens: list[int] = []
    total_tokens = 0

    for scored in ordered:
        if len(selected) >= final_k:
            break
        trimmed = _trim_text(scored.chunk.text, query, constraints)
        trimmed = trimmed.strip()
        if not trimmed:
            continue
        tokens = _estimate_tokens(trimmed)
        remaining = max_tokens - total_tokens
        if tokens > remaining:
            trimmed = _truncate_to_tokens(trimmed, remaining)
            tokens = _estimate_tokens(trimmed)
        if tokens <= 0:
            continue
        selected.append(
            Candidate(
                chunk_id=scored.chunk.chunk_id,
                text=trimmed,
                metadata=scored.chunk.metadata,
                rrf_score=scored.chunk.rrf_score,
            )
        )
        per_chunk_tokens.append(tokens)
        total_tokens += tokens
        if total_tokens >= max_tokens:
            break

    return selected, {"total_tokens": total_tokens, "per_chunk_tokens": per_chunk_tokens}
