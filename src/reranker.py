"""LLM-based reranker with safe heuristic fallback and caching."""

from __future__ import annotations

from dataclasses import dataclass
import math
import hashlib
import json
import re
import time
from typing import Any, Optional

from src.constraint_extractor import Constraints
from src.llm import LLM

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - optional dependency
    CrossEncoder = None

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


@dataclass
class Candidate:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    rrf_score: float


@dataclass
class ScoredCandidate:
    chunk: Candidate
    scores: dict[str, float]
    total: float
    reasons: Optional[str] = None


class RerankCache:
    def __init__(self, ttl_seconds: int = 86400):
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, list[ScoredCandidate]]] = {}

    def get(self, key: str) -> Optional[list[ScoredCandidate]]:
        record = self._store.get(key)
        if not record:
            return None
        ts, value = record
        if time.time() - ts > self.ttl_seconds:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: list[ScoredCandidate]) -> None:
        self._store[key] = (time.time(), value)


def _hash_candidates(query: str, candidates: list[Candidate]) -> str:
    payload = query + "|" + "|".join(
        f"{c.chunk_id}:{hashlib.sha256(c.text.encode('utf-8')).hexdigest()}" for c in candidates
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _format_constraints(constraints: Constraints) -> str:
    parts = []
    if constraints.explicit_datetime:
        parts.append(f"explicit_datetime: {constraints.explicit_datetime}")
    if constraints.relative_time:
        parts.append(f"relative_time: {constraints.relative_time}")
    if constraints.time_of_day:
        parts.append(f"time_of_day: {constraints.time_of_day}")
    if constraints.entities:
        parts.append(f"entities: {', '.join(constraints.entities)}")
    if constraints.scope_hints:
        parts.append(f"scope_hints: {', '.join(constraints.scope_hints)}")
    if constraints.is_time_question:
        parts.append("is_time_question: true")
    return "\n".join(parts) if parts else "(none)"


def _build_prompt(query: str, constraints: Constraints, candidates: list[Candidate]) -> str:
    formatted_constraints = _format_constraints(constraints)
    lines = [
        "You are reranking chunks for a Q&A system over personal notes.",
        "Score each chunk with the rubric below (0-3 per field).",
        "Return ONLY valid JSON with the specified schema.",
        "",
        "Rubric:",
        "direct: Does the chunk directly answer the question? (most important)",
        "specificity: Does it contain concrete facts (names, numbers, dates/times, explicit options)?",
        "constraint_support: If constraints exist, does the chunk support them? If no constraints, use 1.5.",
        "anti_contam: Penalize if clearly different context when constraints exist; if no constraints, use 1.5.",
        "total = 0.55*direct + 0.25*specificity + 0.15*constraint_support + 0.05*anti_contam",
        "",
        f"Question: {query}",
        "",
        f"Constraints:\n{formatted_constraints}",
        "",
        "Chunks:",
    ]
    for idx, cand in enumerate(candidates, start=1):
        meta = cand.metadata or {}
        meta_text = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else "none"
        text = cand.text
        if len(text) > 800:
            text = text[:800] + "…"
        lines.append(f"{idx}. chunk_id: {cand.chunk_id}\nmeta: {meta_text}\ntext: {text}")

    lines.append(
        "\nReturn JSON exactly in this shape:\n"
        "{\"scores\":[{\"chunk_id\":\"...\",\"direct\":3,\"specificity\":2,\"constraint_support\":3,\"anti_contam\":2,\"total\":2.65}]}"
    )
    return "\n".join(lines)


def _safe_json_loads(text: str) -> Optional[dict[str, Any]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _keywords(query: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_-]+", query.lower())
    return [t for t in tokens if t and t not in _STOPWORDS]


def _heuristic_score(query: str, constraints: Constraints, candidate: Candidate) -> ScoredCandidate:
    text_lower = candidate.text.lower()
    keys = _keywords(query)
    key_hits = sum(1 for k in keys if k in text_lower)

    if key_hits >= 2:
        direct = 3.0
    elif key_hits == 1:
        direct = 2.0
    else:
        direct = 0.0

    if constraints.entities:
        if any(ent.lower() in text_lower for ent in constraints.entities):
            direct = min(3.0, direct + 1.0)
        else:
            direct = max(0.0, direct - 1.0)

    specificity = _specificity_score(candidate.text)
    constraint_support, anti_contam = _constraint_scores(constraints, text_lower)

    total = 0.55 * direct + 0.25 * specificity + 0.15 * constraint_support + 0.05 * anti_contam
    return ScoredCandidate(
        chunk=candidate,
        scores={
            "direct": direct,
            "specificity": specificity,
            "constraint_support": constraint_support,
            "anti_contam": anti_contam,
        },
        total=total,
        reasons="heuristic",
    )


def _sort_scored(scored: list[ScoredCandidate], constraints: Constraints) -> list[ScoredCandidate]:
    constraints_exist = bool(
        constraints.explicit_datetime
        or constraints.relative_time
        or constraints.time_of_day
        or constraints.entities
        or constraints.scope_hints
    )

    def sort_key(item: ScoredCandidate) -> tuple[float, float, float, float]:
        return (
            -item.total,
            -item.scores.get("direct", 0.0),
            -item.scores.get("constraint_support", 0.0) if constraints_exist else 0.0,
            -item.chunk.rrf_score,
        )

    return sorted(scored, key=sort_key)


class Reranker:
    def __init__(
        self,
        llm: Optional[LLM],
        cache_ttl_seconds: int = 86400,
        backend: str = "cross-encoder",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.llm = llm
        self.cache = RerankCache(ttl_seconds=cache_ttl_seconds)
        self.backend = backend
        self.cross_encoder_model = cross_encoder_model
        self._cross_encoder = None

    def rerank(
        self,
        query: str,
        constraints: Constraints,
        candidates: list[Candidate],
        max_tokens: int = 512,
    ) -> tuple[list[ScoredCandidate], dict[str, Any]]:
        if not candidates:
            return [], {"used_llm": False, "cache_hit": False}

        key = _hash_candidates(query, candidates)
        cached = self.cache.get(key)
        if cached is not None:
            return cached, {"used_llm": False, "cache_hit": True}

        if self.backend == "cross-encoder":
            scored = self._rerank_with_cross_encoder(query, constraints, candidates)
            if scored is None:
                scored = [_heuristic_score(query, constraints, c) for c in candidates]
                scored = _sort_scored(scored, constraints)
                self.cache.set(key, scored)
                return scored, {
                    "used_llm": False,
                    "cache_hit": False,
                    "fallback": "heuristic",
                    "cross_encoder": "unavailable",
                }
            scored = _sort_scored(scored, constraints)
            self.cache.set(key, scored)
            return scored, {"used_llm": False, "cache_hit": False, "backend": "cross-encoder"}

        if self.backend == "llm":
            if not self.llm:
                scored = [_heuristic_score(query, constraints, c) for c in candidates]
                scored = _sort_scored(scored, constraints)
                self.cache.set(key, scored)
                return scored, {"used_llm": False, "cache_hit": False, "fallback": "heuristic"}

            prompt = _build_prompt(query, constraints, candidates)
            raw = self.llm.generate(prompt, max_tokens=max_tokens)
            data = _safe_json_loads(raw)
            if not data or "scores" not in data:
                scored = [_heuristic_score(query, constraints, c) for c in candidates]
                scored = _sort_scored(scored, constraints)
                self.cache.set(key, scored)
                return scored, {"used_llm": True, "cache_hit": False, "parse_error": True}

            scores_by_id: dict[str, dict[str, float]] = {}
            for row in data.get("scores", []):
                cid = row.get("chunk_id")
                if not cid:
                    continue
                direct = float(row.get("direct", 0.0))
                specificity = float(row.get("specificity", 0.0))
                constraint_support = float(row.get("constraint_support", 0.0))
                anti_contam = float(row.get("anti_contam", 0.0))
                total = row.get("total")
                if total is None:
                    total = (
                        0.55 * direct
                        + 0.25 * specificity
                        + 0.15 * constraint_support
                        + 0.05 * anti_contam
                    )
                scores_by_id[cid] = {
                    "direct": direct,
                    "specificity": specificity,
                    "constraint_support": constraint_support,
                    "anti_contam": anti_contam,
                    "total": float(total),
                }

            scored: list[ScoredCandidate] = []
            for cand in candidates:
                entry = scores_by_id.get(cand.chunk_id)
                if not entry:
                    scored.append(_heuristic_score(query, constraints, cand))
                    continue
                scored.append(
                    ScoredCandidate(
                        chunk=cand,
                        scores={
                            "direct": entry["direct"],
                            "specificity": entry["specificity"],
                            "constraint_support": entry["constraint_support"],
                            "anti_contam": entry["anti_contam"],
                        },
                        total=entry["total"],
                    )
                )

            scored = _sort_scored(scored, constraints)
            self.cache.set(key, scored)
            return scored, {"used_llm": True, "cache_hit": False}

        scored = [_heuristic_score(query, constraints, c) for c in candidates]
        scored = _sort_scored(scored, constraints)
        self.cache.set(key, scored)
        return scored, {"used_llm": False, "cache_hit": False, "backend": "heuristic"}

    def _get_cross_encoder(self):
        if self._cross_encoder is not None:
            return self._cross_encoder
        if CrossEncoder is None:
            return None
        self._cross_encoder = CrossEncoder(self.cross_encoder_model)
        return self._cross_encoder

    def _rerank_with_cross_encoder(
        self,
        query: str,
        constraints: Constraints,
        candidates: list[Candidate],
    ) -> Optional[list[ScoredCandidate]]:
        model = self._get_cross_encoder()
        if model is None:
            return None
        pairs = [(query, cand.text) for cand in candidates]
        try:
            scores = model.predict(pairs)
        except Exception:
            return None
        scored: list[ScoredCandidate] = []
        for cand, score in zip(candidates, scores):
            direct = _scale_cross_score(score, constraints, cand)
            specificity = _specificity_score(cand.text)
            constraint_support, anti_contam = _constraint_scores(constraints, cand.text.lower())
            total = 0.55 * direct + 0.25 * specificity + 0.15 * constraint_support + 0.05 * anti_contam
            scored.append(
                ScoredCandidate(
                    chunk=cand,
                    scores={
                        "direct": direct,
                        "specificity": specificity,
                        "constraint_support": constraint_support,
                        "anti_contam": anti_contam,
                    },
                    total=total,
                )
            )
        return scored


def _specificity_score(text: str) -> float:
    if _TIME_REGEX.search(text) or re.search(r"\\b\\d{1,4}\\b", text):
        return 3.0
    if re.search(r"\\b[A-Z][a-z]+\\b", text):
        return 2.0
    return 1.0


def _constraint_scores(constraints: Constraints, text_lower: str) -> tuple[float, float]:
    constraints_exist = bool(
        constraints.explicit_datetime
        or constraints.relative_time
        or constraints.time_of_day
        or constraints.entities
        or constraints.scope_hints
    )

    if not constraints_exist:
        return 1.5, 1.5

    matches = 0
    if constraints.entities:
        matches += sum(1 for ent in constraints.entities if ent.lower() in text_lower)
    if constraints.time_of_day and constraints.time_of_day in text_lower:
        matches += 1
    if constraints.relative_time and constraints.relative_time in text_lower:
        matches += 1
    if constraints.scope_hints:
        matches += sum(1 for hint in constraints.scope_hints if hint in text_lower)

    if matches > 0:
        return 3.0, 2.0
    return 0.0, 0.0


def _scale_cross_score(score: float, constraints: Constraints, candidate: Candidate) -> float:
    try:
        bounded = max(-5.0, min(5.0, float(score)))
    except (TypeError, ValueError):
        bounded = 0.0
    direct = 3.0 / (1.0 + math.exp(-bounded))

    if constraints.entities:
        text_lower = candidate.text.lower()
        if any(ent.lower() in text_lower for ent in constraints.entities):
            direct = min(3.0, direct + 0.5)
        else:
            direct = max(0.0, direct - 0.5)

    return direct
