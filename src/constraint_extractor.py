"""
Constraint extraction for retrieval reranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Optional

_TIME_REGEX = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s?(?:am|pm)\b", re.IGNORECASE)
_DATE_REGEX = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2})\b",
    re.IGNORECASE,
)

_RELATIVE = [
    "today",
    "yesterday",
    "tomorrow",
    "last week",
    "last month",
    "next week",
    "next month",
    "this week",
    "this month",
    "recent",
]

_TIME_OF_DAY = [
    "morning",
    "afternoon",
    "evening",
    "night",
    "sunrise",
    "sunset",
    "dawn",
    "dusk",
]

_SCOPE_HINTS = [
    "meeting",
    "standup",
    "project",
    "design doc",
    "prd",
    "email",
    "invoice",
    "flight",
    "hotel",
    "ticket",
    "schedule",
    "plan",
    "itinerary",
    "note",
    "summary",
]

_STOP_ENTITY_TOKENS = {
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
}


@dataclass
class Constraints:
    explicit_datetime: Optional[str] = None
    relative_time: Optional[str] = None
    time_of_day: Optional[str] = None
    entities: list[str] = field(default_factory=list)
    scope_hints: list[str] = field(default_factory=list)
    is_time_question: bool = False


def _extract_entities(query: str) -> list[str]:
    entities: list[str] = []
    caps = re.findall(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)\b", query)
    acronyms = re.findall(r"\b[A-Z][A-Z0-9-]{1,}\b", query)
    for ent in caps + acronyms:
        if ent in _STOP_ENTITY_TOKENS:
            continue
        if ent not in entities:
            entities.append(ent)
    return entities


def extract_constraints(query: str) -> Constraints:
    """Extract generic constraints from a query."""
    q = query.strip()
    if not q:
        return Constraints()

    q_lower = q.lower()
    explicit_datetime = None
    date_match = _DATE_REGEX.search(q)
    time_match = _TIME_REGEX.search(q)
    if date_match:
        explicit_datetime = date_match.group(0)
    if time_match:
        explicit_datetime = f"{explicit_datetime} {time_match.group(0)}".strip() if explicit_datetime else time_match.group(0)

    relative_time = None
    for token in _RELATIVE:
        if token in q_lower:
            relative_time = token
            break

    time_of_day = None
    for token in _TIME_OF_DAY:
        if token in q_lower:
            time_of_day = token
            break

    entities = _extract_entities(q)

    scope_hints = [hint for hint in _SCOPE_HINTS if hint in q_lower]

    is_time_question = bool(_TIME_REGEX.search(q))
    if not is_time_question:
        time_tokens = ["when", "time", "schedule", "at", "by", "before", "after", "depart"]
        is_time_question = any(tok in q_lower for tok in time_tokens)

    return Constraints(
        explicit_datetime=explicit_datetime,
        relative_time=relative_time,
        time_of_day=time_of_day,
        entities=entities,
        scope_hints=scope_hints,
        is_time_question=is_time_question,
    )
