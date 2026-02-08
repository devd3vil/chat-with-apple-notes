"""
BM25 index for lexical retrieval.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from rank_bm25 import BM25Okapi

from src.models import Chunk

_PUNCT_RE = re.compile(r"[^\w\s:-]")
BM25_SCHEMA_VERSION = 2
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


def tokenize(text: str) -> list[str]:
    """Deterministic tokenizer for BM25."""
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    tokens = [t for t in cleaned.split() if t and t not in _STOPWORDS]
    return tokens


class BM25Index:
    """Lightweight BM25 index with persistence."""

    def __init__(self, path: Path):
        self.path = path
        self.schema_version = BM25_SCHEMA_VERSION
        self.chunk_ids: list[str] = []
        self.documents: list[str] = []
        self.note_ids: list[str] = []
        self.tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._id_to_idx: dict[str, int] = {}

    def _rebuild(self) -> None:
        self._bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
        self._id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}

    def add_documents(self, chunks: Iterable[Chunk]) -> None:
        updated = False
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            if chunk.id in self._id_to_idx:
                idx = self._id_to_idx[chunk.id]
                self.documents[idx] = chunk.text
                self.note_ids[idx] = chunk.note_id
                self.tokenized_corpus[idx] = tokens
            else:
                self.chunk_ids.append(chunk.id)
                self.documents.append(chunk.text)
                self.note_ids.append(chunk.note_id)
                self.tokenized_corpus.append(tokens)
            updated = True
        if updated:
            self._rebuild()

    def remove_note(self, note_id: str) -> None:
        if not self.chunk_ids:
            return
        keep_indices = [i for i, nid in enumerate(self.note_ids) if nid != note_id]
        if len(keep_indices) == len(self.note_ids):
            return
        self.chunk_ids = [self.chunk_ids[i] for i in keep_indices]
        self.documents = [self.documents[i] for i in keep_indices]
        self.note_ids = [self.note_ids[i] for i in keep_indices]
        self.tokenized_corpus = [self.tokenized_corpus[i] for i in keep_indices]
        self._rebuild()

    def query(self, q: str, top_k: int) -> list[tuple[str, float, str]]:
        if not q or not q.strip():
            return []
        if not self._bm25:
            return []
        tokens = tokenize(q)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        results: list[tuple[str, float, str]] = []
        for idx, score in ranked:
            results.append((self.chunk_ids[idx], float(score), self.documents[idx]))
        return results

    def reset(self) -> None:
        self.chunk_ids = []
        self.documents = []
        self.note_ids = []
        self.tokenized_corpus = []
        self._rebuild()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": self.schema_version,
            "chunk_ids": self.chunk_ids,
            "documents": self.documents,
            "note_ids": self.note_ids,
            "tokenized_corpus": self.tokenized_corpus,
        }
        self.path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        schema_version = int(data.get("schema_version") or 1)
        self.chunk_ids = data.get("chunk_ids", [])
        self.documents = data.get("documents", [])
        self.note_ids = data.get("note_ids", [])
        self.tokenized_corpus = data.get("tokenized_corpus", [])
        self.schema_version = BM25_SCHEMA_VERSION
        self._rebuild()
        if schema_version < self.schema_version:
            self.save()
