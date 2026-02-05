"""
Retriever for RAG system.

Retrieves top-k most relevant chunks from the vector store for a query.
"""

from src.models import Citation
from src.embedder import Embedder
from src.store import VectorStore
from src.bm25 import BM25Index
import re
import numpy as np


class Retriever:
    """Retrieves top-k relevant chunks from vector store."""
    
    def __init__(self, store: VectorStore, embedder: Embedder):
        """
        Initialize retriever.
        
        Args:
            store: Vector store to search.
            embedder: Embedder to convert query text to embedding.
        """
        self.store = store
        self.embedder = embedder
    
    def retrieve(self, query_text: str, top_k: int = 5) -> list[Citation]:
        """
        Retrieve top-k most relevant chunks for a query.
        
        Args:
            query_text: Query text to search for.
            top_k: Number of results to return.
            
        Returns:
            List of Citation objects sorted by relevance (descending).
            
        Raises:
            ValueError: If query_text is empty or retrieval fails.
        """
        if not query_text or not query_text.strip():
            raise ValueError("Cannot retrieve with empty query")
        
        try:
            # Embed the query
            query_embedding = self.embedder.embed(query_text)
            
            # Search the store
            search_results = self.store.search(query_embedding, top_k=top_k)
            
            # Convert to Citation objects
            citations: list[Citation] = []
            for chunk_id, chunk_text, score in search_results:
                citation = Citation(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    score=score,
                )
                citations.append(citation)
            
            return citations
            
        except Exception as e:
            raise ValueError(f"Retrieval failed: {e}")
    
    def health(self) -> bool:
        """
        Check if retriever is healthy.
        
        Returns:
            True if both embedder and store are available.
        """
        try:
            return self.embedder.health() and self.store.health()
        except Exception:
            return False


class HybridRetriever:
    """Hybrid retriever using BM25 + vector search with RRF fusion."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        bm25: BM25Index,
        k_rrf: int = 10,
        w_vec: float = 0.55,
        w_lex: float = 0.45,
        k_vec_raw: int = 80,
        k_vec: int = 50,
        k_lex: int = 50,
        mmr_lambda: float = 0.7,
        enable_mmr: bool = True,
    ):
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        self.k_rrf = k_rrf
        self.w_vec = w_vec
        self.w_lex = w_lex
        self.k_vec_raw = k_vec_raw
        self.k_vec = k_vec
        self.k_lex = k_lex
        self.mmr_lambda = mmr_lambda
        self.enable_mmr = enable_mmr

    def _normalize_query_for_lexical(self, query_text: str) -> str:
        q = query_text.strip()
        if not q:
            return q
        q_lower = q.lower()
        expansions: list[str] = []
        if "sunset" in q_lower:
            expansions.append("golden hour dusk")
        if "sunrise" in q_lower:
            expansions.append("dawn early morning")
        if (
            "schedule" in q_lower
            or "time" in q_lower
            or "when" in q_lower
            or re.search(r"\b\d{1,2}:\d{2}\b", q_lower)
        ):
            expansions.append("pick up tickets depart")
        if expansions:
            q = f"{q} " + " ".join(expansions)
        return q

    def _classify_query(self, query_text: str) -> tuple[bool, bool]:
        q = query_text.lower()
        time_tokens = [
            "time",
            "schedule",
            "when",
            "depart",
            "sunrise",
            "sunset",
            "golden hour",
            "dawn",
        ]
        is_time_query = any(t in q for t in time_tokens) or bool(
            re.search(r"\b\d{1,2}:\d{2}\b", q)
        )
        named_tokens = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "denver",
            "estes",
            "rmnp",
            "vail",
            "georgetown",
            "colorado",
            "springs",
            "pikes",
            "royal",
            "gorge",
            "garden",
            "gods",
            "stanley",
            "riverwalk",
        ]
        is_named_entity_query = any(t in q for t in named_tokens)
        return is_time_query, is_named_entity_query

    def _select_weights(self, query_text: str) -> tuple[float, float, dict]:
        is_time_query, is_named_entity_query = self._classify_query(query_text)
        if is_time_query:
            w_lex, w_vec = 0.65, 0.35
        elif is_named_entity_query:
            w_lex, w_vec = 0.55, 0.45
        else:
            w_lex, w_vec = 0.45, 0.55
        meta = {
            "is_time_query": is_time_query,
            "is_named_entity_query": is_named_entity_query,
        }
        return w_vec, w_lex, meta

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    def _mmr_select(
        self,
        candidates: list[tuple[str, str, float]],
        k: int,
    ) -> list[tuple[str, str, float]]:
        if not candidates:
            return []
        texts = [t for _, t, _ in candidates]
        embeddings = [np.array(self.embedder.embed(t), dtype=np.float32) for t in texts]
        norms = [e / (np.linalg.norm(e) + 1e-8) for e in embeddings]

        selected: list[int] = []
        remaining = list(range(len(candidates)))

        while remaining and len(selected) < k:
            if not selected:
                selected.append(remaining.pop(0))
                continue
            best_idx = None
            best_score = -1e9
            for idx in remaining:
                relevance = candidates[idx][2]
                redundancy = max(
                    self._cosine_sim(norms[idx], norms[s]) for s in selected
                )
                score = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * redundancy
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[i] for i in selected]

    def retrieve_with_trace(self, query_text: str, top_k: int = 5) -> tuple[list[Citation], dict]:
        citations, trace = self.retrieve(query_text, top_k=top_k, include_trace=True)
        return citations, trace

    def retrieve(self, query_text: str, top_k: int = 5, include_trace: bool = False):
        if not query_text or not query_text.strip():
            raise ValueError("Cannot retrieve with empty query")

        try:
            w_vec, w_lex, flags = self._select_weights(query_text)
            query_embedding = self.embedder.embed(query_text)
            vec_results = self.store.search(query_embedding, top_k=self.k_vec_raw)
            vec_results = self._mmr_select(vec_results, self.k_vec) if self.enable_mmr else vec_results[: self.k_vec]
            lex_query = self._normalize_query_for_lexical(query_text)
            lex_results = self.bm25.query(lex_query, top_k=self.k_lex)

            vec_rank: dict[str, int] = {}
            lex_rank: dict[str, int] = {}
            text_map: dict[str, str] = {}

            for i, (chunk_id, text, _score) in enumerate(vec_results, start=1):
                vec_rank[chunk_id] = i
                text_map[chunk_id] = text

            for i, (chunk_id, _score, text) in enumerate(lex_results, start=1):
                lex_rank[chunk_id] = i
                text_map.setdefault(chunk_id, text)

            candidate_ids = set(vec_rank) | set(lex_rank)
            if not candidate_ids:
                return [] if not include_trace else ([], {})

            scored: list[tuple[str, float]] = []
            for chunk_id in candidate_ids:
                score = 0.0
                if chunk_id in vec_rank:
                    score += w_vec / (self.k_rrf + vec_rank[chunk_id])
                if chunk_id in lex_rank:
                    score += w_lex / (self.k_rrf + lex_rank[chunk_id])
                scored.append((chunk_id, score))

            max_score = max(score for _, score in scored) if scored else 0.0

            def sort_key(item: tuple[str, float]) -> tuple[float, int, int, str]:
                chunk_id, score = item
                return (
                    -score,
                    vec_rank.get(chunk_id, 1_000_000),
                    lex_rank.get(chunk_id, 1_000_000),
                    chunk_id,
                )

            scored.sort(key=sort_key)

            citations: list[Citation] = []
            for chunk_id, score in scored[:top_k]:
                normalized = score / max_score if max_score > 0 else 0.0
                citations.append(
                    Citation(
                        chunk_id=chunk_id,
                        text=text_map.get(chunk_id, ""),
                        score=normalized,
                    )
                )
            if not include_trace:
                return citations

            def _note_id(cid: str) -> str:
                return cid.split("_", 1)[0]

            bm25_top = [
                {
                    "chunk_id": cid,
                    "rank_lex": idx + 1,
                    "lex_score": score,
                    "note_id": _note_id(cid),
                }
                for idx, (cid, score, _text) in enumerate(lex_results[:10])
            ]
            vec_top = [
                {
                    "chunk_id": cid,
                    "rank_vec": idx + 1,
                    "vec_score": score,
                    "note_id": _note_id(cid),
                }
                for idx, (cid, _text, score) in enumerate(vec_results[:10])
            ]
            fused_top = [
                {
                    "chunk_id": cid,
                    "rrf_score": score,
                    "rank_lex": lex_rank.get(cid),
                    "rank_vec": vec_rank.get(cid),
                }
                for cid, score in scored[:20]
            ]
            trace = {
                "query": query_text,
                "w_vec": w_vec,
                "w_lex": w_lex,
                "k_rrf": self.k_rrf,
                "k_vec_raw": self.k_vec_raw,
                "k_vec": self.k_vec,
                "k_lex": self.k_lex,
                "lex_query": lex_query,
                "flags": flags,
                "bm25_top": bm25_top,
                "vec_top": vec_top,
                "fused_top": fused_top,
            }
            return citations, trace

        except Exception as e:
            raise ValueError(f"Hybrid retrieval failed: {e}")

    def health(self) -> bool:
        try:
            return self.embedder.health() and self.store.health()
        except Exception:
            return False
