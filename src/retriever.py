"""
Retriever for RAG system.

Retrieves top-k most relevant chunks from the vector store for a query.
"""

from src.models import Citation
from src.embedder import Embedder
from src.store import VectorStore
from src.bm25 import BM25Index
from src.constraint_extractor import extract_constraints, Constraints
from src.context_packer import pack_context
from src.reranker import Reranker, Candidate, ScoredCandidate
from src.llm import LLM
import re
import numpy as np


def _note_id_from_chunk_id(chunk_id: str) -> str:
    return chunk_id.split("_", 1)[0] if "_" in chunk_id else chunk_id


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
                note_id = _note_id_from_chunk_id(chunk_id)
                citation = Citation(
                    chunk_id=chunk_id,
                    note_id=note_id,
                    text=chunk_text,
                    score=score,
                    source={"note_id": note_id},
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
        rerank_enabled: bool = True,
        hybrid_topn: int = 50,
        rerank_candidates_n: int = 30,
        final_context_k: int = 4,
        max_context_tokens: int = 1200,
        rerank_cache_ttl_seconds: int = 86400,
        constraint_mode: str = "soft",
        time_query_require_time_evidence: bool = True,
        rerank_llm: LLM | None = None,
        rerank_backend: str = "cross-encoder",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
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
        self.rerank_enabled = rerank_enabled
        self.hybrid_topn = hybrid_topn
        self.rerank_candidates_n = rerank_candidates_n
        self.final_context_k = final_context_k
        self.max_context_tokens = max_context_tokens
        self.rerank_cache_ttl_seconds = rerank_cache_ttl_seconds
        self.constraint_mode = constraint_mode
        self.time_query_require_time_evidence = time_query_require_time_evidence
        self.reranker = Reranker(
            rerank_llm,
            cache_ttl_seconds=rerank_cache_ttl_seconds,
            backend=rerank_backend,
            cross_encoder_model=cross_encoder_model,
        )

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

    def _constraints_strong(self, constraints: Constraints) -> bool:
        return bool(
            constraints.explicit_datetime
            or constraints.entities
            or constraints.scope_hints
        )

    def _supports_constraints(self, candidate: Candidate, constraints: Constraints) -> bool:
        text_lower = candidate.text.lower()
        if constraints.entities and any(ent.lower() in text_lower for ent in constraints.entities):
            return True
        if constraints.explicit_datetime and constraints.explicit_datetime.lower() in text_lower:
            return True
        if constraints.relative_time and constraints.relative_time in text_lower:
            return True
        if constraints.time_of_day and constraints.time_of_day in text_lower:
            return True
        if constraints.scope_hints and any(hint in text_lower for hint in constraints.scope_hints):
            return True
        return False

    def _sort_scored_candidates(
        self,
        scored: list[ScoredCandidate],
        constraints: Constraints,
    ) -> list[ScoredCandidate]:
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

    def _hybrid_rrf_data(self, query_text: str) -> dict:
        w_vec, w_lex, flags = self._select_weights(query_text)
        query_embedding = self.embedder.embed(query_text)
        vec_results = self.store.search(query_embedding, top_k=self.k_vec_raw)
        if self.enable_mmr:
            vec_results = self._mmr_select(vec_results, self.k_vec)
        else:
            vec_results = vec_results[: self.k_vec]
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
        scored: list[tuple[str, float]] = []
        for chunk_id in candidate_ids:
            score = 0.0
            if chunk_id in vec_rank:
                score += w_vec / (self.k_rrf + vec_rank[chunk_id])
            if chunk_id in lex_rank:
                score += w_lex / (self.k_rrf + lex_rank[chunk_id])
            scored.append((chunk_id, score))

        def sort_key(item: tuple[str, float]) -> tuple[float, int, int, str]:
            chunk_id, score = item
            return (
                -score,
                vec_rank.get(chunk_id, 1_000_000),
                lex_rank.get(chunk_id, 1_000_000),
                chunk_id,
            )

        scored.sort(key=sort_key)

        return {
            "scored": scored,
            "text_map": text_map,
            "vec_rank": vec_rank,
            "lex_rank": lex_rank,
            "vec_results": vec_results,
            "lex_results": lex_results,
            "w_vec": w_vec,
            "w_lex": w_lex,
            "k_rrf": self.k_rrf,
            "lex_query": lex_query,
            "flags": flags,
        }

    def _hybrid_rrf_candidates(self, query_text: str, topn: int) -> tuple[list[Candidate], dict]:
        data = self._hybrid_rrf_data(query_text)
        scored = data["scored"]
        text_map = data["text_map"]

        candidates: list[Candidate] = []
        for chunk_id, score in scored[:topn]:
            note_id = _note_id_from_chunk_id(chunk_id)
            candidates.append(
                Candidate(
                    chunk_id=chunk_id,
                    text=text_map.get(chunk_id, ""),
                    metadata={"note_id": note_id},
                    rrf_score=score,
                )
            )

        data["hybrid_top10"] = [
            {"chunk_id": cid, "rrf_score": score, "note_id": _note_id_from_chunk_id(cid)}
            for cid, score in scored[:10]
        ]
        data["max_rrf"] = max((s for _, s in scored), default=0.0)
        return candidates, data

    def retrieve_for_generation(
        self,
        query_text: str,
        top_k: int | None = None,
        include_trace: bool = False,
    ):
        if not query_text or not query_text.strip():
            raise ValueError("Cannot retrieve with empty query")

        candidates, hybrid_trace = self._hybrid_rrf_candidates(
            query_text,
            topn=self.hybrid_topn,
        )
        if not candidates:
            return [] if not include_trace else ([], {})

        constraints = extract_constraints(query_text)
        rerank_trace: dict = {}
        max_rrf = hybrid_trace.get("max_rrf", 0.0) or 1.0

        scored: list[ScoredCandidate] = []
        if self.rerank_enabled:
            rerank_candidates = candidates[: self.rerank_candidates_n]
            scored, rerank_trace = self.reranker.rerank(
                query_text,
                constraints,
                rerank_candidates,
            )

        scored_ids = {s.chunk.chunk_id for s in scored}
        for cand in candidates:
            if cand.chunk_id in scored_ids:
                continue
            total = 3.0 * (cand.rrf_score / max_rrf) if max_rrf > 0 else 0.0
            scored.append(
                ScoredCandidate(
                    chunk=cand,
                    scores={
                        "direct": 0.0,
                        "specificity": 0.0,
                        "constraint_support": 0.0,
                        "anti_contam": 0.0,
                    },
                    total=total,
                    reasons="rrf",
                )
            )

        scored = self._sort_scored_candidates(scored, constraints)

        if self.constraint_mode == "hard" and self._constraints_strong(constraints):
            supporting = [s for s in scored if self._supports_constraints(s.chunk, constraints)]
            if len(supporting) >= 3:
                scored = supporting

        final_k = min(self.final_context_k, top_k) if top_k else self.final_context_k
        require_time = self.time_query_require_time_evidence and constraints.is_time_question
        packed, pack_trace = pack_context(
            query_text,
            constraints,
            scored,
            final_k=final_k,
            max_tokens=self.max_context_tokens,
            require_time_evidence=require_time,
        )

        scored_map = {s.chunk.chunk_id: s for s in scored}
        max_total = max((s.total for s in scored), default=0.0) or 1.0

        citations: list[Citation] = []
        for cand in packed:
            scored_item = scored_map.get(cand.chunk_id)
            total = scored_item.total if scored_item else 0.0
            normalized = min(1.0, max(0.0, total / max_total))
            note_id = _note_id_from_chunk_id(cand.chunk_id)
            citations.append(
                Citation(
                    chunk_id=cand.chunk_id,
                    note_id=note_id,
                    text=cand.text,
                    score=normalized,
                    source={"note_id": note_id},
                )
            )

        if not include_trace:
            return citations

        rerank_top10 = []
        for item in scored[:10]:
            rerank_top10.append(
                {
                    "chunk_id": item.chunk.chunk_id,
                    "total": item.total,
                    "scores": item.scores,
                    "rrf_score": item.chunk.rrf_score,
                }
            )

        trace = {
            "query": query_text,
            "constraints": constraints.__dict__,
            "hybrid_top10": hybrid_trace.get("hybrid_top10", []),
            "rerank_top10": rerank_top10,
            "packed_context": pack_trace,
            "rerank_trace": rerank_trace,
        }
        return citations, trace

    def warmup_reranker(self) -> bool:
        try:
            return self.reranker.warmup()
        except Exception:
            return False

    def retrieve_with_trace(self, query_text: str, top_k: int = 5) -> tuple[list[Citation], dict]:
        citations, trace = self.retrieve(query_text, top_k=top_k, include_trace=True)
        return citations, trace

    def retrieve(self, query_text: str, top_k: int = 5, include_trace: bool = False):
        if not query_text or not query_text.strip():
            raise ValueError("Cannot retrieve with empty query")

        try:
            data = self._hybrid_rrf_data(query_text)
            scored = data["scored"]
            text_map = data["text_map"]
            vec_rank = data["vec_rank"]
            lex_rank = data["lex_rank"]
            vec_results = data["vec_results"]
            lex_results = data["lex_results"]

            if not scored:
                return [] if not include_trace else ([], {})

            max_score = max(score for _, score in scored) if scored else 0.0

            citations: list[Citation] = []
            for chunk_id, score in scored[:top_k]:
                normalized = score / max_score if max_score > 0 else 0.0
                note_id = _note_id_from_chunk_id(chunk_id)
                citations.append(
                    Citation(
                        chunk_id=chunk_id,
                        note_id=note_id,
                        text=text_map.get(chunk_id, ""),
                        score=normalized,
                        source={"note_id": note_id},
                    )
                )
            if not include_trace:
                return citations

            bm25_top = [
                {
                    "chunk_id": cid,
                    "rank_lex": idx + 1,
                    "lex_score": score,
                    "note_id": _note_id_from_chunk_id(cid),
                }
                for idx, (cid, score, _text) in enumerate(lex_results[:10])
            ]
            vec_top = [
                {
                    "chunk_id": cid,
                    "rank_vec": idx + 1,
                    "vec_score": score,
                    "note_id": _note_id_from_chunk_id(cid),
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
                "w_vec": data["w_vec"],
                "w_lex": data["w_lex"],
                "k_rrf": self.k_rrf,
                "k_vec_raw": self.k_vec_raw,
                "k_vec": self.k_vec,
                "k_lex": self.k_lex,
                "lex_query": data["lex_query"],
                "flags": data["flags"],
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
