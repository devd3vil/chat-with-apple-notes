"""
Retriever for RAG system.

Retrieves top-k most relevant chunks from the vector store for a query.
"""

from src.models import Citation
from src.embedder import Embedder
from src.store import VectorStore
from src.bm25 import BM25Index


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
        w_vec: float = 1.0,
        w_lex: float = 2.0,
    ):
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        self.k_rrf = k_rrf
        self.w_vec = w_vec
        self.w_lex = w_lex

    def retrieve(self, query_text: str, top_k: int = 5) -> list[Citation]:
        if not query_text or not query_text.strip():
            raise ValueError("Cannot retrieve with empty query")

        try:
            top_k_vec = max(50, 5 * top_k)
            top_k_lex = max(50, 5 * top_k)

            query_embedding = self.embedder.embed(query_text)
            vec_results = self.store.search(query_embedding, top_k=top_k_vec)
            lex_results = self.bm25.query(query_text, top_k=top_k_lex)

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
                return []

            scored: list[tuple[str, float]] = []
            for chunk_id in candidate_ids:
                score = 0.0
                if chunk_id in vec_rank:
                    score += self.w_vec / (self.k_rrf + vec_rank[chunk_id])
                if chunk_id in lex_rank:
                    score += self.w_lex / (self.k_rrf + lex_rank[chunk_id])
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
            return citations

        except Exception as e:
            raise ValueError(f"Hybrid retrieval failed: {e}")

    def health(self) -> bool:
        try:
            return self.embedder.health() and self.store.health()
        except Exception:
            return False
