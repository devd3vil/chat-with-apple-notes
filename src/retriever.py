"""
Retriever for RAG system.

Retrieves top-k most relevant chunks from the vector store for a query.
"""

from src.models import Citation
from src.embedder import Embedder
from src.store import VectorStore


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
