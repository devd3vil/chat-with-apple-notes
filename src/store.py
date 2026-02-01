"""
Vector store abstraction for storing and retrieving embedded chunks.

Provides both in-memory (for testing) and Chroma (persistent) implementations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import numpy as np
from src.models import Chunk
from src.embedder import Embedder


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """
        Add chunks to the store with embeddings.
        
        Args:
            chunks: List of chunks to add. Each chunk should have embedding set.
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[str, str, float]]:
        """
        Search for top-k most similar chunks.
        
        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            
        Returns:
            List of tuples (chunk_id, chunk_text, similarity_score).
            Sorted by score descending.
        """
        pass
    
    @abstractmethod
    def delete_note(self, note_id: str) -> None:
        """
        Delete all chunks belonging to a note.
        
        Args:
            note_id: ID of the note whose chunks to delete.
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all chunks from the store."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """
        Get number of chunks in the store.
        
        Returns:
            Total number of chunks.
        """
        pass


class InMemoryStore(VectorStore):
    """
    In-memory vector store using simple cosine similarity.
    
    Suitable for testing and small datasets.
    """
    
    def __init__(self, embedder: Embedder):
        """
        Initialize in-memory store.
        
        Args:
            embedder: Embedder to use for query embedding.
        """
        self.embedder = embedder
        self.chunks: dict[str, Chunk] = {}  # chunk_id -> Chunk
        self.embeddings: dict[str, list[float]] = {}  # chunk_id -> embedding
    
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks with embeddings."""
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} has no embedding")
            
            self.chunks[chunk.id] = chunk
            self.embeddings[chunk.id] = chunk.embedding
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[str, str, float]]:
        """Search using cosine similarity."""
        if not self.chunks:
            return []
        
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm
        else:
            query_vec = query_vec / (query_norm + 1e-8)
        
        scores: list[tuple[str, str, float]] = []
        
        for chunk_id, embedding in self.embeddings.items():
            emb_vec = np.array(embedding, dtype=np.float32)
            emb_norm = np.linalg.norm(emb_vec)
            if emb_norm > 0:
                emb_vec = emb_vec / emb_norm
            else:
                emb_vec = emb_vec / (emb_norm + 1e-8)
            
            # Cosine similarity: [-1, 1] → convert to [0, 1]
            raw_similarity = float(np.dot(query_vec, emb_vec))
            # Clamp to [-1, 1] to avoid numerical errors
            raw_similarity = max(-1.0, min(1.0, raw_similarity))
            # Convert from [-1, 1] to [0, 1]
            similarity = (raw_similarity + 1.0) / 2.0
            
            chunk = self.chunks[chunk_id]
            scores.append((chunk_id, chunk.text, similarity))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[2], reverse=True)
        
        return scores[:top_k]
    
    def delete_note(self, note_id: str) -> None:
        """Delete all chunks for a note."""
        chunk_ids_to_delete = [
            cid for cid in self.chunks if self.chunks[cid].note_id == note_id
        ]
        for cid in chunk_ids_to_delete:
            del self.chunks[cid]
            del self.embeddings[cid]
    
    def clear(self) -> None:
        """Clear all chunks."""
        self.chunks.clear()
        self.embeddings.clear()
    
    def size(self) -> int:
        """Get number of chunks."""
        return len(self.chunks)


class ChromaStore(VectorStore):
    """
    Persistent vector store using Chroma.
    
    Stores embeddings locally on disk.
    """
    
    def __init__(self, db_path: Path, embedder: Embedder):
        """
        Initialize Chroma store.
        
        Args:
            db_path: Path to Chroma database directory.
            embedder: Embedder to use (for embedding queries).
        """
        self.db_path = db_path
        self.embedder = embedder
        
        # Import Chroma here to avoid hard dependency
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")
        
        # Use persistent client
        self.client = chromadb.PersistentClient(path=str(db_path))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="notes",
            metadata={"hnsw:space": "cosine"},
        )
    
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to Chroma."""
        if not chunks:
            return
        
        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} has no embedding")
            
            ids.append(chunk.id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)
            metadatas.append({
                "note_id": chunk.note_id,
                "chunk_idx": str(chunk.chunk_idx),
                "start_char": str(chunk.start_char),
                "end_char": str(chunk.end_char),
            })
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[str, str, float]]:
        """Search Chroma collection."""
        if self.collection.count() == 0:
            return []
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
            
            # Parse results
            output: list[tuple[str, str, float]] = []
            
            if results and results["ids"] and len(results["ids"]) > 0:
                ids = results["ids"][0]
                documents = results["documents"][0]
                distances = results["distances"][0] if results.get("distances") else []
                
                # Convert distance to similarity (Chroma returns distance in cosine space)
                # For cosine similarity: similarity = 1 - distance
                for i, chunk_id in enumerate(ids):
                    doc = documents[i] if i < len(documents) else ""
                    # If no distances returned, use a default score
                    if i < len(distances):
                        similarity = 1.0 - distances[i]
                    else:
                        similarity = 0.0
                    output.append((chunk_id, doc, similarity))
            
            return output
            
        except Exception as e:
            print(f"Error searching Chroma: {e}")
            return []
    
    def delete_note(self, note_id: str) -> None:
        """Delete all chunks for a note from Chroma."""
        try:
            # Query for all chunks with this note_id
            results = self.collection.get(
                where={"note_id": note_id},
            )
            
            if results and results["ids"]:
                self.collection.delete(ids=results["ids"])
                
        except Exception as e:
            print(f"Error deleting note {note_id}: {e}")
    
    def clear(self) -> None:
        """Delete all chunks from the collection."""
        try:
            if self.collection.count() > 0:
                all_ids = self.collection.get()["ids"]
                self.collection.delete(ids=all_ids)
        except Exception as e:
            print(f"Error clearing collection: {e}")
    
    def size(self) -> int:
        """Get number of chunks in collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0
