"""
Embedder abstraction for text-to-vector conversion.

Provides both real (Ollama) and fake (deterministic) implementations for testing.
"""

import hashlib
from abc import ABC, abstractmethod
from typing import Optional
import requests


class Embedder(ABC):
    """Abstract base class for embedders."""
    
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Embed text into a vector.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector as list of floats.
            
        Raises:
            ValueError: If text is empty or embedding fails.
        """
        pass
    
    @abstractmethod
    def health(self) -> bool:
        """
        Check if embedder is healthy and ready.
        
        Returns:
            True if embedder is available and working.
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Get dimensionality of embeddings.
        
        Returns:
            Number of dimensions in embedding vectors.
        """
        pass


class OllamaEmbedder(Embedder):
    """Embedder using Ollama API."""
    
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize Ollama embedder.
        
        Args:
            model_name: Name of the Ollama embedding model.
            base_url: Base URL of Ollama API.
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.embedding_dim: Optional[int] = None
        
        # Try to determine embedding dimension
        try:
            test_embedding = self.embed("test")
            self.embedding_dim = len(test_embedding)
        except Exception:
            # Will be detected on first real use
            pass
    
    def embed(self, text: str) -> list[float]:
        """
        Embed text using Ollama API.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector.
            
        Raises:
            ValueError: If embedding fails or text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=60,
            )
            response.raise_for_status()
            
            data = response.json()
            embedding = data.get("embedding")
            
            if embedding is None:
                raise ValueError("No embedding in response")
            
            # Update embedding dim if not yet set
            if self.embedding_dim is None:
                self.embedding_dim = len(embedding)
            
            return embedding
            
        except requests.RequestException as e:
            raise ValueError(f"Ollama embedder error: {e}")
        except (KeyError, ValueError) as e:
            raise ValueError(f"Failed to parse Ollama response: {e}")
    
    def health(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            data = response.json()
            models = data.get("models", [])
            return any(m.get("name", "").startswith(self.model_name) for m in models)
            
        except Exception:
            return False
    
    def get_embedding_dim(self) -> int:
        """Get dimensionality of embeddings."""
        if self.embedding_dim is not None:
            return self.embedding_dim
        
        # Try to get from a test embedding
        try:
            embedding = self.embed("test")
            self.embedding_dim = len(embedding)
            return self.embedding_dim
        except Exception:
            # Return common default for nomic-embed-text
            return 768


class FakeEmbedder(Embedder):
    """
    Deterministic embedder for testing.
    
    Uses SHA256 hash of text to generate a fake embedding vector.
    Deterministic: same text always produces same embedding.
    """
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize fake embedder.
        
        Args:
            embedding_dim: Dimensionality of output vectors.
        """
        self.embedding_dim = embedding_dim
        self._healthy = True
    
    def embed(self, text: str) -> list[float]:
        """
        Generate deterministic embedding from text hash.
        
        Args:
            text: Text to embed.
            
        Returns:
            Fake embedding vector (deterministic based on text).
            
        Raises:
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        # Hash the text
        hash_obj = hashlib.sha256(text.encode("utf-8"))
        hash_hex = hash_obj.hexdigest()
        
        # Convert hash to embedding vector
        # Split hash into chunks and normalize to [-1, 1]
        embedding = []
        for i in range(self.embedding_dim):
            # Take 2 hex chars per dimension
            chunk_start = (i * 2) % len(hash_hex)
            chunk_end = chunk_start + 2
            if chunk_end > len(hash_hex):
                chunk_end = len(hash_hex)
                chunk_start = len(hash_hex) - 2
            
            hex_chunk = hash_hex[chunk_start:chunk_end]
            # Convert hex to int [0, 255], then normalize to [-1, 1]
            value = int(hex_chunk, 16) / 128.0 - 1.0
            embedding.append(value)
        
        return embedding
    
    def health(self) -> bool:
        """Fake embedder is always healthy."""
        return self._healthy
    
    def get_embedding_dim(self) -> int:
        """Get dimensionality."""
        return self.embedding_dim
    
    def set_health(self, healthy: bool) -> None:
        """
        Set health status for testing failure scenarios.
        
        Args:
            healthy: Whether embedder should be marked as healthy.
        """
        self._healthy = healthy
