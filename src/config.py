"""
Configuration management for the RAG Q&A system.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration loaded from environment variables."""
    
    # Ollama configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_chat_model: str = "neural-chat"
    
    # Vector store configuration
    chroma_db_path: Path = Path("data/chroma_db")
    
    # RAG configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    min_similarity_score: float = 0.8
    bm25_index_path: Path = Path("data/bm25_index.json")
    rerank_enabled: bool = True
    hybrid_topn: int = 50
    rerank_candidates_n: int = 10
    final_context_k: int = 4
    max_context_tokens: int = 1200
    rerank_cache_ttl_seconds: int = 86400
    rerank_backend: str = "cross-encoder"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    constraint_mode: str = "soft"
    time_query_require_time_evidence: bool = True
    
    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    
    # Logging
    debug: bool = False
    
    # Notes export
    notes_export_dir: Optional[Path] = None
    auto_ingest_on_startup: bool = False
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    def __init__(self, **data):
        super().__init__(**data)
        # Create Chroma DB path if it doesn't exist
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
