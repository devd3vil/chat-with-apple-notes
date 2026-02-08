"""
Data models for the RAG Q&A system.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class Note(BaseModel):
    """Represents an Apple Note."""
    
    id: str = Field(..., description="Unique note ID from Apple Notes")
    title: str = Field(..., description="Note title")
    body: str = Field(..., description="Note body (HTML)")
    folder: str = Field(default="Notes", description="Folder path in Notes app")
    created_at: datetime = Field(..., description="Note creation timestamp")
    modified_at: datetime = Field(..., description="Note last modified timestamp")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "p599",
                "title": "My First Note",
                "body": "<html>...</html>",
                "folder": "Notes/Work",
                "created_at": "2024-01-15T10:30:00",
                "modified_at": "2024-01-15T10:30:00",
            }
        }
    )


class Chunk(BaseModel):
    """Represents a chunk of a note (for vectorization)."""
    
    id: str = Field(..., description="Unique chunk ID (format: note_id_chunk_idx)")
    note_id: str = Field(..., description="Parent note ID")
    text: str = Field(..., description="Chunk text content")
    chunk_idx: int = Field(..., description="Index of chunk within note (0-based)")
    start_char: int = Field(..., description="Start character offset in note body")
    end_char: int = Field(..., description="End character offset in note body")
    embedding: Optional[list[float]] = Field(None, description="Vector embedding")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "p599_0",
                "note_id": "p599",
                "text": "This is the first chunk of text...",
                "chunk_idx": 0,
                "start_char": 0,
                "end_char": 512,
                "embedding": [0.1, 0.2, ...],
            }
        }
    )


class Citation(BaseModel):
    """Represents a citation in the RAG response."""
    
    chunk_id: str = Field(..., description="ID of the cited chunk")
    note_id: Optional[str] = Field(
        default=None,
        description="Stable source note ID for this citation.",
    )
    text: str = Field(..., description="Text snippet from the chunk")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")
    source: Optional[dict[str, str]] = Field(
        default=None,
        description="Additional stable source metadata for the citation.",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_id": "p599_0",
                "note_id": "p599",
                "text": "Apple Notes is a note-taking app...",
                "score": 0.95,
                "source": {"note_id": "p599"},
            }
        }
    )


class QAResult(BaseModel):
    """Represents the result of a Q&A query."""
    
    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Generated answer from LLM")
    citations: list[Citation] = Field(..., description="Top-k retrieved chunks with scores")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence (0-1)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is Apple Notes?",
                "answer": "Apple Notes is a note-taking app [1]. It allows you to organize notes in folders [2].",
                "citations": [
                    {"chunk_id": "p599_0", "text": "Apple Notes is a note-taking app", "score": 0.95},
                    {"chunk_id": "p600_1", "text": "You can organize notes in folders", "score": 0.87},
                ],
                "confidence": 0.91,
            }
        }
    )
