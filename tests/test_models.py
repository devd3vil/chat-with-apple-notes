"""
Tests for data models.
"""

import pytest
from datetime import datetime
from src.models import Note, Chunk, Citation, QAResult


class TestNote:
    """Tests for Note model."""
    
    def test_note_creation(self):
        """Test creating a valid Note."""
        note = Note(
            id="p123",
            title="Test Note",
            body="<html>Test body</html>",
            folder="Notes",
            created_at=datetime.now(),
            modified_at=datetime.now(),
        )
        assert note.id == "p123"
        assert note.title == "Test Note"
        assert note.folder == "Notes"
    
    def test_note_default_folder(self):
        """Test Note with default folder."""
        note = Note(
            id="p123",
            title="Test Note",
            body="<html>Test body</html>",
            created_at=datetime.now(),
            modified_at=datetime.now(),
        )
        assert note.folder == "Notes"
    
    def test_note_with_nested_folder(self):
        """Test Note with nested folder path."""
        note = Note(
            id="p123",
            title="Test Note",
            body="<html>Test body</html>",
            folder="Work/Projects/Python",
            created_at=datetime.now(),
            modified_at=datetime.now(),
        )
        assert note.folder == "Work/Projects/Python"


class TestChunk:
    """Tests for Chunk model."""
    
    def test_chunk_creation(self):
        """Test creating a valid Chunk."""
        chunk = Chunk(
            id="p123_0",
            note_id="p123",
            text="This is the first chunk",
            chunk_idx=0,
            start_char=0,
            end_char=24,
        )
        assert chunk.id == "p123_0"
        assert chunk.note_id == "p123"
        assert chunk.chunk_idx == 0
    
    def test_chunk_with_embedding(self):
        """Test Chunk with embedding vector."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        chunk = Chunk(
            id="p123_0",
            note_id="p123",
            text="This is the first chunk",
            chunk_idx=0,
            start_char=0,
            end_char=24,
            embedding=embedding,
        )
        assert chunk.embedding == embedding


class TestCitation:
    """Tests for Citation model."""
    
    def test_citation_creation(self):
        """Test creating a valid Citation."""
        citation = Citation(
            chunk_id="p123_0",
            text="Example citation text",
            score=0.95,
        )
        assert citation.chunk_id == "p123_0"
        assert citation.score == 0.95
    
    def test_citation_score_bounds(self):
        """Test that citation score is bounded [0, 1]."""
        # Valid scores
        Citation(chunk_id="p123_0", text="text", score=0.0)
        Citation(chunk_id="p123_0", text="text", score=1.0)
        Citation(chunk_id="p123_0", text="text", score=0.5)
        
        # Invalid scores should raise validation error
        with pytest.raises(ValueError):
            Citation(chunk_id="p123_0", text="text", score=-0.1)
        
        with pytest.raises(ValueError):
            Citation(chunk_id="p123_0", text="text", score=1.1)


class TestQAResult:
    """Tests for QAResult model."""
    
    def test_qa_result_creation(self):
        """Test creating a valid QAResult."""
        citations = [
            Citation(chunk_id="p123_0", text="text 1", score=0.95),
            Citation(chunk_id="p123_1", text="text 2", score=0.87),
        ]
        result = QAResult(
            query="What is this?",
            answer="This is an answer [1][2].",
            citations=citations,
            confidence=0.91,
        )
        assert result.query == "What is this?"
        assert len(result.citations) == 2
        assert result.confidence == 0.91
    
    def test_qa_result_empty_citations(self):
        """Test QAResult with no citations."""
        result = QAResult(
            query="What is this?",
            answer="I don't know.",
            citations=[],
            confidence=0.0,
        )
        assert len(result.citations) == 0
    
    def test_qa_result_confidence_bounds(self):
        """Test that confidence is bounded [0, 1]."""
        citations = [Citation(chunk_id="p123_0", text="text", score=0.95)]
        
        # Valid confidence
        QAResult(
            query="q", answer="a", citations=citations, confidence=0.0
        )
        QAResult(
            query="q", answer="a", citations=citations, confidence=1.0
        )
        
        # Invalid confidence
        with pytest.raises(ValueError):
            QAResult(
                query="q", answer="a", citations=citations, confidence=-0.1
            )
        
        with pytest.raises(ValueError):
            QAResult(
                query="q", answer="a", citations=citations, confidence=1.1
            )


class TestModelSerialization:
    """Tests for model serialization/deserialization."""
    
    def test_note_json_roundtrip(self):
        """Test Note JSON serialization and deserialization."""
        note = Note(
            id="p123",
            title="Test Note",
            body="<html>Test body</html>",
            folder="Work",
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            modified_at=datetime(2024, 1, 15, 10, 30, 0),
        )
        json_str = note.model_dump_json()
        note_restored = Note.model_validate_json(json_str)
        assert note_restored.id == note.id
        assert note_restored.title == note.title
    
    def test_qa_result_json_roundtrip(self):
        """Test QAResult JSON serialization."""
        citations = [
            Citation(chunk_id="p123_0", text="text 1", score=0.95),
        ]
        result = QAResult(
            query="What is this?",
            answer="This is an answer [1].",
            citations=citations,
            confidence=0.91,
        )
        json_str = result.model_dump_json()
        result_restored = QAResult.model_validate_json(json_str)
        assert result_restored.query == result.query
        assert len(result_restored.citations) == 1
