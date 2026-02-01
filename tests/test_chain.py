"""
Tests for RAGChain and Chunker.

Fast, focused tests for the RAG pipeline.
"""

import pytest
from src.chain import RAGChain, Chunker
from src.models import Citation, Chunk
from src.store import InMemoryStore
from src.embedder import FakeEmbedder
from src.llm import FakeLLM
from src.retriever import Retriever


class TestChunker:
    """Test Chunker class."""
    
    def test_chunk_text_basic(self):
        """Chunker should split text into chunks."""
        chunker = Chunker(chunk_size=10, chunk_overlap=2)
        text = "This is a test"
        
        chunks = chunker.chunk_text(text, "note1")
        
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
    
    def test_chunk_text_empty_returns_empty(self):
        """Chunker should return empty list for empty text."""
        chunker = Chunker()
        chunks = chunker.chunk_text("", "note1")
        
        assert chunks == []
    
    def test_chunk_text_respects_chunk_size(self):
        """Chunks should not exceed chunk_size."""
        chunker = Chunker(chunk_size=10, chunk_overlap=2)
        text = "x" * 30
        
        chunks = chunker.chunk_text(text, "note1")
        
        for chunk in chunks:
            assert len(chunk.text) <= 10
    
    def test_chunk_text_preserves_note_id(self):
        """Chunks should preserve note_id."""
        chunker = Chunker()
        chunks = chunker.chunk_text("test text", "my_note_123")
        
        for chunk in chunks:
            assert chunk.note_id == "my_note_123"


class TestRAGChain:
    """Test RAGChain with fake components."""
    
    def test_ask_returns_qa_result(self):
        """ask() should return a QAResult."""
        embedder = FakeEmbedder(dimension=3)
        store = InMemoryStore()
        llm = FakeLLM()
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        llm.set_response("The answer is 42")
        result = chain.ask("What is the answer?")
        
        assert result.query == "What is the answer?"
        assert "42" in result.answer
        assert 0.0 <= result.confidence <= 1.0
    
    def test_ask_raises_on_empty_query(self):
        """ask() should raise ValueError for empty query."""
        embedder = FakeEmbedder(dimension=3)
        store = InMemoryStore()
        llm = FakeLLM()
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        with pytest.raises(ValueError, match="empty query"):
            chain.ask("")
    
    def test_ask_with_no_retrieved_chunks(self):
        """ask() should handle when no chunks are retrieved."""
        embedder = FakeEmbedder(dimension=3)
        store = InMemoryStore()
        llm = FakeLLM()
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        result = chain.ask("This will match nothing")
        
        assert result.citations == []
        assert result.confidence == 0.0
        assert "could not find" in result.answer.lower()
    
    def test_build_prompt_includes_citations(self):
        """_build_prompt should format snippets correctly."""
        embedder = FakeEmbedder(dimension=3)
        store = InMemoryStore()
        llm = FakeLLM()
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        citations = [
            Citation(chunk_id="1", text="First snippet", score=0.9),
            Citation(chunk_id="2", text="Second snippet", score=0.8),
        ]
        
        prompt = chain._build_prompt("test query", citations)
        
        assert "test query" in prompt
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "First snippet" in prompt
        assert "Second snippet" in prompt
    
    def test_extract_citations_basic(self):
        """_extract_citations should extract citation indices from answer."""
        embedder = FakeEmbedder(dimension=3)
        store = InMemoryStore()
        llm = FakeLLM()
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        citations = [
            Citation(chunk_id="a", text="First", score=0.9),
            Citation(chunk_id="b", text="Second", score=0.8),
        ]
        
        answer = "Answer [1] and [2]"
        result = chain._extract_citations(answer, citations)
        
        assert len(result) == 2
        assert result[0].chunk_id == "a"
        assert result[1].chunk_id == "b"
    
    def test_extract_citations_removes_duplicates(self):
        """_extract_citations should remove duplicate indices."""
        embedder = FakeEmbedder(dimension=3)
        store = InMemoryStore()
        llm = FakeLLM()
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        citations = [Citation(chunk_id="a", text="Info", score=0.9)]
        answer = "[1] says [1] is true and [1] again"
        
        result = chain._extract_citations(answer, citations)
        
        assert len(result) == 1
    
    def test_extract_citations_ignores_invalid(self):
        """_extract_citations should skip invalid indices."""
        embedder = FakeEmbedder(dimension=3)
        store = InMemoryStore()
        llm = FakeLLM()
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        citations = [Citation(chunk_id="a", text="Only one", score=0.9)]
        answer = "Valid [1] but also [99]"
        
        result = chain._extract_citations(answer, citations)
        
        assert len(result) == 1
        assert result[0].chunk_id == "a"
    
    def test_confidence_in_bounds(self):
        """Confidence should always be in [0, 1]."""
        embedder = FakeEmbedder(dimension=3)
        store = InMemoryStore()
        llm = FakeLLM()
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        llm.set_response("Answer")
        result = chain.ask("Query")
        
        assert 0.0 <= result.confidence <= 1.0
