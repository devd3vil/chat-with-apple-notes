"""
Tests for RAGChain and Chunker.

Fast, focused tests for the RAG pipeline.
"""

import pytest
from unittest.mock import Mock
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
        
        chunk = Chunk(
            id="note1_0",
            note_id="note1",
            text="The answer is 42",
            chunk_idx=0,
            start_char=0,
            end_char=15,
            embedding=embedder.embed("The answer is 42"),
        )
        store.add_chunks([chunk])
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        llm.set_response("The answer is 42 [1]")
        result = chain.ask("The answer is 42")
        
        assert result.query == "The answer is 42"
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

    def test_min_score_filters_irrelevant_chunks(self):
        """Chunks below min_score should not be used."""
        embedder = FakeEmbedder(dimension=3)
        store = InMemoryStore()
        llm = FakeLLM()

        chunk = Chunk(
            id="note1_0",
            note_id="note1",
            text="Unrelated snippet",
            chunk_idx=0,
            start_char=0,
            end_char=17,
            embedding=[0.0, 0.0, 0.0],
        )
        store.add_chunks([chunk])

        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm, min_score=0.8)

        result = chain.ask("Completely different query")

        assert result.citations == []
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

    def test_keeps_llm_answer_when_llm_omits_citations(self):
        """Chain should preserve answer text and append a default citation."""
        embedder = FakeEmbedder(dimension=3)
        store = InMemoryStore()
        llm = FakeLLM()
        llm.set_response("Answer without citations.")
        
        chunk = Chunk(
            id="note1_0",
            note_id="note1",
            text="Estes Riverwalk is scenic.",
            chunk_idx=0,
            start_char=0,
            end_char=27,
            embedding=embedder.embed("Estes Riverwalk is scenic."),
        )
        store.add_chunks([chunk])
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        result = chain.ask("Estes Riverwalk is scenic.")
        
        assert result.citations
        assert "Answer without citations." in result.answer
        assert result.answer.strip().endswith("[1]")
        assert "Most relevant snippets:" not in result.answer

    def test_fallback_answer_used_when_llm_returns_empty_text(self):
        """Fallback snippet answer is used only when model text is empty."""
        embedder = FakeEmbedder(dimension=3)
        store = InMemoryStore()
        llm = FakeLLM()
        llm.set_response("")

        chunk = Chunk(
            id="note1_0",
            note_id="note1",
            text="Estes Riverwalk is scenic.",
            chunk_idx=0,
            start_char=0,
            end_char=27,
            embedding=embedder.embed("Estes Riverwalk is scenic."),
        )
        store.add_chunks([chunk])

        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)

        result = chain.ask("Estes Riverwalk is scenic.")

        assert result.citations
        assert result.answer.startswith("Most relevant snippets:")
        assert "[1]" in result.answer

    def test_ask_calls_retriever_and_llm_with_top_k(self):
        """ask() should retrieve top-k and pass snippets into the prompt."""
        retriever = Mock()
        llm = Mock()
        citations = [
            Citation(chunk_id="c1", text="Snippet one", score=0.92),
            Citation(chunk_id="c2", text="Snippet two", score=0.88),
        ]
        retriever.retrieve.return_value = citations
        llm.generate.return_value = "Answer [1]"

        chain = RAGChain(retriever=retriever, llm=llm, top_k=5, min_score=0.8)

        result = chain.ask("What is in my notes?")

        retriever.retrieve.assert_called_once_with("What is in my notes?", top_k=5)
        assert llm.generate.called
        prompt_arg = llm.generate.call_args[0][0]
        assert "Snippet one" in prompt_arg
        assert "Snippet two" in prompt_arg
        assert "What is in my notes?" in prompt_arg
        assert result.citations
