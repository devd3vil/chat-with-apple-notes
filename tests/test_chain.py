"""
Tests for RAGChain and Chunker.

Integration and unit tests for the RAG pipeline.
"""

import pytest
from src.chain import RAGChain, Chunker
from src.models import Citation, Chunk
from src.store import InMemoryStore
from src.embedder import FakeEmbedder
from src.llm import FakeLLM


class TestChunker:
    """Test Chunker class."""
    
    def test_chunk_text_basic(self):
        """Chunker should split text into chunks."""
        chunker = Chunker(chunk_size=10, chunk_overlap=2)
        text = "This is a test text that should be chunked"
        
        chunks = chunker.chunk_text(text, "note1")
        
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
    
    def test_chunk_text_respects_chunk_size(self):
        """Chunks should not exceed chunk_size."""
        chunker = Chunker(chunk_size=20, chunk_overlap=5)
        text = "x" * 100
        
        chunks = chunker.chunk_text(text, "note1")
        
        for chunk in chunks:
            assert len(chunk.text) <= 20
    
    def test_chunk_text_includes_overlap(self):
        """Chunks should overlap by specified amount."""
        chunker = Chunker(chunk_size=10, chunk_overlap=3)
        text = "0123456789abcdefghij"
        
        chunks = chunker.chunk_text(text, "note1")
        
        # First chunk: chars 0-9
        assert chunks[0].text == text[0:10]
        # Second chunk should start at position 7 (10 - 3)
        if len(chunks) > 1:
            assert chunks[1].start_char == 7
    
    def test_chunk_text_empty_input(self):
        """Chunker should return empty list for empty text."""
        chunker = Chunker()
        chunks = chunker.chunk_text("", "note1")
        
        assert chunks == []
    
    def test_chunk_text_preserves_note_id(self):
        """Chunks should preserve note_id."""
        chunker = Chunker()
        chunks = chunker.chunk_text("test text", "my_note_123")
        
        for chunk in chunks:
            assert chunk.note_id == "my_note_123"
    
    def test_chunk_text_generates_unique_ids(self):
        """Each chunk should have a unique ID."""
        chunker = Chunker(chunk_size=5, chunk_overlap=1)
        text = "x" * 20
        
        chunks = chunker.chunk_text(text, "note1")
        chunk_ids = [c.id for c in chunks]
        
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique
    
    def test_chunk_text_small_overlap(self):
        """Chunker should handle small overlap."""
        chunker = Chunker(chunk_size=10, chunk_overlap=1)
        text = "a" * 20
        
        chunks = chunker.chunk_text(text, "note1")
        
        assert len(chunks) >= 2
        # Each chunk should be at most 10 chars
        for chunk in chunks:
            assert len(chunk.text) <= 10


class TestRAGChain:
    """Test RAGChain with fake components."""
    
    @pytest.fixture
    def setup_components(self):
        """Setup RAG components for testing."""
        embedder = FakeEmbedder(dimension=3)
        store = InMemoryStore()
        llm = FakeLLM()
        
        return embedder, store, llm
    
    def test_ask_returns_qa_result(self, setup_components):
        """ask() should return a QAResult."""
        embedder, store, llm = setup_components
        from src.retriever import Retriever
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        llm.set_response("The answer is 42")
        result = chain.ask("What is the answer?")
        
        assert result.query == "What is the answer?"
        assert "42" in result.answer
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0
    
    def test_ask_raises_on_empty_query(self, setup_components):
        """ask() should raise ValueError for empty query."""
        embedder, store, llm = setup_components
        from src.retriever import Retriever
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        with pytest.raises(ValueError, match="empty query"):
            chain.ask("")
    
    def test_ask_with_no_retrieved_chunks(self, setup_components):
        """ask() should handle when no chunks are retrieved."""
        embedder, store, llm = setup_components
        from src.retriever import Retriever
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        result = chain.ask("This will match nothing")
        
        assert result.citations == []
        assert result.confidence == 0.0
        assert "could not find" in result.answer.lower()
    
    def test_ask_includes_retrieved_chunks_in_prompt(self, setup_components):
        """ask() should pass retrieved chunks to LLM prompt."""
        embedder, store, llm = setup_components
        from src.retriever import Retriever
        
        # Add a chunk to store
        chunk_embedding = embedder.embed("test chunk")
        store.add("chunk1", "This is important info", chunk_embedding)
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        # Use a custom LLM to capture the prompt
        received_prompts = []
        
        class CaptureLLM(FakeLLM):
            def generate(self, prompt, max_tokens=512):
                received_prompts.append(prompt)
                return "[1] The important info."
        
        capture_llm = CaptureLLM()
        chain_with_capture = RAGChain(retriever=retriever, llm=capture_llm)
        
        chain_with_capture.ask("What is important?")
        
        assert len(received_prompts) > 0
        # Check that the chunk text is in the prompt
        assert "This is important info" in received_prompts[0]
    
    def test_extract_citations_from_answer(self, setup_components):
        """ask() should extract citations from LLM answer."""
        embedder, store, llm = setup_components
        from src.retriever import Retriever
        
        # Add chunks to store
        chunk1_embedding = embedder.embed("Python is great")
        chunk2_embedding = embedder.embed("Python is versatile")
        store.add("chunk1", "Python is great", chunk1_embedding)
        store.add("chunk2", "Python is versatile", chunk2_embedding)
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        llm.set_response("Python [1] is [2] popular.")
        result = chain.ask("Tell me about Python")
        
        assert len(result.citations) >= 1
        # Should include the cited chunks
        cited_texts = [c.text for c in result.citations]
        assert "Python is great" in cited_texts or "Python is versatile" in cited_texts
    
    def test_confidence_score_based_on_retrieval(self, setup_components):
        """Confidence should reflect retrieval quality."""
        embedder, store, llm = setup_components
        from src.retriever import Retriever
        
        # Add high-quality chunk
        chunk_embedding = embedder.embed("exact answer")
        store.add("chunk1", "exact answer", chunk_embedding)
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        llm.set_response("The answer is yes.")
        result = chain.ask("exact answer")
        
        # Should have reasonable confidence from retrieval
        assert result.confidence > 0.5
    
    def test_confidence_reduced_without_citations(self, setup_components):
        """Confidence should be reduced if LLM answer has no citations."""
        embedder, store, llm = setup_components
        from src.retriever import Retriever
        
        chunk_embedding = embedder.embed("test")
        store.add("chunk1", "test info", chunk_embedding)
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        # Answer without citations
        llm.set_response("Some answer without citations")
        result = chain.ask("test")
        
        # Confidence should be penalized
        assert result.confidence < 0.9
    
    def test_build_prompt_format(self, setup_components):
        """_build_prompt should format snippets correctly."""
        embedder, store, llm = setup_components
        from src.retriever import Retriever
        
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
    
    def test_extract_citations_preserves_order(self, setup_components):
        """Extracted citations should preserve order of indices."""
        embedder, store, llm = setup_components
        from src.retriever import Retriever
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        citations = [
            Citation(chunk_id="a", text="First", score=0.9),
            Citation(chunk_id="b", text="Second", score=0.8),
            Citation(chunk_id="c", text="Third", score=0.7),
        ]
        
        # Answer references [2] then [1]
        answer = "Second [2] and first [1]"
        
        result = chain._extract_citations(answer, citations)
        
        # Should be in order of appearance in answer: [2] then [1]
        assert len(result) == 2
        assert result[0].chunk_id == "b"  # [2] -> index 1
        assert result[1].chunk_id == "a"  # [1] -> index 0
    
    def test_extract_citations_removes_duplicates(self, setup_components):
        """Extracted citations should not include duplicates."""
        embedder, store, llm = setup_components
        from src.retriever import Retriever
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        citations = [
            Citation(chunk_id="a", text="Info", score=0.9),
        ]
        
        # Answer mentions [1] multiple times
        answer = "[1] says that [1] is true and [1] again"
        
        result = chain._extract_citations(answer, citations)
        
        # Should only include [1] once
        assert len(result) == 1
    
    def test_extract_citations_ignores_invalid_indices(self, setup_components):
        """Extracted citations should skip invalid indices."""
        embedder, store, llm = setup_components
        from src.retriever import Retriever
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        citations = [
            Citation(chunk_id="a", text="Only one", score=0.9),
        ]
        
        # Answer cites valid [1] and invalid [99]
        answer = "Valid [1] but also [99]"
        
        result = chain._extract_citations(answer, citations)
        
        # Should only include valid [1]
        assert len(result) == 1
        assert result[0].chunk_id == "a"
    
    def test_ask_integration_with_fakes(self, setup_components):
        """Full integration test with FakeEmbedder and FakeLLM."""
        embedder, store, llm = setup_components
        from src.retriever import Retriever
        
        # Add test data
        chunks = [
            ("chunk1", "Machine learning is powerful", embedder.embed("Machine learning")),
            ("chunk2", "Deep learning is a subset of ML", embedder.embed("Deep learning")),
        ]
        
        for chunk_id, text, embedding in chunks:
            store.add(chunk_id, text, embedding)
        
        # Setup chain
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm, top_k=2)
        
        # Set LLM response
        llm.set_response("Machine learning [1] and deep learning [2] are important.")
        
        # Ask question
        result = chain.ask("What is machine learning?")
        
        # Verify result
        assert result.query == "What is machine learning?"
        assert len(result.citations) >= 1
        assert result.confidence > 0.0
        assert "[1]" in result.answer


class TestRAGChainEdgeCases:
    """Test edge cases for RAGChain."""
    
    def test_ask_with_whitespace_only_query(self):
        """ask() should treat whitespace-only queries as empty."""
        from src.retriever import Retriever
        from src.store import InMemoryStore
        from src.embedder import FakeEmbedder
        from src.llm import FakeLLM
        
        embedder = FakeEmbedder(dimension=3)
        store = InMemoryStore()
        llm = FakeLLM()
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        with pytest.raises(ValueError):
            chain.ask("   \t\n  ")
    
    def test_calculate_confidence_bounds(self):
        """Confidence should always be in [0, 1]."""
        from src.retriever import Retriever
        from src.store import InMemoryStore
        from src.embedder import FakeEmbedder
        from src.llm import FakeLLM
        
        embedder = FakeEmbedder(dimension=3)
        store = InMemoryStore()
        llm = FakeLLM()
        
        # Add high-quality chunks
        for i in range(5):
            store.add(f"chunk{i}", f"Info {i}", embedder.embed(f"Query {i}"))
        
        retriever = Retriever(store=store, embedder=embedder)
        chain = RAGChain(retriever=retriever, llm=llm)
        
        llm.set_response("Very detailed answer with extensive citations [1] [2] [3] [4] [5].")
        result = chain.ask("complex query")
        
        # Even with perfect retrieval and citations, should be <= 1.0
        assert 0.0 <= result.confidence <= 1.0
