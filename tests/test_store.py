"""
Tests for vector store implementations.
"""

import tempfile
from datetime import datetime
from pathlib import Path
import pytest
from src.models import Chunk, Note
from src.embedder import FakeEmbedder
from src.store import InMemoryStore, VectorStore


class TestInMemoryStore:
    """Tests for InMemoryStore."""
    
    @pytest.fixture
    def embedder(self):
        """Create a fake embedder."""
        return FakeEmbedder(embedding_dim=384)
    
    @pytest.fixture
    def store(self, embedder):
        """Create an in-memory store."""
        return InMemoryStore(embedder)
    
    def test_store_creation(self, store):
        """Test creating an in-memory store."""
        assert store.size() == 0
    
    def test_add_single_chunk(self, store, embedder):
        """Test adding a single chunk."""
        chunk = Chunk(
            id="p1_0",
            note_id="p1",
            text="Hello world",
            chunk_idx=0,
            start_char=0,
            end_char=11,
            embedding=embedder.embed("Hello world"),
        )
        
        store.add_chunks([chunk])
        assert store.size() == 1
    
    def test_add_multiple_chunks(self, store, embedder):
        """Test adding multiple chunks."""
        chunks = [
            Chunk(
                id=f"p1_{i}",
                note_id="p1",
                text=f"Chunk {i}",
                chunk_idx=i,
                start_char=i * 10,
                end_char=(i + 1) * 10,
                embedding=embedder.embed(f"Chunk {i}"),
            )
            for i in range(5)
        ]
        
        store.add_chunks(chunks)
        assert store.size() == 5
    
    def test_add_chunk_without_embedding_raises_error(self, store):
        """Test that adding chunk without embedding raises error."""
        chunk = Chunk(
            id="p1_0",
            note_id="p1",
            text="Text",
            chunk_idx=0,
            start_char=0,
            end_char=4,
            embedding=None,
        )
        
        with pytest.raises(ValueError):
            store.add_chunks([chunk])
    
    def test_search_single_chunk(self, store, embedder):
        """Test searching with single chunk in store."""
        text = "Hello world"
        chunk = Chunk(
            id="p1_0",
            note_id="p1",
            text=text,
            chunk_idx=0,
            start_char=0,
            end_char=len(text),
            embedding=embedder.embed(text),
        )
        
        store.add_chunks([chunk])
        
        query_embedding = embedder.embed(text)
        results = store.search(query_embedding, top_k=5)
        
        assert len(results) == 1
        chunk_id, chunk_text, score = results[0]
        assert chunk_id == "p1_0"
        assert chunk_text == text
        assert 0 <= score <= 1
    
    def test_search_similarity_high_for_identical_text(self, store, embedder):
        """Test that identical text has high similarity."""
        text = "The quick brown fox"
        chunk = Chunk(
            id="p1_0",
            note_id="p1",
            text=text,
            chunk_idx=0,
            start_char=0,
            end_char=len(text),
            embedding=embedder.embed(text),
        )
        
        store.add_chunks([chunk])
        
        # Search with exact same text
        query_embedding = embedder.embed(text)
        results = store.search(query_embedding, top_k=1)
        
        assert len(results) == 1
        _, _, score = results[0]
        # Should be very high (close to 1.0)
        assert score > 0.9
    
    def test_search_top_k_respected(self, store, embedder):
        """Test that top_k parameter is respected."""
        chunks = [
            Chunk(
                id=f"p1_{i}",
                note_id="p1",
                text=f"Text number {i}",
                chunk_idx=i,
                start_char=i * 15,
                end_char=(i + 1) * 15,
                embedding=embedder.embed(f"Text number {i}"),
            )
            for i in range(10)
        ]
        
        store.add_chunks(chunks)
        
        query_embedding = embedder.embed("Text")
        
        for k in [1, 3, 5, 10]:
            results = store.search(query_embedding, top_k=k)
            assert len(results) <= k
    
    def test_search_empty_store(self, store, embedder):
        """Test searching empty store."""
        query_embedding = embedder.embed("query")
        results = store.search(query_embedding, top_k=5)
        
        assert results == []
    
    def test_search_returns_sorted_by_score(self, store, embedder):
        """Test that results are sorted by score (descending)."""
        chunks = [
            Chunk(
                id=f"p1_{i}",
                note_id="p1",
                text=f"Text {i}",
                chunk_idx=i,
                start_char=i * 10,
                end_char=(i + 1) * 10,
                embedding=embedder.embed(f"Text {i}"),
            )
            for i in range(5)
        ]
        
        store.add_chunks(chunks)
        
        query_embedding = embedder.embed("Text")
        results = store.search(query_embedding, top_k=5)
        
        # Check scores are sorted descending
        scores = [score for _, _, score in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_delete_note(self, store, embedder):
        """Test deleting all chunks of a note."""
        # Add chunks from two notes
        chunks_p1 = [
            Chunk(
                id=f"p1_{i}",
                note_id="p1",
                text=f"P1 Text {i}",
                chunk_idx=i,
                start_char=i * 10,
                end_char=(i + 1) * 10,
                embedding=embedder.embed(f"P1 Text {i}"),
            )
            for i in range(3)
        ]
        chunks_p2 = [
            Chunk(
                id=f"p2_{i}",
                note_id="p2",
                text=f"P2 Text {i}",
                chunk_idx=i,
                start_char=i * 10,
                end_char=(i + 1) * 10,
                embedding=embedder.embed(f"P2 Text {i}"),
            )
            for i in range(2)
        ]
        
        store.add_chunks(chunks_p1 + chunks_p2)
        assert store.size() == 5
        
        # Delete note p1
        store.delete_note("p1")
        assert store.size() == 2
        
        # Search should only return p2 chunks
        query_embedding = embedder.embed("Text")
        results = store.search(query_embedding, top_k=10)
        assert all(chunk_id.startswith("p2_") for chunk_id, _, _ in results)
    
    def test_delete_nonexistent_note(self, store, embedder):
        """Test deleting a note that doesn't exist."""
        chunk = Chunk(
            id="p1_0",
            note_id="p1",
            text="Text",
            chunk_idx=0,
            start_char=0,
            end_char=4,
            embedding=embedder.embed("Text"),
        )
        
        store.add_chunks([chunk])
        
        # Should not raise error
        store.delete_note("p999")
        assert store.size() == 1
    
    def test_clear_store(self, store, embedder):
        """Test clearing all chunks."""
        chunks = [
            Chunk(
                id=f"p1_{i}",
                note_id="p1",
                text=f"Text {i}",
                chunk_idx=i,
                start_char=i * 10,
                end_char=(i + 1) * 10,
                embedding=embedder.embed(f"Text {i}"),
            )
            for i in range(5)
        ]
        
        store.add_chunks(chunks)
        assert store.size() == 5
        
        store.clear()
        assert store.size() == 0
        
        # Search should return empty
        results = store.search(embedder.embed("query"), top_k=5)
        assert results == []
    
    def test_upsert_behavior(self, store, embedder):
        """Test that adding same chunk ID overwrites."""
        chunk1 = Chunk(
            id="p1_0",
            note_id="p1",
            text="Original text",
            chunk_idx=0,
            start_char=0,
            end_char=13,
            embedding=embedder.embed("Original text"),
        )
        
        chunk2 = Chunk(
            id="p1_0",
            note_id="p1",
            text="Updated text",
            chunk_idx=0,
            start_char=0,
            end_char=12,
            embedding=embedder.embed("Updated text"),
        )
        
        store.add_chunks([chunk1])
        assert store.size() == 1
        
        store.add_chunks([chunk2])
        assert store.size() == 1  # Still 1, not 2
    
    def test_search_result_format(self, store, embedder):
        """Test that search results have correct format."""
        chunk = Chunk(
            id="p1_0",
            note_id="p1",
            text="Sample text",
            chunk_idx=0,
            start_char=0,
            end_char=11,
            embedding=embedder.embed("Sample text"),
        )
        
        store.add_chunks([chunk])
        
        results = store.search(embedder.embed("Sample"), top_k=1)
        
        assert len(results) == 1
        chunk_id, chunk_text, score = results[0]
        
        assert isinstance(chunk_id, str)
        assert isinstance(chunk_text, str)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestVectorStoreInterface:
    """Tests for VectorStore interface contract."""
    
    def test_vector_store_is_abstract(self):
        """Test that VectorStore cannot be instantiated."""
        with pytest.raises(TypeError):
            VectorStore()
    
    def test_in_memory_store_implements_interface(self):
        """Test that InMemoryStore implements VectorStore interface."""
        embedder = FakeEmbedder()
        store = InMemoryStore(embedder)
        
        assert isinstance(store, VectorStore)
        assert hasattr(store, "add_chunks")
        assert hasattr(store, "search")
        assert hasattr(store, "delete_note")
        assert hasattr(store, "clear")
        assert hasattr(store, "size")


class TestInMemoryStoreEdgeCases:
    """Tests for edge cases in InMemoryStore."""
    
    def test_search_with_zero_top_k(self):
        """Test search with top_k=0."""
        embedder = FakeEmbedder()
        store = InMemoryStore(embedder)
        
        chunk = Chunk(
            id="p1_0",
            note_id="p1",
            text="Text",
            chunk_idx=0,
            start_char=0,
            end_char=4,
            embedding=embedder.embed("Text"),
        )
        
        store.add_chunks([chunk])
        results = store.search(embedder.embed("Text"), top_k=0)
        
        assert results == []
    
    def test_search_with_very_large_top_k(self):
        """Test search with top_k larger than store size."""
        embedder = FakeEmbedder()
        store = InMemoryStore(embedder)
        
        chunks = [
            Chunk(
                id=f"p1_{i}",
                note_id="p1",
                text=f"Text {i}",
                chunk_idx=i,
                start_char=i * 10,
                end_char=(i + 1) * 10,
                embedding=embedder.embed(f"Text {i}"),
            )
            for i in range(3)
        ]
        
        store.add_chunks(chunks)
        results = store.search(embedder.embed("Text"), top_k=1000)
        
        # Should return all chunks
        assert len(results) == 3
    
    def test_add_empty_chunk_list(self):
        """Test adding empty chunk list."""
        embedder = FakeEmbedder()
        store = InMemoryStore(embedder)
        
        store.add_chunks([])
        assert store.size() == 0
    
    def test_cosine_similarity_normalization(self):
        """Test that cosine similarity is properly normalized."""
        embedder = FakeEmbedder()
        store = InMemoryStore(embedder)
        
        # Add multiple chunks
        chunks = [
            Chunk(
                id=f"p1_{i}",
                note_id="p1",
                text=f"Different text {i}",
                chunk_idx=i,
                start_char=i * 20,
                end_char=(i + 1) * 20,
                embedding=embedder.embed(f"Different text {i}"),
            )
            for i in range(5)
        ]
        
        store.add_chunks(chunks)
        results = store.search(embedder.embed("query"), top_k=5)
        
        # All scores should be between 0 and 1
        for _, _, score in results:
            assert 0.0 <= score <= 1.0
