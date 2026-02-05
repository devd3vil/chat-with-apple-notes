"""
Tests for Retriever class.

Tests basic retrieval with mocked embedder and store.
"""

import re
import pytest
from unittest.mock import Mock
from src.retriever import Retriever, HybridRetriever
from src.bm25 import BM25Index
from src.models import Chunk
from src.models import Citation


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        note_id=chunk_id.split("_", 1)[0],
        text=text,
        chunk_idx=0,
        start_char=0,
        end_char=len(text),
        embedding=None,
    )


class TestRetriever:
    """Test Retriever class."""
    
    def test_retrieve_returns_citations_from_store(self):
        """Retriever should wrap store search results as Citations."""
        # Setup mocks
        embedder = Mock()
        embedder.embed.return_value = [0.1, 0.2, 0.3]
        
        store = Mock()
        store.search.return_value = [
            ("chunk1", "Some text about Python", 0.95),
            ("chunk2", "Python is great", 0.87),
        ]
        
        # Create retriever
        retriever = Retriever(store=store, embedder=embedder)
        
        # Retrieve
        results = retriever.retrieve("Python", top_k=2)
        
        # Verify
        assert len(results) == 2
        assert isinstance(results[0], Citation)
        assert results[0].chunk_id == "chunk1"
        assert results[0].text == "Some text about Python"
        assert results[0].score == 0.95
        assert results[1].chunk_id == "chunk2"
        assert results[1].text == "Python is great"
        assert results[1].score == 0.87
    
    def test_retrieve_embeds_query(self):
        """Retriever should embed the query before searching."""
        embedder = Mock()
        embedder.embed.return_value = [0.5, 0.5, 0.5]
        
        store = Mock()
        store.search.return_value = []
        
        retriever = Retriever(store=store, embedder=embedder)
        retriever.retrieve("test query", top_k=5)
        
        # Verify embedder was called with query
        embedder.embed.assert_called_once_with("test query")
        
        # Verify store.search was called with embedding
        store.search.assert_called_once_with([0.5, 0.5, 0.5], top_k=5)
    
    def test_retrieve_respects_top_k(self):
        """Retriever should pass top_k to store.search."""
        embedder = Mock()
        embedder.embed.return_value = [0.0]
        
        store = Mock()
        store.search.return_value = []
        
        retriever = Retriever(store=store, embedder=embedder)
        retriever.retrieve("query", top_k=10)
        
        # Verify top_k was passed
        _, kwargs = store.search.call_args
        assert kwargs["top_k"] == 10
    
    def test_retrieve_returns_empty_list_for_no_results(self):
        """Retriever should return empty list if store has no results."""
        embedder = Mock()
        embedder.embed.return_value = [0.0]
        
        store = Mock()
        store.search.return_value = []
        
        retriever = Retriever(store=store, embedder=embedder)
        results = retriever.retrieve("nonexistent query")
        
        assert results == []
    
    def test_retrieve_handles_empty_query(self):
        """Retriever should raise on empty query."""
        embedder = Mock()
        embedder.embed.return_value = [0.0]
        
        store = Mock()
        store.search.return_value = []
        
        retriever = Retriever(store=store, embedder=embedder)
        
        # Should raise on empty query
        with pytest.raises(ValueError, match="empty query"):
            retriever.retrieve("")
    
    def test_health_checks_both_embedder_and_store(self):
        """Health should check both embedder and store."""
        embedder = Mock()
        embedder.health.return_value = True
        
        store = Mock()
        store.health.return_value = True
        
        retriever = Retriever(store=store, embedder=embedder)
        
        assert retriever.health() is True
        embedder.health.assert_called_once()
        store.health.assert_called_once()
    
    def test_health_returns_false_if_embedder_unhealthy(self):
        """Health should return False if embedder is down."""
        embedder = Mock()
        embedder.health.return_value = False
        
        store = Mock()
        store.health.return_value = True
        
        retriever = Retriever(store=store, embedder=embedder)
        
        assert retriever.health() is False
    
    def test_health_returns_false_if_store_unhealthy(self):
        """Health should return False if store is down."""
        embedder = Mock()
        embedder.health.return_value = True
        
        store = Mock()
        store.health.return_value = False
        
        retriever = Retriever(store=store, embedder=embedder)
        
        assert retriever.health() is False
    
    def test_retrieve_with_scored_results(self):
        """Retriever should preserve scores from store."""
        embedder = Mock()
        embedder.embed.return_value = [1.0, 0.0]
        
        store = Mock()
        store.search.return_value = [
            ("id1", "text1", 0.99),
            ("id2", "text2", 0.50),
            ("id3", "text3", 0.01),
        ]
        
        retriever = Retriever(store=store, embedder=embedder)
        results = retriever.retrieve("query")
        
        assert len(results) == 3
        assert results[0].score == 0.99
        assert results[1].score == 0.50
        assert results[2].score == 0.01


def test_hybrid_retriever_rrf_prefers_lexical_match(tmp_path):
    """Hybrid retriever should fuse BM25 + vector with RRF."""
    embedder = Mock()
    embedder.embed.return_value = [0.1, 0.2]

    store = Mock()
    store.search.return_value = [
        ("chunk2", "Colorado Springs Pikes Peak", 0.9),
        ("chunk1", "Estes Park Riverwalk", 0.8),
    ]

    bm25 = BM25Index(tmp_path / "bm25.json")
    bm25.add_documents(
        [
            Chunk(
                id="chunk1",
                note_id="n1",
                text="Estes Park Riverwalk",
                chunk_idx=0,
                start_char=0,
                end_char=20,
                embedding=None,
            ),
        ]
    )

    retriever = HybridRetriever(store=store, embedder=embedder, bm25=bm25)
    results = retriever.retrieve("Estes Park", top_k=1)

    assert results
    assert results[0].chunk_id == "chunk1"


def test_retrieve_for_generation_limits_context_k(tmp_path):
    """retrieve_for_generation should respect final_context_k."""
    embedder = Mock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]

    store = Mock()
    store.search.return_value = [
        ("n1_0", "First chunk text", 0.9),
        ("n2_0", "Second chunk text", 0.8),
        ("n3_0", "Third chunk text", 0.7),
    ]

    bm25 = BM25Index(tmp_path / "bm25.json")
    bm25.add_documents(
        [
            _make_chunk("n1_0", "First chunk text"),
            _make_chunk("n2_0", "Second chunk text"),
            _make_chunk("n3_0", "Third chunk text"),
        ]
    )

    retriever = HybridRetriever(
        store=store,
        embedder=embedder,
        bm25=bm25,
        enable_mmr=False,
        rerank_enabled=False,
        final_context_k=2,
        max_context_tokens=200,
    )

    results = retriever.retrieve_for_generation("test query", top_k=5)
    assert len(results) <= 2


def test_retrieve_for_generation_respects_token_budget(tmp_path):
    """Packed context should not exceed max_context_tokens."""
    embedder = Mock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]

    store = Mock()
    store.search.return_value = [
        ("n1_0", "word " * 50, 0.9),
        ("n2_0", "word " * 50, 0.8),
    ]

    bm25 = BM25Index(tmp_path / "bm25.json")
    bm25.add_documents(
        [
            _make_chunk("n1_0", "word " * 50),
            _make_chunk("n2_0", "word " * 50),
        ]
    )

    retriever = HybridRetriever(
        store=store,
        embedder=embedder,
        bm25=bm25,
        enable_mmr=False,
        rerank_enabled=False,
        final_context_k=4,
        max_context_tokens=10,
    )

    results = retriever.retrieve_for_generation("test query", top_k=5)
    total_tokens = sum(len(r.text.split()) for r in results)
    assert total_tokens <= 10


def test_time_query_requires_time_evidence_when_available(tmp_path):
    """Time questions should include a time-bearing chunk when available."""
    embedder = Mock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]

    store = Mock()
    store.search.return_value = [
        ("n1_0", "Meeting agenda without time", 0.9),
        ("n2_0", "Meeting starts at 3:30 pm in room A", 0.8),
    ]

    bm25 = BM25Index(tmp_path / "bm25.json")
    bm25.add_documents(
        [
            _make_chunk("n1_0", "Meeting agenda without time"),
            _make_chunk("n2_0", "Meeting starts at 3:30 pm in room A"),
        ]
    )

    retriever = HybridRetriever(
        store=store,
        embedder=embedder,
        bm25=bm25,
        enable_mmr=False,
        rerank_enabled=False,
        final_context_k=2,
        max_context_tokens=200,
        time_query_require_time_evidence=True,
    )

    results = retriever.retrieve_for_generation("What time is the meeting?", top_k=5)
    assert any(re.search(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s?(?:am|pm)\b", r.text, re.I) for r in results)


def test_entity_preference_after_rerank(tmp_path):
    """Entity constraints should prefer matching entities after reranking."""
    embedder = Mock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]

    store = Mock()
    store.search.return_value = [
        ("beta_0", "Project Beta status update and risks", 0.9),
        ("alpha_0", "Project Alpha launch timeline and owners", 0.8),
    ]

    bm25 = BM25Index(tmp_path / "bm25.json")
    retriever = HybridRetriever(
        store=store,
        embedder=embedder,
        bm25=bm25,
        enable_mmr=False,
        rerank_enabled=True,
        final_context_k=1,
        max_context_tokens=200,
    )

    results = retriever.retrieve_for_generation("Project Alpha status", top_k=5)
    assert results
    assert results[0].chunk_id == "alpha_0"
