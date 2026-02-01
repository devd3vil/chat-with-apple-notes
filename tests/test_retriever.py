"""
Tests for Retriever class.

Tests basic retrieval with mocked embedder and store.
"""

import pytest
from unittest.mock import Mock
from src.retriever import Retriever
from src.models import Citation


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
