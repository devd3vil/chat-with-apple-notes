"""
Tests for embedder implementations.
"""

import pytest
from src.embedder import Embedder, FakeEmbedder, OllamaEmbedder


class TestFakeEmbedder:
    """Tests for FakeEmbedder."""
    
    def test_fake_embedder_creation(self):
        """Test creating a fake embedder."""
        embedder = FakeEmbedder(embedding_dim=384)
        assert embedder.get_embedding_dim() == 384
    
    def test_fake_embedder_deterministic(self):
        """Test that FakeEmbedder produces deterministic embeddings."""
        embedder = FakeEmbedder(embedding_dim=256)
        
        text = "This is a test sentence."
        embedding1 = embedder.embed(text)
        embedding2 = embedder.embed(text)
        
        assert embedding1 == embedding2
    
    def test_fake_embedder_different_texts(self):
        """Test that different texts produce different embeddings."""
        embedder = FakeEmbedder(embedding_dim=256)
        
        embedding1 = embedder.embed("Text 1")
        embedding2 = embedder.embed("Text 2")
        
        assert embedding1 != embedding2
    
    def test_fake_embedder_dimension(self):
        """Test embedding dimension."""
        for dim in [128, 256, 384, 768]:
            embedder = FakeEmbedder(embedding_dim=dim)
            embedding = embedder.embed("test")
            assert len(embedding) == dim
    
    def test_fake_embedder_value_range(self):
        """Test that embedding values are in reasonable range [-1, 1]."""
        embedder = FakeEmbedder(embedding_dim=256)
        embedding = embedder.embed("test text")
        
        for value in embedding:
            assert -1.0 <= value <= 1.0
    
    def test_fake_embedder_empty_text_error(self):
        """Test that empty text raises error."""
        embedder = FakeEmbedder()
        
        with pytest.raises(ValueError):
            embedder.embed("")
        
        with pytest.raises(ValueError):
            embedder.embed("   ")
    
    def test_fake_embedder_health_check(self):
        """Test health check."""
        embedder = FakeEmbedder()
        assert embedder.health() is True
        
        # Can set health to False for testing
        embedder.set_health(False)
        assert embedder.health() is False
    
    def test_fake_embedder_long_text(self):
        """Test embedding long text."""
        embedder = FakeEmbedder(embedding_dim=384)
        
        long_text = "word " * 1000  # Very long text
        embedding = embedder.embed(long_text)
        
        assert len(embedding) == 384
    
    def test_fake_embedder_special_characters(self):
        """Test embedding text with special characters."""
        embedder = FakeEmbedder()
        
        texts = [
            "Hello, World!",
            "Special @#$% chars",
            "Unicode: 你好世界 🎉",
            "Newlines\nand\ttabs",
        ]
        
        embeddings = [embedder.embed(text) for text in texts]
        
        # All should be different
        assert len(set(tuple(e) for e in embeddings)) == len(embeddings)


class TestEmbedderInterface:
    """Tests for Embedder interface contract."""
    
    def test_embedder_is_abstract(self):
        """Test that Embedder cannot be instantiated."""
        with pytest.raises(TypeError):
            Embedder()
    
    def test_fake_embedder_implements_interface(self):
        """Test that FakeEmbedder implements Embedder interface."""
        embedder = FakeEmbedder()
        
        assert isinstance(embedder, Embedder)
        assert hasattr(embedder, "embed")
        assert hasattr(embedder, "health")
        assert hasattr(embedder, "get_embedding_dim")
    
    def test_embedder_methods_callable(self):
        """Test that all interface methods are callable."""
        embedder = FakeEmbedder()
        
        assert callable(embedder.embed)
        assert callable(embedder.health)
        assert callable(embedder.get_embedding_dim)


class TestOllamaEmbedderConfig:
    """Tests for OllamaEmbedder configuration."""
    
    def test_ollama_embedder_creation_with_defaults(self):
        """Test creating OllamaEmbedder with default config."""
        # Should not raise error on creation
        embedder = OllamaEmbedder()
        assert embedder.model_name == "nomic-embed-text"
        assert "localhost" in embedder.base_url
    
    def test_ollama_embedder_custom_config(self):
        """Test creating OllamaEmbedder with custom config."""
        embedder = OllamaEmbedder(
            model_name="custom-model",
            base_url="http://remote:11434",
        )
        assert embedder.model_name == "custom-model"
        assert embedder.base_url == "http://remote:11434"
    
    def test_ollama_embedder_strips_trailing_slash(self):
        """Test that base_url trailing slash is stripped."""
        embedder = OllamaEmbedder(base_url="http://localhost:11434/")
        assert embedder.base_url == "http://localhost:11434"
    
    def test_ollama_embedder_health_check_offline(self):
        """Test health check when Ollama is offline."""
        embedder = OllamaEmbedder(base_url="http://nonexistent:11434")
        # Should return False, not crash
        assert embedder.health() is False
    
    def test_ollama_embedder_implements_interface(self):
        """Test that OllamaEmbedder implements interface."""
        embedder = OllamaEmbedder()
        
        assert isinstance(embedder, Embedder)
        assert hasattr(embedder, "embed")
        assert hasattr(embedder, "health")
        assert hasattr(embedder, "get_embedding_dim")


class TestEmbedderComparison:
    """Tests comparing FakeEmbedder with Embedder interface."""
    
    def test_fake_vs_ollama_interface_compatible(self):
        """Test that FakeEmbedder and OllamaEmbedder have same interface."""
        fake = FakeEmbedder()
        ollama = OllamaEmbedder()
        
        # Both should have same methods
        fake_methods = set(dir(fake))
        ollama_methods = set(dir(ollama))
        
        # Check core methods
        for method in ["embed", "health", "get_embedding_dim"]:
            assert hasattr(fake, method)
            assert hasattr(ollama, method)


class TestFakeEmbedderDeterminism:
    """Tests for deterministic behavior of FakeEmbedder."""
    
    def test_same_instance_same_embedding(self):
        """Test same instance produces same embedding."""
        embedder = FakeEmbedder()
        text = "consistent embedding test"
        
        results = [embedder.embed(text) for _ in range(5)]
        
        # All should be identical
        for result in results[1:]:
            assert result == results[0]
    
    def test_different_instances_same_embedding(self):
        """Test different instances produce same embedding for same text."""
        text = "shared embedding"
        
        embedder1 = FakeEmbedder(embedding_dim=384)
        embedder2 = FakeEmbedder(embedding_dim=384)
        
        embedding1 = embedder1.embed(text)
        embedding2 = embedder2.embed(text)
        
        assert embedding1 == embedding2
    
    def test_dimension_independence(self):
        """Test that changing dimension affects embedding."""
        text = "test text"
        
        embedder_small = FakeEmbedder(embedding_dim=128)
        embedder_large = FakeEmbedder(embedding_dim=256)
        
        embedding_small = embedder_small.embed(text)
        embedding_large = embedder_large.embed(text)
        
        assert len(embedding_small) == 128
        assert len(embedding_large) == 256
        # The embeddings should be different (different lengths)
        assert embedding_small != embedding_large


class TestFakeEmbedderEdgeCases:
    """Tests for edge cases in FakeEmbedder."""
    
    def test_single_character(self):
        """Test embedding single character."""
        embedder = FakeEmbedder()
        embedding = embedder.embed("a")
        assert len(embedding) > 0
    
    def test_very_long_text(self):
        """Test embedding very long text."""
        embedder = FakeEmbedder()
        long_text = "word " * 10000
        embedding = embedder.embed(long_text)
        assert len(embedding) == 384  # default dim
    
    def test_whitespace_only_texts(self):
        """Test that whitespace-only text is rejected."""
        embedder = FakeEmbedder()
        
        with pytest.raises(ValueError):
            embedder.embed("   ")
        
        with pytest.raises(ValueError):
            embedder.embed("\t\n  ")
    
    def test_numeric_strings(self):
        """Test embedding numeric strings."""
        embedder = FakeEmbedder()
        
        embedding1 = embedder.embed("12345")
        embedding2 = embedder.embed("54321")
        
        assert embedding1 != embedding2
        assert len(embedding1) == 384
