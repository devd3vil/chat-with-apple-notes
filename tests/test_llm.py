"""
Tests for LLM implementations.

Tests OllamaLLM and FakeLLM.
"""

import pytest
from unittest.mock import patch, Mock
import requests
from src.llm import OllamaLLM, FakeLLM


class TestFakeLLM:
    """Test FakeLLM for testing."""
    
    def test_generate_returns_default_response(self):
        """FakeLLM should return a default response by default."""
        llm = FakeLLM()
        response = llm.generate("test prompt")
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_generate_respects_max_tokens(self):
        """FakeLLM should truncate response to ~max_tokens*4 characters."""
        llm = FakeLLM()
        response = llm.generate("prompt", max_tokens=10)
        
        # Rough estimate: 1 token ≈ 4 characters, so char_limit = max_tokens * 4
        assert len(response) <= 40  # 10 * 4
    
    def test_set_response_customizes_output(self):
        """set_response() should customize FakeLLM output."""
        llm = FakeLLM()
        custom_response = "This is a custom response"
        llm.set_response(custom_response)
        
        response = llm.generate("any prompt")
        
        assert response == custom_response
    
    def test_generate_ignores_prompt(self):
        """FakeLLM should ignore the prompt and return preset response."""
        llm = FakeLLM()
        custom = "Static response"
        llm.set_response(custom)
        
        response1 = llm.generate("prompt 1")
        response2 = llm.generate("totally different prompt")
        
        assert response1 == custom
        assert response2 == custom
    
    def test_health_always_returns_true(self):
        """FakeLLM.health() should always return True."""
        llm = FakeLLM()
        assert llm.health() is True
    
    def test_set_response_with_max_tokens_truncation(self):
        """set_response with truncation should respect max_tokens."""
        llm = FakeLLM()
        long_response = "This is a very long response that should be truncated"
        llm.set_response(long_response)
        
        result = llm.generate("prompt", max_tokens=20)
        
        # char_limit = max_tokens * 4 = 80
        assert len(result) <= 80
        assert result == long_response[:80]


class TestOllamaLLM:
    """Test OllamaLLM with mocked requests."""
    
    def test_generate_calls_ollama_endpoint(self):
        """OllamaLLM should POST to /api/generate."""
        with patch("src.llm.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {
                "response": "Generated text"
            }
            
            llm = OllamaLLM(base_url="http://localhost:11434")
            response = llm.generate("test prompt")
            
            assert response == "Generated text"
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "http://localhost:11434/api/generate" in call_args[0]
    
    def test_generate_includes_model_in_request(self):
        """OllamaLLM should include model name in request."""
        with patch("src.llm.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "text"}
            
            llm = OllamaLLM(model_name="neural-chat")
            llm.generate("prompt")
            
            _, kwargs = mock_post.call_args
            assert kwargs["json"]["model"] == "neural-chat"
    
    def test_generate_includes_prompt_in_request(self):
        """OllamaLLM should include prompt in request."""
        with patch("src.llm.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "text"}
            
            llm = OllamaLLM()
            test_prompt = "This is my test prompt"
            llm.generate(test_prompt)
            
            _, kwargs = mock_post.call_args
            assert kwargs["json"]["prompt"] == test_prompt
    
    def test_generate_includes_max_tokens(self):
        """OllamaLLM should include num_predict (max tokens) in request."""
        with patch("src.llm.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "text"}
            
            llm = OllamaLLM()
            llm.generate("prompt", max_tokens=256)
            
            _, kwargs = mock_post.call_args
            assert kwargs["json"]["num_predict"] == 256
    
    def test_generate_sets_stream_to_false(self):
        """OllamaLLM should set stream=false for non-streaming response."""
        with patch("src.llm.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "text"}
            
            llm = OllamaLLM()
            llm.generate("prompt")
            
            _, kwargs = mock_post.call_args
            assert kwargs["json"]["stream"] is False
    
    def test_generate_sets_timeout(self):
        """OllamaLLM should set request timeout."""
        with patch("src.llm.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "text"}
            
            llm = OllamaLLM()
            llm.generate("prompt")
            
            _, kwargs = mock_post.call_args
            assert "timeout" in kwargs
            assert kwargs["timeout"] == 300
    
    def test_health_returns_true_when_ollama_available(self):
        """health() should return True if Ollama is reachable with model."""
        with patch("src.llm.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "models": [{"name": "neural-chat:latest"}]
            }
            
            llm = OllamaLLM(base_url="http://localhost:11434")
            assert llm.health() is True
            
            mock_get.assert_called_once_with(
                "http://localhost:11434/api/tags",
                timeout=5
            )
    
    def test_health_returns_false_when_ollama_unavailable(self):
        """health() should return False if Ollama is not reachable."""
        with patch("src.llm.requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("Connection failed")
            
            llm = OllamaLLM(base_url="http://localhost:11434")
            assert llm.health() is False
    
    def test_generate_handles_missing_response_key(self):
        """OllamaLLM should handle malformed responses gracefully."""
        with patch("src.llm.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"error": "some error"}
            
            llm = OllamaLLM()
            
            # OllamaLLM should raise ValueError when response key is missing
            with pytest.raises(ValueError, match="Empty response"):
                llm.generate("prompt")
    
    def test_generate_handles_request_timeout(self):
        """OllamaLLM should raise ValueError on request timeout."""
        with patch("src.llm.requests.post") as mock_post:
            mock_post.side_effect = requests.Timeout()
            
            llm = OllamaLLM()
            
            # OllamaLLM wraps RequestException in ValueError
            with pytest.raises(ValueError, match="Ollama LLM error"):
                llm.generate("prompt")
    
    def test_ollama_llm_default_values(self):
        """OllamaLLM should use sensible defaults."""
        llm = OllamaLLM()
        
        assert llm.model_name == "neural-chat"
        assert llm.base_url == "http://localhost:11434"
    
    def test_ollama_llm_custom_values(self):
        """OllamaLLM should accept custom model and base_url."""
        llm = OllamaLLM(model_name="custom-model", base_url="http://custom:9999")
        
        assert llm.model_name == "custom-model"
        assert llm.base_url == "http://custom:9999"
