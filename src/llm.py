"""
LLM abstraction for text generation.

Provides both real (Ollama) and fake (deterministic) implementations.
"""

from abc import ABC, abstractmethod
import requests


class LLM(ABC):
    """Abstract base class for language models."""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Generated text.
            
        Raises:
            ValueError: If generation fails.
        """
        pass
    
    @abstractmethod
    def health(self) -> bool:
        """
        Check if LLM is healthy and ready.
        
        Returns:
            True if LLM is available and working.
        """
        pass


class OllamaLLM(LLM):
    """LLM using Ollama API."""
    
    def __init__(
        self,
        model_name: str = "neural-chat",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize Ollama LLM.
        
        Args:
            model_name: Name of the Ollama model.
            base_url: Base URL of Ollama API.
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate text using Ollama API.
        
        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Generated text.
            
        Raises:
            ValueError: If generation fails.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Cannot generate from empty prompt")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "num_predict": max_tokens,
                },
                timeout=300,  # 5 minutes timeout
            )
            response.raise_for_status()
            
            data = response.json()
            text = data.get("response", "")
            
            if not text:
                raise ValueError("Empty response from model")
            
            return text.strip()
            
        except requests.RequestException as e:
            raise ValueError(f"Ollama LLM error: {e}")
        except (KeyError, ValueError) as e:
            raise ValueError(f"Failed to parse Ollama response: {e}")
    
    def health(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            data = response.json()
            models = data.get("models", [])
            return any(m.get("name", "").startswith(self.model_name) for m in models)
            
        except Exception:
            return False


class FakeLLM(LLM):
    """
    Fake LLM for testing.
    
    Returns deterministic canned responses or can be customized.
    """
    
    def __init__(self, response_template: str = ""):
        """
        Initialize fake LLM.
        
        Args:
            response_template: Template for responses. If empty, uses default.
        """
        self.response_template = response_template or (
            "Based on the provided snippets, the answer is: [1][2][3]. "
            "The key information is drawn from the retrieved documents."
        )
        self._healthy = True
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate a deterministic response.
        
        Args:
            prompt: Input prompt (ignored in fake implementation).
            max_tokens: Maximum tokens (respected in output length).
            
        Returns:
            Canned response.
            
        Raises:
            ValueError: If prompt is empty.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Cannot generate from empty prompt")
        
        # Truncate response to approximately max_tokens worth of characters
        # Rough estimate: 1 token ≈ 4 characters
        char_limit = max_tokens * 4
        response = self.response_template[:char_limit]
        
        return response
    
    def health(self) -> bool:
        """Fake LLM is always healthy by default."""
        return self._healthy
    
    def set_health(self, healthy: bool) -> None:
        """
        Set health status for testing failure scenarios.
        
        Args:
            healthy: Whether LLM should be marked as healthy.
        """
        self._healthy = healthy
    
    def set_response(self, response: str) -> None:
        """
        Set custom response for testing.
        
        Args:
            response: Custom response to return.
        """
        self.response_template = response
