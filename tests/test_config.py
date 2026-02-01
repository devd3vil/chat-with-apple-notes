"""
Tests for configuration loading.
"""

import pytest
import tempfile
from pathlib import Path
from src.config import Settings


class TestSettingsDefaults:
    """Tests for default configuration values."""
    
    def test_default_ollama_settings(self):
        """Test default Ollama configuration."""
        settings = Settings(_env_file=None)  # Don't load .env
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.ollama_embedding_model == "nomic-embed-text"
        assert settings.ollama_chat_model == "neural-chat"
    
    def test_default_rag_settings(self):
        """Test default RAG configuration."""
        settings = Settings(_env_file=None)
        assert settings.chunk_size == 512
        assert settings.chunk_overlap == 50
        assert settings.top_k == 5
    
    def test_default_api_settings(self):
        """Test default API configuration."""
        settings = Settings(_env_file=None)
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.api_reload is False
    
    def test_default_debug_false(self):
        """Test debug is off by default."""
        settings = Settings(_env_file=None)
        assert settings.debug is False


class TestSettingsEnvironmentVars:
    """Tests for loading from environment variables."""
    
    def test_load_from_env_vars(self, monkeypatch):
        """Test loading configuration from environment variables."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom-ollama:11434")
        monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "custom-embed")
        monkeypatch.setenv("TOP_K", "10")
        monkeypatch.setenv("DEBUG", "True")
        
        settings = Settings(_env_file=None)
        assert settings.ollama_base_url == "http://custom-ollama:11434"
        assert settings.ollama_embedding_model == "custom-embed"
        assert settings.top_k == 10
        assert settings.debug is True
    
    def test_case_insensitive_env_vars(self, monkeypatch):
        """Test that environment variables are case-insensitive."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://test:11434")
        monkeypatch.setenv("ollama_chat_model", "custom-chat")
        
        settings = Settings(_env_file=None)
        assert settings.ollama_base_url == "http://test:11434"
        assert settings.ollama_chat_model == "custom-chat"


class TestChromaDbPath:
    """Tests for Chroma DB path handling."""
    
    def test_default_chroma_db_path(self):
        """Test default Chroma DB path."""
        settings = Settings(_env_file=None)
        assert settings.chroma_db_path == Path("data/chroma_db")
    
    def test_custom_chroma_db_path(self, monkeypatch, tmp_path):
        """Test custom Chroma DB path."""
        db_path = tmp_path / "custom_chroma"
        monkeypatch.setenv("CHROMA_DB_PATH", str(db_path))
        
        settings = Settings(_env_file=None)
        assert settings.chroma_db_path == db_path
    
    def test_chroma_db_path_created(self, tmp_path):
        """Test that Chroma DB path is created automatically."""
        db_path = tmp_path / "new_dir" / "chroma_db"
        settings = Settings(chroma_db_path=db_path)
        assert db_path.exists()
        assert db_path.is_dir()


class TestNotesExportDir:
    """Tests for notes export directory."""
    
    def test_notes_export_dir_none_by_default(self):
        """Test that notes_export_dir is None by default."""
        settings = Settings(_env_file=None)
        assert settings.notes_export_dir is None
    
    def test_custom_notes_export_dir(self, monkeypatch, tmp_path):
        """Test custom notes export directory."""
        export_path = tmp_path / "notes_export"
        monkeypatch.setenv("NOTES_EXPORT_DIR", str(export_path))
        
        settings = Settings(_env_file=None)
        assert settings.notes_export_dir == export_path


class TestSettingsIntegration:
    """Integration tests for settings."""
    
    def test_multiple_env_vars_together(self, monkeypatch):
        """Test loading multiple environment variables together."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
        monkeypatch.setenv("OLLAMA_CHAT_MODEL", "llama2")
        monkeypatch.setenv("CHUNK_SIZE", "256")
        monkeypatch.setenv("TOP_K", "3")
        monkeypatch.setenv("API_PORT", "9000")
        
        settings = Settings(_env_file=None)
        assert settings.ollama_base_url == "http://ollama:11434"
        assert settings.ollama_chat_model == "llama2"
        assert settings.chunk_size == 256
        assert settings.top_k == 3
        assert settings.api_port == 9000
    
    def test_env_file_loading(self, tmp_path):
        """Test loading from .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "OLLAMA_BASE_URL=http://docker-ollama:11434\n"
            "TOP_K=7\n"
            "DEBUG=true\n"
        )
        
        settings = Settings(_env_file=str(env_file))
        assert settings.ollama_base_url == "http://docker-ollama:11434"
        assert settings.top_k == 7
        assert settings.debug is True
