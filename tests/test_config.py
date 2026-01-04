import pytest
from pydantic import BaseModel
from app.core.config import Settings, get_settings

class ConfigSettings(Settings):
    """Settings subclass that ignores env files for testing."""
    model_config = Settings.model_config.copy()
    model_config['env_file'] = None

def test_settings_defaults():
    """Tests that Settings has correct default values."""
    settings = ConfigSettings()
    
    assert settings.APP_NAME == "RAG Backend Core"
    assert settings.DEBUG is True
    assert settings.QDRANT_URL == "http://qdrant:6333"
    assert settings.EMBEDDING_MODEL == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert settings.LLM_PROVIDER == "ollama"
    assert settings.OLLAMA_BASE_URL == "http://ollama:11434"
    assert settings.OLLAMA_MODEL == "llama3"
    assert settings.GOOGLE_API_KEY is None
    assert settings.GEMINI_MODEL == "gemini-1.5-flash"
    assert settings.USE_RERANKER is False
    assert settings.COHERE_API_KEY is None
    assert settings.RERANK_TOP_K == 3
    assert settings.RETRIEVAL_TOP_K == 10

def test_settings_from_env(monkeypatch):
    """Tests that Settings loads from environment variables."""
    monkeypatch.setenv("APP_NAME", "Test App")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")
    
    settings = Settings()  # Will load from env
    
    assert settings.APP_NAME == "Test App"
    assert settings.DEBUG is False
    assert settings.LLM_PROVIDER == "gemini"
    assert settings.GOOGLE_API_KEY == "test_key"

def test_get_settings_returns_settings_instance():
    """Tests that get_settings returns a Settings instance."""
    settings = get_settings()
    
    assert isinstance(settings, Settings)
    # Note: This may have env values, so don't assert defaults

def test_settings_optional_fields():
    """Tests that optional fields are None by default."""
    settings = ConfigSettings()
    
    assert settings.GOOGLE_API_KEY is None
    assert settings.COHERE_API_KEY is None