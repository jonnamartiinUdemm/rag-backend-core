import pytest
from unittest.mock import MagicMock, patch
from app.services.llm_factory import get_llm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# Mock settings for Gemini to pass validation
MOCK_SETTINGS_GEMINI = MagicMock()
MOCK_SETTINGS_GEMINI.LLM_PROVIDER = "gemini"
MOCK_SETTINGS_GEMINI.GOOGLE_API_KEY = "fake_key"
MOCK_SETTINGS_GEMINI.GEMINI_MODEL = "gemini-2.0-flash"

# Mock settings for Ollama
MOCK_SETTINGS_OLLAMA = MagicMock()
MOCK_SETTINGS_OLLAMA.LLM_PROVIDER = "ollama"
MOCK_SETTINGS_OLLAMA.OLLAMA_BASE_URL = "http://fake-url"
MOCK_SETTINGS_OLLAMA.OLLAMA_MODEL = "llama3"

def test_get_llm_returns_gemini():
    """Tests that the factory returns a Google instance when the provider is gemini."""
    with patch("app.services.llm_factory.settings", MOCK_SETTINGS_GEMINI):
        llm = get_llm()
        assert isinstance(llm, ChatGoogleGenerativeAI)
        assert llm.model == "models/gemini-2.0-flash"

def test_get_llm_returns_ollama():
    """Tests that the factory returns an Ollama instance when the provider is ollama."""
    with patch("app.services.llm_factory.settings", MOCK_SETTINGS_OLLAMA):
        llm = get_llm()
        assert isinstance(llm, ChatOllama)
        assert llm.model == "llama3"

def test_get_llm_raises_error_on_invalid_provider():
    """Tests that the factory raises a ValueError if the provider is invalid."""
    invalid_settings = MagicMock()
    invalid_settings.LLM_PROVIDER = "skynet"
    
    with patch("app.services.llm_factory.settings", invalid_settings):
        # Assert that it raises ValueError
        with pytest.raises(ValueError) as excinfo:
            get_llm()
        assert "Unsupported LLM_PROVIDER" in str(excinfo.value)