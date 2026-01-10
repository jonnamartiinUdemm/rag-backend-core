import os
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from functools import lru_cache

class Settings(BaseSettings):
    # --- App Configuration ---
    APP_NAME: str = Field(default="RAG Backend Core", description="Name of the application")
    DEBUG: bool = Field(default=True, description="Enable debug mode")
    
    # --- Vector DB (Qdrant) ---
    QDRANT_URL: str = Field(default="http://qdrant:6333", description="Internal or external URL for Qdrant")
    QDRANT_API_KEY: str | None = Field(default=None, description="API Key for Qdrant Cloud")
    # --- Redis Configuration ---
    REDIS_URL: str = Field(default="redis://redis:6379/0", description="Redis connection URL for chat history storage")
    
    # --- Embeddings ---
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # --- LLM Selection ---
    LLM_PROVIDER: str = Field(default="ollama", description="Active LLM provider")
    
    # --- Ollama Specifics ---
    OLLAMA_BASE_URL: str = Field(default="http://ollama:11434")
    OLLAMA_MODEL: str = Field(default="llama3", description="Model name in Ollama")
    
    # --- Google Gemini Specifics ---
    GOOGLE_API_KEY: str | None = Field(default=None, description="Required if LLM_PROVIDER is gemini")
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash", description="Google model version to use")

    # --- Reranking (Cohere) ---
    USE_RERANKER: bool = Field(default=False, description="Enable second-stage refinement")
    COHERE_API_KEY: str | None = Field(default=None, description="Required if USE_RERANKER is True")
    RERANK_TOP_K: int = Field(default=3, description="Number of final documents to pass to the LLM")
    RETRIEVAL_TOP_K: int = Field(default=10, description="Number of initial documents to fetch from Qdrant")

    # --- Resilience & Timeouts (NEW) ---
    LLM_TIMEOUT: float = Field(default=60.0, description="Max time (seconds) to wait for LLM generation")
    CONNECT_TIMEOUT: float = Field(default=5.0, description="Max time (seconds) to wait for service connections")

    # --- Chat History Management ---
    MAX_CHAT_HISTORY_LENGTH: int = Field(default=10, description="Max number of messages to retain in chat history")

    @field_validator("REDIS_URL")
    @classmethod
    def clean_redis_url(cls, v: str) -> str:
        if "?ssl_cert_reqs=CERT_NONE" in v:
            return v.replace("?ssl_cert_reqs=CERT_NONE", "")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "ignore" 

@lru_cache()
def get_settings():
    return Settings()