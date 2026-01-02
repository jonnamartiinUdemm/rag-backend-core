import os
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache

class Settings(BaseSettings):
    # --- App Configuration ---
    APP_NAME: str = Field(default="RAG Backend Core", description="Name of the application")
    DEBUG: bool = Field(default=True, description="Enable debug mode")
    
    # --- Vector DB (Qdrant) ---
    QDRANT_URL: str = Field(default="http://qdrant:6333", description="Internal or external URL for Qdrant")
    
    # --- Embeddings ---
    # Model used for vectorization (Must match the dimension in Qdrant)
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # --- LLM Selection ---
    # Defines which "brain" the app uses. Options: "ollama", "gemini"
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

    class Config:
        # Pydantic will read the .env file if it exists
        env_file = ".env"
        env_file_encoding = 'utf-8'
        # Ignores extra variables in .env that are not defined here
        extra = "ignore" 

@lru_cache()
def get_settings():
    return Settings()