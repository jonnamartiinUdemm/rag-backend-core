import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

mock_settings = MagicMock()
mock_settings.EMBEDDING_MODEL = "test-model"
mock_settings.QDRANT_URL = "http://test-qdrant"
mock_settings.REDIS_URL = "redis://mock:6379/0"
mock_settings.USE_RERANKER = False
mock_settings.RETRIEVAL_TOP_K = 5
mock_settings.RERANK_TOP_K = 3
mock_settings.LLM_PROVIDER = "ollama"
mock_settings.LLM_TIMEOUT = 60.0
mock_settings.CONNECT_TIMEOUT = 5.0
mock_settings.GEMINI_MODEL = "gemini-1.5-flash"

@patch("app.api.routes.chat.get_settings")
@patch("app.api.routes.chat.RedisChatMessageHistory")
@patch("app.api.routes.chat.RunnableWithMessageHistory")
@patch("app.api.routes.chat.get_llm")
@patch("app.api.routes.chat.Qdrant")
@patch("qdrant_client.QdrantClient")
@patch("app.api.routes.chat.ChatPromptTemplate")
@patch("app.api.routes.chat.StrOutputParser")
def test_ask_document_success(
    mock_parser, mock_template, mock_qdrant_client, mock_qdrant, mock_get_llm, 
    mock_runnable_history, mock_redis_history, mock_get_settings
):
    mock_get_settings.return_value = mock_settings
    
    mock_history_instance = MagicMock()
    mock_history_instance.messages = []
    mock_redis_history.return_value = mock_history_instance

    mock_qdrant_client.return_value = MagicMock()
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_docs = [MagicMock(page_content="Contenido de prueba", metadata={"source": "test.pdf"})]
    mock_retriever.invoke.return_value = mock_docs
    mock_vector_store.as_retriever.return_value = mock_retriever
    mock_qdrant.return_value = mock_vector_store
    
    mock_chain_instance = MagicMock()
    mock_chain_instance.ainvoke = AsyncMock(return_value="Respuesta de prueba")
    mock_runnable_history.return_value = mock_chain_instance
    
    mock_llm_instance = MagicMock()
    mock_llm_instance.__or__ = MagicMock()
    mock_prompt_instance = MagicMock()
    mock_prompt_instance.__or__ = MagicMock(return_value=mock_llm_instance)
    mock_template.from_