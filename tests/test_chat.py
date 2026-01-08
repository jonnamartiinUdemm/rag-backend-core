import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Mock settings
mock_settings = MagicMock()
mock_settings.EMBEDDING_MODEL = "test-model"
mock_settings.QDRANT_URL = "http://test-qdrant"
mock_settings.USE_RERANKER = False
mock_settings.RETRIEVAL_TOP_K = 5
mock_settings.RERANK_TOP_K = 3
mock_settings.LLM_PROVIDER = "ollama"
mock_settings.LLM_TIMEOUT = 60.0
mock_settings.CONNECT_TIMEOUT = 5.0

@patch("app.api.routes.chat.get_settings")
@patch("app.api.routes.chat.get_llm")
@patch("app.api.routes.chat.Qdrant")
@patch("qdrant_client.QdrantClient")
@patch("app.api.routes.chat.ChatPromptTemplate")
@patch("app.api.routes.chat.StrOutputParser")
def test_ask_document_success(mock_parser, mock_template, mock_qdrant_client, mock_qdrant, mock_get_llm, mock_get_settings):
    """Tests successful query with retrieved documents."""
    mock_get_settings.return_value = mock_settings
    
    # Mock QdrantClient
    mock_qdrant_client.return_value = MagicMock()
    
    # Mock Qdrant
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_docs = [MagicMock(page_content="Test content", metadata={"source": "test.pdf"})]
    mock_retriever.invoke.return_value = mock_docs
    mock_vector_store.as_retriever.return_value = mock_retriever
    mock_qdrant.return_value = mock_vector_store
    
    # Mock chain - create proper awaitable
    async def mock_ainvoke(*args, **kwargs):
        return "Test answer"
    
    mock_chain = MagicMock()
    mock_chain.ainvoke = mock_ainvoke
    
    # Mock the pipe operations: prompt | llm | parser
    mock_llm_instance = MagicMock()
    mock_llm_instance.__or__ = MagicMock(return_value=mock_chain)
    
    mock_prompt_instance = MagicMock()
    mock_prompt_instance.__or__ = MagicMock(return_value=mock_llm_instance)
    
    mock_template.from_template.return_value = mock_prompt_instance
    mock_get_llm.return_value = MagicMock()
    mock_parser.return_value = MagicMock()
    
    response = client.post("/chat/ask", json={"query": "What is AI?"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test answer"
    assert len(data["source_documents"]) > 0

@patch("app.api.routes.chat.get_settings")
@patch("app.api.routes.chat.Qdrant")
@patch("qdrant_client.QdrantClient")
def test_ask_document_no_documents(mock_qdrant_client, mock_qdrant, mock_get_settings):
    """Tests query with no relevant documents."""
    mock_get_settings.return_value = mock_settings
    
    mock_qdrant_client.return_value = MagicMock()
    
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    mock_vector_store.as_retriever.return_value = mock_retriever
    mock_qdrant.return_value = mock_vector_store
    
    response = client.post("/chat/ask", json={"query": "Unknown topic"})
    
    assert response.status_code == 200
    data = response.json()
    assert "No relevant information found" in data["answer"]
    assert data["source_documents"] == []

@patch("app.api.routes.chat.get_settings")
@patch("app.api.routes.chat.Qdrant")
@patch("qdrant_client.QdrantClient")
def test_ask_document_qdrant_error(mock_qdrant_client, mock_qdrant, mock_get_settings):
    """Tests error handling when Qdrant fails (Expect 503)."""
    mock_get_settings.return_value = mock_settings
    
    # Mock Qdrant error to trigger the 503 block
    mock_qdrant_client.side_effect = Exception("Qdrant connection failed")
    
    response = client.post("/chat/ask", json={"query": "Test query"})
    
    # UPDATED: Expect 503 for infrastructure failure
    assert response.status_code == 503
    assert response.json()["detail"] == "Vector Database Unavailable"

@patch("app.api.routes.chat.get_settings")
@patch("app.api.routes.chat.get_llm")
@patch("app.api.routes.chat.Qdrant")
@patch("qdrant_client.QdrantClient")
@patch("app.api.routes.chat.ChatPromptTemplate")
@patch("app.api.routes.chat.StrOutputParser")
def test_ask_document_llm_error(mock_parser, mock_template, mock_qdrant_client, mock_qdrant, mock_get_llm, mock_get_settings):
    """Tests error handling when LLM fails (Generic 500)."""
    mock_get_settings.return_value = mock_settings
    
    mock_qdrant_client.return_value = MagicMock()
    
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_docs = [MagicMock(page_content="Test content", metadata={"source": "test.pdf"})]
    mock_retriever.invoke.return_value = mock_docs
    mock_vector_store.as_retriever.return_value = mock_retriever
    mock_qdrant.return_value = mock_vector_store
    
    # Mock chain with error
    async def mock_ainvoke_error(*args, **kwargs):
        raise Exception("LLM failed")
    
    mock_chain = MagicMock()
    mock_chain.ainvoke = mock_ainvoke_error
    
    mock_llm_instance = MagicMock()
    mock_llm_instance.__or__ = MagicMock(return_value=mock_chain)
    
    mock_prompt_instance = MagicMock()
    mock_prompt_instance.__or__ = MagicMock(return_value=mock_llm_instance)
    
    mock_template.from_template.return_value = mock_prompt_instance
    mock_get_llm.return_value = MagicMock()
    mock_parser.return_value = MagicMock()
    
    response = client.post("/chat/ask", json={"query": "Test query"})
    
    assert response.status_code == 500
    # UPDATED: The new code catches generic errors and returns "Internal Server Error"
    assert response.json()["detail"] == "Internal Server Error"