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
mock_settings.LLM_PROVIDER = "ollama"
mock_settings.LLM_TIMEOUT = 60.0
mock_settings.CONNECT_TIMEOUT = 5.0

@pytest.mark.asyncio
@patch("app.api.routes.chat.get_settings")
@patch("app.api.routes.chat.RedisChatMessageHistory")
@patch("app.api.routes.chat.RunnableWithMessageHistory")
@patch("app.api.routes.chat.get_llm")
@patch("app.api.routes.chat.Qdrant")
@patch("qdrant_client.QdrantClient")
@patch("app.api.routes.chat.ChatPromptTemplate")
@patch("app.api.routes.chat.StrOutputParser")
@patch("app.api.routes.documents.process_document.delay")
async def test_full_rag_pipeline_integration(
    mock_delay, mock_parser, mock_template, mock_qdrant_client, mock_qdrant, 
    mock_get_llm, mock_runnable, mock_redis_history, mock_get_settings
):
    mock_get_settings.return_value = mock_settings
    
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_delay.return_value = mock_task
    
    mock_qdrant_client.return_value = MagicMock()
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_docs = [MagicMock(page_content="Test content from uploaded PDF", metadata={"source": "test.pdf"})]
    mock_retriever.invoke.return_value = mock_docs
    mock_vector_store.as_retriever.return_value = mock_retriever
    mock_qdrant.return_value = mock_vector_store
    
    mock_chain_instance = MagicMock()
    mock_chain_instance.ainvoke = AsyncMock(return_value="Answer based on uploaded PDF content")
    mock_runnable.return_value = mock_chain_instance
    
    mock_llm_instance = MagicMock()
    mock_llm_instance.__or__ = MagicMock()
    mock_prompt_instance = MagicMock()
    mock_prompt_instance.__or__ = MagicMock(return_value=mock_llm_instance)
    mock_template.from_messages.return_value = mock_prompt_instance
    
    mock_get_llm.return_value = MagicMock()
    mock_parser.return_value = MagicMock()
    
    file_content = b"fake pdf content"
    files = {"file": ("test.pdf", file_content, "application/pdf")}
    upload_response = client.post("/documents/upload", files=files)
    
    assert upload_response.status_code == 200
    upload_data = upload_response.json()
    assert upload_data["filename"] == "test.pdf"
    assert "task_id" in upload_data
    
    query_response = client.post("/chat/ask", json={
        "query": "What is in the PDF?",
        "user_id": "integration-user",
        "chat_id": "integration-chat"
    })
    
    assert query_response.status_code == 200