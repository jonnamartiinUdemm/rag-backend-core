import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from app.main import app

client = TestClient(app)

# Mock settings for integration
mock_settings = MagicMock()
mock_settings.EMBEDDING_MODEL = "test-model"
mock_settings.QDRANT_URL = "http://test-qdrant"
mock_settings.USE_RERANKER = False
mock_settings.RETRIEVAL_TOP_K = 5
mock_settings.RERANK_TOP_K = 3
mock_settings.LLM_PROVIDER = "ollama"

@pytest.mark.asyncio
@patch("app.api.routes.chat.get_settings")
@patch("app.api.routes.chat.get_llm")
@patch("app.api.routes.chat.Qdrant")
@patch("qdrant_client.QdrantClient")
@patch("app.api.routes.chat.ChatPromptTemplate")
@patch("app.api.routes.chat.StrOutputParser")
@patch("app.api.routes.documents.process_document.delay")
async def test_full_rag_pipeline_integration(mock_delay, mock_parser, mock_template, mock_qdrant_client, mock_qdrant, mock_get_llm, mock_get_settings):
    """Integration test for the full RAG pipeline: upload -> process -> query."""
    mock_get_settings.return_value = mock_settings
    
    # Mock task delay
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_delay.return_value = mock_task
    
    # Mock Qdrant for chat
    mock_qdrant_client.return_value = MagicMock()
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_docs = [MagicMock(page_content="Test content from uploaded PDF", metadata={"source": "test.pdf"})]
    mock_retriever.invoke.return_value = mock_docs
    mock_vector_store.as_retriever.return_value = mock_retriever
    mock_qdrant.return_value = mock_vector_store
    
    # Mock LLM chain for chat
    async def mock_ainvoke(*args, **kwargs):
        return "Answer based on uploaded PDF content"
    
    mock_chain = MagicMock()
    mock_chain.ainvoke = mock_ainvoke
    
    # Mock the chain creation: prompt | llm | parser
    mock_intermediate = MagicMock()
    mock_intermediate.__or__ = MagicMock(return_value=mock_chain)  # intermediate | parser -> chain
    
    mock_prompt_instance = MagicMock()
    mock_prompt_instance.__or__ = MagicMock(return_value=mock_intermediate)  # prompt | llm -> intermediate
    mock_template.from_template.return_value = mock_prompt_instance
    mock_get_llm.return_value = MagicMock()
    mock_parser.return_value = MagicMock()
    
    # Step 1: Upload PDF
    file_content = b"fake pdf content"
    files = {"file": ("test.pdf", file_content, "application/pdf")}
    upload_response = client.post("/documents/upload", files=files)
    
    assert upload_response.status_code == 200
    upload_data = upload_response.json()
    assert upload_data["filename"] == "test.pdf"
    assert "task_id" in upload_data  # Assuming task_id is added back
    
    # Step 2: Simulate task completion (in real integration, wait for Celery)
    # For test, assume task completes
    
    # Step 3: Query the uploaded content
    query_response = client.post("/chat/ask", json={"query": "What is in the PDF?"})
    
    assert query_response.status_code == 200
    query_data = query_response.json()
    assert "Answer based on uploaded PDF content" in query_data["answer"]
    assert len(query_data["source_documents"]) > 0