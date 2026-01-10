import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from app.main import app, check_infrastructure

client = TestClient(app)

@pytest.mark.asyncio
@patch("app.main.settings")
@patch("app.main.httpx.AsyncClient")
@patch("app.main.QdrantClient")
@patch("app.main.Redis")
async def test_check_infrastructure_gemini_security(
    mock_redis, mock_qdrant, mock_httpx_client, mock_settings
):
    mock_settings.LLM_PROVIDER = "gemini"
    mock_settings.GOOGLE_API_KEY = "secret-key-123"
    mock_settings.QDRANT_URL = "http://qdrant"
    mock_settings.CONNECT_TIMEOUT = 5.0
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    
    mock_client_instance = AsyncMock()
    mock_client_instance.get.return_value = mock_response
    mock_httpx_client.return_value.__aenter__.return_value = mock_client_instance

    status = await check_infrastructure()

    assert status["services"]["gemini"] == "up"
    
    mock_client_instance.get.assert_called_once()
    call_args = mock_client_instance.get.call_args
    
    expected_url = "https://generativelanguage.googleapis.com/v1beta/models"
    assert call_args[0][0] == expected_url
    
    headers = call_args[1]["headers"]
    assert headers["x-goog-api-key"] == "secret-key-123"

@patch("app.main.check_infrastructure", new_callable=AsyncMock)
def test_health_endpoint_structure(mock_check_infra):
    mock_check_infra.return_value = {
        "status": "ok", 
        "services": {"qdrant": "up", "redis": "up", "gemini": "up"}
    }
    
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json()["status"] == "ok"