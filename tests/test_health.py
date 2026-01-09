from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@patch("app.main.check_infrastructure", new_callable=AsyncMock)
def test_health_check(mock_check_infra):
    """
    Basic sanity check to ensure the container starts and API is responsive.
    Crucial for CD pipelines to verify deployment success.
    """
    # Mock the infrastructure check to return healthy status
    mock_check_infra.return_value = {
        "status": "ok", 
        "services": {"qdrant": "up", "redis": "up", "ollama": "up"}
    }
    
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    # Optional: Verify services are reported as up
    assert response.json()["services"]["qdrant"] == "up"