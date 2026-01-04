from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    """
    Basic sanity check to ensure the container starts and API is responsive.
    Crucial for CD pipelines to verify deployment success.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"