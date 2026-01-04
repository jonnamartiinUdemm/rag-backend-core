import pytest
from unittest.mock import MagicMock, patch, mock_open
from fastapi.testclient import TestClient
from fastapi import HTTPException
from app.main import app
from app.tasks.tasks import process_document

client = TestClient(app)

# Mock para process_document.delay
mock_task = MagicMock()
mock_task.id = "test-task-id"

@patch("app.api.routes.documents.process_document.delay")
def test_upload_document_valid_pdf(mock_delay):
    """Tests successful upload of a valid PDF file."""
    mock_delay.return_value = mock_task
    
    # Simular un archivo PDF
    file_content = b"fake pdf content"
    files = {"file": ("test.pdf", file_content, "application/pdf")}
    
    response = client.post("/documents/upload", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.pdf"
    assert "test.pdf" in data["file_path"]
    assert "File uploaded successfully" in data["status"]
    mock_delay.assert_called_once()

@patch("app.api.routes.documents.process_document.delay")
def test_upload_document_invalid_file_type(mock_delay):
    """Tests upload rejection for non-PDF files."""
    files = {"file": ("test.txt", b"content", "text/plain")}
    
    response = client.post("/documents/upload", files=files)
    
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]
    mock_delay.assert_not_called()

@patch("app.api.routes.documents.process_document.delay")
@patch("builtins.open", new_callable=mock_open)
@patch("os.makedirs")
def test_upload_document_file_save_error(mock_makedirs, mock_file, mock_delay):
    """Tests error handling when file save fails."""
    mock_file.side_effect = Exception("Save failed")
    
    files = {"file": ("test.pdf", b"content", "application/pdf")}
    
    response = client.post("/documents/upload", files=files)
    
    assert response.status_code == 500
    assert "Failed to upload file" in response.json()["detail"]
    mock_delay.assert_not_called()

@patch("app.api.routes.documents.celery_app.AsyncResult")
def test_get_task_status_success(mock_async_result):
    """Tests retrieving status for a successful task."""
    mock_result = MagicMock()
    mock_result.status = "SUCCESS"
    mock_result.result = "Processed 10 chunks"
    mock_result.ready.return_value = True
    mock_async_result.return_value = mock_result
    
    response = client.get("/documents/status/test-task-id")
    
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "test-task-id"
    assert data["status"] == "SUCCESS"
    assert data["result"] == "Processed 10 chunks"

@patch("app.api.routes.documents.celery_app.AsyncResult")
def test_get_task_status_pending(mock_async_result):
    """Tests retrieving status for a pending task."""
    mock_result = MagicMock()
    mock_result.status = "PENDING"
    mock_result.ready.return_value = False
    mock_async_result.return_value = mock_result
    
    response = client.get("/documents/status/test-task-id")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "PENDING"
    assert data["result"] is None

@patch("app.api.routes.documents.celery_app.AsyncResult")
def test_get_task_status_not_found(mock_async_result):
    """Tests 404 for non-existent task."""
    mock_async_result.return_value = None
    
    response = client.get("/documents/status/invalid-id")
    
    assert response.status_code == 404
    assert "Task not found" in response.json()["detail"]