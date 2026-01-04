import pytest
from unittest.mock import MagicMock, patch
from app.tasks.tasks import process_document

@patch("app.tasks.tasks.os.path.basename")
@patch("app.tasks.tasks.Qdrant.from_documents")
@patch("app.tasks.tasks.QdrantClient")
@patch("app.tasks.tasks.FastEmbedEmbeddings")
@patch("app.tasks.tasks.RecursiveCharacterTextSplitter")
@patch("app.tasks.tasks.PyPDFLoader")
def test_process_document_success(mock_loader, mock_splitter, mock_embeddings, mock_qdrant_client, mock_qdrant_from, mock_basename):
    """Tests successful document processing."""
    # Mock loader
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = [MagicMock(page_content="Test page")]
    mock_loader.return_value = mock_loader_instance
    
    # Mock splitter
    mock_splitter_instance = MagicMock()
    mock_splitter_instance.split_documents.return_value = [MagicMock(page_content="Chunk 1"), MagicMock(page_content="Chunk 2")]
    mock_splitter.return_value = mock_splitter_instance
    
    # Mock embeddings
    mock_embeddings.return_value = MagicMock()
    
    # Mock QdrantClient
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_qdrant_client.return_value = mock_client
    
    # Mock basename
    mock_basename.return_value = "test.pdf"
    
    # Call the function
    result = process_document("test_path.pdf")
    
    assert result == "Processed 2 chunks from test.pdf"
    mock_loader.assert_called_once_with("test_path.pdf")
    mock_splitter.assert_called_once()
    mock_embeddings.assert_called_once()
    mock_qdrant_from.assert_called_once()

@patch("app.tasks.tasks.PyPDFLoader")
def test_process_document_load_error(mock_loader):
    """Tests error during PDF loading."""
    # Mock loader to raise exception
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.side_effect = Exception("PDF load failed")
    mock_loader.return_value = mock_loader_instance
    
    with pytest.raises(Exception) as excinfo:
        process_document("test_path.pdf")
    
    assert "PDF load failed" in str(excinfo.value)

@patch("app.tasks.tasks.os.path.basename")
@patch("app.tasks.tasks.Qdrant.from_documents")
@patch("app.tasks.tasks.QdrantClient")
@patch("app.tasks.tasks.FastEmbedEmbeddings")
@patch("app.tasks.tasks.RecursiveCharacterTextSplitter")
@patch("app.tasks.tasks.PyPDFLoader")
def test_process_document_store_error(mock_loader, mock_splitter, mock_embeddings, mock_qdrant_client, mock_qdrant_from, mock_basename):
    """Tests error during document storage."""
    # Mock loader
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = [MagicMock(page_content="Test page")]
    mock_loader.return_value = mock_loader_instance
    
    # Mock splitter
    mock_splitter_instance = MagicMock()
    mock_splitter_instance.split_documents.return_value = [MagicMock(page_content="Chunk 1")]
    mock_splitter.return_value = mock_splitter_instance
    
    # Mock embeddings
    mock_embeddings.return_value = MagicMock()
    
    # Mock QdrantClient
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_qdrant_client.return_value = mock_client
    
    # Mock Qdrant.from_documents to raise exception
    mock_qdrant_from.side_effect = Exception("Storage failed")
    
    with pytest.raises(Exception) as excinfo:
        process_document("test_path.pdf")
    
    assert "Storage failed" in str(excinfo.value)