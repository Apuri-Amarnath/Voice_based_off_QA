import pytest
from unittest.mock import patch, MagicMock
import os
from PyPDF2 import PdfWriter
from src.vector_store import create_vector_store, save_vector_store, load_vector_store
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

@pytest.fixture
def dummy_pdf_file(tmpdir):
    """Create a dummy PDF file for testing."""
    file_path = os.path.join(tmpdir, "test.pdf")
    writer = PdfWriter()
    writer.add_blank_page(width=210, height=297)
    with open(file_path, "wb") as f:
        writer.write(f)
    return file_path

@patch("src.vector_store.PyPDFLoader")
@patch("src.vector_store.HuggingFaceEmbeddings")
@patch("src.vector_store.FAISS")
def test_create_vector_store(mock_faiss, mock_embeddings, mock_loader, dummy_pdf_file):
    """
    Test the create_vector_store function with mocked dependencies.
    """
    # Arrange
    mock_loader.return_value.load.return_value = [Document(page_content="This is a test document.")]
    mock_faiss.from_documents.return_value = "mock_vector_store"

    # Act
    vector_store = create_vector_store(dummy_pdf_file, mock_embeddings)

    # Assert
    mock_loader.assert_called_once_with(dummy_pdf_file)
    mock_faiss.from_documents.assert_called_once()
    assert vector_store == "mock_vector_store"

def test_create_vector_store_file_not_found():
    """
    Test that create_vector_store raises FileNotFoundError for a non-existent file.
    """
    with pytest.raises(FileNotFoundError):
        # Pass a dummy mock for the embeddings object
        create_vector_store("non_existent_file.pdf", MagicMock())

@patch("src.vector_store.FAISS")
def test_save_and_load_vector_store(mock_faiss, tmpdir):
    """
    Test saving and loading a vector store.
    """
    # Arrange
    store_path = os.path.join(tmpdir, "test_store")
    mock_vs = MagicMock()
    mock_embeddings = MagicMock()
    
    # Create a dummy file for the store to be "found" by os.path.exists
    os.makedirs(store_path, exist_ok=True)
    with open(os.path.join(store_path, "index.faiss"), "w") as f:
        f.write("dummy_data")

    mock_faiss.load_local.return_value = "loaded_vector_store"

    # Act
    save_vector_store(mock_vs, store_path)
    loaded_vs = load_vector_store(store_path, mock_embeddings)

    # Assert
    mock_vs.save_local.assert_called_once_with(store_path)
    mock_faiss.load_local.assert_called_once_with(store_path, mock_embeddings, allow_dangerous_deserialization=True)
    assert loaded_vs == "loaded_vector_store"
