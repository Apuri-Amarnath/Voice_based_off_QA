import pytest
from unittest.mock import patch, MagicMock
import os
from src.llm_handler import load_llm

@pytest.fixture
def dummy_model_file(tmpdir):
    """Create a dummy model file for testing."""
    file_path = os.path.join(tmpdir, "test.gguf")
    with open(file_path, "w") as f:
        f.write("dummy model content")
    return file_path

@patch("src.llm_handler.LlamaCpp")
def test_load_llm(mock_llama_cpp, dummy_model_file):
    """
    Test the load_llm function with a mocked LlamaCpp class.
    """
    # Arrange
    mock_llama_cpp.return_value = "mock_llm"

    # Act
    llm = load_llm(dummy_model_file, n_threads=4, temperature=0.5, max_tokens=100)

    # Assert
    mock_llama_cpp.assert_called_once_with(
        model_path=dummy_model_file,
        n_ctx=2048,  # Default value
        temperature=0.5,
        max_tokens=100,
        n_threads=4,
        n_batch=512,  # Default value
        verbose=False,
        streaming=False,
        n_gpu_layers=0  # Default value
    )
    assert llm == "mock_llm"

@patch("src.llm_handler.os.path.exists", return_value=False)
def test_load_llm_file_not_found(mock_exists):
    """
    Test that load_llm raises FileNotFoundError for a non-existent model file.
    """
    with pytest.raises(FileNotFoundError):
        load_llm("non_existent_model.gguf")




