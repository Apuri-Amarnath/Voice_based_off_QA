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
    llm = load_llm(dummy_model_file, n_gpu_layers=10, n_batch=1024, verbose=False)

    # Assert
    assert llm == "mock_llm"
    mock_llama_cpp.assert_called_once()
    args, kwargs = mock_llama_cpp.call_args
    assert kwargs['model_path'] == dummy_model_file
    assert kwargs['n_gpu_layers'] == 10
    assert kwargs['n_batch'] == 1024
    assert not kwargs['verbose']

def test_load_llm_file_not_found():
    """
    Test that load_llm raises FileNotFoundError for a non-existent model file.
    """
    with pytest.raises(FileNotFoundError):
        load_llm("non_existent_model.gguf")




