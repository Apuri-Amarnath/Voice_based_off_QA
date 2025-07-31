import pytest
from unittest.mock import patch, MagicMock
import os
import wave
import numpy as np
from src.audio_transcriber import transcribe_audio

@pytest.fixture
def dummy_wav_file(tmpdir):
    """Create a dummy WAV file for testing."""
    file_path = os.path.join(tmpdir, "test.wav")
    with wave.open(file_path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(np.random.randint(-32768, 32767, 16000).astype(np.int16).tobytes())
    return file_path

@patch('src.audio_transcriber.WhisperModel')
def test_transcribe_audio(MockWhisperModel, dummy_wav_file):
    """
    Test the transcribe_audio function with a mocked WhisperModel.
    """
    # Arrange
    mock_segment = MagicMock()
    mock_segment.text = "This is a test transcription."
    mock_model_instance = MockWhisperModel.return_value
    mock_model_instance.transcribe.return_value = ([mock_segment], MagicMock())

    # Act
    result = transcribe_audio(dummy_wav_file, model_name="tiny")

    # Assert
    assert result == "This is a test transcription."
    MockWhisperModel.assert_called_once_with(
        "tiny",
        device="cpu",
        compute_type="int8",
        download_root="whisper_models"  # Check for default value
    )
    mock_model_instance.transcribe.assert_called_once_with(dummy_wav_file, beam_size=5)

def test_transcribe_audio_file_not_found():
    """
    Test that transcribe_audio returns an empty string for a non-existent file.
    """
    # The function now catches the exception and returns an empty string
    result = transcribe_audio("non_existent_file.wav", model_name="tiny")
    assert result == ""




