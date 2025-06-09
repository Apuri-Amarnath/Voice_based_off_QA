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

@patch("src.audio_transcriber.whisper.load_model")
def test_transcribe_audio(mock_load_model, dummy_wav_file):
    """
    Test the transcribe_audio function with a mocked whisper model.
    """
    # Arrange
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "This is a test transcription."}
    mock_load_model.return_value = mock_model

    # Act
    transcription = transcribe_audio(dummy_wav_file)

    # Assert
    assert transcription == "This is a test transcription."
    mock_load_model.assert_called_once_with("base")
    mock_model.transcribe.assert_called_once_with(dummy_wav_file)

def test_transcribe_audio_file_not_found():
    """
    Test that transcribe_audio raises FileNotFoundError for a non-existent file.
    """
    with pytest.raises(FileNotFoundError):
        transcribe_audio("non_existent_file.wav")




