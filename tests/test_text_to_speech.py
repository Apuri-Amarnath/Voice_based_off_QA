import pytest
from unittest.mock import patch, MagicMock
from src.text_to_speech import TextToSpeech

@patch('pyttsx3.init')
def test_speak(mock_init):
    """
    Test that the speak method calls the TTS engine correctly.
    """
    # Arrange
    mock_engine = MagicMock()
    mock_init.return_value = mock_engine
    
    tts = TextToSpeech()
    text_to_speak = "Hello, world!"
    
    # Act
    tts.speak(text_to_speak)
    
    # Assert
    mock_init.assert_called_once()
    mock_engine.say.assert_called_once_with(text_to_speak)
    mock_engine.runAndWait.assert_called_once()

@patch('pyttsx3.init')
def test_speak_with_empty_text(mock_init):
    """
    Test that the speak method does not call the engine if the text is empty.
    """
    # Arrange
    mock_engine = MagicMock()
    mock_init.return_value = mock_engine
    
    tts = TextToSpeech()
    
    # Act
    tts.speak("")
    
    # Assert
    mock_engine.say.assert_not_called()
    mock_engine.runAndWait.assert_not_called() 