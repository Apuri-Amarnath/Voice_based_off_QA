import pyttsx3

class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()

    def speak(self, text):
        """
        Speaks the given text using the TTS engine.
        """
        if text:
            self.engine.say(text)
            self.engine.runAndWait()

def get_tts_engine():
    """Factory function to get a TTS engine instance."""
    return TextToSpeech() 