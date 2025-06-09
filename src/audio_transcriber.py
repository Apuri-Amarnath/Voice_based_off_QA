from faster_whisper import WhisperModel
import os

# Store the model globally to avoid reloading
model_cache = {}

def transcribe_audio(audio_path: str, model_name: str = "base", download_root: str = "whisper_models"):
    """
    Transcribes an audio file using faster-whisper.

    Args:
        audio_path (str): The path to the audio file.
        model_name (str): Name of the Whisper model to use (e.g., "tiny", "base", "small").
        download_root (str): The directory to cache the Whisper models.

    Returns:
        str: The transcribed text.
    """
    global model_cache
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return ""

    try:
        if model_name not in model_cache:
            # Use int8 for faster inference on CPU
            model = WhisperModel(
                model_name, 
                device="cpu", 
                compute_type="int8",
                download_root=download_root
            )
            model_cache[model_name] = model
        else:
            model = model_cache[model_name]

        segments, _ = model.transcribe(audio_path, beam_size=5)
        
        transcribed_text = "".join(segment.text for segment in segments)

        return transcribed_text.strip()
    except Exception as e:
        print(f"Error during audio transcription: {e}")
        return ""



