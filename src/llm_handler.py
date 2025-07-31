from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

def load_llm(model_path: str, n_ctx: int = 2048, temperature: float = 0.75, max_tokens: int = 1024, n_threads: int = 10):
    """
    Loads the LlamaCpp language model.

    Args:
        model_path (str): The path to the GGUF model file.
        n_ctx (int): The context window size for the model.
        temperature (float): The temperature for sampling.
        max_tokens (int): The maximum number of tokens to generate.
        n_threads (int): The number of CPU threads to use.

    Returns:
        LlamaCpp: The loaded language model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=n_ctx,
        temperature=temperature,
        max_tokens=max_tokens,
        n_threads=n_threads,
        n_batch=512,
        verbose=False,
        streaming=False,
        # Set to a higher value if you have more RAM
        n_gpu_layers=0  # Offload no layers to GPU
    )
    
    return llm
