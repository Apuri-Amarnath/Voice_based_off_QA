import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(file_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Creates a FAISS vector store from a PDF file.

    Args:
        file_path (str): The path to the PDF file.
        model_name (str): The name of the Hugging Face model to use for embeddings.

    Returns:
        FAISS: The created vector store.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}
    )

    # Create the vector store
    vector_store = FAISS.from_documents(texts, embeddings)

    return vector_store

def save_vector_store(vector_store: FAISS, file_path: str):
    """
    Saves a FAISS vector store to a file.

    Args:
        vector_store (FAISS): The vector store to save.
        file_path (str): The path to save the vector store to.
    """
    vector_store.save_local(file_path)

def load_vector_store(file_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Loads a FAISS vector store from a file.

    Args:
        file_path (str): The path to the vector store file.
        model_name (str): The name of the Hugging Face model to use for embeddings.

    Returns:
        FAISS: The loaded vector store.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}
    )
    vector_store = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store
