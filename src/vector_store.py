import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

def create_vector_store(file_path: str, embeddings: Embeddings):
    """
    Creates a FAISS vector store from a PDF file.

    Args:
        file_path (str): The path to the PDF file.
        embeddings (Embeddings): The embedding model to use.

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

    # Create the vector store using the provided embeddings
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

def load_vector_store(file_path: str, embeddings: Embeddings):
    """
    Loads a FAISS vector store from a file.

    Args:
        file_path (str): The path to the vector store file.
        embeddings (Embeddings): The embedding model to use.

    Returns:
        FAISS: The loaded vector store.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    vector_store = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store
