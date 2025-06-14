import streamlit as st
import os
import tempfile
import hashlib
from pathlib import Path

# Import project modules
from src.vector_store import create_vector_store, save_vector_store, load_vector_store
from src.audio_transcriber import transcribe_audio
from src.llm_handler import load_llm
from src.text_to_speech import TextToSpeech
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from streamlit_mic_recorder import mic_recorder
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION & INITIALIZATION ---
st.set_page_config(layout="wide", page_title="Voice-Based Document Q&A")

# Custom CSS to color the mic recorder prompts and the processing button
st.markdown("""
<style>
    /* Mic recorder button - Start state. Targets the p element inside the button. */
    div[data-testid="stToolbar"] button[title="Start recording"] p {
        color: #28a745 !important; /* Green */
        font-weight: bold;
    }
    /* Mic recorder button - Stop state. Targets the p element inside the button. */
    div[data-testid="stToolbar"] button[title="Stop recording"] p {
        color: #dc3545 !important; /* Red */
        font-weight: bold;
    }
    /* Disabled "Processing" button */
    div[data-testid="stButton"] button[disabled] {
        color: #ffc107 !important; /* Amber */
        border-color: #ffc107 !important;
        background-color: #31333F !important;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_CACHE_DIR = "vector_store_cache"
WHISPER_MODEL_DIR = "whisper_models"

# Create directories if they don't exist
Path(VECTOR_STORE_CACHE_DIR).mkdir(exist_ok=True)
Path(WHISPER_MODEL_DIR).mkdir(exist_ok=True)

# --- PROMPT TEMPLATE ---
QA_TEMPLATE_STR = """
Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
---
Context: {context}
---
Question: {question}
---
Helpful Answer:
"""
QA_TEMPLATE = PromptTemplate(template=QA_TEMPLATE_STR, input_variables=["context", "question"])

# --- SESSION STATE MANAGEMENT ---
def initialize_session_state():
    """Initialize all session state variables."""
    # App state
    st.session_state.setdefault('uploaded_files', [])
    st.session_state.setdefault('active_pdf_hash', None)
    st.session_state.setdefault('qa_chain', None)
    st.session_state.setdefault('last_audio_bytes', None)
    st.session_state.setdefault('processing_query', False)

    # Chat history
    st.session_state.setdefault('query_text', "")
    st.session_state.setdefault('answer', "")
    st.session_state.setdefault('source_documents', [])

    # Models & engines
    st.session_state.setdefault('llm', None)
    st.session_state.setdefault('embedding_model', None)
    
    # UI settings / sidebar defaults
    st.session_state.setdefault('auto_process_answer', True)
    st.session_state.setdefault('whisper_model_size', 'base')
    st.session_state.setdefault('num_retriever_docs', 4)
    st.session_state.setdefault('llm_temp', 0.75)
    st.session_state.setdefault('llm_ctx', 2048)
    st.session_state.setdefault('llm_max_tokens', 512)

initialize_session_state()

# --- HELPER FUNCTIONS ---
def get_file_hash(file):
    """Computes SHA256 hash of file bytes."""
    return hashlib.sha256(file.getvalue()).hexdigest()

def reset_session():
    """Resets the Streamlit session state."""
    st.session_state.clear()
    st.rerun()

# --- CORE LOGIC ---
@st.cache_resource
def load_llm_cached(n_ctx, temperature, max_tokens, n_threads):
    """Cached function to load the LLM."""
    if not os.path.exists(MODEL_PATH):
        st.sidebar.error(f"LLM file not found at {MODEL_PATH}.")
        return None
    return load_llm(MODEL_PATH, n_ctx=n_ctx, temperature=temperature, max_tokens=max_tokens, n_threads=n_threads)

@st.cache_resource
def load_embedding_model_cached(model_name: str):
    """Cached function to load the embedding model."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}
    )

def build_qa_chain():
    """Builds or rebuilds the QA chain based on session state."""
    if st.session_state.llm and st.session_state.active_pdf_hash:
        vector_store_path = Path(VECTOR_STORE_CACHE_DIR) / f"{st.session_state.active_pdf_hash}.faiss"
        if vector_store_path.exists():
            try:
                with st.spinner("Loading vector store..."):
                    vector_store = load_vector_store(str(vector_store_path), st.session_state.embedding_model)
                retriever = vector_store.as_retriever(search_kwargs={'k': st.session_state.num_retriever_docs})
                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=st.session_state.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": QA_TEMPLATE}
                )
                st.sidebar.success(f"Ready to chat with '{st.session_state.get('active_pdf_name', 'document')}'!")
            except Exception as e:
                st.sidebar.error(f"Failed to load vector store: {e}")
        else:
            st.session_state.qa_chain = None
    else:
        st.session_state.qa_chain = None

def process_uploaded_files(uploaded_files):
    """Processes newly uploaded PDF files."""
    for uploaded_file in uploaded_files:
        file_hash = get_file_hash(uploaded_file)
        if not any(f['hash'] == file_hash for f in st.session_state.uploaded_files):
            vector_store_path = Path(VECTOR_STORE_CACHE_DIR) / f"{file_hash}.faiss"
            
            with st.spinner(f"Processing '{uploaded_file.name}'... This may take a moment."):
                if vector_store_path.exists():
                    st.toast(f"'{uploaded_file.name}' already processed. Loading from cache.", icon="📁")
                else:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.getvalue())
                        
                        vector_store = create_vector_store(tmp.name, st.session_state.embedding_model)
                        save_vector_store(vector_store, str(vector_store_path))
                        os.remove(tmp.name)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                        continue
            
            st.session_state.uploaded_files.append({"name": uploaded_file.name, "hash": file_hash})

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("Configuration")

    with st.expander("1. Document Management", expanded=True):
        st.subheader("Upload Documents")
        new_files = st.file_uploader(
            "Upload one or more PDF documents.", 
            type="pdf", 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if new_files:
            process_uploaded_files(new_files)

        if st.session_state.uploaded_files:
            st.subheader("Select Document")
            
            pdf_options = {f['hash']: f['name'] for f in st.session_state.uploaded_files}
            
            def format_func(pdf_hash):
                return pdf_options.get(pdf_hash, "Unknown PDF")

            active_hash = st.selectbox(
                "Choose a document to chat with:",
                options=list(pdf_options.keys()),
                format_func=format_func,
                index=0 if not st.session_state.active_pdf_hash else list(pdf_options.keys()).index(st.session_state.active_pdf_hash),
                label_visibility="collapsed"
            )

            if active_hash != st.session_state.active_pdf_hash:
                st.session_state.active_pdf_hash = active_hash
                st.session_state.active_pdf_name = pdf_options[active_hash]
                st.session_state.qa_chain = None

    with st.expander("2. Settings", expanded=True):
        st.subheader("Transcription Model")
        whisper_options = ["tiny", "base", "small"]
        whisper_default_index = whisper_options.index(st.session_state.whisper_model_size) if st.session_state.whisper_model_size in whisper_options else 1
        st.session_state.whisper_model_size = st.selectbox(
            "Whisper Model", whisper_options, index=whisper_default_index,
            help="Model for transcribing your voice. 'tiny' is fastest, 'small' is most accurate.",
            label_visibility="collapsed"
        )

        st.subheader("Retrieval Settings")
        num_retriever_docs = st.slider(
            "Source Chunks", min_value=2, max_value=8, value=st.session_state.num_retriever_docs,
            help="How many text chunks from the PDF to give the AI for context."
        )

        st.subheader("LLM Settings")
        temp = st.slider(
            "LLM Temperature", min_value=0.0, max_value=1.0, value=st.session_state.llm_temp,
            help="Controls the creativity of the AI. Lower values are more factual."
        )
        ctx = st.slider(
            "LLM Context Size", min_value=1024, max_value=4096, value=st.session_state.llm_ctx,
            help="The amount of conversation history and context the AI can remember."
        )
        max_tokens = st.slider(
            "Max Answer Length", min_value=256, max_value=2048, value=st.session_state.llm_max_tokens,
            help="The maximum length of the generated answer."
        )
        
        st.markdown("---")
        st.session_state.auto_process_answer = st.toggle("Auto-generate answer", value=True, help="If on, the answer will be generated right after transcription.")
        
        if st.button("Reset Session", use_container_width=True, type="secondary"):
            reset_session()
        
        n_threads = 10

# Load models (cached) and build QA chain if necessary
if not st.session_state.embedding_model:
    st.session_state.embedding_model = load_embedding_model_cached(EMBEDDING_MODEL_NAME)

if st.session_state.llm is None or st.session_state.llm_temp != temp or st.session_state.llm_ctx != ctx or st.session_state.llm_max_tokens != max_tokens:
    st.session_state.llm = load_llm_cached(ctx, temp, max_tokens, n_threads)
    st.session_state.llm_temp = temp
    st.session_state.llm_ctx = ctx
    st.session_state.llm_max_tokens = max_tokens
    st.session_state.qa_chain = None

if st.session_state.num_retriever_docs != num_retriever_docs:
    st.session_state.num_retriever_docs = num_retriever_docs
    st.session_state.qa_chain = None # Force rebuild for new k value

# Initial check to build QA chain on first load after PDF selection
if st.session_state.llm and st.session_state.embedding_model and not st.session_state.qa_chain and st.session_state.active_pdf_hash:
    build_qa_chain()

# --- MAIN PAGE UI ---
st.title("Voice-Based Document Q&A")

if not st.session_state.uploaded_files:
    st.info("Welcome! Please upload a PDF document in the sidebar to get started.")
elif not st.session_state.qa_chain:
    st.warning(f"Preparing chat for '{st.session_state.get('active_pdf_name', 'document')}'... Please wait.")
    st.info("The QA chain is being built. This may take a moment, especially on the first run.")
else:
    main_tab, inspect_tab = st.tabs(["Chat", "Inspect Document"])

    with main_tab:
        # Create a single container for the chat interface
        chat_container = st.container()

        with chat_container:
            # Top section for asking a question and listening to the answer
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Ask a Question")
                if st.session_state.get('processing_query', False):
                    st.button("Processing... Please wait", disabled=True, use_container_width=True)
                    audio_info = None
                else:
                    audio_info = mic_recorder(
                        start_prompt="Start Recording",
                        stop_prompt="Stop Recording",
                        key='recorder',
                        use_container_width=True
                    )
            with col2:
                st.subheader("Listen to Last Answer")
                if st.session_state.answer:
                    if st.button("Read Answer Aloud", use_container_width=True):
                        with st.spinner("Generating audio..."):
                            tts = TextToSpeech()
                            tts.speak(st.session_state.answer)
                else:
                    st.write("No answer yet.")

            st.markdown("---")

            # This is the core logic block that handles transcription and triggers the QA chain
            if audio_info and audio_info['bytes'] and audio_info['bytes'] != st.session_state.get('last_audio_bytes'):
                st.session_state.last_audio_bytes = audio_info['bytes']
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_info['bytes'])
                    audio_path = tmp.name

                with st.spinner(f"Transcribing with '{st.session_state.whisper_model_size}' model..."):
                    try:
                        st.session_state.query_text = transcribe_audio(
                            audio_path,
                            model_name=st.session_state.whisper_model_size,
                            download_root=WHISPER_MODEL_DIR
                        )
                        st.session_state.processing_query = True
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
                        st.session_state.query_text = ""
                
                os.remove(audio_path)
                st.rerun()

            # Display area for the question, answer, and spinner
            if st.session_state.query_text:
                st.subheader("Your Question")
                st.code(st.session_state.query_text, language=None)

                if st.session_state.processing_query:
                    with st.spinner("Searching for answers..."):
                        try:
                            result = st.session_state.qa_chain.invoke({"query": st.session_state.query_text})
                            st.session_state.answer = result.get("result", "No answer found.")
                            st.session_state.source_documents = result.get("source_documents", [])
                        except Exception as e:
                            st.error(f"QA Chain failed: {e}")
                    
                    st.session_state.processing_query = False
                    st.rerun()

            if st.session_state.answer:
                st.subheader("Answer")
                st.success(st.session_state.answer)

                with st.expander("View Sources", expanded=True):
                    if st.session_state.source_documents:
                        for doc in st.session_state.source_documents:
                            st.info(f"**Page {doc.metadata.get('page', 'N/A')}:**\n\n{doc.page_content}")
                    else:
                        st.write("No source documents were retrieved for this answer.")
    
    with inspect_tab:
        st.header(f"Document Contents: {st.session_state.get('active_pdf_name', 'N/A')}")
        st.info("This shows a sample of the text chunks from your document that the AI will search through to find answers.")
        
        try:
            vector_store_path = Path(VECTOR_STORE_CACHE_DIR) / f"{st.session_state.active_pdf_hash}.faiss"
            if vector_store_path.exists():
                vector_store = load_vector_store(str(vector_store_path), st.session_state.embedding_model)
                docstore = vector_store.docstore._dict
                if docstore:
                    st.write(f"The vector store contains {len(docstore)} text chunks.")
                    sample_docs = list(docstore.values())[:5]
                    for i, doc in enumerate(sample_docs):
                        st.markdown(f"---")
                        st.write(f"**Chunk {i+1} (from Page {doc.metadata.get('page', 'N/A')})**")
                        st.caption(doc.page_content)
                else:
                    st.write("No documents found in the vector store.")
            else:
                st.warning("No vector store found for this document yet.")
        except Exception as e:
            st.error(f"Could not inspect the vector store: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Built with offline-first AI models.") 