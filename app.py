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

# --- CONFIGURATION & INITIALIZATION ---
st.set_page_config(layout="wide", page_title="Voice-Based Document Q&A")

MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
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

    # Chat history
    st.session_state.setdefault('query_text', "")
    st.session_state.setdefault('answer', "")
    st.session_state.setdefault('source_documents', [])

    # Models & engines
    st.session_state.setdefault('llm', None)
    st.session_state.setdefault('tts', TextToSpeech())
    
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
def load_llm_cached(n_ctx, temperature, max_tokens):
    """Cached function to load the LLM."""
    if not os.path.exists(MODEL_PATH):
        st.sidebar.error(f"LLM file not found at {MODEL_PATH}.")
        return None
    return load_llm(MODEL_PATH, n_ctx=n_ctx, temperature=temperature, max_tokens=max_tokens)

def build_qa_chain():
    """Builds or rebuilds the QA chain based on session state."""
    if st.session_state.llm and st.session_state.active_pdf_hash:
        vector_store_path = Path(VECTOR_STORE_CACHE_DIR) / f"{st.session_state.active_pdf_hash}.faiss"
        if vector_store_path.exists():
            try:
                vector_store = load_vector_store(str(vector_store_path))
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
            
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                if vector_store_path.exists():
                    st.toast(f"'{uploaded_file.name}' already processed. Loading from cache.", icon="📁")
                else:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.getvalue())
                        
                        vector_store = create_vector_store(tmp.name)
                        save_vector_store(vector_store, str(vector_store_path))
                        os.remove(tmp.name)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                        continue
            
            st.session_state.uploaded_files.append({"name": uploaded_file.name, "hash": file_hash})

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Step 1: Upload PDFs")
    new_files = st.file_uploader(
        "Upload one or more PDF documents.", 
        type="pdf", 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if new_files:
        process_uploaded_files(new_files)

    if st.session_state.uploaded_files:
        st.subheader("Step 2: Select Document")
        
        # Create a list of PDF names for the selectbox
        pdf_options = {f['hash']: f['name'] for f in st.session_state.uploaded_files}
        
        # Function to format the selectbox options
        def format_func(pdf_hash):
            return pdf_options.get(pdf_hash, "Unknown PDF")

        # Select the active PDF
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
            st.session_state.qa_chain = None  # Force QA chain to rebuild

    st.subheader("Step 3: Adjust Settings")
    
    whisper_options = ["tiny", "base", "small"]
    whisper_default_index = whisper_options.index(st.session_state.whisper_model_size) if st.session_state.whisper_model_size in whisper_options else 1
    st.session_state.whisper_model_size = st.selectbox(
        "Whisper Model", whisper_options, index=whisper_default_index,
        help="Model for transcribing your voice. 'tiny' is fastest, 'small' is most accurate."
    )

    num_retriever_docs = st.slider(
        "Source Chunks", min_value=2, max_value=8, value=st.session_state.num_retriever_docs,
        help="How many text chunks from the PDF to give the AI for context. More chunks can yield better answers but take longer."
    )

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
        help="The maximum number of tokens (words) for the generated answer. Shorter answers are faster."
    )
    
    st.session_state.auto_process_answer = st.toggle("Auto-generate answer", value=True, help="If on, the answer will be generated right after transcription.")

    st.markdown("---")
    if st.button("⚠️ Reset Session", use_container_width=True):
        reset_session()

# Load LLM (cached) and build QA chain if necessary
if st.session_state.llm_temp != temp or st.session_state.llm_ctx != ctx or st.session_state.llm_max_tokens != max_tokens:
    st.session_state.llm = load_llm_cached(ctx, temp, max_tokens)
    st.session_state.llm_temp = temp
    st.session_state.llm_ctx = ctx
    st.session_state.llm_max_tokens = max_tokens
    st.session_state.qa_chain = None

if st.session_state.num_retriever_docs != num_retriever_docs:
    st.session_state.num_retriever_docs = num_retriever_docs
    st.session_state.qa_chain = None # Force rebuild for new k value

if st.session_state.llm and not st.session_state.qa_chain:
    build_qa_chain()

# --- MAIN PAGE UI ---
st.title("🎙️ Voice-Based Document Q&A")

if not st.session_state.uploaded_files:
    st.info("Welcome! Please upload a PDF document in the sidebar to get started.")
elif not st.session_state.qa_chain:
    st.warning(f"Processing '{st.session_state.get('active_pdf_name', 'document')}'... Please wait.")
    st.info("The QA chain is being built. This may take a moment.")
else:
    main_tab, inspect_tab = st.tabs(["💬 Chat", "🔬 Inspect Document"])

    with main_tab:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Step 4: Ask a Question")
            audio_info = mic_recorder(
                start_prompt="Click to Record",
                stop_prompt="Click to Stop",
                key='recorder'
            )
        with col2:
            st.subheader("Listen to Last Answer")
            if st.session_state.answer:
                if st.button("Read Answer Aloud", use_container_width=True):
                    with st.spinner("Generating audio..."):
                        st.session_state.tts.speak(st.session_state.answer)
            else:
                st.write("_No answer yet._")
        
        if audio_info and audio_info['bytes'] and audio_info['bytes'] != st.session_state.last_audio_bytes:
            st.session_state.last_audio_bytes = audio_info['bytes']
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_info['bytes'])
                audio_path = tmp.name

            with st.spinner(f"Transcribing with '{st.session_state.whisper_model_size}' model..."):
                st.session_state.query_text = transcribe_audio(
                    audio_path,
                    model_name=st.session_state.whisper_model_size,
                    download_root=WHISPER_MODEL_DIR
                )
            os.remove(audio_path)
            
            # Clear previous answer
            st.session_state.answer = ""
            st.session_state.source_documents = []
            
            if st.session_state.auto_process_answer:
                 st.rerun() # This will trigger the answer generation below
            else:
                 st.rerun()

        if st.session_state.query_text:
            st.markdown("---")
            st.subheader("Your Question")
            st.code(st.session_state.query_text, language=None)
            
            # If auto-process is off, show a button. If on, just run it.
            if not st.session_state.auto_process_answer:
                if st.button("Get Answer", use_container_width=True):
                    with st.spinner("Searching for answers..."):
                        result = st.session_state.qa_chain.invoke({"query": st.session_state.query_text})
                        st.session_state.answer = result["result"]
                        st.session_state.source_documents = result["source_documents"]
                        st.rerun()
            elif not st.session_state.answer: # Auto-process is on, and we don't have an answer yet
                 with st.spinner("Searching for answers..."):
                    result = st.session_state.qa_chain.invoke({"query": st.session_state.query_text})
                    st.session_state.answer = result["result"]
                    st.session_state.source_documents = result["source_documents"]
                    st.rerun()

        if st.session_state.answer:
            st.markdown("---")
            st.subheader("📝 Answer")
            st.success(st.session_state.answer)

            with st.expander("📄 View Sources"):
                if st.session_state.source_documents:
                    for doc in st.session_state.source_documents:
                        st.info(f"**Page {doc.metadata.get('page', 'N/A')}:**\n\n{doc.page_content}")
                else:
                    st.write("No source documents found for this answer.")
    
    with inspect_tab:
        st.header(f"🔍 Document Contents: {st.session_state.get('active_pdf_name', 'N/A')}")
        st.info("This shows a sample of the text chunks from your document that the AI will search through to find answers.")
        
        try:
            vector_store_path = Path(VECTOR_STORE_CACHE_DIR) / f"{st.session_state.active_pdf_hash}.faiss"
            if vector_store_path.exists():
                vector_store = load_vector_store(str(vector_store_path))
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