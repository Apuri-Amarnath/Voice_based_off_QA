# Voice-Based Document Q&A

## Overview

Voice-Based Document Q&A is an offline-first, multimodal application that lets you chat with your PDF documents using your voice. All AI models run locally for privacy and offline use.

---

## Features

- Voice-based interaction with PDF documents
- Offline-first: No data leaves your machine
- Multi-PDF support and seamless switching
- Audio feedback (text-to-speech)
- Efficient, CPU-optimized transcription and LLM
- Configurable AI parameters (temperature, context, answer length)
- Clean, step-by-step Streamlit UI

---

## Architecture & Tech Stack

- **Frontend:** Streamlit
- **Audio Recording:** streamlit-mic-recorder
- **Orchestration:** LangChain
- **LLM:** Mistral-7B-Instruct (GGUF) via llama-cpp-python
- **Transcription:** faster-whisper
- **Vector Store:** FAISS
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **Text-to-Speech:** pyttsx3

---

## Tested Environment

- **OS:** Windows 10 (Build 19045)
- **CPU:** Intel Core i5-1235U (10 cores)
- **RAM:** 8 GB (minimum), 16 GB (recommended)
- **Python:** 3.10
- **Streamlit:** 1.32+
- **Browser:** Chrome, Edge, Firefox

## Minimum System Requirements

- **CPU:** 4-core Intel/AMD (8+ threads recommended)
- **RAM:** 8 GB minimum
- **Disk:** 2 GB free
- **OS:** Windows 10/11, macOS 12+, or Linux
- **Python:** 3.9+
- **Microphone:** Required
- **No GPU required**

---

## Setup and Installation

1. **Clone the Repository**
    ```bash
    git clone <repository_url>
    cd Voice_based_off_QA
    ```

2. **Create and Activate a Virtual Environment**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # Windows
    # or
    source .venv/bin/activate  # macOS/Linux
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download AI Models**
    - Place the Mistral GGUF model in `models/`
    - Whisper models will be downloaded automatically on first run, or you can manually place them in `whisper_models/`

---

## Running the Application

```bash
streamlit run app.py --server.fileWatcherType none
```

---

## How to Use

1. **Upload PDFs:** Use the file uploader in the sidebar to upload one or more PDF documents.
2. **Select Document:** Choose the document you want to query from the dropdown menu.
3. **Adjust Settings:** Choose Whisper model, number of source chunks, and LLM parameters as needed.
4. **Ask a Question:**
    - Click "Start Recording" and speak your question.
    - Click "Stop Recording" when finished.
    - Your transcribed question will appear, and the AI will automatically start generating an answer.
5. **Listen and Review:**
    - The AI's answer will be displayed on the screen.
    - Click "Read Answer Aloud" to hear the response.
    - Expand the "View Sources" section to see the exact text chunks from the PDF that were used to generate the answer.

---

## Troubleshooting

- **Model Not Found:** Ensure the `.gguf` file is in `models/`
- **Slow Performance:** Use smaller models, smaller PDFs, or a faster CPU
- **Streamlit File Watcher Error:** Use `--server.fileWatcherType none` on Windows
- **Audio Issues:** Check your microphone and browser permissions

---

## Contributing

Pull requests and issues are welcome.

---

