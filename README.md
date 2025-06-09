# 🎙️ Voice-Based Document Q&A

This project is an offline-first, multimodal application that allows you to chat with your PDF documents using your voice. It leverages local AI models for transcription, language understanding, and text-to-speech, ensuring your data remains private and the application runs without an internet connection.

## ✨ Features

*   **Voice-Based Interaction**: Ask questions about your documents by speaking directly into your microphone.
*   **Offline-First**: All AI models (Transcription, LLM, Embeddings) run locally on your machine. No data is sent to the cloud.
*   **Multi-PDF Support**: Upload multiple PDF documents and seamlessly switch between them to ask questions.
*   **Audio Feedback**: Hear the AI's answer spoken back to you.
*   **Performance Optimized**: 
    *   Uses `faster-whisper` for efficient, CPU-based audio transcription.
    *   Caches processed documents (vector stores) to avoid reprocessing the same file.
    *   Allows selection of different Whisper model sizes to balance speed and accuracy.
*   **Configurable AI**: Adjust LLM parameters like temperature, context size, and maximum answer length to fine-tune the AI's behavior.
*   **Intuitive UI**: A clean, step-by-step interface built with Streamlit guides you through the process.

## 🏗️ Architecture & Tech Stack

The application is built with a modular architecture using the following key technologies:

*   **Frontend**: `Streamlit` for the user interface and interactive components.
*   **Audio Recording**: `streamlit-mic-recorder` for capturing microphone input in the browser.
*   **Orchestration**: `LangChain` to chain together the different components (retriever, LLM, prompt).
*   **LLM**: `Mistral-7B-Instruct` (GGUF format) running via `llama-cpp-python` for CPU-based inference.
*   **Transcription**: `faster-whisper` for highly efficient speech-to-text.
*   **Vector Store**: `FAISS` for creating and storing searchable document embeddings.
*   **Embeddings Model**: `sentence-transformers/all-MiniLM-L6-v2` for converting text chunks into vectors.
*   **Text-to-Speech**: `pyttsx3` for generating audio responses.

## 🚀 Setup and Installation

Follow these steps to get the application running on your local machine.

### 1. Clone the Repository

```bash
git clone <repository_url>
cd Voice_Based_Document_Navigation_and_QA
```

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Download AI Models

The application requires two sets of models: the Language Model (LLM) and the Whisper transcription model.

**a) Language Model (Mistral 7B)**

*   Create a directory named `models` in the root of the project.
*   Download the GGUF model file from [Hugging Face: TheBloke/Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf).
*   Place the downloaded `.gguf` file inside the `models/` directory. The final path should be `models/mistral-7b-instruct-v0.1.Q4_K_M.gguf`.

**b) Whisper Model**

*   The first time you run the app, the selected Whisper model (`base` by default) will be downloaded and cached automatically into the `whisper_models/` directory.
*   If your machine has restricted internet access, you can download the model files manually from [Hugging Face:Systran](https://huggingface.co/Systran/faster-whisper-base/tree/main) and place them in a sub-folder inside `whisper_models`.

## ▶️ How to Run the Application

Once the setup is complete, you can run the application using Streamlit.

```bash
streamlit run app.py --server.fileWatcherType none
```

The `--server.fileWatcherType none` flag is recommended to prevent potential conflicts between Streamlit's file watcher and underlying libraries like PyTorch.

## 📖 How to Use the App

1.  **Step 1: Upload PDFs**: Use the file uploader in the sidebar to upload one or more PDF documents.
2.  **Step 2: Select Document**: Once uploaded, choose the document you want to query from the dropdown menu.
3.  **Step 3: Adjust Settings**:
    *   **Whisper Model**: Choose the transcription model size. `tiny` is the fastest but least accurate, while `small` is the most accurate but slowest. `base` is a good balance.
    *   **Source Chunks**: Control how many pieces of the document are used as context for the answer.
    *   **LLM Parameters**: Adjust Temperature, Context Size, and Max Answer Length to control the AI's output.
4.  **Step 4: Ask a Question**:
    *   Click the "Click to Record" button and speak your question.
    *   Click "Click to Stop" when you're done.
    *   Your transcribed question will appear, and the AI will automatically start generating an answer.
5.  **Listen and Review**:
    *   The AI's answer will be displayed on the screen.
    *   Click "Read Answer Aloud" to hear the response.
    *   Expand the "View Sources" section to see the exact text chunks from the PDF that were used to generate the answer.

