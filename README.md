# CyberBron - T-Level AI Assistant

A personalized, local AI study assistant built to support a Cybersecurity T-Level curriculum. This application runs entirely on your local machine, ensuring privacy and full control over your data. It leverages a powerful local LLM via Ollama and uses Retrieval-Augmented Generation (RAG) to provide answers based on your own course notes, textbooks, and documents.

## 🚀 Key Features

*   **🧠 Conversational Memory:** Remembers the context of your conversation for intelligent, human-like follow-up questions (with configurable history limit to optimize memory).
*   **📚 Custom Knowledge Base:** Ingests your personal study materials, including PDFs, Word documents, PowerPoint slides, and Markdown notes.
*   **🔒 100% Local & Private:** Your documents and conversations never leave your computer. Powered by Ollama.
*   **💬 Multi-Conversation UI:** Create, save, switch between, and delete multiple chat sessions, all stored locally.
*   **💻 GPU Accelerated:** Optimized to run on NVIDIA GPUs for significantly faster response times.
*   **⚙️ Configurable Settings:** Easily customize model parameters, chunk sizes, retrieval settings, and more via `config.yaml`.
*   **🛡️ Health Checks:** Automatic verification that Ollama is running before attempting to use the assistant.
*   **📊 Progress Indicators:** Visual feedback during document ingestion with detailed logging.
*   **🔧 Advanced Controls:** Adjustable retrieval parameters (number of documents to retrieve) directly in the UI.
*   **🏀 Personality:** A unique "CyberBron" persona, blending expert cybersecurity knowledge with LeBron James-inspired humor and motivation.

## 🛠️ Tech Stack

*   **LLM Server:** [Ollama](https://ollama.com/)
*   **LLM Model:** `llama3:8b-instruct-q8_0` (or your preferred model)
*   **UI Framework:** Streamlit
*   **AI Orchestration:** LangChain
*   **Vector Store:** ChromaDB (local)
*   **Embedding Model:** `nomic-embed-text`
*   **Core Language:** Python 3

## ⚙️ Setup & Installation

Follow these steps to get CyberBron running on your local machine.

### Prerequisites

*   Python 3.9+
*   [Git](https://git-scm.com/downloads)
*   An NVIDIA GPU with CUDA drivers installed (recommended for GPU acceleration, but CPU works too)
*   [Ollama](https://ollama.com/) installed and running

### 1. Clone the Repository

Open your terminal and clone this repository to your local machine:
```bash
git clone https://github.com/ga14ctic/CyberBron.git
cd CyberBron
```

### 2. Install Dependencies

Install all the required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3. Set Up Ollama

1.  Download and install **[Ollama](https://ollama.com/)** for your operating system.
2.  Start the Ollama service (it should run in the background).
3.  Pull the necessary models. This will download the main chat model and the embedding model.
    ```bash
    ollama pull llama3:8b-instruct-q8_0
    ollama pull nomic-embed-text
    ```

### 4. Configure Settings (Optional)

CyberBron comes with sensible defaults in `config.yaml`. You can customize:
*   Model names and parameters
*   Chunk sizes for document processing
*   Number of documents to retrieve (k parameter)
*   Response length preferences
*   Humor level and personality settings
*   Web sources for ingestion

Edit `config.yaml` to adjust these settings, or create `config.local.yaml` for personal overrides (this file is ignored by git).

### 5. Build Your Knowledge Base

1.  Create a `data/` folder if it doesn't exist:
    ```bash
    mkdir -p data
    ```
2.  Add your personal study documents (`.pdf`, `.docx`, `.pptx`, `.md`, `.txt`, etc.) into the `data/` folder.
3.  Run the ingestion script to process your documents and build the local vector database:
    ```bash
    python ingest.py
    ```
    This will create a `chroma_db/` folder containing your knowledge base. The script includes:
    *   Progress bars showing ingestion status
    *   Batch processing to handle large document sets efficiently
    *   Error handling for individual file failures
    *   Detailed logging to `ingest.log`

### 6. Run the Application

Make sure the Ollama application is running in the background. Then, launch the Streamlit app:
```bash
streamlit run app.py
```
Your browser should open with the CyberBron chat interface ready to go!

### Troubleshooting

*   **"Ollama is not running"**: Ensure Ollama is installed and the service is started. Check that it's accessible at `http://localhost:11434`.
*   **"Vector database not found"**: Run `python ingest.py` to build your knowledge base first.
*   **Memory issues**: Adjust `chunk_size` and `max_history_messages` in `config.yaml` to lower values.
*   **Check logs**: Review `cyberbron.log` and `ingest.log` for detailed error information.

## 📂 Project Structure

```
.
├── 📄 app.py                # Main Streamlit application with RAG logic
├── 📄 ingest.py             # Document processing and vector store creation
├── 📄 config.yaml           # Configuration file for all settings
├── 📄 requirements.txt      # Python dependencies
├── 📁 data/                 # Your PDF, DOCX, PPTX, MD course files
├── 📁 conversations/        # Saved chat session JSON files (gitignored)
├── 📁 chroma_db/           # Vector database storage (gitignored)
├── 📄 cyberbron.log        # Application logs (gitignored)
├── 📄 ingest.log           # Ingestion logs (gitignored)
└── 📄 .gitignore           # Git ignore rules
```

## 🎯 Usage Tips

### In the Chat Interface

*   **New Conversation**: Click the "➕ New Conversation" button in the sidebar
*   **Delete Conversations**: Use the 🗑️ button next to any conversation
*   **Adjust Retrieval**: Click the ⚙️ icon in the sidebar to adjust how many documents are retrieved (1-10)
*   **Long Responses**: CyberBron is configured to provide detailed, thorough answers - perfect for learning!

### Managing Your Knowledge Base

*   Add new documents to `data/` and re-run `python ingest.py`
*   The ingestion process is idempotent - it will rebuild the entire knowledge base
*   Check `ingest.log` for detailed information about document processing

### Customizing CyberBron

Edit `config.yaml` to customize:
*   **Model**: Switch to different Ollama models (e.g., `llama2`, `mistral`, `codellama`)
*   **Personality**: Adjust humor level and joke frequency
*   **Performance**: Tune chunk sizes, retrieval count, and history limits
*   **Web Sources**: Add or remove URLs to scrape during ingestion

## 🔧 Configuration Reference

Key settings in `config.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `models.llm` | `llama3:8b-instruct-q8_0` | Main language model |
| `models.embeddings` | `nomic-embed-text` | Embedding model for RAG |
| `models.temperature` | `0.7` | Response creativity (0-1) |
| `rag.chunk_size` | `1000` | Document chunk size in characters |
| `rag.chunk_overlap` | `200` | Overlap between chunks |
| `rag.retrieval_k` | `5` | Number of documents to retrieve |
| `rag.max_history_messages` | `20` | Conversation history limit |
| `response.min_words` | `300` | Target response length |
| `persona.min_jokes_per_response` | `2` | Basketball jokes per answer |

## 📈 Recent Improvements

### Version 2.0 Updates

-   ✅ **Configuration System:** All settings now manageable via `config.yaml`
-   ✅ **Health Checks:** Automatic Ollama connectivity verification
-   ✅ **Better Error Handling:** Graceful handling of file loading, JSON parsing, and network errors
-   ✅ **Memory Optimization:** Batch processing for large document sets, limited conversation history
-   ✅ **Progress Indicators:** Visual feedback with tqdm during document ingestion
-   ✅ **Advanced Logging:** Detailed logs in `cyberbron.log` and `ingest.log`
-   ✅ **Conversation Management:** Delete conversations directly from the UI
-   ✅ **Configurable Retrieval:** Adjust the number of retrieved documents in real-time
-   ✅ **Input Validation:** Prevents empty messages and provides clear error messages
-   ✅ **Improved Prompt:** Safer, more balanced personality with adjustable humor levels

### Future Improvements (Roadmap)

-   [ ] **True Streaming Responses:** Implement token-by-token streaming for faster perceived response times
-   [ ] **Web Search Agent:** Integrate DuckDuckGo or similar for current events
-   [ ] **Citation Display:** Show which documents were used to answer questions
-   [ ] **Multi-Model Support:** Easy switching between different LLMs in the UI
-   [ ] **Export Conversations:** Save chats as PDF or Markdown
-   [ ] **Voice Input:** Integrate speech-to-text for hands-free questions
