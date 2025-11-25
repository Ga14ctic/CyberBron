# CyberBron - T-Level AI Assistant

A personalized, local AI study assistant built to support a Cybersecurity T-Level curriculum. This application runs entirely on your local machine, ensuring privacy and full control over your data. It leverages a powerful local LLM via Ollama and uses Retrieval-Augmented Generation (RAG) to provide answers based on your own course notes, textbooks, and documents.

## 🚀 Key Features

*   **🧠 Conversational Memory:** Remembers the context of your conversation for intelligent, human-like follow-up questions.
*   **📚 Custom Knowledge Base:** Ingests your personal study materials, including PDFs, Word documents, PowerPoint slides, and Markdown notes.
*   **🔒 100% Local & Private:** Your documents and conversations never leave your computer. Powered by Ollama.
*   **💬 Multi-Conversation UI:** Create, save, and switch between multiple chat sessions, all stored locally.
*   **💻 GPU Accelerated:** Optimized to run on NVIDIA GPUs for significantly faster response times.
*   ** personality:** A unique "CyberBron" persona, blending expert cybersecurity knowledge with LeBron James-inspired humor and motivation.

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
*   An NVIDIA GPU with CUDA drivers installed (for GPU acceleration)

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
2.  Pull the necessary models. This will download the main chat model and the embedding model.
    ```bash
    ollama pull llama3:8b-instruct-q8_0
    ollama pull nomic-embed-text
    ```

### 4. Build Your Knowledge Base

1.  Add your personal study documents (`.pdf`, `.docx`, `.pptx`, `.md`, `.txt`, etc.) into the `data/` folder.
2.  Run the ingestion script to process your documents and build the local vector database. This only needs to be done once, or whenever you add new files to the `data` folder.
    ```bash
    python ingest.py
    ```
    This will create a `chroma_db/` folder containing your knowledge base.

### 5. Run the Application

Make sure the Ollama application is running in the background. Then, launch the Streamlit app:
```bash
streamlit run app.py
```
Your browser should open with the CyberBron chat interface ready to go!

## 📂 Project Structure

```
.
├── 📄 app.py              # The main Streamlit application logic and UI
├── 📄 ingest.py            # Script to process documents into the vector store
├── 📁 data/                # Folder for your PDF, DOCX, MD course files
├── 📁 conversations/       # Stores saved chat session JSON files
├── 📄 requirements.txt     # Python dependencies for pip
└── 📄 .gitignore          # Specifies files for Git to ignore (like chroma_db)
```

## 📈 Future Improvements (Roadmap)

-   [ ] **Streaming Responses:** Implement `st.write_stream` to display the AI's response word-by-word for a much faster user experience.
-   [ ] **Web Search Agent:** Integrate a web search tool (e.g., DuckDuckGo) to allow the AI to answer questions about current events or topics not in the local documents.
-   [ ] **Automated Note Syncing:** Enhance the `sync_obsidian.py` script to run automatically or with a button in the UI.
