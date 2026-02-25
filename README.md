# CyberBron - AI-Powered Cybersecurity Study Platform 🛡️

A comprehensive, local AI study platform built specifically for T-Level Cybersecurity students. CyberBron combines document-based RAG, real-time web search, intelligent memory, and AI-powered study tools into a sleek Flask web app — no Streamlit, no cloud, fully local.

---

## 🌟 Key Features

### 💬 **Intelligent Chat Assistant**
- **Hybrid Knowledge Mode**: Combines your course documents, AI knowledge, and real-time web search
- **Streaming responses**: Tokens stream live to the browser as the model generates them
- **Conversational Memory**: Maintains context across your session
- **Auto Web Search**: Automatically searches for current events, CVEs, and recent threats
- **Source Citations**: Clear indicators showing where information comes from (📚 docs, 🧠 AI, 🌐 web)
- **Quick Actions**: Save responses to notes or generate flashcards with one click
- **GPU Busy indicator**: Shows when the AI is occupied so you know when to wait

### 📝 **Complete Notes Management**
- Create, edit, search, and organise study notes
- Tag and folder organisation system
- Markdown rendering in-browser
- AI-powered features: summarisation, expansion, flashcard generation, quiz generation
- Save chat responses directly to notes

### 🎴 **Flashcard System with Spaced Repetition**
- Create manually or generate with AI (up to 100 cards)
- 3D CSS card flip animations
- Spaced repetition algorithm (Easy: 7 days, Medium: 3 days, Hard: 1 day)
- Deck management and progress tracking
- Generate from notes or conversation history

### 📊 **Quiz Mode with AI Grading**
- Multiple choice, true/false, and short answer questions
- AI-powered grading for short answers
- Detailed explanations and score history
- Generate quizzes from your own study materials

### 🎯 **Presentation Generator**
- Generate professional PowerPoint (`.pptx`) files on any cybersecurity topic
- Multiple themes: Professional, Modern, Minimal, Dark
- Optional web research to enrich content
- Configurable slide count and detail level
- Download directly from the browser

### 🌐 **Web Search Integration**
- DuckDuckGo integration for real-time info
- Auto-triggers on keywords: "latest", "recent", "CVE-", year references
- Results shown with source links

### 🧠 **Long-term Memory**
- Remembers your frequently studied topics across sessions
- Tracks learning preferences and progress
- Stores flashcard mastery and quiz history

### 🔒 **Security Features (LAN sharing)**
- Token-based auth for network clients — localhost always bypasses
- GPU protection: only 1 AI job runs at a time (semaphore queue)
- Per-IP rate limiting (5 chat requests/min, 60 API requests/min)
- CSRF protection on all state-changing endpoints
- Input sanitisation and length caps
- Admin endpoints are localhost-only (403 for everyone else)
- File upload validation (type + extension whitelist, size cap)
- Path traversal protection on file downloads

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.com/) installed and running locally
- 8 GB+ RAM recommended
- (Optional) NVIDIA GPU with CUDA for faster generation

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Ga14ctic/CyberBron.git
cd CyberBron
```

**2. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**3. Pull the required Ollama models**
```bash
# Main language model
ollama pull mistral:latest

# Embedding model (for RAG)
ollama pull nomic-embed-text
```

**4. Build your knowledge base** *(optional but recommended)*
```bash
# Drop your PDFs, DOCX, PPTX, MD, or TXT files into data/
mkdir -p data
# Then ingest them:
python ingest.py
```

**5. Launch CyberBron**
```bash
python app.py
```

Open your browser at **http://localhost:5000**

> On Windows you can also double-click `CyberBron.bat`

---

## 🌐 Network Sharing (Share with Classmates)

CyberBron can be shared over your local network so classmates can use the AI without installing Ollama themselves.

**Steps:**
1. Open the **Admin** page at `http://localhost:5000/admin`
2. Click **"Enable Network Sharing"** — this sets `network_mode: true` in `config.yaml` and shows your LAN IP
3. Set an access token in `config.yaml` under `flask.access_token` (default: `changeme` — **change it!**)
4. Restart the app: `python app.py`
5. Share the URL `http://<your-LAN-IP>:5000` and the token with your classmates

Classmates connect by adding the token to their requests via the in-app login prompt or `Authorization: Bearer <token>` header.

> **Note:** Admin and document ingest endpoints are always localhost-only regardless of network mode.

---

## 📁 Project Structure

```
CyberBron/
├── app.py                          # Flask application (main entry point)
├── ingest.py                       # Document ingestion script
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
│
├── templates/                      # Jinja2 HTML templates
│   ├── base.html                   # Base layout with nav & dark theme
│   ├── index.html                  # Chat page
│   ├── notes.html                  # Notes manager
│   ├── flashcards.html             # Flashcard study & creation
│   ├── quiz.html                   # Quiz mode
│   ├── presentations.html          # Presentation generator
│   └── admin.html                  # Admin dashboard (localhost only)
│
├── static/
│   ├── css/main.css                # Dark cyber theme
│   └── js/
│       ├── app.js                  # Shared utilities & CSRF injection
│       ├── chat.js                 # Streaming chat (SSE)
│       ├── notes.js                # Notes CRUD
│       ├── flashcards.js           # Card flip & study logic
│       └── quiz.js                 # Quiz flow & scoring
│
├── services/                       # Backend service classes
│   ├── search_service.py           # DuckDuckGo web search
│   ├── memory_service.py           # Long-term memory
│   ├── notes_service.py            # Notes CRUD
│   ├── flashcard_service.py        # Flashcard management
│   ├── quiz_service.py             # Quiz management
│   └── presentation_service.py     # Presentation request handling
│
├── generators/                     # AI-powered content generators
│   ├── pptx_generator.py           # PowerPoint file generation
│   ├── flashcard_generator.py      # AI flashcard generation
│   └── quiz_generator.py           # AI quiz generation
│
├── data/                           # Your course materials (gitignored)
├── chroma_db/                      # Vector database (gitignored)
├── notes/                          # Saved notes (gitignored)
├── flashcards/                     # Flashcard storage (gitignored)
├── memory/                         # Long-term memory (gitignored)
├── output/                         # Generated presentations (gitignored)
└── exports/                        # Exported notes (gitignored)
```

---

## ⚙️ Configuration

Edit `config.yaml` to customise CyberBron:

```yaml
models:
  llm: "mistral:latest"          # Main language model
  embeddings: "nomic-embed-text" # Embedding model for RAG
  temperature: 0.7               # Response creativity (0–1)

ollama:
  base_url: "http://localhost:11434"
  timeout: 120

rag:
  retrieval_k: 5                 # Documents to retrieve per query
  max_history_messages: 20       # Conversation context window
  hybrid_mode: true              # Use docs + AI knowledge + web

search:
  enabled: true
  max_results: 5

flask:
  secret_key: "change-me-in-production"  # Flask session secret
  port: 5000
  debug: false
  network_mode: false            # true = bind 0.0.0.0 for LAN sharing
  access_token: "changeme"       # Token required from network clients
  max_upload_size_mb: 20
  rate_limit_chat: "5 per minute"
  rate_limit_api: "60 per minute"
  max_prompt_length: 4000

notes:
  auto_tag: true
  default_folder: "General"

flashcards:
  cards_per_generation: 10
  spaced_repetition: true

quiz:
  questions_per_quiz: 10
  default_difficulty: "medium"   # easy / medium / hard

presentations:
  default_slides: 7
  default_theme: "professional"
  enable_search: true
  output_dir: "output"
```

---

## 🔑 API Reference

All endpoints return JSON. State-changing endpoints require the `X-CSRFToken` header (injected automatically by `app.js`).

When `network_mode: true`, remote clients must include:
```
Authorization: Bearer <your-access-token>
```
Localhost requests never require a token.

### Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Send a message (streaming SSE response) |
| `GET`  | `/api/models` | List available Ollama models |
| `GET`  | `/api/queue/status` | Check GPU queue depth |

### Notes
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/notes` | List all notes |
| `POST` | `/api/notes` | Create a note |
| `GET`  | `/api/notes/<id>` | Get a note |
| `PUT`  | `/api/notes/<id>` | Update a note |
| `DELETE` | `/api/notes/<id>` | Delete a note |

### Flashcards
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/flashcards` | List all flashcards |
| `POST` | `/api/flashcards` | Create a flashcard |
| `PUT`  | `/api/flashcards/<id>` | Update a flashcard |
| `DELETE` | `/api/flashcards/<id>` | Delete a flashcard |
| `POST` | `/api/flashcards/<id>/review` | Submit a review result |
| `POST` | `/api/generate/flashcards` | AI-generate flashcards from text |

### Quizzes
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/quizzes` | List all quizzes |
| `POST` | `/api/quizzes` | Create a quiz |
| `POST` | `/api/quizzes/<id>/submit` | Submit answers |
| `POST` | `/api/generate/quiz` | AI-generate quiz from text |

### Presentations
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/presentations/generate` | Generate a `.pptx` file |
| `GET`  | `/api/presentations/<filename>` | Download a generated `.pptx` |

### Admin *(localhost only)*
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/admin` | Admin dashboard |
| `POST` | `/admin/ingest` | Upload & ingest documents |
| `GET/POST` | `/admin/network` | View / toggle network mode |
| `GET`  | `/admin/logs` | View last 50 lines of `cyberbron.log` |

---

## 🎯 Usage Guide

### Chat
1. Type your cybersecurity question in the chat box
2. The AI streams its response live — watch tokens appear in real time
3. If the GPU is busy (another generation running), a yellow banner appears — your request will queue automatically
4. On the last assistant message, click **"Save to Notes"** or **"Make Flashcards"**

### Notes
1. Click **"New Note"** or use the quick-create form
2. Organise with folders and tags
3. Search notes using the search bar
4. Click a note to expand, edit, or delete it

### Flashcards
1. Go to **Flashcards → Study** and select a deck
2. Flip cards with a click — mark "Know it" or "Don't know it"
3. Generate new cards from text under **Create → Generate with AI**

### Quiz
1. Select a quiz from **Take Quiz** or generate one under **Generate Quiz**
2. Answer question by question
3. See your score and explanations at the end

### Presentations
1. Enter a topic and configure slides, theme, and detail level
2. Click **Generate** and wait (may take 30–60 seconds)
3. Click **Download** to save the `.pptx`

---

## 🚧 Troubleshooting

### "Ollama is not running"
- Start Ollama: run `ollama serve` or open the Ollama desktop app
- Confirm it's accessible: `curl http://localhost:11434/api/tags`

### "Vector database not found" / no document context in chat
- Run `python ingest.py` to build the knowledge base
- Make sure you have files in the `data/` directory

### "Model not found"
- Pull the model: `ollama pull mistral:latest`
- Or change the model in `config.yaml` under `models.llm`

### Port 5000 already in use
- Change `flask.port` in `config.yaml` to e.g. `5001`

### Web search not working
- Check your internet connection
- DuckDuckGo may rate-limit; try again in a minute
- Disable entirely: set `search.enabled: false` in `config.yaml`

### Performance / memory issues
- Reduce `rag.retrieval_k` (try `3`)
- Lower `rag.max_history_messages` (try `10`)
- Use a smaller model: `ollama pull mistral:7b` and update `config.yaml`

### Network clients get 401
- Confirm `flask.network_mode: true` in `config.yaml` and restart
- Make sure clients are sending `Authorization: Bearer <token>` with the correct token from `flask.access_token`

---

## 🎨 Themes

### UI Theme
Dark mode with cybersecurity aesthetics (applied globally):
- **Background**: `#0d1117`
- **Cards/Panels**: `#161b22`
- **Primary accent**: `#00ff88` (Cyber Green)
- **Secondary accent**: `#00d4ff` (Cyber Cyan)
- **Text**: `#c9d1d9`

### Presentation Themes
| Theme | Style |
|-------|-------|
| **Professional** | White background, navy title, blue accent |
| **Modern** | Light grey, contemporary feel |
| **Minimal** | Clean white, minimal colour |
| **Dark** | Black background, green title — matches the UI |

---

## 🤝 Contributing

This is a student project for T-Level Cybersecurity. Contributions, bug reports, and suggestions are welcome!

---

## 🙏 Acknowledgments

- Powered by [Ollama](https://ollama.com/) and Mistral AI
- RAG pipeline using [LangChain](https://langchain.com/)
- Vector storage with [ChromaDB](https://www.trychroma.com/)
- Web search via [DuckDuckGo](https://duckduckgo.com/)
- Web framework: [Flask](https://flask.palletsprojects.com/)
- Presentation generation inspired by SlideBron

---

**Built with ❤️ for T-Level Cybersecurity Students**

*Transform your cybersecurity studies with AI-powered learning!* 🛡️