# CyberBron - AI-Powered Cybersecurity Study Platform 🛡️

A comprehensive, local AI study platform built specifically for T-Level Cybersecurity students. CyberBron combines document-based RAG, real-time web search, intelligent memory, and AI-powered study tools to create the ultimate cybersecurity learning companion.

## 🌟 Key Features

### 💬 **Intelligent Chat Assistant**
- **Hybrid Knowledge Mode**: Combines your course documents, AI knowledge, and web search
- **Conversational Memory**: Remembers context throughout your conversation
- **Auto Web Search**: Automatically searches for current events, CVEs, and recent threats
- **Source Citations**: Clear indicators showing where information comes from (📚 docs, 🧠 AI, 🌐 web)
- **Quick Actions**: Save responses to notes, generate flashcards, or create presentations

### 📝 **Complete Notes Management**
- Create, edit, and organize study notes
- Search across all notes with full-text search
- Tag and folder organization system
- Export notes to Markdown
- AI-powered note features (summarization, flashcard generation)
- Save conversation responses directly to notes

### 🎴 **Flashcard System with Spaced Repetition**
- Create flashcards manually or generate with AI
- Spaced repetition algorithm (Easy: 7 days, Medium: 3 days, Hard: 1 day)
- Deck management and organization
- Study mode with card flipping interface
- Generate flashcards from notes or conversations
- Track review progress and mastery

### 📊 **Quiz Mode with AI Grading**
- Generate quizzes from your study materials using AI
- Multiple choice, true/false, and short answer questions
- AI-powered grading for short answers
- Detailed explanations for each question
- Score tracking and history
- Review incorrect answers with feedback

### 🎯 **Presentation Generator**
- Generate professional PowerPoint presentations
- Multiple themes: Professional, Modern, Minimal, Dark (cybersecurity theme)
- Optional web research for additional content
- Configurable number of slides and detail level
- Download presentations directly
- Based on SlideBron architecture

### 🌐 **Web Search Integration**
- DuckDuckGo integration for real-time information
- Automatic search triggers for keywords like "latest", "recent", "CVE-"
- Cybersecurity-focused search with curated sources
- Search results displayed with sources

### 🧠 **Long-term Memory System**
- Remember user preferences and learning style
- Track frequently studied topics
- Store learned facts and corrections
- Cross-session memory persistence
- Progress tracking across all features

### 🎨 **Modern Cybersecurity UI**
- Dark theme with green/cyan accents
- Tabbed interface for easy navigation
- Status indicators for system health
- Responsive design
- Custom CSS styling for professional look

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.com/) installed and running
- 8GB+ RAM recommended
- (Optional) NVIDIA GPU with CUDA for faster processing

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Ga14ctic/CyberBron.git
cd CyberBron
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up Ollama and pull required models**
```bash
# Install Ollama from https://ollama.com/

# Pull the Mistral model (recommended)
ollama pull mistral:latest

# Pull the embedding model
ollama pull nomic-embed-text
```

4. **Build your knowledge base**
```bash
# Add your study materials to the data/ directory
mkdir -p data
# Copy your PDFs, DOCX, PPTX, MD, TXT files to data/

# Run the ingestion script
python ingest.py
```

5. **Launch CyberBron**
```bash
streamlit run app.py
```

Your browser should automatically open to `http://localhost:8501`

## 📁 Project Structure

```
CyberBron/
├── app.py                          # Main Streamlit application
├── ingest.py                       # Document ingestion script
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
│
├── services/                       # Backend services
│   ├── search_service.py          # DuckDuckGo web search
│   ├── memory_service.py          # Long-term memory
│   ├── notes_service.py           # Notes CRUD operations
│   ├── flashcard_service.py       # Flashcard management
│   ├── quiz_service.py            # Quiz management
│   └── presentation_service.py    # Presentation requests
│
├── generators/                     # AI-powered generators
│   ├── pptx_generator.py          # PowerPoint generation
│   ├── flashcard_generator.py     # AI flashcard generation
│   └── quiz_generator.py          # AI quiz generation
│
├── ui/                            # UI components
│   ├── styles.py                  # Custom CSS theme
│   ├── chat_tab.py                # Chat interface
│   ├── notes_tab.py               # Notes management UI
│   ├── flashcards_tab.py          # Flashcard study UI
│   ├── quiz_tab.py                # Quiz interface
│   └── presentations_tab.py       # Presentation generator UI
│
├── data/                          # Your course materials (gitignored)
├── chroma_db/                     # Vector database (gitignored)
├── conversations/                 # Chat history (gitignored)
├── notes/                         # User notes (gitignored)
├── flashcards/                    # Flashcard storage (gitignored)
├── memory/                        # Long-term memory (gitignored)
├── output/                        # Generated presentations (gitignored)
└── exports/                       # Exported notes (gitignored)
```

## ⚙️ Configuration

Edit `config.yaml` to customize CyberBron:

### Key Settings

```yaml
models:
  llm: "mistral:latest"              # Main language model
  embeddings: "nomic-embed-text"     # Embedding model
  temperature: 0.7                    # Response creativity (0-1)

search:
  enabled: true                       # Enable/disable web search
  max_results: 5                      # Number of search results

rag:
  retrieval_k: 5                      # Documents to retrieve
  max_history_messages: 20            # Conversation context length
  hybrid_mode: true                   # Use docs + AI knowledge + web

notes:
  auto_tag: true                      # Auto-tag notes with AI
  default_folder: "General"           # Default folder for new notes

flashcards:
  cards_per_generation: 10            # Default cards to generate
  spaced_repetition: true             # Enable spaced repetition

quiz:
  questions_per_quiz: 10              # Default quiz length
  default_difficulty: "medium"        # easy, medium, hard

presentations:
  default_slides: 7                   # Default number of slides
  default_theme: "professional"       # Theme selection
  enable_search: true                 # Web search for content
```

## 🎯 Usage Guide

### Chat Tab
1. Ask questions about cybersecurity topics
2. CyberBron will search your documents, use AI knowledge, and web search
3. Use quick action buttons to save responses to notes or generate flashcards
4. Web search automatically triggers for current events and CVEs

### Notes Tab
1. Click "➕ New Note" to create a note
2. Organize with folders and tags
3. Search notes using the search bar
4. Export notes to Markdown
5. Generate flashcards from any note

### Flashcards Tab
- **Study**: Review flashcards with spaced repetition
- **Create**: Make cards manually or generate with AI
- **Decks**: Manage your flashcard collections

### Quiz Tab
- **Take Quiz**: Select and complete a quiz
- **Generate Quiz**: Create quizzes from your study materials
- **Results**: View your quiz history and progress

### Presentations Tab
1. Enter your topic
2. Configure slides, theme, and options
3. Generate presentation with AI
4. Download the .pptx file
5. Optional web research enhances content

## 🔧 Advanced Features

### Web Search Keywords
Automatic web search triggers for:
- "latest", "recent", "current"
- Years: "2024", "2025", "2026"
- CVE identifiers: "CVE-"

### Memory System
CyberBron remembers:
- Your frequently studied topics
- Learning preferences
- Quiz scores and progress
- Flashcard mastery levels

### Curated Cybersecurity Sources
Built-in knowledge of:
- OWASP Top 10
- CISA alerts
- MITRE ATT&CK
- CVE database
- NIST Cybersecurity Framework

## 🎨 Themes

### Available Presentation Themes
- **Professional**: Classic corporate style
- **Modern**: Clean, contemporary design
- **Minimal**: Simple and elegant
- **Dark**: Cybersecurity-themed with green accents

### UI Theme
Dark mode with cybersecurity aesthetics:
- Primary: Cyber Green (#00ff88)
- Accent: Cyber Cyan (#00d4ff)
- Background: Dark (#0d1117)

## 🚧 Troubleshooting

### "Ollama is not running"
- Start Ollama service
- Check it's accessible at `http://localhost:11434`
- Test with: `curl http://localhost:11434/api/tags`

### "Vector database not found"
- Run `python ingest.py` to build your knowledge base
- Ensure you have documents in the `data/` directory

### "Model not found"
- Download the model: `ollama pull mistral:latest`
- Or use a different model in `config.yaml`

### Web search not working
- Check internet connection
- DuckDuckGo may have rate limits
- Disable in config if needed: `search.enabled: false`

### Memory/Performance issues
- Reduce `retrieval_k` in config (try 3 instead of 5)
- Lower `max_history_messages` (try 10 instead of 20)
- Use a smaller model like `mistral:7b`

## 🤝 Contributing

This is a student project for T-Level Cybersecurity. Contributions, suggestions, and feedback are welcome!

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Ollama](https://ollama.com/) and Mistral AI
- RAG implementation using [LangChain](https://langchain.com/)
- Vector storage with [ChromaDB](https://www.trychroma.com/)
- Web search via [DuckDuckGo](https://duckduckgo.com/)
- Presentation generation inspired by SlideBron

---

**Built with ❤️ for T-Level Cybersecurity Students**

*Transform your cybersecurity studies with AI-powered learning!* 🛡️
