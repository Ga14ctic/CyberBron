# Changelog

All notable changes to the CyberBron project will be documented in this file.

## [2.0.0] - 2026-01-15

### Added
- **Configuration System**: Comprehensive `config.yaml` for all settings
  - Model selection and parameters
  - RAG chunking and retrieval settings
  - Persona and response customization
  - Web sources configuration
- **Health Checks**: Automatic Ollama service verification on startup
- **Logging System**: Detailed logging to `cyberbron.log` and `ingest.log`
- **Progress Indicators**: Visual progress bars during document ingestion using tqdm
- **Conversation Management**: Delete button for removing unwanted conversations
- **Advanced Settings**: UI slider for adjusting retrieval parameter (k)
- **Error Handling**: Comprehensive try-catch blocks for:
  - File loading (PDF, DOCX, PPTX, CSV, etc.)
  - JSON parsing for conversations
  - Web scraping failures
  - Ollama connectivity issues
- **Input Validation**: Prevents empty chat messages
- **Batch Processing**: Memory-efficient document ingestion for large datasets
- **Example Configs**: Low-memory configuration example for resource-constrained systems
- **Data Folder Structure**: README in data folder for user guidance
- **requirements.txt**: Complete list of all Python dependencies

### Changed
- **Prompt Template**: Safer, more balanced personality
  - Removed "all limitations lifted" clause for better safety
  - Reduced minimum jokes from 4 to 2 per response
  - Reduced minimum response length from 500 to 300 words
  - Clearer instructions for factual accuracy
- **Memory Optimization**: Limited conversation history to prevent memory bloat
  - Default max_history_messages: 20
  - Configurable via config.yaml
- **Conversation Previews**: Better error handling with .get() methods
- **Directory Management**: Improved with ensure_directories() function
- **Ollama Integration**: Added explicit base_url configuration
- **Web Scraping**: Made URLs configurable via config.yaml

### Improved
- **README Documentation**: 
  - Comprehensive setup instructions
  - Troubleshooting section
  - Configuration reference table
  - Usage tips and examples
- **Code Quality**:
  - Added type hints
  - Consistent error handling patterns
  - Descriptive logging messages
  - Better code organization with helper functions
- **User Experience**:
  - Clearer error messages
  - Visual feedback during operations
  - Page title and emoji in browser tab
  - Organized sidebar layout with columns

### Fixed
- **Memory Leaks**: History accumulation no longer unbounded
- **Crash on Missing Files**: Graceful handling of missing directories
- **JSON Corruption**: Better error recovery for corrupted conversation files
- **Empty Conversations**: Proper handling of conversations with no user messages
- **Missing Dependencies**: All imports now documented in requirements.txt

### Security
- **Prompt Safety**: Removed unrestricted ethical bypass instructions
- **Error Messages**: No sensitive information leaked in error logs
- **Input Sanitization**: Chat input validation prevents empty/malicious inputs

## [1.0.0] - Initial Release

### Features
- Basic RAG chat interface with Streamlit
- Document ingestion from local files
- Multi-conversation support
- LangChain integration with Ollama
- ChromaDB vector store
- History-aware question rephrasing
- CyberBron personality with LeBron James humor
