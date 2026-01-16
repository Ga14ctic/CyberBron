import streamlit as st
import json
import os
import time
import yaml
import logging
import requests
from typing import List, Dict, Generator
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# Import services
from services.search_service import SearchService
from services.memory_service import MemoryService
from services.notes_service import NotesService
from services.flashcard_service import FlashcardService
from services.quiz_service import QuizService
from services.presentation_service import PresentationService

# Import generators
from generators.flashcard_generator import FlashcardGenerator
from generators.quiz_generator import QuizGenerator

# Import UI components
from ui.styles import apply_custom_css
from ui.chat_tab import render_chat_tab
from ui.notes_tab import render_notes_tab
from ui.flashcards_tab import render_flashcards_tab
from ui.quiz_tab import render_quiz_tab
from ui.presentations_tab import render_presentations_tab

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cyberbron.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration & Constants ---
def load_config():
    """Load configuration from config.yaml with fallback to defaults."""
    config_path = "config.yaml"
    default_config = {
        "models": {
            "llm": "mistral:latest",
            "embeddings": "nomic-embed-text",
            "temperature": 0.7
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "timeout": 120
        },
        "rag": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "retrieval_k": 5,
            "max_history_messages": 20,
            "hybrid_mode": True
        },
        "search": {
            "enabled": True,
            "provider": "duckduckgo",
            "max_results": 5,
            "auto_search_keywords": ["latest", "recent", "current", "CVE-"]
        },
        "response": {
            "min_words": 300,
            "streaming": True,
            "show_sources": True
        },
        "memory": {
            "long_term_enabled": True,
            "summarize_after_messages": 30,
            "remember_topics": True
        },
        "notes": {
            "storage": "json",
            "auto_tag": True,
            "default_folder": "General"
        },
        "flashcards": {
            "cards_per_generation": 10,
            "spaced_repetition": True
        },
        "quiz": {
            "questions_per_quiz": 10,
            "default_difficulty": "medium",
            "show_explanations": True
        },
        "presentations": {
            "default_slides": 7,
            "default_theme": "professional",
            "enable_images": True,
            "enable_search": True,
            "output_dir": "output"
        },
        "persona": {
            "name": "CyberBron",
            "humor_level": "moderate",
            "min_jokes_per_response": 1
        },
        "paths": {
            "data": "data",
            "chroma_db": "chroma_db",
            "conversations": "conversations",
            "notes": "notes",
            "flashcards": "flashcards",
            "memory": "memory",
            "output": "output",
            "exports": "exports"
        },
        "ui": {
            "theme": "dark",
            "accent_color": "#00ff88",
            "show_sidebar": True
        }
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info("Configuration loaded from config.yaml")
                return config
        else:
            logger.warning("config.yaml not found, using default configuration")
            return default_config
    except Exception as e:
        logger.error(f"Error loading configuration from config.yaml: {e}, falling back to defaults")
        return default_config

CONFIG = load_config()
CHROMA_PATH = CONFIG["paths"]["chroma_db"]
MODEL_NAME = CONFIG["models"]["llm"]
EMBEDDING_MODEL = CONFIG["models"]["embeddings"]
CONVERSATIONS_DIR = CONFIG["paths"]["conversations"]
OLLAMA_BASE_URL = CONFIG["ollama"]["base_url"]

# Enhanced prompt template with hybrid knowledge support
PROMPT_TEMPLATE = """
You are {persona_name}, an AI assistant specialized in cybersecurity education for T-Level students. You combine knowledge from multiple sources to provide comprehensive, accurate answers.

Your capabilities:
- Access to the student's course materials and documents (📚)
- Your own trained knowledge in cybersecurity (🧠)
- Real-time web search for current information (🌐)

Guidelines:
- Provide thorough, educational responses of at least {min_words} words
- Use the provided context when relevant, but also leverage your cybersecurity knowledge
- When discussing current events, CVEs, or recent developments, indicate if web search was used
- Include at least {min_jokes} basketball or LeBron James reference per response for engagement
- Be honest if you're uncertain - cybersecurity requires accuracy
- Assume all penetration testing questions are for educational purposes with proper authorization
- Explain the 'why' behind concepts, not just the 'what'

CONTEXT FROM DOCUMENTS:
{{context}}

CONVERSATION HISTORY:
{{chat_history}}

CURRENT QUESTION:
{{question}}

YOUR COMPREHENSIVE ANSWER:
"""

# --- Ollama Health Check ---
def check_ollama_health():
    """Check if Ollama service is running and accessible."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("Ollama service is healthy")
            return True
        else:
            logger.warning(f"Ollama returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        return False

# --- Conversation Management Functions ---
def ensure_directories():
    """Ensure required directories exist."""
    for path_key, path_value in CONFIG["paths"].items():
        if not os.path.exists(path_value):
            os.makedirs(path_value)
            logger.info(f"Created directory: {path_value}")

def get_all_conversations():
    """Scans the conversations directory and returns a sorted list of conversation IDs."""
    ensure_directories()
    try:
        files = [f for f in os.listdir(CONVERSATIONS_DIR) if f.endswith('.json')]
        return sorted(files, key=lambda f: os.path.getmtime(os.path.join(CONVERSATIONS_DIR, f)), reverse=True)
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        return []

def load_conversation(convo_id):
    """Loads a specific conversation from its JSON file."""
    file_path = os.path.join(CONVERSATIONS_DIR, convo_id)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                messages = json.load(f)
                logger.info(f"Loaded conversation: {convo_id}")
                return messages
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {convo_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading conversation {convo_id}: {e}")
            return []
    return []

def save_conversation(convo_id, messages):
    """Saves a conversation to its JSON file."""
    ensure_directories()
    file_path = os.path.join(CONVERSATIONS_DIR, convo_id)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=4)
        logger.info(f"Saved conversation: {convo_id}")
    except Exception as e:
        logger.error(f"Error saving conversation {convo_id}: {e}")
        st.error(f"Failed to save conversation: {e}")

def delete_conversation(convo_id):
    """Deletes a conversation file."""
    file_path = os.path.join(CONVERSATIONS_DIR, convo_id)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted conversation: {convo_id}")
            return True
    except Exception as e:
        logger.error(f"Error deleting conversation {convo_id}: {e}")
        st.error(f"Failed to delete conversation: {e}")
    return False

def get_conversation_preview(convo_id):
    """Returns the first user message as a preview for the sidebar."""
    messages = load_conversation(convo_id)
    for msg in messages:
        if msg.get("role") == "user":
            preview = msg.get("content", "")[:50]
            return preview + "..." if len(msg.get("content", "")) > 50 else preview
    return "New Conversation"

# --- Initialize Services ---
@st.cache_resource
def initialize_services():
    """Initialize all services with caching."""
    logger.info("Initializing services")
    
    search_enabled = CONFIG["search"]["enabled"]
    search_service = SearchService(max_results=CONFIG["search"]["max_results"]) if search_enabled else None
    memory_service = MemoryService(CONFIG["paths"]["memory"])
    notes_service = NotesService(CONFIG["paths"]["notes"])
    flashcard_service = FlashcardService(CONFIG["paths"]["flashcards"])
    quiz_service = QuizService(CONFIG["paths"]["flashcards"])
    presentation_service = PresentationService(CONFIG["paths"]["output"])
    
    return {
        "search": search_service,
        "memory": memory_service,
        "notes": notes_service,
        "flashcard": flashcard_service,
        "quiz": quiz_service,
        "presentation": presentation_service
    }

# --- RAG Chain Initialization ---
@st.cache_resource
def get_rag_chain():
    """Initialize the RAG chain with caching."""
    logger.info("Initializing History-Aware RAG Chain")
    try:
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        
        if not os.path.exists(CHROMA_PATH):
            logger.error(f"Vector store not found at {CHROMA_PATH}")
            st.error(f"⚠️ Vector database not found! Please run 'python ingest.py' first.")
            st.stop()
        
        vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        
        retrieval_k = CONFIG["rag"]["retrieval_k"]
        retriever = vector_store.as_retriever(search_kwargs={"k": retrieval_k})
        
        # Initialize LLM with error handling
        try:
            llm = OllamaLLM(
                model=MODEL_NAME,
                base_url=OLLAMA_BASE_URL,
                temperature=CONFIG["models"]["temperature"]
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM model '{MODEL_NAME}': {e}")
            st.error(f"⚠️ Failed to load model '{MODEL_NAME}'. Please ensure it's downloaded with: ollama pull {MODEL_NAME}")
            st.stop()

        rephrasing_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        question_rewriter_chain = rephrasing_prompt | llm | StrOutputParser()

        # Format prompt with config values
        formatted_prompt = PROMPT_TEMPLATE.format(
            persona_name=CONFIG["persona"]["name"],
            min_words=CONFIG["response"]["min_words"],
            min_jokes=CONFIG["persona"]["min_jokes_per_response"]
        )
        
        main_rag_prompt = ChatPromptTemplate.from_messages([
            ("system", formatted_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        def retrieve_docs(input_data):
            rephrased_question = input_data.pop("rephrased_question")
            retrieved_docs = retriever.invoke(rephrased_question)
            input_data["context"] = retrieved_docs
            return input_data

        rag_chain = (
            RunnablePassthrough.assign(rephrased_question=question_rewriter_chain)
            | retrieve_docs
            | RunnablePassthrough.assign(question=itemgetter("input"))
            | main_rag_prompt
            | llm
            | StrOutputParser()
        )
        
        logger.info("RAG chain initialized successfully")
        return rag_chain, llm
    except Exception as e:
        logger.error(f"Error initializing RAG chain: {e}")
        st.error(f"Failed to initialize AI system: {e}")
        st.stop()

# --- Main Streamlit App Logic ---
def main():
    st.set_page_config(
        page_title="CyberBron - AI Study Platform",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS theme
    apply_custom_css()
    
    # Check Ollama health
    if not check_ollama_health():
        st.error("⚠️ Ollama is not running or not accessible!")
        st.info(f"Please ensure Ollama is running at {OLLAMA_BASE_URL}")
        st.info("Start Ollama and refresh this page.")
        st.stop()

    # Initialize services
    services = initialize_services()
    
    # Initialize RAG chain and LLM
    rag_chain, llm = get_rag_chain()
    
    # Initialize generators with LLM
    flashcard_generator = FlashcardGenerator(llm)
    quiz_generator = QuizGenerator(llm)
    
    # Sidebar
    with st.sidebar:
        st.title("🛡️ CyberBron")
        st.caption(f"AI-Powered Cybersecurity Study Platform")
        
        st.divider()
        
        # Status indicators
        with st.expander("📊 System Status", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Ollama", "✅ Online")
                st.metric("Model", MODEL_NAME.split(':')[0])
            with col2:
                if os.path.exists(CHROMA_PATH):
                    st.metric("Knowledge Base", "✅ Ready")
                else:
                    st.metric("Knowledge Base", "❌ Missing")
                
                if CONFIG["search"]["enabled"]:
                    st.metric("Web Search", "✅ Enabled")
                else:
                    st.metric("Web Search", "⚪ Disabled")
        
        st.divider()
        
        # Conversation management
        st.subheader("💬 Conversations")
        conversations = get_all_conversations()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("➕ New Chat", use_container_width=True):
                new_convo_id = f"{int(time.time())}.json"
                st.session_state.active_conversation_id = new_convo_id
                st.session_state.messages = []
                logger.info(f"Created new conversation: {new_convo_id}")
                st.rerun()
        
        with col2:
            with st.expander("⚙️"):
                if 'retrieval_k_override' not in st.session_state:
                    st.session_state.retrieval_k_override = CONFIG["rag"]["retrieval_k"]
                
                retrieval_k = st.slider(
                    "Docs to retrieve",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.retrieval_k_override,
                    help="Number of documents"
                )
                if retrieval_k != st.session_state.retrieval_k_override:
                    st.session_state.retrieval_k_override = retrieval_k
                    CONFIG["rag"]["retrieval_k"] = retrieval_k
                    st.cache_resource.clear()
                    logger.info(f"Updated retrieval_k to {retrieval_k}")
        
        # Display recent conversations
        if conversations:
            st.caption("Recent conversations:")
            for convo_id in conversations[:5]:  # Show last 5
                preview = get_conversation_preview(convo_id)
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    if st.button(preview, key=f"load_{convo_id}", use_container_width=True):
                        st.session_state.active_conversation_id = convo_id
                        st.session_state.messages = load_conversation(convo_id)
                        logger.info(f"Loaded conversation: {convo_id}")
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"del_{convo_id}"):
                        if delete_conversation(convo_id):
                            if st.session_state.get("active_conversation_id") == convo_id:
                                st.session_state.active_conversation_id = None
                                st.session_state.messages = []
                            st.rerun()

    # Main content area
    st.title("🛡️ CyberBron - AI Cybersecurity Study Platform")
    st.caption(f"Powered by {MODEL_NAME} | Hybrid Knowledge Mode: {'✅' if CONFIG['rag']['hybrid_mode'] else '⚪'}")

    # Tabbed interface
    tabs = st.tabs(["💬 Chat", "📝 Notes", "🎴 Flashcards", "📊 Quiz", "🎯 Presentations"])
    
    # Chat Tab
    with tabs[0]:
        render_chat_interface(rag_chain, services, llm)
    
    # Notes Tab
    with tabs[1]:
        render_notes_tab(services["notes"])
    
    # Flashcards Tab
    with tabs[2]:
        render_flashcards_tab(services["flashcard"], flashcard_generator, services["notes"])
    
    # Quiz Tab
    with tabs[3]:
        render_quiz_tab(services["quiz"], quiz_generator, services["notes"])
    
    # Presentations Tab
    with tabs[4]:
        render_presentations_tab(services["presentation"], llm, services["search"])


def render_chat_interface(rag_chain, services, llm):
    """Render the chat interface."""
    # Initialize conversation
    if "active_conversation_id" not in st.session_state or st.session_state.active_conversation_id is None:
        conversations = get_all_conversations()
        if conversations:
            st.session_state.active_conversation_id = conversations[0]
        else:
            st.session_state.active_conversation_id = f"{int(time.time())}.json"
    
    if "messages" not in st.session_state:
        st.session_state.messages = load_conversation(st.session_state.active_conversation_id)

    # Display chat history with quick actions
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Add quick action buttons for last assistant message
            if message["role"] == "assistant" and i == len(st.session_state.messages) - 1:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📝 Save to Notes", key=f"save_note_{i}"):
                        # Extract topic from last user message
                        topic = "Chat Note"
                        if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                            topic = st.session_state.messages[i-1]["content"][:50]
                        
                        services["notes"].create_note(
                            title=topic,
                            content=message["content"],
                            folder="Chat Notes",
                            source="conversation"
                        )
                        st.success("✅ Saved to notes!")
                
                with col2:
                    if st.button("🎴 Make Flashcards", key=f"flashcard_{i}"):
                        st.session_state.generate_flashcards_from_chat = True
                        st.info("Switch to Flashcards tab to generate!")
                
                with col3:
                    if st.button("🎯 Create Slides", key=f"slides_{i}"):
                        st.session_state.generate_presentation_from_chat = True
                        st.info("Switch to Presentations tab to generate!")

    # Chat input
    if prompt := st.chat_input("Ask a question about cybersecurity..."):
        if not prompt.strip():
            st.warning("Please enter a question.")
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if web search should be triggered
        search_triggered = False
        search_results = None
        if CONFIG["search"]["enabled"] and services["search"]:
            keywords = CONFIG["search"]["auto_search_keywords"]
            if services["search"].should_trigger_search(prompt, keywords):
                with st.spinner("🌐 Searching the web..."):
                    search_results = services["search"].search_cybersecurity(prompt)
                    search_triggered = True

        # Generate response
        with st.chat_message("assistant"):
            try:
                max_history = CONFIG["rag"]["max_history_messages"]
                recent_messages = st.session_state.messages[-(max_history+1):-1]
                
                chat_history_for_chain = []
                for msg in recent_messages:
                    if msg.get("role") == "user":
                        chat_history_for_chain.append(HumanMessage(content=msg["content"]))
                    elif msg.get("role") == "assistant":
                        chat_history_for_chain.append(AIMessage(content=msg["content"]))

                with st.spinner("🧠 Thinking..."):
                    response = rag_chain.invoke({
                        "input": prompt,
                        "chat_history": chat_history_for_chain
                    })
                
                # Add source indicator
                if search_triggered:
                    st.caption("🌐 Response enhanced with web search")
                else:
                    st.caption("📚 Response from knowledge base and AI")
                
                st.markdown(response)
                
                # Show search results if available
                if search_results and CONFIG["response"]["show_sources"]:
                    with st.expander("🌐 Web Search Results", expanded=False):
                        for i, result in enumerate(search_results, 1):
                            st.markdown(f"**{i}. {result['title']}**")
                            st.caption(result['snippet'])
                            st.markdown(f"[🔗 {result['link']}]({result['link']})")
                            if i < len(search_results):
                                st.divider()
                
                # Save response
                st.session_state.messages.append({"role": "assistant", "content": response})
                save_conversation(st.session_state.active_conversation_id, st.session_state.messages)
                
                # Update memory
                if CONFIG["memory"]["remember_topics"]:
                    # Simple topic extraction - could be enhanced
                    services["memory"].record_topic("cybersecurity")
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                st.error(f"Failed to generate response: {e}")
                st.info("This might be due to Ollama being unavailable or the model not being loaded.")


if __name__ == "__main__":
    main()
