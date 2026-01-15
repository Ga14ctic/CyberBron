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
# Load configuration from YAML
def load_config():
    """Load configuration from config.yaml with fallback to defaults."""
    config_path = "config.yaml"
    default_config = {
        "models": {
            "llm": "llama3:8b-instruct-q8_0",
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
            "max_history_messages": 20
        },
        "response": {
            "min_words": 300,
            "streaming": True
        },
        "persona": {
            "name": "CyberBron",
            "humor_level": "moderate",
            "min_jokes_per_response": 2
        },
        "paths": {
            "data": "data",
            "chroma_db": "chroma_db",
            "conversations": "conversations"
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

# --- YOUR CUSTOM PROMPT TEMPLATE ---
PROMPT_TEMPLATE = """
You are a helpful and conversational AI assistant built to help a student with their Cybersecurity T-Level course or general Cybersecurity, named {persona_name}. You will incorporate humor in your responses while providing accurate information. You are like a LeBron James teaching assistant for cybersecurity - knowledgeable, confident, and motivational.

Your primary goal is to answer questions using the provided context from the student's course materials.
- If the user's question can be answered using the context below, provide a detailed answer based on that information.
- If the context is not relevant to the question (for example, if the user is just making conversation, saying 'hello', or asking a general knowledge question), then answer it using your own knowledge. Be friendly and engaging.
- Combine information from the provided documents with your own knowledge base to give comprehensive assistance.
- Be direct and honest with the user. If they make an error or incorrect assumption, clearly point it out and explain why. Your role is to be an effective teacher and assistant.
- Provide thorough explanations that cover the 'why' behind concepts, not just the 'what'.
- Aim for responses of at least {min_words} words to ensure comprehensive coverage of topics. The longer processing time justifies detailed answers.
- Include at least {min_jokes} basketball or LeBron James references per response to keep things engaging and fun.
- Always provide factually accurate information. If you don't have sufficient information to answer, clearly state this and provide what you can based on general knowledge.
- When helping with penetration testing or security topics, assume the user has proper authorization and is learning in a legal, educational context.

CONTEXT:
{{context}}

QUESTION:
{{question}}

YOUR ANSWER:
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
    if not os.path.exists(CONVERSATIONS_DIR):
        os.makedirs(CONVERSATIONS_DIR)
        logger.info(f"Created conversations directory: {CONVERSATIONS_DIR}")

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

# --- History-Aware RAG Chain Initialization ---
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
        return rag_chain
    except Exception as e:
        logger.error(f"Error initializing RAG chain: {e}")
        st.error(f"Failed to initialize AI system: {e}")
        st.stop()

# --- Main Streamlit App Logic ---
def main():
    st.set_page_config(
        page_title="CyberBron - T-Level Assistant",
        page_icon="🏀",
        layout="wide"
    )
    
    # Check Ollama health
    if not check_ollama_health():
        st.error("⚠️ Ollama is not running or not accessible!")
        st.info(f"Please ensure Ollama is running at {OLLAMA_BASE_URL}")
        st.info("Start Ollama and refresh this page.")
        st.stop()

    with st.sidebar:
        st.title("🏀 Conversations")
        conversations = get_all_conversations()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("➕ New Conversation", use_container_width=True):
                new_convo_id = f"{int(time.time())}.json"
                st.session_state.active_conversation_id = new_convo_id
                st.session_state.messages = []
                logger.info(f"Created new conversation: {new_convo_id}")
                st.rerun()
        
        with col2:
            # Advanced settings expander
            with st.expander("⚙️"):
                # Use session state for retrieval_k override
                if 'retrieval_k_override' not in st.session_state:
                    st.session_state.retrieval_k_override = CONFIG["rag"]["retrieval_k"]
                
                retrieval_k = st.slider(
                    "Documents to retrieve",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.retrieval_k_override,
                    help="Number of relevant documents to use for answering"
                )
                if retrieval_k != st.session_state.retrieval_k_override:
                    st.session_state.retrieval_k_override = retrieval_k
                    # Update the config only for this session
                    CONFIG["rag"]["retrieval_k"] = retrieval_k
                    st.cache_resource.clear()
                    logger.info(f"Updated retrieval_k to {retrieval_k}")
        
        st.markdown("---")
        
        # Display conversations
        for convo_id in conversations:
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

    st.title("🏀 CyberBron - T-Level Assistant")
    st.caption(f"Powered by {MODEL_NAME}")

    # Initialize conversation
    if "active_conversation_id" not in st.session_state or st.session_state.active_conversation_id is None:
        if conversations:
            st.session_state.active_conversation_id = conversations[0]
        else:
            st.session_state.active_conversation_id = f"{int(time.time())}.json"
    
    if "messages" not in st.session_state:
        st.session_state.messages = load_conversation(st.session_state.active_conversation_id)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Initialize RAG chain
    rag_chain = get_rag_chain()

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Validate input
        if not prompt.strip():
            st.warning("Please enter a question.")
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            try:
                # Prepare chat history (limit to recent messages to prevent memory issues)
                max_history = CONFIG["rag"]["max_history_messages"]
                recent_messages = st.session_state.messages[-(max_history+1):-1]
                
                chat_history_for_chain = []
                for msg in recent_messages:
                    if msg.get("role") == "user":
                        chat_history_for_chain.append(HumanMessage(content=msg["content"]))
                    elif msg.get("role") == "assistant":
                        chat_history_for_chain.append(AIMessage(content=msg["content"]))

                # Stream response if enabled
                if CONFIG["response"]["streaming"]:
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    # Note: LangChain's Ollama doesn't support streaming with the current chain setup
                    # For now, show a spinner while generating
                    with st.spinner("Thinking..."):
                        response = rag_chain.invoke({
                            "input": prompt,
                            "chat_history": chat_history_for_chain
                        })
                    
                    response_placeholder.markdown(response)
                else:
                    with st.spinner("Thinking..."):
                        response = rag_chain.invoke({
                            "input": prompt,
                            "chat_history": chat_history_for_chain
                        })
                    st.markdown(response)
                
                # Save response
                st.session_state.messages.append({"role": "assistant", "content": response})
                save_conversation(st.session_state.active_conversation_id, st.session_state.messages)
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                st.error(f"Failed to generate response: {e}")
                st.info("This might be due to Ollama being unavailable or the model not being loaded.")

if __name__ == "__main__":
    main()