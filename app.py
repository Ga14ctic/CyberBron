"""
CyberBron Flask Application
Replaces the Streamlit UI with a secure Flask web application.
"""
import os
import sys
import json
import html
import time
import logging
import secrets
import threading
import socket
from datetime import datetime
from functools import wraps
from pathlib import Path

import yaml
import requests
from urllib.parse import urlparse

from flask import (
    Flask, render_template, request, jsonify,
    Response, send_file, abort, session, redirect,
    url_for, stream_with_context
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect, generate_csrf
from werkzeug.utils import secure_filename

from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaEmbeddings

from claude_code_llm import ClaudeCodeLLM

from services.search_service import SearchService
from services.memory_service import MemoryService
from services.notes_service import NotesService
from services.flashcard_service import FlashcardService
from services.quiz_service import QuizService
from services.presentation_service import PresentationService
from generators.flashcard_generator import FlashcardGenerator
from generators.quiz_generator import QuizGenerator
from generators.pptx_generator import PPTXGenerator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("cyberbron.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = "config.yaml"

_DEFAULT_CONFIG = {
    "flask": {
        "secret_key": "change-me-in-production",
        "network_mode": False,
        "host": "127.0.0.1",
        "port": 5000,
        "debug": False,
    },
    "security": {
        "access_token": "cyberbron-access",
        "max_input_length": 5000,
        "max_file_size_mb": 10,
        "allowed_upload_types": [".pdf", ".docx", ".txt", ".md", ".pptx"],
        "rate_limit_ai": "5 per minute",
        "rate_limit_global": "60 per minute",
    },
    "models": {
        "llm": "mistral:latest",
        "embeddings": "nomic-embed-text",
        "temperature": 0.7,
    },
    "ollama": {"base_url": "http://localhost:11434", "timeout": 120},
    "rag": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "retrieval_k": 5,
        "max_history_messages": 20,
        "hybrid_mode": True,
    },
    "search": {
        "enabled": True,
        "provider": "duckduckgo",
        "max_results": 5,
        "auto_search_keywords": ["latest", "recent", "current", "CVE-"],
    },
    "response": {"min_words": 300, "streaming": True, "show_sources": True},
    "memory": {
        "long_term_enabled": True,
        "summarize_after_messages": 30,
        "remember_topics": True,
    },
    "notes": {"storage": "json", "auto_tag": True, "default_folder": "General"},
    "flashcards": {"cards_per_generation": 10, "spaced_repetition": True},
    "quiz": {
        "questions_per_quiz": 10,
        "default_difficulty": "medium",
        "show_explanations": True,
    },
    "presentations": {
        "default_slides": 7,
        "default_theme": "professional",
        "enable_images": True,
        "enable_search": True,
        "output_dir": "output",
    },
    "persona": {
        "name": "CyberBron",
        "humor_level": "moderate",
        "min_jokes_per_response": 1,
    },
    "paths": {
        "data": "data",
        "chroma_db": "chroma_db",
        "conversations": "conversations",
        "notes": "notes",
        "flashcards": "flashcards",
        "memory": "memory",
        "output": "output",
        "exports": "exports",
    },
    "ui": {"theme": "dark", "accent_color": "#00ff88", "show_sidebar": True},
}


def load_config():
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f) or {}
            # Deep-merge with defaults
            merged = dict(_DEFAULT_CONFIG)
            for key, val in cfg.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
                    merged[key] = {**merged[key], **val}
                else:
                    merged[key] = val
            return merged
    except Exception as e:
        logger.error(f"Error loading config.yaml: {e}")
    return dict(_DEFAULT_CONFIG)


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        logger.error(f"Error saving config.yaml: {e}")


CONFIG = load_config()

# Convenience constants
CHROMA_PATH = CONFIG["paths"]["chroma_db"]
EMBEDDING_MODEL = CONFIG["models"]["embeddings"]
CONVERSATIONS_DIR = CONFIG["paths"]["conversations"]
OUTPUT_DIR = CONFIG["paths"]["output"]
OLLAMA_BASE_URL = CONFIG["ollama"]["base_url"]

# ---------------------------------------------------------------------------
# Ensure directories
# ---------------------------------------------------------------------------

def ensure_directories():
    for path_value in CONFIG["paths"].values():
        os.makedirs(path_value, exist_ok=True)


ensure_directories()

# ---------------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    CONFIG["flask"].get("secret_key", secrets.token_hex(32)),
)
app.config["WTF_CSRF_TIME_LIMIT"] = None  # No CSRF token expiry
app.config["WTF_CSRF_CHECK_DEFAULT"] = True
app.config["MAX_CONTENT_LENGTH"] = (
    CONFIG["security"]["max_file_size_mb"] * 1024 * 1024
)

csrf = CSRFProtect(app)

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[CONFIG["security"]["rate_limit_global"]],
    storage_uri="memory://",
)

# ---------------------------------------------------------------------------
# Generation queue (GPU protection – only 1 concurrent LLM call)
# ---------------------------------------------------------------------------

class _GenerationQueue:
    def __init__(self):
        self._sem = threading.Semaphore(1)
        self._lock = threading.Lock()
        self._active = 0
        self._waiting = 0

    def __enter__(self):
        with self._lock:
            self._waiting += 1
        self._sem.acquire()
        with self._lock:
            self._waiting -= 1
            self._active += 1
        return self

    def __exit__(self, *_):
        with self._lock:
            self._active -= 1
        self._sem.release()

    @property
    def status(self):
        with self._lock:
            return {"active": self._active > 0, "queue_depth": self._waiting}


gen_queue = _GenerationQueue()

# ---------------------------------------------------------------------------
# AI Services & Generators (initialised once at startup)
# ---------------------------------------------------------------------------

_services = None
_rag_chain = None
_llm = None
_flashcard_gen = None
_quiz_gen = None


def get_services():
    global _services
    if _services is None:
        search_svc = (
            SearchService(max_results=CONFIG["search"]["max_results"])
            if CONFIG["search"]["enabled"]
            else None
        )
        _services = {
            "search": search_svc,
            "memory": MemoryService(CONFIG["paths"]["memory"]),
            "notes": NotesService(CONFIG["paths"]["notes"]),
            "flashcard": FlashcardService(CONFIG["paths"]["flashcards"]),
            "quiz": QuizService(CONFIG["paths"]["flashcards"]),
            "presentation": PresentationService(CONFIG["paths"]["output"]),
        }
        logger.info("Services initialised")
    return _services


def get_rag_chain():
    global _rag_chain, _llm, _flashcard_gen, _quiz_gen
    if _rag_chain is not None:
        return _rag_chain, _llm

    try:
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL
        )

        if not os.path.exists(CHROMA_PATH):
            logger.warning("Vector store not found – RAG disabled until ingest is run")
            return None, None

        vector_store = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=embeddings
        )
        retriever = vector_store.as_retriever(
            search_kwargs={"k": CONFIG["rag"]["retrieval_k"]}
        )

        llm = ClaudeCodeLLM(model=CONFIG["models"].get("claude_model", "sonnet"))

        persona_name = CONFIG["persona"]["name"]
        min_words    = CONFIG["response"]["min_words"]
        min_jokes    = CONFIG["persona"]["min_jokes_per_response"]

        system_prompt = (
            f"You are {persona_name}, an AI assistant specialised in "
            "cybersecurity education for T-Level students.\n\n"
            "Guidelines:\n"
            f"- Provide thorough educational responses of at least {min_words} words\n"
            f"- Include at least {min_jokes} LeBron James reference per response\n"
            "- Use provided context when relevant; leverage your own knowledge too\n"
            "- Be honest about uncertainty – cybersecurity requires accuracy\n"
            "- Treat all penetration testing questions as educational\n\n"
            "CONTEXT FROM DOCUMENTS:\n{context}\n"
        )

        def rag_invoke(input_data):
            question = input_data["input"]
            chat_history = input_data.get("chat_history", [])
            docs = retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)
            history_str = ""
            for msg in chat_history[-10:]:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                history_str += f"{role}: {msg.content}\n"
            prompt = system_prompt.replace("{context}", context)
            prompt += f"\nCONVERSATION HISTORY:\n{history_str}\n"
            prompt += f"\nCURRENT QUESTION:\n{question}\n"
            return llm.invoke(prompt)

        class _RAGChain:
            def invoke(self, data):
                return rag_invoke(data)

        _rag_chain = _RAGChain()
        _llm = llm
        _flashcard_gen = FlashcardGenerator(llm)
        _quiz_gen = QuizGenerator(llm)
        logger.info("RAG chain initialised")
        return _rag_chain, _llm
    except Exception as e:
        logger.error(f"Error initialising RAG chain: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def sanitize_input(text, max_length=None):
    if not isinstance(text, str):
        return ""
    if max_length is None:
        max_length = CONFIG["security"]["max_input_length"]
    text = text[:max_length]
    text = text.replace("\x00", "")
    return text


def get_client_ip():
    """Return raw REMOTE_ADDR – do not trust X-Forwarded-For for auth decisions."""
    return request.remote_addr or ""


def is_localhost(ip=None):
    if ip is None:
        ip = get_client_ip()
    return ip in ("127.0.0.1", "::1", "0:0:0:0:0:0:0:1")


def is_network_mode():
    return bool(CONFIG.get("flask", {}).get("network_mode", False))


def get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


def require_auth(f):
    """Require authentication when running in network mode for non-localhost clients."""

    @wraps(f)
    def decorated(*args, **kwargs):
        if not is_network_mode() or is_localhost():
            return f(*args, **kwargs)

        # Session-based auth (browser)
        if session.get("authenticated"):
            return f(*args, **kwargs)

        # Header-based auth (API / programmatic)
        provided = ""
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            provided = auth[7:]
        else:
            provided = request.headers.get("X-Access-Token", "")

        expected = CONFIG.get("security", {}).get("access_token", "")
        if expected and provided and secrets.compare_digest(provided, expected):
            return f(*args, **kwargs)

        # API requests → 401 JSON
        if request.is_json or request.path.startswith("/api/"):
            return jsonify({"error": "Authentication required"}), 401

        # Browser requests → login page
        return redirect(url_for("login", next=request.path))

    return decorated


def admin_only(f):
    """Restrict endpoint to localhost."""

    @wraps(f)
    def decorated(*args, **kwargs):
        if not is_localhost():
            abort(403)
        return f(*args, **kwargs)

    return decorated


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------

def get_all_conversations():
    try:
        files = [f for f in os.listdir(CONVERSATIONS_DIR) if f.endswith(".json")]
        return sorted(
            files,
            key=lambda f: os.path.getmtime(os.path.join(CONVERSATIONS_DIR, f)),
            reverse=True,
        )
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        return []


def load_conversation(convo_id):
    # Prevent path traversal
    safe_id = secure_filename(convo_id)
    file_path = os.path.join(CONVERSATIONS_DIR, safe_id)
    real_path = os.path.realpath(file_path)
    if not real_path.startswith(os.path.realpath(CONVERSATIONS_DIR)):
        return []
    if os.path.exists(real_path):
        try:
            with open(real_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading conversation {convo_id}: {e}")
    return []


def save_conversation(convo_id, messages):
    safe_id = secure_filename(convo_id)
    file_path = os.path.join(CONVERSATIONS_DIR, safe_id)
    real_path = os.path.realpath(file_path)
    if not real_path.startswith(os.path.realpath(CONVERSATIONS_DIR)):
        return
    try:
        with open(real_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")


def delete_conversation(convo_id):
    safe_id = secure_filename(convo_id)
    file_path = os.path.join(CONVERSATIONS_DIR, safe_id)
    real_path = os.path.realpath(file_path)
    if not real_path.startswith(os.path.realpath(CONVERSATIONS_DIR)):
        return False
    try:
        if os.path.exists(real_path):
            os.remove(real_path)
            return True
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
    return False


def get_conversation_preview(convo_id):
    messages = load_conversation(convo_id)
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return content[:60] + "..." if len(content) > 60 else content
    return "New Conversation"


# ---------------------------------------------------------------------------
# Ollama health
# ---------------------------------------------------------------------------

def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def get_ollama_models():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Presentation content generation (extracted from Streamlit ui/)
# ---------------------------------------------------------------------------

def generate_presentation_content(llm, topic, num_slides, custom_content=None,
                                   detail_level="Moderate", enable_search=False,
                                   search_service=None):
    try:
        from enhanced_content import PearsonTLevelIntegration
        specifics = PearsonTLevelIntegration.get_technical_specifics(topic)
        curriculum_context = PearsonTLevelIntegration.get_topic_context(topic)
        specifics_text = ""
        if specifics:
            specifics_text = "\n\nTechnical Specifics Database:\n"
            for cat, items in specifics.items():
                specifics_text += f"\n{cat.upper()}:\n"
                if isinstance(items, list):
                    for item in items[:5]:
                        specifics_text += f"- {item}\n"
        curriculum_text = ""
        if curriculum_context:
            curriculum_text = (
                f"\n\nT-Level Curriculum: "
                f"{curriculum_context.get('unit','')} - "
                f"{curriculum_context.get('title','')}"
            )
    except (ImportError, Exception):
        specifics_text = ""
        curriculum_text = ""

    search_context = ""
    if enable_search and search_service:
        try:
            results = search_service.search_cybersecurity(topic)
            if results:
                search_context = "\n\nWeb Research Context:\n"
                for r in results[:5]:
                    search_context += f"- {r['title']}: {r['snippet']}\n"
        except Exception as e:
            logger.warning(f"Web search failed: {e}")

    content_instruction = ""
    if custom_content:
        safe_custom = sanitize_input(custom_content, max_length=8000)
        content_instruction = f"\n\nBase the presentation on this content:\n{safe_custom}\n"

    detail_map = {
        "Brief": "Provide 5-8 concise bullet points per slide.",
        "Moderate": "Provide 8-12 detailed points per slide with examples.",
        "Detailed": "Provide 12-20 in-depth points per slide covering theory, practice, and examples.",
    }
    detail_instructions = detail_map.get(detail_level, detail_map["Moderate"])

    prompt = f"""Create a detailed professional presentation for: "{topic}"
Generate exactly {num_slides} content slides.
{curriculum_text}

{detail_instructions}

{content_instruction}{specifics_text}{search_context}

Format as JSON array:
[
  {{
    "title": "Slide Title",
    "content": ["Point 1 with details...", "Point 2 with examples...", "..."]
  }}
]
"""

    try:
        response = llm.invoke(prompt)
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end > start:
            slides = json.loads(response[start:end])
            return [
                s for s in slides[:num_slides]
                if isinstance(s, dict) and "title" in s and "content" in s
            ]
    except Exception as e:
        logger.error(f"Error generating presentation content: {e}")
    return []


# ---------------------------------------------------------------------------
# CSRF token inject context processor
# ---------------------------------------------------------------------------

@app.context_processor
def inject_csrf():
    return {"csrf_token": generate_csrf}


# ---------------------------------------------------------------------------
# Routes – Auth
# ---------------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if is_localhost() or not is_network_mode():
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        token = request.form.get("token", "")
        expected = CONFIG.get("security", {}).get("access_token", "")
        if expected and secrets.compare_digest(token, expected):
            session["authenticated"] = True
            next_url = request.args.get("next", "")
            # Prevent open-redirect: reject any URL with a netloc or scheme
            parsed = urlparse(next_url)
            if parsed.netloc or parsed.scheme or not next_url.startswith("/"):
                next_url = url_for("index")
            return redirect(next_url or url_for("index"))
        error = "Invalid access token"

    return render_template("login.html", error=error)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Routes – Pages
# ---------------------------------------------------------------------------

@app.route("/")
@require_auth
def index():
    conversations = get_all_conversations()
    previews = {c: get_conversation_preview(c) for c in conversations[:10]}
    ollama_ok = check_ollama()
    rag_ready = os.path.exists(CHROMA_PATH)
    return render_template(
        "index.html",
        conversations=conversations[:10],
        previews=previews,
        ollama_ok=ollama_ok,
        rag_ready=rag_ready,
        model_name="Claude Code",
        network_mode=is_network_mode(),
    )


@app.route("/notes")
@require_auth
def notes_page():
    return render_template("notes.html", network_mode=is_network_mode())


@app.route("/flashcards")
@require_auth
def flashcards_page():
    return render_template("flashcards.html", network_mode=is_network_mode())


@app.route("/quiz")
@require_auth
def quiz_page():
    return render_template("quiz.html", network_mode=is_network_mode())


@app.route("/presentations")
@require_auth
def presentations_page():
    svc = get_services()["presentation"]
    themes = svc.get_available_themes()
    return render_template(
        "presentations.html", themes=themes, network_mode=is_network_mode()
    )


# ---------------------------------------------------------------------------
# Routes – Admin (localhost only)
# ---------------------------------------------------------------------------

@app.route("/admin")
@admin_only
def admin():
    status = gen_queue.status
    lan_ip = get_lan_ip()
    network_mode = is_network_mode()
    ollama_ok = check_ollama()
    rag_ready = os.path.exists(CHROMA_PATH)
    return render_template(
        "admin.html",
        queue_status=status,
        lan_ip=lan_ip,
        network_mode=network_mode,
        ollama_ok=ollama_ok,
        rag_ready=rag_ready,
        model_name="Claude Code",
        port=CONFIG["flask"].get("port", 5000),
    )


@app.route("/admin/ingest", methods=["POST"])
@admin_only
def admin_ingest():
    allowed_exts = set(CONFIG["security"]["allowed_upload_types"])

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    original_name = secure_filename(file.filename)
    ext = Path(original_name).suffix.lower()
    if ext not in allowed_exts:
        return jsonify({"error": "File type not allowed"}), 400

    data_dir = CONFIG["paths"]["data"]
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, original_name)
    real_path = os.path.realpath(save_path)
    if not real_path.startswith(os.path.realpath(data_dir)):
        return jsonify({"error": "Invalid filename"}), 400

    file.save(real_path)
    logger.info(f"Admin uploaded file: {original_name}")
    return jsonify({"message": f"File '{original_name}' uploaded. Run ingest.py to index it."})


@app.route("/admin/network", methods=["GET", "POST"])
@admin_only
def admin_network():
    if request.method == "POST":
        action = request.form.get("action", "")
        cfg = load_config()
        if action == "enable":
            cfg["flask"]["network_mode"] = True
            save_config(cfg)
            CONFIG["flask"]["network_mode"] = True
            logger.info("Network mode enabled")
            return jsonify({"network_mode": True, "message": "Network mode enabled. Restart app to rebind."})
        elif action == "disable":
            cfg["flask"]["network_mode"] = False
            save_config(cfg)
            CONFIG["flask"]["network_mode"] = False
            logger.info("Network mode disabled")
            return jsonify({"network_mode": False, "message": "Network mode disabled. Restart app to rebind."})
        return jsonify({"error": "Invalid action"}), 400

    return jsonify({"network_mode": is_network_mode(), "lan_ip": get_lan_ip()})


@app.route("/admin/logs")
@admin_only
def admin_logs():
    try:
        n = int(request.args.get("n", 100))
        n = min(n, 500)
        with open("cyberbron.log", "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        # Return only last n lines (no full path info that might be sensitive)
        return jsonify({"lines": [l.rstrip() for l in lines[-n:]]})
    except FileNotFoundError:
        return jsonify({"lines": []})
    except Exception:
        return jsonify({"error": "Could not read logs"}), 500


# ---------------------------------------------------------------------------
# Routes – API: Queue status
# ---------------------------------------------------------------------------

@app.route("/api/queue/status")
@require_auth
def queue_status():
    return jsonify(gen_queue.status)


# ---------------------------------------------------------------------------
# Routes – API: Models
# ---------------------------------------------------------------------------

@app.route("/api/models")
@require_auth
def api_models():
    return jsonify({"models": get_ollama_models()})


# ---------------------------------------------------------------------------
# Routes – API: Chat (SSE streaming)
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
@require_auth
@limiter.limit(CONFIG["security"]["rate_limit_ai"])
def api_chat():
    data = request.get_json(silent=True) or {}
    message = sanitize_input(data.get("message", ""))
    session_id = sanitize_input(data.get("session_id", ""), max_length=100)

    if not message:
        return jsonify({"error": "Message is required"}), 400

    rag_chain, llm_obj = get_rag_chain()

    if rag_chain is None:
        # No RAG – LLM only fallback
        if llm_obj is None:
            return jsonify({"error": "AI system unavailable"}), 503

    if not session_id:
        session_id = f"{int(time.time())}.json"

    messages = load_conversation(session_id)

    # Build LangChain history
    max_hist = CONFIG["rag"]["max_history_messages"]
    history = []
    for msg in messages[-(max_hist + 1):-1]:
        if msg.get("role") == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            history.append(AIMessage(content=msg["content"]))

    # Web search
    search_info = None
    svc = get_services()
    if CONFIG["search"]["enabled"] and svc["search"]:
        keywords = CONFIG["search"]["auto_search_keywords"]
        if svc["search"].should_trigger_search(message, keywords):
            try:
                results = svc["search"].search_cybersecurity(message)
                if results:
                    search_info = results
            except Exception as e:
                logger.warning(f"Search failed: {e}")

    def generate():
        full_response = []
        try:
            with gen_queue:
                if rag_chain is not None:
                    chain = rag_chain
                    for chunk in chain.stream({"input": message, "chat_history": history}):
                        if chunk:
                            full_response.append(chunk)
                            payload = json.dumps({"chunk": chunk})
                            yield f"data: {payload}\n\n"
                else:
                    for chunk in llm_obj.stream(message):
                        if chunk:
                            full_response.append(chunk)
                            payload = json.dumps({"chunk": chunk})
                            yield f"data: {payload}\n\n"

            # Save conversation
            messages.append({"role": "user", "content": message})
            messages.append({"role": "assistant", "content": "".join(full_response)})
            save_conversation(session_id, messages)

            if CONFIG["memory"]["remember_topics"]:
                try:
                    svc["memory"].record_topic("cybersecurity")
                except Exception:
                    pass

            # Send search results and done signal
            done_payload = {"done": True, "session_id": session_id}
            if search_info and CONFIG["response"]["show_sources"]:
                # Only send title+snippet+link – no internal paths
                safe_results = [
                    {
                        "title": html.escape(r.get("title", "")[:200]),
                        "snippet": html.escape(r.get("snippet", "")[:500]),
                        "link": r.get("link", "")[:500],
                    }
                    for r in search_info[:5]
                ]
                done_payload["search_results"] = safe_results

            yield f"data: {json.dumps(done_payload)}\n\n"

        except Exception as e:
            logger.error(f"Chat generation error: {e}")
            yield f"data: {json.dumps({'error': 'Generation failed. Please try again.'})}\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Routes – API: Conversations
# ---------------------------------------------------------------------------

@app.route("/api/conversations", methods=["GET"])
@require_auth
def api_conversations():
    conversations = get_all_conversations()
    result = []
    for c in conversations[:20]:
        result.append({"id": c, "preview": get_conversation_preview(c)})
    return jsonify(result)


@app.route("/api/conversations/<convo_id>", methods=["GET"])
@require_auth
def api_get_conversation(convo_id):
    convo_id = sanitize_input(convo_id, max_length=100)
    messages = load_conversation(convo_id)
    return jsonify(messages)


@app.route("/api/conversations/<convo_id>", methods=["DELETE"])
@require_auth
def api_delete_conversation(convo_id):
    convo_id = sanitize_input(convo_id, max_length=100)
    success = delete_conversation(convo_id)
    if success:
        return jsonify({"message": "Deleted"})
    return jsonify({"error": "Not found"}), 404


# ---------------------------------------------------------------------------
# Routes – API: Notes
# ---------------------------------------------------------------------------

@app.route("/api/notes", methods=["GET"])
@require_auth
def api_get_notes():
    svc = get_services()["notes"]
    q = sanitize_input(request.args.get("q", ""), max_length=200)
    folder = sanitize_input(request.args.get("folder", ""), max_length=100)
    if q:
        notes = svc.search_notes(q)
    elif folder:
        notes = svc.get_notes_by_folder(folder)
    else:
        notes = svc.get_all_notes()
    return jsonify(notes)


@app.route("/api/notes", methods=["POST"])
@require_auth
def api_create_note():
    data = request.get_json(silent=True) or {}
    title = sanitize_input(data.get("title", ""), max_length=300)
    content = sanitize_input(data.get("content", ""), max_length=50000)
    folder = sanitize_input(data.get("folder", "General"), max_length=100)
    tags_raw = data.get("tags", [])
    if not isinstance(tags_raw, list):
        tags_raw = []
    tags = [sanitize_input(t, max_length=50) for t in tags_raw[:20]]

    if not title:
        return jsonify({"error": "Title is required"}), 400

    note = get_services()["notes"].create_note(
        title=title, content=content, folder=folder, tags=tags
    )
    return jsonify(note), 201


@app.route("/api/notes/<note_id>", methods=["GET"])
@require_auth
def api_get_note(note_id):
    note_id = sanitize_input(note_id, max_length=100)
    note = get_services()["notes"].get_note(note_id)
    if note is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(note)


@app.route("/api/notes/<note_id>", methods=["PUT"])
@require_auth
def api_update_note(note_id):
    note_id = sanitize_input(note_id, max_length=100)
    data = request.get_json(silent=True) or {}
    title = sanitize_input(data.get("title", ""), max_length=300) or None
    content = sanitize_input(data.get("content", ""), max_length=50000) or None
    folder = sanitize_input(data.get("folder", ""), max_length=100) or None
    tags_raw = data.get("tags")
    tags = None
    if isinstance(tags_raw, list):
        tags = [sanitize_input(t, max_length=50) for t in tags_raw[:20]]

    note = get_services()["notes"].update_note(
        note_id, title=title, content=content, folder=folder, tags=tags
    )
    if note is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(note)


@app.route("/api/notes/<note_id>", methods=["DELETE"])
@require_auth
def api_delete_note(note_id):
    note_id = sanitize_input(note_id, max_length=100)
    success = get_services()["notes"].delete_note(note_id)
    if success:
        return jsonify({"message": "Deleted"})
    return jsonify({"error": "Not found"}), 404


# ---------------------------------------------------------------------------
# Routes – API: Flashcards
# ---------------------------------------------------------------------------

@app.route("/api/flashcards", methods=["GET"])
@require_auth
def api_get_flashcards():
    svc = get_services()["flashcard"]
    deck = sanitize_input(request.args.get("deck", ""), max_length=100)
    if deck:
        cards = svc.get_flashcards_by_deck(deck)
    else:
        cards = svc.get_all_flashcards()
    decks = svc.get_all_decks()
    return jsonify({"cards": cards, "decks": decks})


@app.route("/api/flashcards", methods=["POST"])
@require_auth
def api_create_flashcard():
    data = request.get_json(silent=True) or {}
    question = sanitize_input(data.get("question", ""), max_length=2000)
    answer = sanitize_input(data.get("answer", ""), max_length=5000)
    deck = sanitize_input(data.get("deck", "General"), max_length=100)
    topic = sanitize_input(data.get("topic", ""), max_length=100) or None

    if not question or not answer:
        return jsonify({"error": "Question and answer are required"}), 400

    card = get_services()["flashcard"].create_flashcard(
        question=question, answer=answer, deck=deck, topic=topic
    )
    return jsonify(card), 201


@app.route("/api/flashcards/<card_id>", methods=["PUT"])
@require_auth
def api_update_flashcard(card_id):
    card_id = sanitize_input(card_id, max_length=100)
    data = request.get_json(silent=True) or {}
    question = sanitize_input(data.get("question", ""), max_length=2000) or None
    answer = sanitize_input(data.get("answer", ""), max_length=5000) or None
    deck = sanitize_input(data.get("deck", ""), max_length=100) or None
    topic = sanitize_input(data.get("topic", ""), max_length=100) or None

    card = get_services()["flashcard"].update_flashcard(
        card_id, question=question, answer=answer, deck=deck, topic=topic
    )
    if card is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(card)


@app.route("/api/flashcards/<card_id>", methods=["DELETE"])
@require_auth
def api_delete_flashcard(card_id):
    card_id = sanitize_input(card_id, max_length=100)
    success = get_services()["flashcard"].delete_flashcard(card_id)
    if success:
        return jsonify({"message": "Deleted"})
    return jsonify({"error": "Not found"}), 404


@app.route("/api/flashcards/<card_id>/review", methods=["POST"])
@require_auth
def api_review_flashcard(card_id):
    card_id = sanitize_input(card_id, max_length=100)
    data = request.get_json(silent=True) or {}
    difficulty = sanitize_input(data.get("difficulty", "medium"), max_length=20)
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"
    get_services()["flashcard"].record_review(card_id, difficulty)
    return jsonify({"message": "Review recorded"})


@app.route("/api/generate/flashcards", methods=["POST"])
@require_auth
@limiter.limit(CONFIG["security"]["rate_limit_ai"])
def api_generate_flashcards():
    global _flashcard_gen
    data = request.get_json(silent=True) or {}
    text = sanitize_input(data.get("text", ""), max_length=10000)
    num_cards = int(data.get("num_cards", 10))
    num_cards = max(1, min(num_cards, 30))
    deck = sanitize_input(data.get("deck", "Generated"), max_length=100)
    topic = sanitize_input(data.get("topic", ""), max_length=100) or None

    if not text:
        return jsonify({"error": "Text is required"}), 400

    _, llm_obj = get_rag_chain()
    if llm_obj is None:
        return jsonify({"error": "AI system unavailable"}), 503

    if _flashcard_gen is None:
        _flashcard_gen = FlashcardGenerator(llm_obj)

    try:
        with gen_queue:
            cards_data = _flashcard_gen.generate_from_text(text, num_cards=num_cards, topic=topic)
    except Exception as e:
        logger.error(f"Flashcard generation error: {e}")
        return jsonify({"error": "Generation failed"}), 500

    svc = get_services()["flashcard"]
    created = []
    for card in cards_data:
        q = sanitize_input(card.get("question", ""), max_length=2000)
        a = sanitize_input(card.get("answer", ""), max_length=5000)
        if q and a:
            created.append(svc.create_flashcard(question=q, answer=a, deck=deck, topic=topic, source="generated"))

    return jsonify({"cards": created, "count": len(created)})


# ---------------------------------------------------------------------------
# Routes – API: Quizzes
# ---------------------------------------------------------------------------

@app.route("/api/quizzes", methods=["GET"])
@require_auth
def api_get_quizzes():
    return jsonify(get_services()["quiz"].get_all_quizzes())


@app.route("/api/quizzes", methods=["POST"])
@require_auth
def api_create_quiz():
    data = request.get_json(silent=True) or {}
    title = sanitize_input(data.get("title", ""), max_length=300)
    questions = data.get("questions", [])
    topic = sanitize_input(data.get("topic", ""), max_length=100) or None
    difficulty = sanitize_input(data.get("difficulty", "medium"), max_length=20)

    if not title:
        return jsonify({"error": "Title is required"}), 400
    if not isinstance(questions, list):
        return jsonify({"error": "Questions must be a list"}), 400

    quiz = get_services()["quiz"].create_quiz(
        title=title, questions=questions, topic=topic, difficulty=difficulty
    )
    return jsonify(quiz), 201


@app.route("/api/quizzes/<quiz_id>/submit", methods=["POST"])
@require_auth
def api_submit_quiz(quiz_id):
    quiz_id = sanitize_input(quiz_id, max_length=100)
    data = request.get_json(silent=True) or {}
    answers = data.get("answers", {})
    score = data.get("score", 0)
    total = data.get("total_questions", 0)

    result = get_services()["quiz"].submit_quiz_result(
        quiz_id=quiz_id, answers=answers, score=score, total_questions=total
    )
    return jsonify(result)


@app.route("/api/generate/quiz", methods=["POST"])
@require_auth
@limiter.limit(CONFIG["security"]["rate_limit_ai"])
def api_generate_quiz():
    global _quiz_gen
    data = request.get_json(silent=True) or {}
    text = sanitize_input(data.get("text", ""), max_length=10000)
    num_questions = int(data.get("num_questions", 10))
    num_questions = max(1, min(num_questions, 30))
    difficulty = sanitize_input(data.get("difficulty", "medium"), max_length=20)
    title = sanitize_input(data.get("title", "Generated Quiz"), max_length=300)
    topic = sanitize_input(data.get("topic", ""), max_length=100) or None

    if not text:
        return jsonify({"error": "Text is required"}), 400

    _, llm_obj = get_rag_chain()
    if llm_obj is None:
        return jsonify({"error": "AI system unavailable"}), 503

    if _quiz_gen is None:
        _quiz_gen = QuizGenerator(llm_obj)

    try:
        with gen_queue:
            questions = _quiz_gen.generate_quiz(
                text, num_questions=num_questions, difficulty=difficulty, topic=topic
            )
    except Exception as e:
        logger.error(f"Quiz generation error: {e}")
        return jsonify({"error": "Generation failed"}), 500

    quiz = get_services()["quiz"].create_quiz(
        title=title, questions=questions, topic=topic, difficulty=difficulty
    )
    return jsonify(quiz)


# ---------------------------------------------------------------------------
# Routes – API: Presentations
# ---------------------------------------------------------------------------

@app.route("/api/presentations/generate", methods=["POST"])
@require_auth
@limiter.limit(CONFIG["security"]["rate_limit_ai"])
def api_generate_presentation():
    data = request.get_json(silent=True) or {}
    topic = sanitize_input(data.get("topic", ""), max_length=500)
    custom_content = sanitize_input(data.get("custom_content", ""), max_length=10000) or None
    num_slides = int(data.get("num_slides", CONFIG["presentations"]["default_slides"]))
    num_slides = max(3, min(num_slides, 30))
    theme = sanitize_input(data.get("theme", "professional"), max_length=50)
    detail_level = sanitize_input(data.get("detail_level", "Moderate"), max_length=20)
    enable_search = bool(data.get("enable_search", False))

    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    _, llm_obj = get_rag_chain()
    if llm_obj is None:
        return jsonify({"error": "AI system unavailable"}), 503

    svc = get_services()
    search_svc = svc["search"] if enable_search else None

    try:
        with gen_queue:
            slides_content = generate_presentation_content(
                llm=llm_obj,
                topic=topic,
                num_slides=num_slides,
                custom_content=custom_content,
                detail_level=detail_level,
                enable_search=enable_search,
                search_service=search_svc,
            )
    except Exception as e:
        logger.error(f"Presentation generation error: {e}")
        return jsonify({"error": "Generation failed"}), 500

    if not slides_content:
        return jsonify({"error": "No slides generated"}), 500

    generator = PPTXGenerator(theme=theme)
    safe_topic = secure_filename(topic[:50]) or "presentation"
    filename = f"{safe_topic}_{int(time.time())}.pptx"
    output_path = os.path.join(OUTPUT_DIR, filename)
    real_output = os.path.realpath(output_path)
    if not real_output.startswith(os.path.realpath(OUTPUT_DIR)):
        return jsonify({"error": "Invalid output path"}), 400

    try:
        generator.create_presentation(
            title=topic, slides_content=slides_content, output_path=real_output
        )
    except Exception as e:
        logger.error(f"PPTX creation error: {e}")
        return jsonify({"error": "Failed to create presentation file"}), 500

    return jsonify({"filename": filename, "slides": len(slides_content)})


@app.route("/api/presentations", methods=["GET"])
@require_auth
def api_list_presentations():
    try:
        files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".pptx")]
        files.sort(
            key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True
        )
        result = []
        for f in files[:20]:
            fp = os.path.join(OUTPUT_DIR, f)
            result.append({
                "filename": f,
                "size_kb": round(os.path.getsize(fp) / 1024, 1),
                "created": datetime.fromtimestamp(os.path.getmtime(fp)).isoformat(),
            })
        return jsonify(result)
    except Exception:
        return jsonify([])


@app.route("/api/presentations/<filename>")
@require_auth
def api_download_presentation(filename):
    safe_name = secure_filename(filename)
    if not safe_name or not safe_name.endswith(".pptx"):
        abort(404)
    filepath = os.path.realpath(os.path.join(OUTPUT_DIR, safe_name))
    if not filepath.startswith(os.path.realpath(OUTPUT_DIR)):
        abort(403)
    if not os.path.exists(filepath):
        abort(404)
    return send_file(
        filepath,
        as_attachment=True,
        download_name=safe_name,
        mimetype="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(403)
def forbidden(e):
    if request.is_json or request.path.startswith("/api/"):
        return jsonify({"error": "Forbidden"}), 403
    return render_template("base.html", error_code=403, error_msg="Access denied"), 403


@app.errorhandler(404)
def not_found(e):
    if request.is_json or request.path.startswith("/api/"):
        return jsonify({"error": "Not found"}), 404
    return render_template("base.html", error_code=404, error_msg="Page not found"), 404


@app.errorhandler(429)
def rate_limited(e):
    return jsonify({"error": "Too many requests. Please slow down."}), 429


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CyberBron Flask App")
    parser.add_argument(
        "--network",
        action="store_true",
        help="Bind to 0.0.0.0 (LAN accessible)",
    )
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    if args.network:
        CONFIG["flask"]["network_mode"] = True

    network_mode = CONFIG["flask"].get("network_mode", False)
    host = "0.0.0.0" if network_mode else "127.0.0.1"
    port = args.port or CONFIG["flask"].get("port", 5000)
    debug = CONFIG["flask"].get("debug", False)

    logger.info(f"Starting CyberBron on {host}:{port} (network_mode={network_mode})")

    # Warn if default access token is used in network mode
    if network_mode:
        default_token = "cyberbron-access"
        if CONFIG.get("security", {}).get("access_token", "") == default_token:
            logger.warning(
                "SECURITY WARNING: Default access token is in use. "
                "Set a strong 'access_token' in config.yaml or .env before sharing the URL."
            )

    # Initialise services and RAG chain at startup
    get_services()
    get_rag_chain()

    app.run(host=host, port=port, debug=debug, threaded=True)
