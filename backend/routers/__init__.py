"""Router package initialization"""

from .auth import router as auth_router
from .chat import router as chat_router
from .notes import router as notes_router
from .flashcards import router as flashcards_router
from .quiz import router as quiz_router
from .presentations import router as presentations_router
from .admin import router as admin_router

__all__ = [
    "auth_router",
    "chat_router",
    "notes_router",
    "flashcards_router",
    "quiz_router",
    "presentations_router",
    "admin_router",
]
