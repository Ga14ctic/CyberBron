"""
Configuration management for CyberBron Backend
"""

from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1", "*"]
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501"
    ]
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./cyberbron.db")
    
    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral:latest"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    
    # Paths
    DATA_DIR: str = "data"
    CHROMA_DB_DIR: str = "chroma_db"
    UPLOAD_DIR: str = "uploads"
    EXPORT_DIR: str = "exports"
    OUTPUT_DIR: str = "output"
    
    # Features
    ENABLE_WEB_SEARCH: bool = True
    ENABLE_MEMORY: bool = True
    ENABLE_RATE_LIMITING: bool = True
    
    # Content generation
    MAX_FLASHCARDS_PER_REQUEST: int = 20
    MAX_QUIZ_QUESTIONS: int = 50
    MAX_PRESENTATION_SLIDES: int = 30
    
    # Pearson T-Level Integration
    PEARSON_SPEC_URL: str = "https://qualifications.pearson.com/content/dam/pdf/T-Level/digital-production-design-development/2020/specification/Digital-TLevel-Specification-Issue-4.pdf"
    PEARSON_SPEC_PATH: str = "data/pearson_cybersecurity_spec.pdf"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
