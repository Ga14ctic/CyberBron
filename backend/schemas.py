"""
Pydantic schemas for request/response validation
"""

from pydantic import BaseModel, EmailStr, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

# User schemas
class UserBase(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=100)

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(UserBase):
    id: int
    is_active: bool
    is_admin: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

# Note schemas
class NoteBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    folder: str = "General"
    tags: List[str] = []
    source: str = "manual"

class NoteCreate(NoteBase):
    pass

class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    folder: Optional[str] = None
    tags: Optional[List[str]] = None

class NoteResponse(NoteBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True

# Flashcard schemas
class FlashcardBase(BaseModel):
    deck: str = "Default"
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)

class FlashcardCreate(FlashcardBase):
    pass

class FlashcardUpdate(BaseModel):
    deck: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None

class FlashcardResponse(FlashcardBase):
    id: int
    user_id: int
    next_review: Optional[datetime]
    interval_days: int
    ease_factor: float
    review_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class FlashcardReview(BaseModel):
    card_id: int
    difficulty: str = Field(..., pattern="^(easy|medium|hard)$")

class FlashcardGenerateRequest(BaseModel):
    content: str = Field(..., min_length=50)
    num_cards: int = Field(10, ge=1, le=20)
    deck: str = "Generated"

# Quiz schemas
class QuizQuestion(BaseModel):
    type: str  # multiple_choice, true_false, short_answer
    question: str
    options: Optional[List[str]] = None
    correct_answer: str
    explanation: str

class QuizBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    questions: List[QuizQuestion]
    difficulty: str = "medium"

class QuizCreate(QuizBase):
    pass

class QuizResponse(QuizBase):
    id: int
    user_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class QuizAttemptCreate(BaseModel):
    quiz_id: int
    answers: Dict[int, str]

class QuizAttemptResponse(BaseModel):
    id: int
    quiz_id: int
    score: int
    max_score: int
    percentage: float
    completed_at: datetime
    
    class Config:
        from_attributes = True

class QuizGenerateRequest(BaseModel):
    content: str = Field(..., min_length=100)
    num_questions: int = Field(10, ge=5, le=50)
    difficulty: str = "medium"
    title: str

# Chat schemas
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: Optional[int] = None
    use_web_search: bool = True

class ChatResponse(BaseModel):
    message: str
    conversation_id: int
    sources: List[str] = []
    web_search_used: bool = False

# Presentation schemas
class PresentationRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    num_slides: int = Field(7, ge=3, le=30)
    theme: str = "professional"
    detail_level: str = "moderate"
    enable_web_search: bool = True
    custom_content: Optional[str] = None

class PresentationResponse(BaseModel):
    filename: str
    download_url: str
    num_slides: int
    created_at: datetime

# Generic response schemas
class SuccessResponse(BaseModel):
    success: bool = True
    message: str
    data: Optional[Any] = None

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None

# Pagination schemas
class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
