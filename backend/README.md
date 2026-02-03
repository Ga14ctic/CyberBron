# CyberBron Backend API

FastAPI-based backend for the CyberBron AI-powered learning platform.

## Features

- **Authentication**: User registration, login with JWT tokens
- **Chat**: AI-powered conversations with context retention
- **Notes**: Full CRUD operations with search, tagging, and organization
- **Flashcards**: Spaced repetition learning with AI generation
- **Quizzes**: Quiz creation, AI generation, and attempt tracking
- **Presentations**: AI-powered presentation generation
- **Admin**: User management and system statistics

## Quick Start

### Prerequisites

- Python 3.8+
- SQLite (or PostgreSQL for production)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional)
cp .env.example .env
# Edit .env with your configuration
```

### Running the Server

```bash
# Development mode
python -m backend.main

# Or using uvicorn directly
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

## API Endpoints

### Authentication (`/api/auth`)

- `POST /api/auth/register` - Register a new user
- `POST /api/auth/login` - Login and get access token

### Chat (`/api`)

- `POST /api/chat` - Send a message and get AI response
- `GET /api/conversations` - List user's conversations
- `GET /api/conversations/{id}` - Get specific conversation
- `DELETE /api/conversations/{id}` - Delete conversation

### Notes (`/api`)

- `GET /api/notes` - List all notes (with pagination, filtering)
- `POST /api/notes` - Create a note
- `GET /api/notes/{id}` - Get a note
- `PUT /api/notes/{id}` - Update a note
- `DELETE /api/notes/{id}` - Delete a note
- `GET /api/notes/search` - Search notes

### Flashcards (`/api`)

- `GET /api/flashcards` - List flashcards
- `POST /api/flashcards` - Create flashcard
- `PUT /api/flashcards/{id}` - Update flashcard
- `DELETE /api/flashcards/{id}` - Delete flashcard
- `POST /api/flashcards/generate` - AI generate flashcards
- `POST /api/flashcards/{id}/review` - Submit review response
- `GET /api/flashcards/due` - Get cards due for review

### Quizzes (`/api`)

- `GET /api/quizzes` - List quizzes
- `POST /api/quizzes` - Create quiz
- `POST /api/quizzes/generate` - AI generate quiz
- `GET /api/quizzes/{id}` - Get quiz
- `POST /api/quizzes/{id}/attempt` - Submit quiz attempt
- `GET /api/quizzes/attempts` - Get user's quiz history

### Presentations (`/api`)

- `POST /api/presentations/generate` - Generate presentation
- `GET /api/presentations` - List presentations
- `GET /api/presentations/{filename}/download` - Download presentation

### Admin (`/api/admin`)

- `GET /api/admin/users` - List all users (admin only)
- `PUT /api/admin/users/{id}/activate` - Activate/deactivate user (admin only)
- `PUT /api/admin/users/{id}/admin` - Grant/revoke admin privileges (admin only)
- `GET /api/admin/stats` - Get system statistics (admin only)
- `DELETE /api/admin/users/{id}` - Delete user (admin only)

## Authentication

Most endpoints require authentication using JWT tokens. Include the token in the Authorization header:

```
Authorization: Bearer <your_token_here>
```

### Getting a Token

```bash
# Register
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "username": "user123",
    "password": "securepass123"
  }'

# Login
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user123",
    "password": "securepass123"
  }'
```

## Configuration

Configure the backend using environment variables or `.env` file:

```env
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Security
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Database
DATABASE_URL=sqlite+aiosqlite:///./cyberbron.db

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:latest

# Features
ENABLE_WEB_SEARCH=True
ENABLE_MEMORY=True
MAX_FLASHCARDS_PER_REQUEST=20
MAX_QUIZ_QUESTIONS=50
MAX_PRESENTATION_SLIDES=30
```

## Database

The backend uses SQLAlchemy with async support. By default, it uses SQLite for development.

### Database Models

- **User**: User accounts and authentication
- **Note**: Study notes with tags and folders
- **Flashcard**: Flashcards with spaced repetition data
- **Quiz**: Quizzes with questions
- **QuizAttempt**: Quiz attempt records and scores
- **Conversation**: Chat conversation history

### Migrations

Database tables are automatically created on first run. For production, consider using Alembic for migrations.

## Error Handling

All endpoints follow consistent error response format:

```json
{
  "success": false,
  "error": "Error type",
  "detail": "Detailed error message"
}
```

Common HTTP status codes:
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error

## Development

### Project Structure

```
backend/
├── main.py              # FastAPI application
├── config.py            # Configuration settings
├── database.py          # Database models and session
├── schemas.py           # Pydantic schemas
├── routers/            
│   ├── __init__.py
│   ├── auth.py          # Authentication endpoints
│   ├── chat.py          # Chat endpoints
│   ├── notes.py         # Notes endpoints
│   ├── flashcards.py    # Flashcard endpoints
│   ├── quiz.py          # Quiz endpoints
│   ├── presentations.py # Presentation endpoints
│   └── admin.py         # Admin endpoints
├── utils/
│   └── auth.py          # Authentication utilities
└── middleware/          # Custom middleware
```

### Adding New Endpoints

1. Create/modify router file in `backend/routers/`
2. Define Pydantic schemas in `backend/schemas.py`
3. Add database models to `backend/database.py` if needed
4. Register router in `backend/main.py`

### Testing

```bash
# Run tests (when implemented)
pytest tests/

# Test specific endpoint
curl -X GET http://localhost:8000/health
```

## Production Deployment

### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Gunicorn + Uvicorn

```bash
gunicorn backend.main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Environment Variables for Production

- Set `DEBUG=False`
- Use strong `SECRET_KEY`
- Configure proper `DATABASE_URL` (PostgreSQL recommended)
- Set appropriate `CORS_ORIGINS`
- Enable rate limiting: `ENABLE_RATE_LIMITING=True`

## License

Part of the CyberBron project.
