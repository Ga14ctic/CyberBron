# CyberBron Backend - Quick Start Guide

## Setup & Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment (optional but recommended)
cp .env.example .env
# Edit .env with your configuration

# 3. Run the server
python -m backend.main

# Or with auto-reload for development
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

## Testing the API

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. View API Documentation
Open in browser: http://localhost:8000/api/docs

### 3. Run Example Script
```bash
python backend/example_usage.py
```

## Quick API Reference

### Authentication
```bash
# Register
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "username": "user123",
    "password": "SecurePass123!",
    "full_name": "John Doe"
  }'

# Response includes token
{
  "access_token": "eyJ0eXAiOiJKV1...",
  "token_type": "bearer",
  "user": {...}
}

# Login
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user123", "password": "SecurePass123!"}'

# Use token in subsequent requests
curl -X GET http://localhost:8000/api/notes \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

### Notes
```bash
# Create note
curl -X POST http://localhost:8000/api/notes \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Python Basics",
    "content": "Python is awesome!",
    "folder": "Programming",
    "tags": ["python", "learning"]
  }'

# List notes
curl http://localhost:8000/api/notes?page=1&page_size=10 \
  -H "Authorization: Bearer $TOKEN"

# Search notes
curl "http://localhost:8000/api/notes/search?q=python" \
  -H "Authorization: Bearer $TOKEN"

# Get note
curl http://localhost:8000/api/notes/1 \
  -H "Authorization: Bearer $TOKEN"

# Update note
curl -X PUT http://localhost:8000/api/notes/1 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "Updated Title"}'

# Delete note
curl -X DELETE http://localhost:8000/api/notes/1 \
  -H "Authorization: Bearer $TOKEN"
```

### Flashcards
```bash
# Create flashcard
curl -X POST http://localhost:8000/api/flashcards \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "deck": "Python",
    "question": "What is a list?",
    "answer": "An ordered collection of items"
  }'

# Get due flashcards
curl http://localhost:8000/api/flashcards/due?limit=10 \
  -H "Authorization: Bearer $TOKEN"

# Review flashcard
curl -X POST http://localhost:8000/api/flashcards/1/review \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"card_id": 1, "difficulty": "easy"}'
```

### Quizzes
```bash
# Create quiz
curl -X POST http://localhost:8000/api/quizzes \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Python Quiz",
    "difficulty": "easy",
    "questions": [
      {
        "type": "multiple_choice",
        "question": "What is Python?",
        "options": ["A snake", "A language", "A tool"],
        "correct_answer": "A language",
        "explanation": "Python is a programming language"
      }
    ]
  }'

# Submit quiz attempt
curl -X POST http://localhost:8000/api/quizzes/1/attempt \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"quiz_id": 1, "answers": {"0": "A language"}}'

# Get attempt history
curl http://localhost:8000/api/quizzes/attempts \
  -H "Authorization: Bearer $TOKEN"
```

### Chat
```bash
# Send message
curl -X POST http://localhost:8000/api/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain Python lists",
    "use_web_search": true
  }'

# List conversations
curl http://localhost:8000/api/conversations \
  -H "Authorization: Bearer $TOKEN"

# Get conversation
curl http://localhost:8000/api/conversations/1 \
  -H "Authorization: Bearer $TOKEN"
```

### Presentations
```bash
# Generate presentation
curl -X POST http://localhost:8000/api/presentations/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Introduction to Python",
    "num_slides": 10,
    "theme": "professional"
  }'

# List presentations
curl http://localhost:8000/api/presentations \
  -H "Authorization: Bearer $TOKEN"

# Download presentation
curl http://localhost:8000/api/presentations/filename.pptx/download \
  -H "Authorization: Bearer $TOKEN" \
  -o presentation.pptx
```

### Admin (Admin Only)
```bash
# Get system statistics
curl http://localhost:8000/api/admin/stats \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# List all users
curl http://localhost:8000/api/admin/users?page=1 \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Deactivate user
curl -X PUT "http://localhost:8000/api/admin/users/2/activate?activate=false" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

## Common Response Formats

### Success (2xx)
```json
{
  "id": 1,
  "title": "Note title",
  "content": "Note content",
  "created_at": "2024-01-01T12:00:00Z"
}
```

### Paginated List
```json
{
  "items": [...],
  "total": 50,
  "page": 1,
  "page_size": 20,
  "total_pages": 3
}
```

### Error (4xx/5xx)
```json
{
  "success": false,
  "error": "Not found",
  "detail": "Note with id 999 not found"
}
```

## Development Tips

### 1. Interactive API Testing
Use the built-in Swagger UI at http://localhost:8000/api/docs
- Try out endpoints
- See request/response schemas
- Test authentication

### 2. Database Management
```python
# Reset database (caution: deletes all data)
import os
os.remove("cyberbron.db")
# Restart server to recreate tables
```

### 3. Logging
Logs are written to:
- Console (stdout)
- `cyberbron.log` file

Set `DEBUG=True` in `.env` for detailed logs.

### 4. Testing Authentication
```python
# In Python REPL or script
import requests

# Login
response = requests.post("http://localhost:8000/api/auth/login", json={
    "username": "user123",
    "password": "SecurePass123!"
})
token = response.json()["access_token"]

# Use token
headers = {"Authorization": f"Bearer {token}"}
response = requests.get("http://localhost:8000/api/notes", headers=headers)
print(response.json())
```

## Common Issues

### Issue: "Could not validate credentials"
**Solution**: Token expired or invalid. Login again to get new token.

### Issue: "403 Forbidden"
**Solution**: 
- For user endpoints: Check if user is active
- For admin endpoints: Check if user has admin privileges

### Issue: "422 Unprocessable Entity"
**Solution**: Check request body format. See error details for specific validation errors.

### Issue: Database locked
**Solution**: Only one process can write to SQLite at a time. Use PostgreSQL for production.

## Environment Variables

Key variables for `.env`:
```env
# Development
DEBUG=True
SECRET_KEY=dev-secret-key-change-in-production

# Production
DEBUG=False
SECRET_KEY=<generate-strong-key>
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/cyberbron
```

## Next Steps

1. ✅ Backend is ready for use
2. 📝 Integrate AI services (Ollama) for content generation
3. 🧪 Add tests (pytest)
4. 🎨 Build/update frontend to use API
5. 🚀 Deploy to production

## Resources

- **API Docs**: http://localhost:8000/api/docs
- **Full Documentation**: `backend/README.md`
- **Examples**: `backend/example_usage.py`
- **Implementation Details**: `BACKEND_IMPLEMENTATION.md`

## Support

For issues or questions:
1. Check the API documentation at `/api/docs`
2. Review `backend/README.md`
3. Run `example_usage.py` for working examples
4. Check logs in `cyberbron.log`
