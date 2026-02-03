#!/bin/bash
# CyberBron v2.0 - Complete Application Launcher
# This script starts all components of the full-stack application

set -e

echo "🛡️ CyberBron v2.0 - Full Stack Web Application"
echo "================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from .env.example...${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}✅ Created .env file. Please review and update it.${NC}"
    else
        echo -e "${RED}❌ .env.example not found. Please create .env manually.${NC}"
        exit 1
    fi
fi

# Check Ollama
echo -e "${BLUE}🔍 Checking Ollama service...${NC}"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Ollama is running${NC}"
else
    echo -e "${YELLOW}⚠️  Ollama is not running. Please start Ollama first.${NC}"
    echo "   Visit: https://ollama.com for installation"
    exit 1
fi

# Check if models are available
echo -e "${BLUE}🔍 Checking AI models...${NC}"
if ollama list | grep -q "mistral"; then
    echo -e "${GREEN}✅ Mistral model found${NC}"
else
    echo -e "${YELLOW}⚠️  Mistral model not found. Pulling model...${NC}"
    ollama pull mistral:latest
fi

if ollama list | grep -q "nomic-embed-text"; then
    echo -e "${GREEN}✅ Embedding model found${NC}"
else
    echo -e "${YELLOW}⚠️  Embedding model not found. Pulling model...${NC}"
    ollama pull nomic-embed-text
fi

# Check Python dependencies
echo -e "${BLUE}🔍 Checking Python dependencies...${NC}"
if python3 -c "import fastapi" 2>/dev/null; then
    echo -e "${GREEN}✅ Backend dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  Installing backend dependencies...${NC}"
    pip install -r requirements.txt
fi

# Check if chroma_db exists
if [ ! -d "chroma_db" ]; then
    echo -e "${YELLOW}⚠️  Vector database not found${NC}"
    echo "   Would you like to ingest documents now? (y/n)"
    read -r response
    if [[ "$response" == "y" || "$response" == "Y" ]]; then
        if [ -d "data" ] && [ "$(ls -A data)" ]; then
            echo -e "${BLUE}📚 Ingesting documents...${NC}"
            python ingest.py
            echo -e "${GREEN}✅ Documents ingested${NC}"
        else
            echo -e "${RED}❌ No documents found in data/ directory${NC}"
            echo "   Please add documents to data/ and run: python ingest.py"
        fi
    fi
fi

# Start the application
echo ""
echo "🚀 Starting CyberBron v2.0..."
echo "================================"
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down CyberBron...${NC}"
    kill $(jobs -p) 2>/dev/null
    echo -e "${GREEN}✅ Stopped all services${NC}"
    exit 0
}

trap cleanup INT TERM

# Start backend
echo -e "${BLUE}🔧 Starting Backend API...${NC}"
python -m backend.main > backend.log 2>&1 &
BACKEND_PID=$!
echo -e "${GREEN}✅ Backend started (PID: $BACKEND_PID)${NC}"
echo "   Logs: backend.log"
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/api/docs"

# Wait for backend to be ready
echo -e "${BLUE}⏳ Waiting for backend to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Backend failed to start. Check backend.log${NC}"
        cleanup
    fi
    sleep 1
done

# Check if frontend exists
if [ -d "frontend" ]; then
    echo ""
    echo -e "${BLUE}🎨 Starting Frontend...${NC}"
    
    # Check if node_modules exists
    if [ ! -d "frontend/node_modules" ]; then
        echo -e "${YELLOW}⚠️  Installing frontend dependencies...${NC}"
        cd frontend && npm install && cd ..
    fi
    
    cd frontend
    npm run dev > ../frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ..
    echo -e "${GREEN}✅ Frontend started (PID: $FRONTEND_PID)${NC}"
    echo "   Logs: frontend.log"
    echo "   URL: http://localhost:3000"
else
    echo -e "${YELLOW}⚠️  Frontend not found. Skipping...${NC}"
fi

# Optional: Start Streamlit (legacy UI)
echo ""
echo "Would you like to start the legacy Streamlit UI? (y/n)"
read -r response
if [[ "$response" == "y" || "$response" == "Y" ]]; then
    echo -e "${BLUE}🎭 Starting Streamlit UI...${NC}"
    streamlit run app.py > streamlit.log 2>&1 &
    STREAMLIT_PID=$!
    echo -e "${GREEN}✅ Streamlit started (PID: $STREAMLIT_PID)${NC}"
    echo "   Logs: streamlit.log"
    echo "   URL: http://localhost:8501"
fi

# Display summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✨ CyberBron v2.0 is running!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📍 Access Points:"
echo "   • Backend API:  http://localhost:8000"
echo "   • API Docs:     http://localhost:8000/api/docs"
if [ -d "frontend" ]; then
    echo "   • Frontend:     http://localhost:3000"
fi
if [[ "$response" == "y" || "$response" == "Y" ]]; then
    echo "   • Streamlit:    http://localhost:8501"
fi
echo ""
echo "📝 Logs:"
echo "   • Backend:      tail -f backend.log"
if [ -d "frontend" ]; then
    echo "   • Frontend:     tail -f frontend.log"
fi
if [[ "$response" == "y" || "$response" == "Y" ]]; then
    echo "   • Streamlit:    tail -f streamlit.log"
fi
echo ""
echo "🛑 To stop: Press Ctrl+C"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Keep script running
wait
