#!/bin/bash
# CyberBron Installation and Startup Script

echo "=========================================="
echo "  🛡️ CyberBron Setup & Launch Script"
echo "=========================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
major_version=$(echo $python_version | cut -d. -f1)
minor_version=$(echo $python_version | cut -d. -f2)

if [ "$major_version" -ge 3 ] && [ "$minor_version" -ge 9 ]; then
    echo "   ✅ Python $python_version (OK)"
else
    echo "   ❌ Python 3.9+ required (found $python_version)"
    exit 1
fi

# Check if Ollama is running
echo ""
echo "2. Checking Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "   ✅ Ollama is running"
else
    echo "   ❌ Ollama is not running"
    echo "   Please start Ollama and run this script again"
    exit 1
fi

# Check if Mistral model is available
echo ""
echo "3. Checking Mistral model..."
if ollama list | grep -q "mistral"; then
    echo "   ✅ Mistral model found"
else
    echo "   ⚠️  Mistral model not found"
    echo "   Downloading mistral:latest (this may take a while)..."
    ollama pull mistral:latest
    if [ $? -eq 0 ]; then
        echo "   ✅ Mistral downloaded successfully"
    else
        echo "   ❌ Failed to download Mistral"
        exit 1
    fi
fi

# Check embedding model
echo ""
echo "4. Checking embedding model..."
if ollama list | grep -q "nomic-embed-text"; then
    echo "   ✅ Nomic Embed Text model found"
else
    echo "   ⚠️  Nomic Embed Text model not found"
    echo "   Downloading nomic-embed-text..."
    ollama pull nomic-embed-text
    if [ $? -eq 0 ]; then
        echo "   ✅ Nomic Embed Text downloaded successfully"
    else
        echo "   ❌ Failed to download embedding model"
        exit 1
    fi
fi

# Install Python dependencies
echo ""
echo "5. Installing Python dependencies..."
if pip install -q -r requirements.txt; then
    echo "   ✅ Dependencies installed"
else
    echo "   ❌ Failed to install dependencies"
    exit 1
fi

# Check if knowledge base exists
echo ""
echo "6. Checking knowledge base..."
if [ -d "chroma_db" ]; then
    echo "   ✅ Knowledge base found"
else
    echo "   ⚠️  Knowledge base not found"
    if [ -d "data" ] && [ "$(ls -A data 2>/dev/null)" ]; then
        echo "   Building knowledge base from documents in data/..."
        python3 ingest.py
        if [ $? -eq 0 ]; then
            echo "   ✅ Knowledge base created"
        else
            echo "   ❌ Failed to create knowledge base"
            exit 1
        fi
    else
        echo "   ℹ️  No documents found in data/ directory"
        echo "   You can add documents later and run: python3 ingest.py"
    fi
fi

# Test application imports
echo ""
echo "7. Testing application..."
if python3 -c "import app; print('Import test passed')" 2>&1 | grep -q "Import test passed"; then
    echo "   ✅ Application is ready"
else
    echo "   ❌ Application test failed"
    exit 1
fi

# All checks passed
echo ""
echo "=========================================="
echo "  ✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Starting CyberBron..."
echo ""
echo "📝 Note: The application will open in your browser"
echo "         Press Ctrl+C to stop the server"
echo ""
echo "=========================================="
echo ""

# Launch the application
streamlit run app.py
