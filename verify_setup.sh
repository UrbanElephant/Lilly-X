#!/bin/bash
# Verification script for LLIX setup after model update

echo "============================================"
echo "LLIX System Verification"
echo "Model: mistral-nemo:12b"
echo "============================================"
echo ""

# Check 1: Qdrant
echo "1️⃣ Checking Qdrant..."
if curl -s http://localhost:6333/healthz > /dev/null 2>&1; then
    echo "   ✅ Qdrant is running"
else
    echo "   ❌ Qdrant is NOT running"
    echo "   Run: podman run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage:z qdrant/qdrant:latest"
fi
echo ""

# Check 2: Ollama Service
echo "2️⃣ Checking Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "   ✅ Ollama is running"
    
    # Check 3: mistral-nemo model
    echo ""
    echo "3️⃣ Checking for mistral-nemo:12b model..."
    if ollama list 2>/dev/null | grep -q "mistral-nemo"; then
        echo "   ✅ mistral-nemo model found"
        ollama list | grep mistral-nemo | head -1
    else
        echo "   ❌ mistral-nemo:12b not found"
        echo "   Run: ollama pull mistral-nemo:12b"
    fi
else
    echo "   ❌ Ollama is NOT running"
    echo "   Please start Ollama service"
fi
echo ""

# Check 4: Virtual Environment
echo "4️⃣ Checking virtual environment..."
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    VENV_DIR="venv"
fi

if [ -d "$VENV_DIR" ]; then
    echo "   ✅ $VENV_DIR directory exists"
    
    # Check if streamlit is installed
    if [ -f "$VENV_DIR/bin/streamlit" ]; then
        echo "   ✅ Streamlit is installed"
    else
        echo "   ⚠️  Streamlit not found in $VENV_DIR"
        echo "   Run: source $VENV_DIR/bin/activate && pip install -r requirements.txt"
    fi
else
    echo "   ❌ venv directory not found"
    echo "   Run: python -m venv .venv"
fi
echo ""

# Check 5: Configuration files
echo "5️⃣ Checking configuration..."
if [ -f "src/config.py" ]; then
    MODEL=$(grep 'default="mistral-nemo' src/config.py)
    if [ ! -z "$MODEL" ]; then
        echo "   ✅ src/config.py updated with mistral-nemo:12b"
    else
        echo "   ⚠️  src/config.py may need updating"
    fi
fi

if [ -f ".env" ]; then
    echo "   ✅ .env file exists"
    if grep -q "mistral-nemo" .env 2>/dev/null; then
        echo "   ✅ .env configured for mistral-nemo:12b"
    fi
else
    echo "   ℹ️  .env file not found (using defaults from config.py)"
fi
echo ""

# Summary
echo "============================================"
echo "Summary"
echo "============================================"
echo ""
echo "To start LLIX:"
echo "  cd /home/gerrit/Antigravity/LLIX && bash start.sh"
echo ""
echo "Or use the all-in-one script:"
echo "  cd /home/gerrit/Antigravity/LLIX && bash start_all.sh"
echo ""
