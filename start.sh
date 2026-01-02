#!/bin/bash
# LLIX Startup Script - Activates venv and runs Streamlit

echo "🚀 Starting LLIX..."
echo ""

# Activate virtual environment
source venv/bin/activate

# Verify streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found in venv. Installing dependencies..."
    pip install -r requirements.txt
fi

# Display current model
echo "📊 Current LLM Model: mistral-nemo:12b"
echo ""

# Set Python path to project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start Streamlit
echo "🌐 Starting Streamlit UI..."
streamlit run src/app.py
