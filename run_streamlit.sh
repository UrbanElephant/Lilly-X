#!/bin/bash
# Wrapper script to run Streamlit with output logging

cd /home/gerrit/Antigravity/LLIX

echo "========================================" | tee -a streamlit_launch.log
echo "🚀 Starting LLIX Streamlit App" | tee -a streamlit_launch.log
echo "Time: $(date)" | tee -a streamlit_launch.log
echo "========================================" | tee -a streamlit_launch.log

# Activate virtual environment
source venv/bin/activate 2>&1 | tee -a streamlit_launch.log

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing dependencies..." | tee -a streamlit_launch.log
    pip install -r requirements.txt 2>&1 | tee -a streamlit_launch.log
else
    echo "✅ Streamlit is installed" | tee -a streamlit_launch.log
    streamlit --version 2>&1 | tee -a streamlit_launch.log
fi

echo "" | tee -a streamlit_launch.log
echo "📊 Current LLM Model: ibm/granite4:32b-a9b-h" | tee -a streamlit_launch.log
echo "🌐 Starting Streamlit UI..." | tee -a streamlit_launch.log
echo "" | tee -a streamlit_launch.log

# Run Streamlit
streamlit run src/app.py 2>&1 | tee -a streamlit_launch.log
