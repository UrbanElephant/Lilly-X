# LLIX Application - Start Instructions

## Quick Start

To start the LLIX Streamlit application, run:

```bash
bash start.sh
```

Or make it executable and run directly:

```bash
chmod +x start.sh
./start.sh
```

## What Happens When You Start

The `start.sh` script performs these steps:

1. **Activates the virtual environment** (`venv`)
2. **Verifies Streamlit is installed** (installs if missing)
3. **Sets PYTHONPATH** to the project root
4. **Launches the Streamlit UI** on `http://localhost:8501`

## Prerequisites Checklist

Before starting, ensure:

- ✅ **Virtual environment exists**: `venv/` directory should be present
- ✅ **Dependencies installed**: Run `pip install -r requirements.txt` inside venv
- ✅ **Qdrant is running**: Run `podman compose up -d` to start the vector database
- ✅ **Ollama is running**: The mistral-nemo:12b model should be available

## Manual Start (if start.sh has issues)

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 3. Run Streamlit
streamlit run src/app.py
```

## Verify Prerequisites

### Check Qdrant Status
```bash
# Start Qdrant
podman compose up -d

# Verify it's running
curl http://localhost:6333/healthz
```

### Check Ollama Status
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Check if the model is available
ollama list | grep mistral-nemo
```

### Check Python Environment
```bash
source venv/bin/activate
python --version  # Should be 3.10+
pip list | grep streamlit  # Should show streamlit is installed
```

## Accessing the Application

Once started, the Streamlit UI will be available at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://[your-ip]:8501

The terminal will display these URLs when Streamlit starts successfully.

## Troubleshooting

### Streamlit Not Found
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Qdrant Connection Error
```bash
podman compose down
podman compose up -d
```

### Ollama Model Not Available
```bash
ollama pull mistral-nemo:12b
```

### Port Already in Use
```bash
# Kill any existing Streamlit instance
pkill -f streamlit

# Or specify a different port
streamlit run src/app.py --server.port 8502
```

## Expected Behavior

When you run `bash start.sh`, you should see output similar to:

```
🚀 Starting LLIX...

📊 Current LLM Model: mistral-nemo:12b

🌐 Starting Streamlit UI...

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

## Features Available

Once the UI loads:
- 📚 **Ask questions** about your ingested documents
- 🔍 **View sources** for each answer with relevance scores
- ⚙️ **Check status** in the sidebar (model info, connection status)
- 💬 **Chat history** is maintained during your session

## Stopping the Application

Press `Ctrl+C` in the terminal where Streamlit is running.

## Full Setup from Scratch

If starting fresh:

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Qdrant
podman compose up -d

# 4. Verify Ollama is running
ollama list

# 5. Start the application
bash start.sh
```
