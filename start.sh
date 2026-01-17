#!/bin/bash

# 1. System-Limits fixen
ulimit -n 65536

# 2. Hardware-Tuning (CPU Fokus für Ollama)
# Nutzt 16 deiner 32 Threads intensiv für die Inferenz
export OLLAMA_NUM_THREAD=16
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1

# 3. Start-Befehl (Explizit Python 3.12 nutzen)
echo "🚀 Starte Ingest auf Max+ 395 (Hardware-optimiert)..."
./venv/bin/python3.12 -m src.ingest
