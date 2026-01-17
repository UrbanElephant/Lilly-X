#!/bin/bash
# ============================================================
# LLIX Golden Start Script - Max+ 395 / Fedora 42 Release
# ============================================================

# 1. System-Limits für massive Parallelität (Fix 'Too many open files')
ulimit -n 65536

# 2. Hardware-Zwangsschaltung (CPU-Only & Stabilität)
export OLLAMA_LLM_LIBRARY=cpu      # Erzwingt den stabilen CPU-Runner
export ROCR_VISIBLE_DEVICES=""     # Versteckt die iGPU vor ROCm (verhindert Abstürze)
export OLLAMA_NUM_THREAD=16        # Nutzt 16 physische Kerne deines Max+ 395
export OLLAMA_NUM_PARALLEL=1       # Verhindert Context-Konflikte beim Ingest
export OLLAMA_MAX_LOADED_MODELS=1  # Spart RAM/VRAM

# 3. Python-Umgebung festnageln (Stabilität auf Fedora 42)
PYTHON_BIN="./venv/bin/python3.12" 

# Prüfung auf venv
if [ ! -f "$PYTHON_BIN" ]; then
    echo "❌ Virtual Environment nicht gefunden!"
    echo "👉 Bitte ausführen: /usr/bin/python3.12 -m venv venv && ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

case "$1" in
    ingest)
        echo "🚀 Starte Ingest auf 16 CPU-Kernen (Max+ 395)..."
        $PYTHON_BIN -m src.ingest
        ;;
    ui)
        echo "🎨 Starte Streamlit UI..."
        ./venv/bin/streamlit run src/app.py
        ;;
    *)
        echo "Usage: ./run_llix.sh {ingest|ui}"
        exit 1
        ;;
esac
