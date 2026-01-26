#!/bin/bash
set -e

echo "🔧 PATCHING INGESTION LAUNCHER..."

cat << 'EOF' > run_ingestion.sh
#!/bin/bash
set -e

# 0. Safety: Check Venv
if [ ! -f ".venv/bin/activate" ]; then
    echo "❌ Error: .venv not found!"
    echo "   👉 Run: bash scripts/fix_runtime_312.sh"
    exit 1
fi

# 1. Activate Environment
source .venv/bin/activate

# 2. Optimize System Limits
# Prevents 'OSError: Too many open files' during parallel ingestion
ulimit -n 4096 2>/dev/null || echo "⚠️ Warning: Could not increase ulimit. Using system default."

# 3. Configuration
export INGEST_WORKERS=4
export QDRANT__STORAGE__PERFORMANCE__MMAP_THRESHOLD_KB=0 

# 4. CRITICAL: Fix Python Path
# Ensures 'from src.config import...' works correctly
export PYTHONPATH=.

# 5. Execution
echo "🚀 Starting Ingestion Pipeline..."
echo "   - Workers: $INGEST_WORKERS"
echo "   - Python: $(which python)"
echo "   - Path: $(pwd)"

python src/ingest.py

echo "✅ Ingestion Complete."
EOF

chmod +x run_ingestion.sh
echo "✅ Launcher Patched."
echo "👉 Now run: ./run_ingestion.sh"
