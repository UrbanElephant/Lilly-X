#!/bin/bash
# Final commit script for Lilly-X migration to mistral-nemo:12b

set -e  # Exit on any error

echo "============================================"
echo "Final Lilly-X Migration Commit"
echo "mistral-nemo:12b with 1024-token chunks"
echo "============================================"
echo ""

cd /home/gerrit/Antigravity/LLIX

# Ensure models directory exists
mkdir -p models
echo "✅ models/ directory ready for embedding cache"
echo ""

# Check for debug files being tracked
echo "🔍 Checking for debug files..."
DEBUG_FILES=$(git ls-files | grep -E "debug_.*\.md|.*\.log" || true)
if [ -n "$DEBUG_FILES" ]; then
    echo "⚠️  Warning: Debug files found in git:"
    echo "$DEBUG_FILES"
    echo ""
    echo "Removing them from git index..."
    echo "$DEBUG_FILES" | xargs git rm --cached
    echo "✅ Debug files removed from tracking"
else
    echo "✅ No debug files being tracked"
fi
echo ""

# Stage the core files
echo "📦 Staging files for commit..."
git add src/config.py
git add src/rag_engine.py
git add src/ingest.py
git add .env.template

# Add any new documentation/scripts that should be committed
git add START_INSTRUCTIONS.md 2>/dev/null || true
git add MIGRATION_COMPLETE.md 2>/dev/null || true
git add MODEL_UPDATE_2026-01-02.md 2>/dev/null || true
git add QUICKSTART.md 2>/dev/null || true
git add start.sh 2>/dev/null || true
git add start_all.sh 2>/dev/null || true
git add verify_setup.sh 2>/dev/null || true
git add verify_migration.sh 2>/dev/null || true

echo "✅ Files staged"
echo ""

# Show what's being committed
echo "📝 Files to be committed:"
git diff --cached --name-status
echo ""

# Verify critical settings
echo "🔍 Verifying critical settings..."
echo ""
echo "1. LLM Model:"
grep -n "mistral-nemo:12b" src/config.py | head -1
echo ""
echo "2. Embedding Model:"
grep -n "BAAI/bge-m3" src/config.py | head -1
echo ""
echo "3. Context Window:"
grep -n "context_window=8192" src/rag_engine.py | head -1
echo ""
echo "4. num_ctx Parameter:"
grep -n "num_ctx.*8192" src/rag_engine.py | head -1
echo ""
echo "5. Chunk Size:"
grep -n "chunk_size.*1024" src/config.py | head -1
echo ""
echo "6. Chunk Overlap:"
grep -n "chunk_overlap.*200" src/config.py | head -1
echo ""

# Create the commit
echo "💾 Creating final commit..."
git commit -m "feat: finalize Lilly-X migration to mistral-nemo:12b with optimized 1024-token chunks

Core Changes:
- LLM model: mistral-nemo:12b (from ibm/granite4:32b-a9b-h)
- Embedding model: BAAI/bge-m3 (1024 dimensions)
- Context window: 8192 tokens with num_ctx=8192
- Chunk configuration: 1024 tokens with 200 overlap

Optimizations:
- Configured for Ryzen AI MAX-395 with 128GB RAM
- AMD Radeon 8060S iGPU with 32GB VRAM
- ROCm acceleration (HSA_OVERRIDE_GFX_VERSION=11.0.2)
- High-performance local RAG setup

Files Updated:
- src/config.py: Model and chunk settings
- src/rag_engine.py: Context window optimization
- src/ingest.py: Verified settings usage
- .env.template: Updated environment template
- Documentation and utility scripts

This commit officially closes the migration to mistral-nemo:12b."

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Commit created successfully!"
    echo ""
    
    # Show the commit
    echo "📊 Commit Summary:"
    git log -1 --stat --format="%h - %s%n%nAuthor: %an <%ae>%nDate: %ad%n" --date=format:"%Y-%m-%d %H:%M:%S"
    echo ""
    
    echo "============================================"
    echo "✅ Migration Finalized!"
    echo "============================================"
    echo ""
    echo "Next steps:"
    echo "1. Pull model: ollama pull mistral-nemo:12b"
    echo "2. Start system: bash start_all.sh"
    echo "3. Push to remote: git push origin main"
    echo ""
else
    echo ""
    echo "❌ Commit failed!"
    exit 1
fi
