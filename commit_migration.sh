#!/bin/bash
# Migration completion and Git commit script for LLIX

echo "============================================"
echo "LLIX Migration to mistral-nemo:12b"
echo "Git Commit Automation"
echo "============================================"
echo ""

cd /home/gerrit/Antigravity/LLIX

# Show current git status
echo "📋 Current Git Status:"
git status --short
echo ""

# Stage the core configuration and code files
echo "📦 Staging modified files..."
git add src/config.py
git add src/rag_engine.py 
git add src/ingest.py
git add .env.template
git add start.sh
git add START_INSTRUCTIONS.md

# Also add documentation files we created
git add MODEL_UPDATE_2026-01-02.md
git add verify_setup.sh
git add start_all.sh
git add run_streamlit.sh

echo "✅ Files staged"
echo ""

# Show what's staged
echo "📝 Staged files:"
git diff --staged --name-only
echo ""

# Show specific changes
echo "🔍 Key changes:"
echo ""
echo "1. src/config.py changes:"
git diff --staged src/config.py | grep -A2 -B2 "mistral-nemo\|BAAI/bge-m3" | head -20
echo ""

echo "2. .env.template changes:"
git diff --staged .env.template | grep -A2 -B2 "mistral-nemo\|BAAI/bge-m3" | head -15
echo ""

# Commit with the specified message
echo "💾 Creating commit..."
git commit -m "feat: complete migration to mistral-nemo:12b, optimize context window for 1024 chunks and update project config

- Update LLM model from ibm/granite4:32b-a9b-h to mistral-nemo:12b
- Optimize for Ryzen AI MAX-395 with 128GB RAM and AMD iGPU (Radeon 8060S)
- Ensure context_window=8192 with num_ctx=8192 for optimal 1024-token chunks
- Update embedding model to BAAI/bge-m3 (1024 dimensions)
- Verify chunk_size=1024 and chunk_overlap=200 across all components
- Add comprehensive documentation and verification scripts

Hardware context: Fedora workstation with ROCm (HSA_OVERRIDE_GFX_VERSION=11.0.2)
and 32GB allocated VRAM for iGPU acceleration."

echo ""
echo "✅ Commit created successfully!"
echo ""

# Show the commit
echo "📊 Commit details:"
git log -1 --stat
echo ""

echo "============================================"
echo "Migration Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Pull mistral-nemo model: ollama pull mistral-nemo:12b"
echo "2. Start the system: bash start.sh"
echo "3. Optional: Push to remote with: git push origin main"
echo ""
