#!/usr/bin/env bash
#
# Git Preparation Script for Lilly X Advanced RAG System
# Initializes repository and creates initial commit
#

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════════════════════════════════════"
echo "  🚀 Lilly X Advanced RAG — Git Preparation"
echo "════════════════════════════════════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# Colors for output
# ============================================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ============================================================================
# Check if already initialized
# ============================================================================

if [ -d ".git" ]; then
    echo -e "${YELLOW}⚠️  Git repository already initialized${NC}"
    echo ""
    read -p "Do you want to continue and create a new commit? (y/N): " CONTINUE
    CONTINUE=${CONTINUE:-n}
    
    if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
        echo ""
        echo "Aborted."
        exit 0
    fi
else
    echo -e "${CYAN}📦 Initializing Git repository...${NC}"
    git init
    echo -e "${GREEN}✅ Git repository initialized${NC}"
    echo ""
fi

# ============================================================================
# Check .gitignore exists
# ============================================================================

if [ ! -f ".gitignore" ]; then
    echo -e "${RED}❌ Error: .gitignore not found${NC}"
    echo "Please ensure .gitignore is created before running this script."
    exit 1
fi

echo -e "${CYAN}📋 Verifying .gitignore configuration...${NC}"
echo -e "${GREEN}✅ .gitignore found${NC}"
echo ""

# ============================================================================
# Stage files
# ============================================================================

echo -e "${CYAN}📁 Staging files for commit...${NC}"
echo ""

# Add all project files (respecting .gitignore)
git add .

# Show what will be committed
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Files to be committed:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
git status --short
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================================================
# Verify critical files are included
# ============================================================================

echo -e "${CYAN}🔍 Verifying critical files are staged...${NC}"
echo ""

CRITICAL_FILES=(
    "README.md"
    "requirements.txt"
    "src/config.py"
    "src/ingest.py"
    "src/rag_engine.py"
    "src/app.py"
    "src/advanced_rag/query_transform.py"
    "src/advanced_rag/retrieval.py"
    "src/advanced_rag/fusion.py"
    "src/advanced_rag/rerank.py"
    "src/advanced_rag/pipeline.py"
    "scripts/install_dependencies.sh"
    "scripts/fix_llm.sh"
)

ALL_FOUND=true

for file in "${CRITICAL_FILES[@]}"; do
    if git ls-files --staged | grep -q "^$file$"; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        if [ -f "$file" ]; then
            echo -e "  ${YELLOW}⚠${NC} $file (exists but not staged)"
        else
            echo -e "  ${RED}✗${NC} $file (missing)"
            ALL_FOUND=false
        fi
    fi
done

echo ""

if [ "$ALL_FOUND" = false ]; then
    echo -e "${RED}❌ Some critical files are missing${NC}"
    echo "Please verify the project structure before committing."
    echo ""
    read -p "Continue anyway? (y/N): " FORCE_CONTINUE
    FORCE_CONTINUE=${FORCE_CONTINUE:-n}
    
    if [[ ! "$FORCE_CONTINUE" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# ============================================================================
# Check for sensitive files
# ============================================================================

echo -e "${CYAN}🔒 Checking for sensitive files...${NC}"
echo ""

SENSITIVE_PATTERNS=(".env" "*.pyc" "__pycache__" "storage/" "qdrant_storage/" "neo4j_data/")
FOUND_SENSITIVE=false

for pattern in "${SENSITIVE_PATTERNS[@]}"; do
    if git ls-files --staged | grep -q "$pattern"; then
        echo -e "  ${RED}⚠️  WARNING: $pattern is staged!${NC}"
        FOUND_SENSITIVE=true
    fi
done

if [ "$FOUND_SENSITIVE" = true ]; then
    echo ""
    echo -e "${RED}❌ Sensitive files detected in staged files!${NC}"
    echo "These files should be excluded via .gitignore"
    echo ""
    read -p "Abort and review? (Y/n): " ABORT_SENSITIVE
    ABORT_SENSITIVE=${ABORT_SENSITIVE:-y}
    
    if [[ "$ABORT_SENSITIVE" =~ ^[Yy]$ ]]; then
        echo "Aborted. Please review and update .gitignore"
        exit 1
    fi
else
    echo -e "${GREEN}✅ No sensitive files detected${NC}"
fi

echo ""

# ============================================================================
# Create initial commit
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📝 Commit Message"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Default commit message
DEFAULT_MESSAGE="🚀 Initial commit: Lilly X Advanced RAG Architecture

Features:
- Hybrid Search: Qdrant (Vector) + Neo4j (Graph) + BM25 (Keyword)
- Query Transformation: Decomposition, HyDE, Rewriting
- Reciprocal Rank Fusion: Multi-retriever result merging
- Cross-Encoder Re-ranking: BAAI/bge-reranker-v2-m3
- Containerized Inference: Ollama/Podman deployment
- Hardware Optimization: AMD Ryzen AI MAX-395 + Fedora 42

Tech Stack:
- LlamaIndex Core (RAG orchestration)
- Qdrant (vector store)
- Neo4j (graph store)
- Ollama (local LLM)
- Python 3.12

Platform:
- Fedora 42
- AMD Ryzen AI MAX-395 (32 cores, 128GB RAM, 32GB iGPU)
- Podman containerization"

echo "$DEFAULT_MESSAGE"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

read -p "Use this commit message? (Y/n/edit): " COMMIT_CHOICE
COMMIT_CHOICE=${COMMIT_CHOICE:-y}

if [[ "$COMMIT_CHOICE" =~ ^[Ee]$ ]]; then
    echo ""
    echo "Opening editor for custom commit message..."
    TEMP_MSG=$(mktemp)
    echo "$DEFAULT_MESSAGE" > "$TEMP_MSG"
    ${EDITOR:-nano} "$TEMP_MSG"
    COMMIT_MESSAGE=$(cat "$TEMP_MSG")
    rm "$TEMP_MSG"
elif [[ "$COMMIT_CHOICE" =~ ^[Nn]$ ]]; then
    echo ""
    echo "Enter your commit message (press Ctrl+D when done):"
    COMMIT_MESSAGE=$(cat)
else
    COMMIT_MESSAGE="$DEFAULT_MESSAGE"
fi

echo ""
echo -e "${CYAN}💾 Creating commit...${NC}"

if git commit -m "$COMMIT_MESSAGE"; then
    echo ""
    echo -e "${GREEN}✅ Commit created successfully!${NC}"
else
    echo ""
    echo -e "${RED}❌ Failed to create commit${NC}"
    exit 1
fi

# ============================================================================
# Summary and next steps
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════════════════════════════"
echo "  ✨ Git Preparation Complete!"
echo "════════════════════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Repository Status:"
git log --oneline -1
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next Steps:"
echo ""
echo "1. ${CYAN}Add remote repository:${NC}"
echo "   git remote add origin https://github.com/yourusername/lilly-x.git"
echo ""
echo "2. ${CYAN}Push to GitHub:${NC}"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. ${CYAN}Create additional branches (optional):${NC}"
echo "   git checkout -b development"
echo "   git checkout -b feature/your-feature"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}🎉 Your repository is ready for the world!${NC}"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════════════════════"

exit 0
