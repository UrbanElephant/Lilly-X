#!/bin/bash
# prepare_git.sh - Pre-commit repository cleanup script

set -e

echo "========================================"
echo "🧹 LLIX Repository Preparation"
echo "========================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env is tracked
echo "🔍 Checking for sensitive files..."
if git ls-files --error-unmatch .env >/dev/null 2>&1; then
    echo -e "${RED}⚠️  WARNING: .env is currently tracked by git!${NC}"
    echo ""
    echo "To remove it from git (without deleting the file):"
    echo -e "${YELLOW}  git rm --cached .env${NC}"
    echo ""
    echo "Add to .gitignore (already done) and commit:"
    echo -e "${YELLOW}  git commit -m 'Remove .env from tracking'${NC}"
    echo ""
    exit 1
else
    echo -e "${GREEN}✅ .env is not tracked (good!)${NC}"
fi

# Check if feedback.json exists and is tracked
if [ -f "feedback.json" ]; then
    if git ls-files --error-unmatch feedback.json >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  feedback.json is tracked. Consider removing with:${NC}"
        echo -e "${YELLOW}  git rm --cached feedback.json${NC}"
    else
        echo -e "${GREEN}✅ feedback.json is not tracked${NC}"
    fi
fi

# Check if data/docs/ is tracked
if [ -d "data/docs" ]; then
    if git ls-files data/docs/ 2>/dev/null | grep -q .; then
        echo -e "${YELLOW}⚠️  data/docs/ contains tracked files${NC}"
        echo "If these are private documents, remove with:"
        echo -e "${YELLOW}  git rm --cached -r data/docs/${NC}"
    else
        echo -e "${GREEN}✅ data/docs/ is not tracked${NC}"
    fi
fi

echo ""
echo "📦 Checking dependencies..."

# Option 1: Manual requirements.txt update
echo ""
echo "Current requirements.txt includes:"
grep -E "^(llama-index|sentence-transformers|json-repair|streamlit)" requirements.txt || echo "  (checking...)"

echo ""
echo -e "${YELLOW}Note: To freeze exact versions, run:${NC}"
echo -e "${YELLOW}  pip freeze > requirements.txt${NC}"
echo ""
echo "But this will include ALL installed packages."
echo "Current requirements.txt uses flexible versioning (recommended)."

echo ""
echo "🧪 Verifying critical imports..."

# Test critical imports
python3 << 'PYEOF'
import sys

try:
    from sentence_transformers import CrossEncoder
    print("✅ sentence-transformers: OK")
except ImportError:
    print("❌ sentence-transformers: MISSING")
    sys.exit(1)

try:
    from json_repair import repair_json
    print("✅ json-repair: OK")
except ImportError:
    print("❌ json-repair: MISSING")
    sys.exit(1)

try:
    import streamlit
    print(f"✅ streamlit: OK (version {streamlit.__version__})")
except ImportError:
    print("❌ streamlit: MISSING")
    sys.exit(1)

try:
    from llama_index.core import VectorStoreIndex
    print("✅ llama-index: OK")
except ImportError:
    print("❌ llama-index: MISSING")
    sys.exit(1)

print("\n✅ All critical dependencies installed")
PYEOF

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}❌ Dependency check failed!${NC}"
    echo "Install missing dependencies:"
    echo -e "${YELLOW}  pip install -r requirements.txt${NC}"
    exit 1
fi

echo ""
echo "🔍 Checking for common issues..."

# Check for __pycache__ in git
if git ls-files | grep -q "__pycache__"; then
    echo -e "${YELLOW}⚠️  __pycache__ directories are tracked${NC}"
    echo "Remove with:"
    echo -e "${YELLOW}  git rm --cached -r **/__pycache__${NC}"
else
    echo -e "${GREEN}✅ No __pycache__ tracked${NC}"
fi

# Check for .pyc files
if git ls-files | grep -q "\.pyc$"; then
    echo -e "${YELLOW}⚠️  .pyc files are tracked${NC}"
    echo "Remove with:"
    echo -e "${YELLOW}  git rm --cached **/*.pyc${NC}"
else
    echo -e "${GREEN}✅ No .pyc files tracked${NC}"
fi

echo ""
echo "========================================"
echo -e "${GREEN}🎉 Repository is clean and ready!${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Review changes:"
echo -e "   ${YELLOW}git status${NC}"
echo ""
echo "2. Stage your changes:"
echo -e "   ${YELLOW}git add .${NC}"
echo ""
echo "3. Commit:"
echo -e "   ${YELLOW}git commit -m \"Feat: Enterprise-grade RAG with 2-stage retrieval and strict citations\"${NC}"
echo ""
echo "4. Push to remote:"
echo -e "   ${YELLOW}git push origin main${NC}"
echo ""

exit 0
