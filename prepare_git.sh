#!/bin/bash
# =============================================================================
# prepare_git.sh - Prepare Lilly-X repository for Git commit
# =============================================================================
# This script cleans the repository and checks for common issues before commit.
#
# Usage: bash prepare_git.sh
# =============================================================================

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "           🧹 Lilly-X Git Preparation Script                   "
echo "════════════════════════════════════════════════════════════════"
echo ""

# =============================================================================
# 1. CLEANUP
# =============================================================================
echo "📦 Step 1/4: Cleaning up temporary files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name ".DS_Store" -delete 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
echo "   ✅ Cleanup complete!"
echo ""

# =============================================================================
# 2. GITIGNORE CHECK
# =============================================================================
echo "📝 Step 2/4: Checking .gitignore..."

if [ ! -f .gitignore ]; then
    echo "   ⚠️  .gitignore not found! Creating default..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.pyc
.pytest_cache/

# Virtual Environment
venv/
.venv/
env/

# Environment Variables
.env
.env.local

# IDEs
.vscode/
.idea/
*.swp

# OS
.DS_Store

# Databases
qdrant_storage/
neo4j_data/

# Logs
*.log
EOF
    echo "   ✅ Created .gitignore"
else
    echo "   ✅ .gitignore exists"
fi

# Ensure .env is ignored
if ! grep -q "^\.env$" .gitignore 2>/dev/null; then
    echo ".env" >> .gitignore
    echo "   ✅ Added .env to .gitignore"
fi

echo ""

# =============================================================================
# 3. .ENV SAFETY CHECK
# =============================================================================
echo "🔒 Step 3/4: .env Safety Check..."

if [ -f .env ]; then
    echo "   ⚠️  Found .env file!"
    echo "   💡 Make sure it contains only generic values, NOT real secrets!"
    echo ""
    echo "   ✅ .env is git-ignored"
else
    echo "   ✅ No .env file found"
fi

# Check for .env.template
if [ ! -f .env.template ] && [ -f .env ]; then
    echo "   💡 Creating .env.template from .env (with placeholder values)..."
    cat .env | sed -E 's/=.*/=YOUR_VALUE_HERE/g' > .env.template
    echo "   ✅ Created .env.template"
fi

echo ""

# =============================================================================
# 4. LARGE FILES CHECK
# =============================================================================
echo "📊 Step 4/4: Checking for large files (>50MB)..."
large_files=$(find . -type f -size +50M -not -path "./.git/*" -not -path "./venv/*" -not -path "./.venv/*" -not -path "./qdrant_storage/*" -not -path "./neo4j_data/*" 2>/dev/null || true)

if [ -n "$large_files" ]; then
    echo "   ⚠️  Found large files:"
    echo "$large_files" | while read -r file; do
        size=$(du -h "$file" | cut -f1)
        echo "       - $file ($size)"
    done
    echo ""
    echo "   💡 Consider adding to .gitignore or using Git LFS"
else
    echo "   ✅ No large files found"
fi

echo ""

# =============================================================================
# SUMMARY
# =============================================================================
echo "════════════════════════════════════════════════════════════════"
echo "                    ✅ Preparation Complete!                    "
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "🚀 Your repository is ready for Git!"
echo ""
echo "💡 Next Steps:"
echo "   1. git add ."
echo "   2. git commit -m 'feat: Complete Hybrid RAG System'"
echo "   3. git push origin main"
echo ""
