#!/bin/bash
# Pre-commit verification for LLIX migration

echo "============================================"
echo "LLIX Migration Verification"
echo "Pre-Commit Checks"
echo "============================================"
echo ""

ERRORS=0

# Check 1: Verify config.py has mistral-nemo
echo "1️⃣ Checking src/config.py..."
if grep -q 'default="mistral-nemo:12b"' src/config.py; then
    echo "   ✅ LLM model set to mistral-nemo:12b"
else
    echo "   ❌ ERROR: LLM model not set correctly"
    ERRORS=$((ERRORS + 1))
fi

if grep -q 'default="BAAI/bge-m3"' src/config.py; then
    echo "   ✅ Embedding model set to BAAI/bge-m3"
else
    echo "   ❌ ERROR: Embedding model not set correctly"
    ERRORS=$((ERRORS + 1))
fi

if grep -q 'default=1024' src/config.py | grep chunk_size; then
    echo "   ✅ Chunk size set to 1024"
else
    echo "   ⚠️  WARNING: Verify chunk_size=1024"
fi

echo ""

# Check 2: Verify RAG engine has context window optimization
echo "2️⃣ Checking src/rag_engine.py..."
if grep -q 'context_window=8192' src/rag_engine.py; then
    echo "   ✅ Context window set to 8192"
else
    echo "   ❌ ERROR: Context window not set"
    ERRORS=$((ERRORS + 1))
fi

if grep -q 'num_ctx.*8192' src/rag_engine.py; then
    echo "   ✅ num_ctx parameter set to 8192"
else
    echo "   ❌ ERROR: num_ctx not set correctly"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# Check 3: Verify .env.template
echo "3️⃣ Checking .env.template..."
if grep -q 'LLM_MODEL=mistral-nemo:12b' .env.template; then
    echo "   ✅ Template has correct LLM model"
else
    echo "   ❌ ERROR: Template LLM model incorrect"
    ERRORS=$((ERRORS + 1))
fi

if grep -q 'EMBEDDING_MODEL=BAAI/bge-m3' .env.template; then
    echo "   ✅ Template has correct embedding model"
else
    echo "   ❌ ERROR: Template embedding model incorrect"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# Check 4: Verify ingestion uses settings
echo "4️⃣ Checking src/ingest.py..."
if grep -q 'settings.chunk_size' src/ingest.py; then
    echo "   ✅ Ingestion uses settings.chunk_size"
else
    echo "   ❌ ERROR: Ingestion not using global chunk_size"
    ERRORS=$((ERRORS + 1))
fi

if grep -q 'settings.chunk_overlap' src/ingest.py; then
    echo "   ✅ Ingestion uses settings.chunk_overlap"
else
    echo "   ❌ ERROR: Ingestion not using global chunk_overlap"
    ERRORS=$((ERRORS + 1))
fi

if grep -q 'settings.embedding_model' src/ingest.py; then
    echo "   ✅ Ingestion uses settings.embedding_model"
else
    echo "   ❌ ERROR: Ingestion not using global embedding_model"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# Summary
echo "============================================"
if [ $ERRORS -eq 0 ]; then
    echo "✅ All checks passed!"
    echo "============================================"
    echo ""
    echo "Ready to commit. Run:"
    echo "  bash commit_migration.sh"
    echo ""
    exit 0
else
    echo "❌ $ERRORS error(s) found!"
    echo "============================================"
    echo ""
    echo "Please fix the errors before committing."
    echo ""
    exit 1
fi
