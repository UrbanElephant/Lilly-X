# GraphRAG Verification Scripts

This directory contains test and verification scripts for the Microsoft-style GraphRAG implementation.

## 📋 Scripts Overview

### Core Verification

#### `verify_lightweight.py` ⭐ **Recommended**
Lightweight verification that bypasses RAGEngine initialization issues.
- ✅ Works with Python 3.14
- Tests community detection independently
- Shows sample community summaries
- **Use this for quick verification**

```bash
python3 tests/verification/verify_lightweight.py
```

#### `verify_final.py`
Full verification including RAGEngine global_search.
- ⚠️ Requires Python 3.11/3.12 (Pydantic v1 issue)
- Tests complete global search pipeline
- Validates routing logic

```bash
python3 tests/verification/verify_final.py
```

### Diagnostic Tools

#### `diagnose_graphrag.py`
Comprehensive component diagnostics.
- Checks all imports
- Verifies GraphOperations methods
- Tests Neo4j connection
- Validates GDS availability

```bash
python3 tests/verification/diagnose_graphrag.py
```

#### `test_global_access.py`
Quick diagnostic for global_search method availability.
- Checks if global_search exists
- Verifies community count
- Provides manual test instructions

```bash
python3 tests/verification/test_global_access.py
```

### Pipeline Execution

#### `run_community_summarization.py`
Main pipeline script to generate community summaries.
- Runs Leiden/Louvain community detection
- Generates LLM-based summaries
- Stores results in Neo4j

```bash
python3 tests/verification/run_community_summarization.py
```

### Integration Tests

#### `test_global.py`
Integration test for query routing.
- Tests GLOBAL_DISCOVERY vs standard retrieval
- Validates intent detection
- Shows routing decisions

```bash
python3 tests/verification/test_global.py
```

#### `test_ollama_settings.py`
Validates Ollama configuration.
- Tests LLM initialization
- Verifies embedding model setup
- Quick LLM response test

```bash
python3 tests/verification/test_ollama_settings.py
```

## 🚀 Recommended Workflow

### 1. Initial Verification (Post-Installation)
```bash
# Quick check
python3 tests/verification/diagnose_graphrag.py

# Verify communities exist
python3 tests/verification/verify_lightweight.py
```

### 2. Run Community Detection (First Time)
```bash
# Generate community summaries
python3 tests/verification/run_community_summarization.py
```

### 3. Test Global Search (Python 3.11/3.12)
```bash
# Full integration test
python3 tests/verification/verify_final.py
```

## ⚠️ Python Version Notes

- **Python 3.14**: Use `verify_lightweight.py` (Pydantic v1 compatibility issue)
- **Python 3.11/3.12**: All scripts work, including full RAGEngine tests
- **Recommended**: Python 3.11 (see `.python-version` file in project root)

## 📊 Expected Results

### verify_lightweight.py Output
```
✅ Found 8 Community nodes
✅ Retrieved community summaries
✅ CommunitySummary model works
🎉 Microsoft GraphRAG Community Detection: OPERATIONAL
```

### run_community_summarization.py Output
```
🔍 Starting community detection using LEIDEN algorithm...
✅ Neo4j GDS plugin is available
📊 Projecting graph with types: [...] (UNDIRECTED)
✅ Community Detection complete! Found 8 communities
🤖 Step 3: Generating summaries...
✅ Community Summarization Pipeline Complete!
```

## 🔧 Troubleshooting

### "No communities found"
Run the pipeline first:
```bash
python3 tests/verification/run_community_summarization.py
```

### "GDS plugin not available"
Check Neo4j configuration:
```bash
bash verify_gds.sh
```

### "ModuleNotFoundError"
Install dependencies:
```bash
pip install -r requirements.txt
```

## 📚 Related Documentation

- [`/home/gerrit/.gemini/.../verification_walkthrough.md`](../../../.gemini/antigravity/brain/6308b599-9d58-4d27-a0b8-a19db66b58e5/verification_walkthrough.md) - Complete verification results
- `NEO4J_GDS_SETUP.md` - GDS plugin setup guide
- `GLOBAL_SEARCH_INTEGRATION.md` - Global search documentation
