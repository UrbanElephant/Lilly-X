# Python Connection Verification Guide

## Quick Test

### Method 1: Database Module Test (Recommended)
```bash
cd /home/gerrit/Antigravity/Lilly-X
source venv/bin/activate
python -m src.database
```

**Expected Output:**
```
==================================================
Lilly-X - Qdrant Connection Test
==================================================

📡 Connecting to Qdrant at http://127.0.0.1:6333...
✓ Connected to Qdrant at http://127.0.0.1:6333

📋 Fetching collections...
✓ Found 0 collection(s):

🔍 Checking collection 'tech_books'...
ℹ Collection 'tech_books' does not exist yet (will be created during ingestion)

==================================================
✅ Connection test SUCCESSFUL!
==================================================

Configuration:
  - Qdrant URL: http://127.0.0.1:6333
  - Collection: tech_books
  - Embedding Model: BAAI/bge-large-en-v1.5
  - LLM Model: llama3:70b

✓ Qdrant client connection closed
```

### Method 2: Standalone Test Script
```bash
cd /home/gerrit/Antigravity/Lilly-X
source venv/bin/activate
python test_connection.py
```

### Method 3: Infrastructure Verification
```bash
cd /home/gerrit/Antigravity/Lilly-X
./verify_qdrant.sh
```

## Configuration Summary

All files now point to `http://127.0.0.1:6333`:
- ✓ `src/config.py` - Default value updated
- ✓ `.env` - Environment file updated
- ✓ `.env.template` - Template updated

## Next Steps After Successful Verification

1. **Install Full Dependencies**:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Ready for Module 2**: Document ingestion pipeline
3. **Add Documents**: Place PDFs/EPUBs in `/home/gerrit/Antigravity/Lilly-X/data/books/`

## Troubleshooting

If connection fails:
1. Verify Qdrant is running: `podman ps`
2. Test direct connection: `curl http://127.0.0.1:6333`
3. Check container logs: `podman logs qdrant`
4. Restart container if needed: `podman restart qdrant`
