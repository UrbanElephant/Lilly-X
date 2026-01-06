# Contributing to Lilly-X

Thank you for your interest in contributing to Lilly-X! This document outlines our development workflow and engineering principles.

## Engineering Philosophy

### No Magic, Just Engineering
- **Explicit over Implicit**: No hidden state, no magic globals
- **Type Safety**: Use Pydantic models and type annotations everywhere
- **Observability**: Comprehensive logging at every stage
- **Idempotent Operations**: Scripts should be safely re-runnable

### Head Chef Pattern
- **Human as Architect**: You design the system, make architectural decisions
- **AI as Sous-Chef**: LLMs extract metadata, but validated with Pydantic schemas
- **TDD Where It Matters**: Critical paths (config, retrieval, indexing) have tests
- **Fail Gracefully**: Every external call (LLM, Qdrant, Ollama) has try-except blocks

---

## Development Workflow ("Inner Loop")

### 1. Setup Development Environment

```bash
# Clone and setup
git clone <repo-url> LLIX
cd LLIX

# Create venv with Python 3.12
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.template .env
```

### 2. Make Changes

**Code Structure**:
- `src/config.py`: Settings (use Pydantic Field with descriptions)
- `src/database.py`: Qdrant client (singleton pattern)
- `src/rag_engine.py`: RAG logic (two-stage retrieval)
- `src/ingest.py`: Ingestion pipeline (incremental indexing)
- `src/app.py`: Streamlit UI

**Key Conventions**:
- All settings via `.env` or environment variables (no hardcoded configs)
- Log all state changes: `logger.info()` for milestones, `logger.debug()` for details
- Use `@field_validator` for flexible Pydantic models (handle LLM output variability)
- Hash-based tracking for any cache/state (see `IngestionState` class)

### 3. Test Locally

```bash
# Start infrastructure
bash start_all.sh

# Run ingestion with your changes
bash run_ingestion.sh

# Test retrieval in UI
# Access http://localhost:8501
```

### 4. Verify Idempotency

Critical test: Can you run it twice safely?

```bash
# Run ingestion twice
bash run_ingestion.sh
bash run_ingestion.sh  # Should skip unchanged files

# Restart system
bash start_all.sh      # Should handle existing container
```

### 5. Commit

```bash
git add <files>
git commit -m "feat: Add XYZ feature

- Implemented ABC
- Added validation for DEF
- Handles edge case GHI gracefully"
```

---

## Code Style

### Python Conventions

```python
# ✅ Good: Explicit types, validated config
class MyExtractor(BaseExtractor):
    llm: Ollama = LlamaField(description="LLM for extraction")
    
    async def aextract(self, nodes: List[BaseNode]) -> List[Dict[str, str]]:
        results = []
        for node in nodes:
            try:
                response = await self.llm.acomplete(prompt)
                results.append({"extracted": str(response)})
            except Exception as e:
                logger.warning(f"Extraction failed: {e}")
                results.append({"extracted": "Unknown"})
        return results

# ❌ Bad: No types, no error handling
def extract(nodes):
    results = []
    for n in nodes:
        results.append(llm.complete(n.text))
    return results
```

### Configuration Pattern

```python
# ✅ Good: Pydantic setting with description
new_feature: bool = Field(
    default=True,
    description="Enable feature XYZ for improved ABC"
)

# ❌ Bad: Hardcoded or magic value
USE_XYZ = True  # What is this? Where is it used?
```

### Error Handling Pattern

```python
# ✅ Good: Graceful degradation with logging
try:
    reranker = HuggingFaceRerank(model=settings.reranker_model)
except Exception as e:
    logger.warning(f"Failed to load reranker: {e}. Falling back to single-stage.")
    reranker = None

# Later...
if reranker:
    query_engine = index.as_query_engine(node_postprocessors=[reranker])
else:
    query_engine = index.as_query_engine()

# ❌ Bad: Silent failure or crash
reranker = HuggingFaceRerank(model=settings.reranker_model)  # Crashes if fails
```

---

## Adding New Features

### Adding a Metadata Extractor

1. **Define Pydantic Model**:
```python
class NewMetadata(BaseModel):
    field_name: Union[str, List[str]] = PydanticField(default="Unknown")
    
    @field_validator('field_name', mode='before')
    @classmethod
    def flatten_list(cls, v: Union[str, List[str]]) -> str:
        if isinstance(v, list):
            return ", ".join([str(item) for item in v])
        return v
```

2. **Create Extractor Class**:
```python
class NewExtractor(BaseExtractor):
    llm: object = LlamaField(description="LLM for extraction")
    
    async def aextract(self, nodes: list[BaseNode]) -> list[dict]:
        # Implementation with error handling
        pass
```

3. **Add to Pipeline**:
```python
# In get_advanced_pipeline()
IngestionPipeline(
    transformations=[
        # ... existing extractors
        NewExtractor(llm=llm),
        embed_model,
    ]
)
```

### Adding a Configuration Setting

1. **Update `src/config.py`**:
```python
new_setting: int = Field(
    default=10,
    description="Clear description of what this controls"
)
```

2. **Update `.env.template`**:
```bash
# New Feature Settings
NEW_SETTING=10  # Description
```

3. **Use in code**:
```python
from src.config import settings

value = settings.new_setting
```

---

## Testing Checklist

Before submitting changes:

- [ ] Code runs without errors
- [ ] Startup script handles existing containers
- [ ] Ingestion is idempotent (run twice safely)
- [ ] New settings documented in `.env.template`
- [ ] Error handling for external services (LLM, Qdrant)
- [ ] Logging added for new milestones
- [ ] No hardcoded paths or configs
- [ ] Type annotations on new functions

---

## Questions?

Open an issue or start a discussion. We're building a robust system together! 🚀
