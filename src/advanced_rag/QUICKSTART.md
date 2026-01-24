# Advanced RAG Modules - Quick Start Guide

## 📦 Installation

Install optional dependencies for full functionality:

```bash
cd /home/gerrit/Antigravity/LLIX

# Activate your virtual environment if using one
# source venv/bin/activate

# Install BM25 retriever
pip install llama-index-retrievers-bm25

# Install reranker (recommended)
pip install llama-index-postprocessor-flag-embedding-reranker

# Install JSON parsing helper
pip install json-repair
```

## 🧪 Testing Individual Modules

Each module includes self-contained tests:

```bash
# Query Transformation
/usr/bin/python3.12 src/advanced_rag/query_transform.py

# Hybrid Retrieval
/usr/bin/python3.12 src/advanced_rag/retrieval.py

# Reciprocal Rank Fusion
/usr/bin/python3.12 src/advanced_rag/fusion.py

# Reranking
/usr/bin/python3.12 src/advanced_rag/rerank.py
```

## 📊 Performance Baseline

Run the reranker performance verification:

```bash
/usr/bin/python3.12 tests/verification/verify_reranker_performance.py
```

This will:
- Load the BGE reranker model (first run downloads ~500MB)
- Test reranking 50 documents on CPU
- Generate baseline metrics for future iGPU comparison
- Save results to `tests/verification/reranker_baseline_results.txt`

## 🔍 Basic Usage Examples

### Query Decomposition

```python
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from advanced_rag.query_transform import QueryDecomposer

# Setup LLM
Settings.llm = Ollama(model="mistral-nemo:12b", base_url="http://localhost:11434")

# Decompose complex query
decomposer = QueryDecomposer()
sub_queries = decomposer.decompose(
    "What are vector databases and how do they differ from traditional databases?"
)

print(f"Generated {len(sub_queries)} sub-queries:")
for i, sq in enumerate(sub_queries, 1):
    print(f"{i}. {sq}")
```

### HyDE (Hypothetical Document Embeddings)

```python
from advanced_rag.query_transform import HyDEGenerator

hyde = HyDEGenerator()
hypothetical_doc = hyde.generate("Explain RAG architecture")

print("Hypothetical answer:")
print(hypothetical_doc)
# Use this for embedding-based retrieval instead of the question
```

### Hybrid Retrieval

```python
from llama_index.core import VectorStoreIndex, Document
from advanced_rag.retrieval import HybridRetriever

# Create or load your index
docs = [Document(text="Your document content...")]
index = VectorStoreIndex.from_documents(docs)

# Setup hybrid retriever
retriever = HybridRetriever(
    vector_index=index,
    vector_top_k=10,
    bm25_top_k=10,
    enable_vector=True,
    enable_bm25=True,
    enable_graph=False,  # Set to True if you have graph_retriever
)

# Retrieve
results = retriever.retrieve("Your query")
print(f"Retrieved {len(results)} results from hybrid search")
```

### Reciprocal Rank Fusion

```python
from advanced_rag.fusion import ReciprocalRankFusion

# Combine results from multiple retrievers
vector_results = [...]  # List[NodeWithScore]
bm25_results = [...]    # List[NodeWithScore]
graph_results = [...]   # List[NodeWithScore]

rrf = ReciprocalRankFusion(k=60)
fused = rrf.fuse(
    [vector_results, bm25_results, graph_results],
    top_n=20
)

print(f"Fused to {len(fused)} unique results")
```

### Reranking

```python
from advanced_rag.rerank import ReRanker

# Rerank fused results
reranker = ReRanker(
    model="BAAI/bge-reranker-v2-m3",
    top_n=5,
    device="cpu"
)

final_results = reranker.rerank(
    nodes=fused,
    query="Your original query",
    top_n=5
)

print(f"Top {len(final_results)} after reranking:")
for i, result in enumerate(final_results, 1):
    print(f"{i}. [Score: {result.score:.4f}] {result.node.text[:100]}...")
```

## 🔗 Full Pipeline Example

```python
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from advanced_rag import (
    QueryDecomposer,
    HyDEGenerator,
    HybridRetriever,
    ReciprocalRankFusion,
    ReRanker,
)

# 1. Setup
Settings.llm = Ollama(model="mistral-nemo:12b", base_url="http://localhost:11434")
index = VectorStoreIndex.from_documents(your_docs)

# 2. Initialize components
decomposer = QueryDecomposer()
hyde = HyDEGenerator()
retriever = HybridRetriever(vector_index=index)
rrf = ReciprocalRankFusion(k=60)
reranker = ReRanker(top_n=5)

# 3. Process query
query = "How does RAG improve LLM accuracy?"

# 3a. Transform query
sub_queries = decomposer.decompose(query, max_subqueries=3)
hyde_doc = hyde.generate(query)

# 3b. Retrieve with multiple strategies
all_results = []
for sq in sub_queries:
    results = retriever.retrieve(sq)
    all_results.append(results)

# Also retrieve with HyDE
hyde_results = retriever.retrieve(hyde_doc)
all_results.append(hyde_results)

# 3c. Fuse results
fused = rrf.fuse(all_results, top_n=25)

# 3d. Rerank
final = reranker.rerank(fused, query, top_n=5)

# 4. Use final results for generation
context = "\n\n".join([node.node.text for node in final])
# Pass context to LLM for answer generation...
```

## 📝 Module Reference

### Query Transformation (`query_transform.py`)

| Class | Purpose | Key Method |
|-------|---------|------------|
| `QueryDecomposer` | Split complex queries | `decompose(query, max_subqueries=3)` |
| `HyDEGenerator` | Generate hypothetical answers | `generate(query)` |
| `QueryRewriter` | Create query variations | `rewrite(query, include_original=True)` |

### Hybrid Retrieval (`retrieval.py`)

| Class | Purpose | Key Method |
|-------|---------|------------|
| `HybridRetriever` | Multi-strategy retrieval | `retrieve(query)` |
| `SimpleGraphRetriever` | Neo4j graph search | `retrieve(query)` |

### Fusion (`fusion.py`)

| Class | Purpose | Key Method |
|-------|---------|------------|
| `ReciprocalRankFusion` | Combine ranked lists | `fuse(results_list, top_n=None)` |

### Reranking (`rerank.py`)

| Class | Purpose | Key Method |
|-------|---------|------------|
| `ReRanker` | Cross-encoder scoring | `rerank(nodes, query, top_n=None)` |

## 🐛 Troubleshooting

### BM25 not working
```
[HybridRetriever] Warning: BM25Retriever not available
```
**Solution**: `pip install llama-index-retrievers-bm25`

### Reranker initialization failed
```
[ReRanker] Warning: Could not initialize reranker
```
**Solution**: `pip install llama-index-postprocessor-flag-embedding-reranker`

### JSON parsing errors in query transformation
**Solution**: `pip install json-repair`

### Ollama connection errors
**Ensure**:
1. Ollama is running: `systemctl status ollama` or `ollama serve`
2. Model is pulled: `ollama pull mistral-nemo:12b`
3. Correct base URL: `http://localhost:11434`

## 📚 Next Steps

1. **Integration**: Incorporate into `src/rag_engine.py`
2. **Configuration**: Add settings to `src/config.py`
3. **UI**: Add Advanced RAG controls to `src/app.py`
4. **Optimization**: Migrate reranker to iGPU for 5-10x speedup

## 📖 Documentation

Full implementation details: See `implementation_plan.md` in artifacts directory

Performance baseline: `tests/verification/reranker_baseline_results.txt`
