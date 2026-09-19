# Retrieval and vector stores

Retrieval selects the document chunks that become evidence for generation. Muffakir
lets you change both the storage backend and the retrieval strategy independently.

---

## Quick start

```python
from Muffakir import MuffakirRAG

rag = MuffakirRAG(
    data_dir="./knowledge-base",
    vector_db_provider="chroma",
    db_path="./muffakir_db",
    collection_name="arabic-books",
    retrieval_method="similarity",
    k=5,
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    api_key="sk-...",
    embedding_provider="sentence_transformers",
    embedding_model="mohamed2811/Muffakir_Embedding",
)

answer = rag.ask("ما أحكام عقد الإيجار في المذهب الحنبلي؟")
```

---

## Vector store backends

### `chroma` — local persistent store

**Best for:** Development, single-machine deployments, or any workflow where an
embedded database with zero infrastructure is sufficient.

Persists to a local directory via `langchain-chroma`. Chroma 0.4+ auto-persists;
older versions call `persist()` explicitly (non-fatal if unavailable).

```python
rag = MuffakirRAG(
    vector_db_provider="chroma",
    db_path="./muffakir_db",           # persist directory
    collection_name="arabic-books",    # collection name within the store
    ...
)
```

Via `create_vector_db` directly:

```python
from VectorDB import create_vector_db

db = create_vector_db(
    provider="chroma",
    path="./muffakir_db",
    collection_name="arabic-books",
)
db.add_documents(documents)
results = db.similarity_search("شروط الإجارة", k=5)
```

| Parameter | Config key | Default | Description |
| --- | --- | --- | --- |
| `path` | `db_path` | `"./muffakir_db"` | Persist directory. |
| `collection_name` | `collection_name` | `"ArabicBooks"` | Chroma collection name. |

**Dependency:** `pip install langchain-chroma chromadb`  
**Alias:** `chromadb`

---

### `faiss` — in-memory + disk

**Best for:** High-throughput local similarity search where you want flat-file
portability without a server.

Builds an in-memory FAISS index and saves it to disk (`index.faiss` + `index.pkl`).
On construction, loads an existing index from `folder_path` if one is present.
MMR is supported via LangChain's wrapper.

```python
rag = MuffakirRAG(
    vector_db_provider="faiss",
    db_path="./faiss_index",          # folder_path
    ...
)
```

| Parameter | Config key | Default | Description |
| --- | --- | --- | --- |
| `folder_path` | `db_path` | `"./faiss_index"` | Directory for `index.faiss` and `index.pkl`. |
| `index_name` | `index_name` | `"index"` | Filename prefix. |

> **Note:** FAISS has no native document store. `load_all_documents()` returns
> the in-memory cache from the current process. For BM25 hybrid search after a
> process restart, re-index documents explicitly.

**Dependency:** `pip install faiss-cpu langchain-community`  
*(Use `faiss-gpu` for GPU acceleration.)*

---

### `qdrant` — server or local file

**Best for:** Production deployments, filtering on metadata, or when you need a
persistent REST/gRPC service with advanced filtering.

Connects to a running Qdrant instance (HTTP or gRPC) or an on-disk path for Qdrant
local mode.

```python
# Remote Qdrant Cloud / self-hosted
rag = MuffakirRAG(
    vector_db_provider="qdrant",
    url="https://my-cluster.cloud.qdrant.io",
    api_key="qd-...",
    collection_name="arabic-books",
    ...
)

# On-disk (Qdrant local)
rag = MuffakirRAG(
    vector_db_provider="qdrant",
    db_path="./qdrant_local",
    collection_name="arabic-books",
    ...
)
```

| Parameter | Config key | Default | Description |
| --- | --- | --- | --- |
| `location` | `url` | `"http://localhost:6333"` | Qdrant server URL. Ignored when `path` is set. |
| `api_key` | `api_key` | `None` | Qdrant Cloud API key. |
| `path` | `db_path` | `None` | On-disk path for local Qdrant. |
| `collection_name` | `collection_name` | `"ArabicBooks"` | Collection name. |

**Dependency:** `pip install qdrant-client langchain-qdrant`  
**Alias:** `quadrant`

---

### `pinecone` — managed cloud

**Best for:** Serverless deployments where you want a fully managed, scalable
vector search service without operating your own server.

```python
import os
rag = MuffakirRAG(
    vector_db_provider="pinecone",
    index_name="muffakir-index",
    api_key=os.environ["PINECONE_API_KEY"],
    ...
)
```

| Parameter | Config key | Default | Description |
| --- | --- | --- | --- |
| `index_name` | `index_name` | `"muffakir-index"` | Pinecone index name (must exist). |
| `api_key` | `api_key` | `None` | Pinecone API key. Never written to `os.environ`. |

> **Note:** Pinecone does not support bulk document retrieval via the SDK.
> `load_all_documents()` returns the in-memory cache only. Re-index documents
> after a process restart for BM25 hybrid search.

**Dependency:** `pip install langchain-pinecone`

---

### `milvus` — local lite or cluster

**Best for:** Large-scale deployments using Milvus Lite for zero-infrastructure
local testing, or a full Milvus cluster for production.

```python
# Milvus Lite (local file)
rag = MuffakirRAG(
    vector_db_provider="milvus",
    db_path="./milvus_local.db",
    collection_name="arabic-books",
    ...
)

# Remote Milvus cluster
rag = MuffakirRAG(
    vector_db_provider="milvus",
    connection_args={"uri": "http://localhost:19530"},
    collection_name="arabic-books",
    ...
)
```

| Parameter | Config key | Default | Description |
| --- | --- | --- | --- |
| `collection_name` | `collection_name` | `"ArabicBooks"` | Milvus collection name. |
| `connection_args` | `connection_args` | `{"uri": "./milvus_local.db"}` | Dict passed directly to the Milvus client. Use `db_path` as a shorthand for `uri`. |

> **Note:** Milvus has no bulk-retrieval API. `load_all_documents()` returns the
> in-memory cache only.

**Dependency:** `pip install langchain-milvus pymilvus`  
**Alias:** `milvus_lite`

---

## Backend comparison

| Backend | Persistence | Infra needed | `load_all_documents()` | Best for |
| --- | --- | --- | --- | --- |
| `chroma` | Local dir | None | ✅ Full | Development, small deployments |
| `faiss` | Flat files | None | ⚠️ In-memory only | High-throughput local search |
| `qdrant` | Server / local | Optional | ✅ Full (scroll) | Production, metadata filtering |
| `pinecone` | Managed cloud | None | ⚠️ In-memory only | Serverless / managed |
| `milvus` | Local file / cluster | Optional | ⚠️ In-memory only | Large-scale |

---

## Retrieval methods

Set `retrieval_method` (or pass it directly to `rag.ask`) to choose how chunks are
selected.

| Method | Description | Key parameters |
| --- | --- | --- |
| `similarity` / `similarity_search` | Standard ANN cosine/dot-product search. Strong baseline. | `k` |
| `mmr` / `max_marginal_relevance` | Maximally Marginal Relevance — trades some relevance for result diversity. | `k`, `fetch_k` |
| `hybrid` | Combines BM25 lexical scores with vector scores. Requires `load_all_documents()` support. | `k`, `fetch_k` |
| `contextual` | Query-transform-aware selection with context fusion. | `k` |

```python
# MMR retrieval — more diverse results
rag = MuffakirRAG(
    retrieval_method="mmr",
    k=5,
    fetch_k=20,       # candidate pool before MMR re-ranking
    ...
)

# Hybrid retrieval — BM25 + vector
rag = MuffakirRAG(
    retrieval_method="hybrid",
    k=5,
    ...
)
```

> **Tip:** Do not tune retrieval from intuition alone. Compare candidates with
> Recall@k, Precision@k, MRR, and nDCG via [evaluation](../evaluate/evaluation.md)
> or add `retrieval_method` to a [Composer search space](../evaluate/composer.md).

---

## Standalone usage

Use `create_vector_db` when you want a vector store outside a full RAG pipeline:

```python
from VectorDB import create_vector_db
from Embedding import create_embedding_provider

emb = create_embedding_provider(
    provider="sentence_transformers",
    model_name="mohamed2811/Muffakir_Embedding",
)

db = create_vector_db(
    provider="chroma",
    embedding_provider=emb,
    path="./muffakir_db",
    collection_name="arabic-books",
)

# Ingest
db.add_documents(documents)

# Search
hits = db.similarity_search("حكم بيع الغرر", k=5)
diverse = db.max_marginal_relevance_search("حكم بيع الغرر", k=5, fetch_k=20)

# Full document load (for BM25 re-indexing)
all_docs = db.get_all_documents()
```

---

## Custom backend

Subclass `BaseVectorDBManager` to wrap any LangChain-compatible vector store:

```python
from VectorDB.base import BaseVectorDBManager
from langchain_core.vectorstores import VectorStore
from typing import List
from langchain_core.documents import Document

class MyCustomDBManager(BaseVectorDBManager):
    def __init__(self, embedding_provider, **kwargs):
        super().__init__(embedding_provider)
        self._store = None  # initialize your store here

    @property
    def vector_store(self) -> VectorStore:
        return self._store

    def add_documents(self, documents: List[Document]) -> None:
        # ingest into your backend
        ...
        self._all_documents.extend(documents)

    def load_all_documents(self) -> List[Document]:
        # fetch all persisted documents
        return list(self._all_documents)
```

---

## See also

- [Embeddings](embeddings.md) — embedding providers used for indexing and search
- [Reranking](reranking.md) — post-retrieval reranking strategies
- [Configuration reference](../reference/configuration.md#retrieval-and-reranking)
- [Public API](../reference/api.md#vectordb)
