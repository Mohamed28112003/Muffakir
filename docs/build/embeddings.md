# Embeddings in the Python library

In a Retrieval-Augmented Generation (RAG) pipeline, the embedding model translates document chunks and user queries into dense numerical vectors. The quality of semantic retrieval depends directly on how effectively the embedding model captures semantic relationships—especially in morphologically rich languages like Arabic.

Muffakir provides a unified **`Embedding`** module that standardizes local sentence transformers, commercial cloud APIs, and custom embeddings behind a common LangChain-compliant interface (`BaseEmbeddingProvider`). The module includes built-in dual-layer caching, automatic batching, hardware acceleration, and per-sample latency tracking.

For visual configuration and automated model comparison in Composer, see [Embeddings in ComposerUI](embeddings-ui.md).

---

## Start with a RAG application

Configure `embedding_provider` and `embedding_model` in `MuffakirRAG`:

=== "Local (Default Arabic Model)"

    ```python
    from Muffakir import MuffakirRAG

    rag = MuffakirRAG(
        data_dir="./knowledge-base",
        embedding_provider="sentence_transformers",
        embedding_model="mohamed2811/Muffakir_Embedding",
        device="auto",  # Uses CUDA if available, otherwise CPU
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        language="ar",
    )

    response = rag.ask("ما هي شروط القبول في البرنامج؟")
    print(response["answer"])
    ```

=== "OpenAI (Cloud)"

    ```python
    import os
    from Muffakir import MuffakirRAG

    rag = MuffakirRAG(
        data_dir="./knowledge-base",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        language="ar",
    )
    ```

=== "Cohere (Multilingual Cloud)"

    ```python
    import os
    from Muffakir import MuffakirRAG

    rag = MuffakirRAG(
        data_dir="./knowledge-base",
        embedding_provider="cohere",
        embedding_model="embed-multilingual-v3.0",
        api_key=os.environ["COHERE_API_KEY"],
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        language="ar",
    )
    ```

---

## Supported embedding providers

| Provider Key | Type | Default Model | Key Advantages | Install Extra |
|---|---|---|---|---|
| **`sentence_transformers`** (aliases: `local`, `huggingface`) | Local PyTorch | **`mohamed2811/Muffakir_Embedding`** | Fully offline, zero API fees, privacy-preserving, supports GPU acceleration (`cuda`). | `pip install "Muffakir[local]"` |
| **`openai`** (alias: `langchain_openai`) | Cloud API | `text-embedding-3-small` | High-throughput cloud service, 1536 dimensions, cost-efficient. | `pip install "Muffakir[openai]"` |
| **`cohere`** (alias: `langchain_cohere`) | Cloud API | `embed-multilingual-v3.0` | SOTA multilingual semantic representation across 100+ languages including Arabic. | `pip install "Muffakir[cohere]"` |
| **`custom`** | Injected | Injected model name | Wrap any custom LangChain `Embeddings` instance with Muffakir caching and batching. | Built-in |

---

## Standalone usage (`create_embedding_provider`)

You can instantiate and use embedding providers independently of a full RAG pipeline:

```python
from Embedding import create_embedding_provider

# Instantiate local Arabic embedder
embedder = create_embedding_provider(
    provider="sentence_transformers",
    model_name="mohamed2811/Muffakir_Embedding",
    device="auto",
    batch_size=32,
)

# 1. Embed a single query
query_vector = embedder.embed_query("الذكاء الاصطناعي التوليدي")
print(f"Query vector dimensions: {len(query_vector)}")

# 2. Embed multiple documents in batches
documents = [
    "الذكاء الاصطناعي هو سلوك وخصائص معينة تتسم بها البرامج الحاسوبية.",
    "تعتمد خطة البحث على جمع البيانات من المصادر الأولية.",
    "تستخدم النماذج اللغوية الكبيرة معالجة اللغة الطبيعية لتوليد النصوص.",
]
doc_vectors = embedder.embed_documents(documents)
print(f"Embedded {len(doc_vectors)} documents.")
```

---

## Detailed provider guides

### 1. Local SentenceTransformers (`sentence_transformers`)

Local sentence transformers run directly on your hardware without external network calls or cloud costs.

```bash
pip install "Muffakir[local]"
```

#### Default Arabic Model: `mohamed2811/Muffakir_Embedding`
Muffakir defaults to `mohamed2811/Muffakir_Embedding`, an embedding model specifically trained and evaluated on Arabic retrieval benchmarks.

#### Popular alternative models:
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: Compact, fast, 384 dimensions.
- `intfloat/multilingual-e5-base` / `intfloat/multilingual-e5-large`: High-performing general multilingual models.
- `BAAI/bge-m3`: Dense, multi-function multilingual model.

#### Hardware device resolution (`device`):
Muffakir resolves devices intelligently via `Muffakir.device.resolve_device`:
- `"auto"` (default): Automatically detects and uses CUDA GPU if available; falls back to CPU.
- `"cpu"`: Enforces CPU execution.
- `"cuda"` or `"cuda:0"`: Enforces NVIDIA GPU acceleration.

```python
embedder = create_embedding_provider(
    provider="sentence_transformers",
    model_name="mohamed2811/Muffakir_Embedding",
    device="cuda",  # Enforce GPU acceleration
)
```

---

### 2. OpenAI (`openai`)

Uses OpenAI's cloud embedding endpoint through `langchain-openai`.

```bash
pip install "Muffakir[openai]"
```

```python
import os
from Embedding import create_embedding_provider

embedder = create_embedding_provider(
    provider="openai",
    model_name="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"],
)
```

Supported models include:
- `text-embedding-3-small`: Default, 1536 dimensions, highly cost-effective.
- `text-embedding-3-large`: 3072 dimensions, higher semantic fidelity.
- `text-embedding-ada-002`: Legacy embedding model.

---

### 3. Cohere (`cohere`)

Cohere's multilingual embedding models excel at cross-lingual and Arabic semantic search.

```bash
pip install "Muffakir[cohere]"
```

```python
import os
from Embedding import create_embedding_provider

embedder = create_embedding_provider(
    provider="cohere",
    model_name="embed-multilingual-v3.0",
    api_key=os.environ["COHERE_API_KEY"],
)
```

---

### 4. Custom LangChain Embeddings injection (`custom`)

If your organization uses a proprietary, fine-tuned, or private Hugging Face endpoint not directly exposed by Muffakir, pass any instance implementing LangChain's `Embeddings` interface via `custom_embeddings`:

```python
from langchain_community.embeddings import FakeEmbeddings
from Embedding import create_embedding_provider

custom_inst = FakeEmbeddings(size=768)

embedder = create_embedding_provider(
    provider="custom",
    custom_embeddings=custom_inst,
    model_name="my-custom-model",
    cache_dir=".embedding_cache",
)
```

Muffakir wraps your custom instance in `_CustomWrapper`, immediately providing dual-layer caching, batching, and observability.

---

## Dual-layer caching architecture

Embedding large document corpora or repeated test queries can be computationally expensive and time-consuming. `BaseEmbeddingProvider` incorporates a built-in, two-tier caching engine:

```
User Query / Document Chunk
           │
           ▼
┌──────────────────────────────────────┐
│  L1: In-Memory LRU Cache             │ ◄── Zero I/O, thread-safe, 1024 entries
└──────────────────┬───────────────────┘
                   │ Cache Miss
                   ▼
┌──────────────────────────────────────┐
│  L2: Persistent On-Disk JSON Cache   │ ◄── Atomic replace, .embedding_cache/
└──────────────────┬───────────────────┘
                   │ Cache Miss
                   ▼
┌──────────────────────────────────────┐
│  Embedding Model (Local / Cloud API) │ ◄── Compute only for new texts
└──────────────────────────────────────┘
```

### Cache Key Computation
Every text is hashed with SHA-256 using the provider name, model identifier, and string content:
$$\text{key} = \text{SHA256}(\text{provider} + ":" + \text{model} + ":" + \text{text})$$

This prevents collisions when comparing multiple models against the same corpus.

### Atomic Disk Persistence
When saving an embedding to disk, Muffakir writes to a temporary file via `tempfile.mkstemp` and performs an atomic rename (`os.replace`). This eliminates corrupted JSON files caused by sudden process termination or concurrent worker processes.

### Customizing Cache Directory
```python
embedder = create_embedding_provider(
    provider="sentence_transformers",
    cache_dir="./custom_cache_storage",
)
```

---

## Batching and performance tuning

The `batch_size` parameter governs how many uncached texts are bundled in a single call to the underlying model:

```python
embedder = create_embedding_provider(
    provider="sentence_transformers",
    batch_size=64,  # Larger batches for high-RAM GPU environments
)
```

- When `embed_documents()` receives a list of texts, it checks L1 and L2 caches first.
- Only missing texts are grouped into sub-batches of size `batch_size`.
- Results are merged back into the original order, guaranteeing deterministic alignment with input documents.

---

## Observability and query embedding latency

Muffakir tracks query embedding latency via `EmbeddingTimingTracker`. Every call to `embed_query()` is automatically timed and attributed to the active evaluation sample:

```python
totals = embedder.timing.get_totals()
print(f"Total embedding time: {totals['total_ms']:.2f} ms")
print(f"Calls made: {totals['count']}")
```

In Composer runs, this timing surfaces as `mean_query_embedding_ms` in the trial telemetry and comparison dashboards.

---

## Multiprocessing and pickling support

When running grid searches with parallel workers (`n_jobs > 1`), embedding providers must cross process boundaries. `BaseEmbeddingProvider` implements custom `__getstate__` and `__setstate__` methods that decouple non-picklable thread locks during serialization and reinitialize them safely in child worker processes.
