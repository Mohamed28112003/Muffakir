# Embeddings overview

The embedding model translates text chunks and user queries into dense vectors for semantic similarity search in vector databases.

For in-depth guides, see:
- [Embeddings in the Python library](embeddings.md) for local models (`mohamed2811/Muffakir_Embedding`), OpenAI, Cohere, dual-layer caching, and device acceleration.
- [Embeddings in ComposerUI](embeddings-ui.md) for visual search space configuration, multi-model comparison, and latency tracking.

---

## Quick example

```python
from Muffakir import MuffakirRAG

rag = MuffakirRAG(
    data_dir="./knowledge-base",
    embedding_provider="sentence_transformers",
    embedding_model="mohamed2811/Muffakir_Embedding",  # Default Arabic model
    device="auto",                                    # CUDA GPU if available
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    language="ar",
)
```

See the [Configuration reference](../reference/configuration.md) and [Public API](../reference/api.md) for complete options.
