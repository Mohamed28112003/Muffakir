# Reranking — Python library

Reranking reorders retrieved candidates before answer generation. It is most useful when
initial retrieval is broad enough to surface relevant passages but not precise enough to
rank the best evidence first. Muffakir provides six reranking strategies through a
uniform `BaseReranker` interface, all accessible via a single factory.

---

## Quick start — via `MuffakirRAG`

```python
from Muffakir import MuffakirRAG

rag = MuffakirRAG(
    data_dir="./knowledge-base",
    reranking_method="cross_encoder",
    reranking_model="BAAI/bge-reranker-v2-m3",  # any HF CrossEncoder model
    device="auto",
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    api_key="sk-...",
    embedding_provider="sentence_transformers",
    embedding_model="mohamed2811/Muffakir_Embedding",
    vector_db_provider="chroma",
)

answer = rag.ask("ما أحكام عقد الإيجار في الفقه الحنبلي؟")
```

Set `reranking_method` to any of the six strategy names listed below.

---

## Standalone usage

Use the `Reranker` wrapper or `create_reranker` factory when you want reranking
outside a full pipeline:

```python
from Reranker import create_reranker
from LLMProvider import LLMProvider

reranker = create_reranker(
    method="cross_encoder",
    model_name="BAAI/bge-reranker-v2-m3",
)

# documents: List[langchain_core.documents.Document]
reranked = reranker.rerank(query="شروط الإجارة", documents=docs, top_k=3)
```

`rerank` returns a `List[Document]` sorted by descending relevance. To inspect
raw scores, call `score` directly — it returns `List[Tuple[Document, float]]`.

---

## Strategy reference

### `semantic_similarity` — Embedding cosine reranker

**Best for:** Keeping reranking in-pipeline without an extra model download.

Reuses the configured embedding model to score cosine similarity between the
query embedding and each document embedding. No additional dependencies beyond
`sentence-transformers`.

```python
from Reranker import SemanticSimilarityReranker
from Embedding import create_embedding_provider

emb = create_embedding_provider(
    provider="sentence_transformers",
    model_name="mohamed2811/Muffakir_Embedding",
)

reranker = SemanticSimilarityReranker(embedding_provider=emb)
ranked = reranker.rerank("العقد الصوري", documents=docs)
```

| Parameter | Default | Description |
| --- | --- | --- |
| `embedding_provider` | Auto-created | `EmbeddingProvider` instance; creates `Muffakir_Embedding` if omitted. |
| `model_name` | `"mohamed2811/Muffakir_Embedding"` | Fallback model ID when no provider is passed. |

**Aliases:** `cosine`, `embedding`

---

### `bm25` — BM25 lexical reranker

**Best for:** Queries with rare keywords or technical identifiers where semantic
models may over-generalise.

Scores candidates using Okapi BM25 over whitespace-tokenized text. Requires
`rank-bm25` (`pip install rank-bm25`).

```python
from Reranker import BM25Reranker

reranker = BM25Reranker()
ranked = reranker.rerank("قانون رقم 17 لسنة 1983", documents=docs)
```

No constructor parameters. Tokenization is whitespace-based and language-agnostic,
making it usable for Arabic without a dedicated tokenizer.

**Alias:** `bm25_reranker`

---

### `cross_encoder` — Joint query-document scoring

**Best for:** Highest quality reranking where latency is acceptable.

Feeds each `(query, document)` pair jointly into a `sentence_transformers.CrossEncoder`
model. The model reads both texts together, producing a precise pairwise relevance
score. Default model: `BAAI/bge-reranker-base`. Requires `sentence-transformers`.

```python
from Reranker import CrossEncoderReranker

reranker = CrossEncoderReranker(
    model_name="BAAI/bge-reranker-v2-m3",  # multilingual, good for Arabic
    device="auto",
)
ranked = reranker.rerank("شروط صحة العقد", documents=docs, top_k=5)
```

| Parameter | Default | Description |
| --- | --- | --- |
| `model_name` | `"BAAI/bge-reranker-base"` | Any HF `CrossEncoder`-compatible sequence-classification model. |
| `device` | `"auto"` | `"auto"` (CUDA if available, else CPU), `"cpu"`, or `"cuda"`. |

**Alias:** `crossencoder`

---

### `pointwise` — Independent sigmoid-scored ranking

**Best for:** When a calibrated 0–1 relevance probability is needed per document.

Scores each document independently (not pairwise) using a `CrossEncoder` model with
a Sigmoid activation, producing a probability in `[0.0, 1.0]`. Supports a
`relevance_threshold` to push sub-threshold documents to the end of the list.
Default model: `BAAI/bge-reranker-base`. Requires `sentence-transformers` and `torch`.

```python
from Reranker import PointwiseReranker

reranker = PointwiseReranker(
    model_name="BAAI/bge-reranker-v2-m3",
    relevance_threshold=0.4,  # docs below 0.4 are placed at end
    device="auto",
)
ranked = reranker.rerank("التزامات المستأجر", documents=docs)
```

| Parameter | Default | Description |
| --- | --- | --- |
| `model_name` | `"BAAI/bge-reranker-base"` | HF CrossEncoder-compatible model. |
| `relevance_threshold` | `0.0` | Documents scoring below this threshold are ranked last. |
| `device` | `"auto"` | Device selection. |

**Aliases:** `pointwise_ltr`, `pointwise_l2r`

---

### `llm` — LLM relevance judge

**Best for:** Domains with specialized vocabulary where local models lack coverage,
or when you want bilingual (Arabic / English) relevance reasoning.

Uses the configured LLM to assign each document a continuous relevance score in
`[0.0, 1.0]`. Prefers Pydantic structured output (`with_structured_output`) for
JSON-enforced scoring. Falls back to plain text generation with regex score extraction
for providers that do not support function calling. Temperature is forced to `0.0`
for deterministic scoring. Prompt is loaded from `MuffakirPrompt` under the
`reranker_scoring` key.

```python
from Reranker import LLMReranker
from LLMProvider import LLMProvider
from PromptManager import MuffakirPrompt

llm = LLMProvider(provider="openai", model="gpt-4o-mini", api_key="sk-...")
prompt = MuffakirPrompt(language="ar")

reranker = LLMReranker(
    llm_provider=llm,
    prompt_manager=prompt,
)
ranked = reranker.rerank("الفسخ في عقد البيع", documents=docs, top_k=4)
```

| Parameter | Default | Description |
| --- | --- | --- |
| `llm_provider` | *Required* | Instantiated `LLMProvider`. |
| `prompt_manager` | Auto-created (`ar`) | `MuffakirPrompt` instance for prompt lookup. |
| `prompt_key` | `"reranker_scoring"` | Prompt template key in `MuffakirPrompt`. |
| `use_structured_output` | `True` | Prefer `with_structured_output` for JSON scoring. |

**Aliases:** `llm_reranker`, `llm_based`, `llm-based`

---

### `custom` — Remote HTTP reranker

**Best for:** Cohere Rerank, vLLM reranking endpoints, or any Cohere-compatible
HTTP service.

Calls a user-provided HTTP endpoint with a `POST` request containing the query and
document list. Accepts two response formats:

**Format A — index-based (Cohere-style):**
```json
{
  "results": [
    {"index": 0, "relevance_score": 0.92},
    {"index": 2, "relevance_score": 0.74},
    {"index": 1, "relevance_score": 0.31}
  ]
}
```

**Format B — scores array:**
```json
{ "scores": [0.92, 0.31, 0.74] }
```

```python
from Reranker import RemoteReranker

reranker = RemoteReranker(
    base_url="https://api.cohere.com/v2/rerank",
    api_key="co-...",
    model="rerank-multilingual-v3.0",
    timeout_seconds=30.0,
)
ranked = reranker.rerank("شروط انعقاد الزواج", documents=docs)
```

Via `MuffakirRAG`:

```python
rag = MuffakirRAG(
    reranking_method="custom",
    remote_base_url="https://api.cohere.com/v2/rerank",
    remote_api_key="co-...",
    remote_model="rerank-multilingual-v3.0",
    remote_timeout=30.0,
    ...
)
```

| Parameter | Config key | Default | Description |
| --- | --- | --- | --- |
| `base_url` | `remote_base_url` | *Required* | Full URL of the reranking endpoint. |
| `api_key` | `remote_api_key` | `None` | Bearer token sent in `Authorization` header. |
| `model` | `remote_model` | `None` | Optional model identifier passed in the request body. |
| `timeout_seconds` | `remote_timeout` | `30.0` | Per-request timeout in seconds. |
| `options` | `remote_options` | `{}` | Extra keys merged into the request body. |

**Aliases:** `remote`, `http`

---

## Custom strategy

Subclass `BaseReranker` and either use it directly or register it in the factory:

```python
from Reranker.base import BaseReranker
from Reranker.factory import register_reranker
from typing import List, Tuple
from langchain_core.documents import Document

class KeywordBoostReranker(BaseReranker):
    def __init__(self, boost_terms: list):
        self.boost_terms = boost_terms

    @property
    def name(self) -> str:
        return "keyword_boost"

    def score(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        scored = []
        for doc in documents:
            hits = sum(term in doc.page_content for term in self.boost_terms)
            scored.append((doc, float(hits)))
        return sorted(scored, key=lambda x: x[1], reverse=True)

# Use directly:
reranker = KeywordBoostReranker(boost_terms=["عقد", "فسخ"])
ranked = reranker.rerank(query, documents=docs)

# Or register for use via create_reranker("keyword_boost", ...):
register_reranker(
    "keyword_boost",
    lambda **ctx: KeywordBoostReranker(ctx.get("boost_terms", [])),
    description="Boosts documents containing specific terms.",
)
```

---

## Strategy comparison

| Strategy | Quality | Speed | Extra dependency | Best use case |
| --- | --- | --- | --- | --- |
| `semantic_similarity` | Good | Fast | None (reuses embedding) | Low-latency reranking |
| `bm25` | Moderate | Fast | `rank-bm25` | Keyword-sensitive queries |
| `cross_encoder` | Excellent | Moderate | `sentence-transformers` | Production quality |
| `pointwise` | Excellent | Moderate | `sentence-transformers` + `torch` | Threshold-filtered ranking |
| `llm` | Excellent | Slow | LLM API | Domain-specific, multilingual |
| `custom` | Depends | Depends | `requests` + endpoint | Managed reranker services |

---

## See also

- [Reranking — ComposerUI](reranking-ui.md)
- [Configuration reference](../reference/configuration.md#retrieval-and-reranking)
- [Public API](../reference/api.md#reranker)
