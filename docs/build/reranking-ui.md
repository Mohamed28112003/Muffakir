# Reranking — ComposerUI

ComposerUI exposes all six reranking strategies as a searchable dimension so you can
compare them against your evaluation dataset in a single Composer run.

---

## Enabling reranking

Open your Composer notebook and go to the **Search Space** tab.
Under the **Reranking** section, toggle it on and select one or more methods:

| Method | Composer token | Notes |
| --- | --- | --- |
| Disabled (baseline) | *(unselected)* | No reranking applied. |
| Semantic similarity | `semantic_similarity` | Reuses the embedding model — no extra download. |
| BM25 | `bm25` | Requires `rank-bm25` on the worker. |
| Cross encoder | `cross_encoder` | Requires `sentence-transformers`; select model(s) below. |
| Pointwise | `pointwise` | Requires `sentence-transformers` + `torch`; select model(s) below. |
| LLM reranker | `llm` | Uses the configured answer-generation LLM. |
| Custom endpoint | `custom` | Provide `remote_base_url` in the Config tab. |

---

## Local model selection

When `cross_encoder` or `pointwise` is selected, a **Reranking model** field
appears below the strategy selector. Add one or more Hugging Face model IDs to
compare:

- `BAAI/bge-reranker-base` — default; compact and fast.
- `BAAI/bge-reranker-v2-m3` — multilingual, recommended for Arabic corpora.
- Any `sentence_transformers`-compatible `CrossEncoder` model.

Each model ID creates a separate Composer trial under the same strategy. Other
strategies (`semantic_similarity`, `bm25`, `llm`, `custom`) are evaluated once and
are not duplicated per model.

---

## Reading the results

After the run, open the **Results** tab:

- **Reranking method** column — shows the strategy and model for each candidate.
- **Score columns** — faithfulness, answer relevance, context precision, and
  the composite score.
- **Latency** — reranking adds inference time; the Results tab shows total
  pipeline latency per candidate.

Sort by composite score to identify the reranking configuration that best balances
quality and cost. An expensive reranker is valuable only when the quality improvement
justifies its latency.

---

## Programmatic search space configuration

```python
from Composer import MuffakirComposer

composer = MuffakirComposer(
    data_dir="./knowledge-base",
    eval_dataset="./dataset.json",
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    api_key="sk-...",
    embedding_provider="sentence_transformers",
    embedding_model="mohamed2811/Muffakir_Embedding",
    vector_db_provider="chroma",
    search_space={
        "reranking_method": [
            "semantic_similarity",
            "bm25",
            "cross_encoder",
            "llm",
        ],
        "reranking_model": [
            "BAAI/bge-reranker-base",
            "BAAI/bge-reranker-v2-m3",
        ],
        # combine with other dimensions:
        "k": [5, 10],
        "retrieval_method": ["similarity", "hybrid"],
    },
    metrics=["faithfulness", "answer_relevance", "context_precision"],
)

results = composer.fit()
print(results.best_config)
```

> **Note:** `reranking_model` applies only to `cross_encoder` and `pointwise` trials.
> Other strategies use the listed models as a search hint but do not duplicate trials.

---

## Custom remote endpoint

To evaluate a managed reranking service (Cohere, vLLM, etc.):

```python
composer = MuffakirComposer(
    search_space={
        "reranking_method": ["custom"],
    },
    remote_base_url="https://api.cohere.com/v2/rerank",
    remote_api_key="co-...",
    remote_model="rerank-multilingual-v3.0",
    ...
)
```

---

## Loading the best configuration

Composer serializes the winning configuration to `checkpoint.json`:

```python
from Muffakir import MuffakirRAG
import json

best = json.load(open("checkpoint.json"))["best_config"]
rag = MuffakirRAG(**best)
```

---

## See also

- [Reranking — Python library](reranking.md)
- [Composer search](../evaluate/composer.md)
- [Configuration reference](../reference/configuration.md#retrieval-and-reranking)
