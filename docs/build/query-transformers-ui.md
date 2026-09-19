# Query transformation — ComposerUI

The **Query Expansion** dimension in ComposerUI lets you evaluate every
transformation strategy in a single Composer run and automatically identify which
strategy best improves retrieval quality for your corpus.

---

## Enabling query transformation

Open your Composer notebook and locate the **Search Space** tab.
Under the **Query Expansion** section, activate the toggle and select one or more
strategies to compare:

| Strategy | Composer token |
| --- | --- |
| No transformation (baseline) | `none` |
| Query rewriting | `rewrite` |
| Multi-query expansion | `multi_query` |
| Query decomposition | `decomposition` |
| Hypothetical Document Embeddings | `hyde` |
| Step-back prompting | `step_back` |

Each selected strategy becomes a separate configuration candidate. Composer runs
retrieval and (optionally) answer generation for every candidate and scores them
against your evaluation dataset.

---

## Reading the results

After the run completes, open the **Results** tab:

- **Strategy column** — shows the `query_expansion` value for each candidate.
- **Score columns** — faithfulness, answer relevance, context precision, and the
  composite score (weighted sum of all selected metrics).
- **Transformed query** — click any sample row in the detail view to see the exact
  query that was sent to the vector store for that strategy.

Sort by the composite score to find the winning strategy, then copy its
configuration directly from the **Best config** panel.

---

## Inspecting a single strategy run

Switch to the **Run Detail** view for any candidate to see per-sample traces:

1. **Original query** — the raw user question from your evaluation dataset.
2. **Transformed query** — what the strategy produced (or a list, for multi-output
   strategies like `multi_query`, `decomposition`, and `step_back`).
3. **Retrieved chunks** — the documents returned by vector search using the
   transformed query.
4. **Generated answer** (full-RAG mode only) — the LLM answer produced from those
   chunks.

Use the transformed query field to answer three diagnostic questions:

1. Did the model preserve the user's intent?
2. Did it add terminology that exists in the corpus?
3. Did it broaden the search so much that retrieval became less precise?

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
        "query_expansion": ["none", "rewrite", "multi_query", "hyde", "step_back"],
        # combine with other dimensions:
        "retrieval_method": ["similarity", "hybrid"],
        "k": [3, 5],
    },
    metrics=["faithfulness", "answer_relevance", "context_precision"],
)

results = composer.fit()
print(results.best_config)
```

Composer serialises the winning configuration to `checkpoint.json`. Load it directly
into `MuffakirRAG`:

```python
from Muffakir import MuffakirRAG
import json

best = json.load(open("checkpoint.json"))["best_config"]
rag = MuffakirRAG(**best)
```

---

## Retrieval-only mode

When `pipeline_mode="retrieval_only"` is set, a query-transform LLM must be
configured explicitly because there is no answer-generation LLM to fall back on:

```python
composer = MuffakirComposer(
    pipeline_mode="retrieval_only",
    llm_provider="openai",          # used exclusively for query transformation
    llm_model="gpt-4o-mini",
    api_key="sk-...",
    search_space={
        "query_expansion": ["rewrite", "multi_query"],
    },
    ...
)
```

---

## See also

- [Query transformation — Python library](query-transformers.md)
- [Composer search](../evaluate/composer.md) — complete guide to the Composer search space
- [Configuration reference](../reference/configuration.md#query-transformation-and-web-search)
