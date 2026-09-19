# Composer search

`MuffakirComposer` provides an automated, scikit-learn-like hyperparameter search framework
(`composer.fit()`) for Arabic RAG pipelines. It systematically evaluates combinations of
query transformers, retrieval strategies, vector backends, rerankers, models, and chunking
parameters against an evaluation dataset to identify the optimal architecture.

---

## Tune the same model with different parameters

Each `search_space.llm` entry can include a `parameters` dictionary. A variant is one
search choice, not an automatic grid of individual parameter values:

```python
search_space = {
    "llm": [
        {"provider": "openai", "model": "gpt-4o-mini",
         "parameters": {"temperature": temperature, "max_tokens": 800}}
        for temperature in [0, 0.3, 0.7]
    ],
    "k": [3, 5],
}
# Pass search_space to composer.fit(...): 3 variants × 2 depths = 6 trials.
```

Variants override base generation parameters. Keep comparisons controlled by setting
`judge_llm_parameters`, `dataset_llm_parameters`, `query_transform_llm_parameters`,
and `reranker_llm_parameters` explicitly in the base configuration for enabled stages.
ComposerUI does this automatically for new runs; older SDK configurations retain their
previous inheritance behavior. Auxiliary parameter sets are not search dimensions.
Existing `{provider, model}` entries still work. Exact effective duplicates are rejected.

See [the provider guide](../build/llm-providers.md#portable-generation-parameters) for
parameter names and limitations, or [ComposerUI](../build/llm-providers-ui.md#compare-generation-variants)
for the visual editor. Variants share compatible document indexes and pricing entries.

## Quick start

```python
import os
from Muffakir import MuffakirComposer

# 1. Configure the base system and evaluation credentials
composer = MuffakirComposer(
    config={
        "data_dir": "./knowledge-base",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": os.environ["OPENAI_API_KEY"],
        "embedding_provider": "sentence_transformers",
        "embedding_model": "mohamed2811/Muffakir_Embedding",
    }
)

# 2. Define the search space and fit
report = composer.fit(
    search_space={
        "query_expansion": ["none", "rewrite", "hyde"],
        "retrieval": ["similarity_search", "hybrid"],
        "reranking": ["none", "cross_encoder"],
        "reranking_model": ["BAAI/bge-reranker-base", "BAAI/bge-reranker-v2-m3"],
        "k": [3, 5],
    },
    eval_dataset="./arabic_rag_eval.jsonl",
    metrics=["recall", "faithfulness", "answer_correctness", "llm_judge_rating"],
    metric_weights={
        "faithfulness": 0.4,
        "answer_correctness": 0.3,
        "recall": 0.2,
        "llm_judge_rating": 0.1,
    },
    n_jobs=4,
    checkpoint_dir="./muffakir_checkpoints",
    resume=True,
)

# 3. Inspect the winning pipeline
print(f"Best trial ID:       {report.best_trial.trial_id}")
print(f"Best composite score: {report.best_score:.4f}")
print("Best configuration:")
for key, value in report.best_config.items():
    print(f"  {key}: {value}")
```

---

## Search space dimensions

`MuffakirComposer` accepts a flexible `search_space` dictionary mapping pipeline dimensions
to lists of candidate options:

| Dimension | Type | Supported values / Examples | Description |
| --- | --- | --- | --- |
| `query_expansion` | `str` | `"none"`, `"rewrite"`, `"multi_query"`, `"decomposition"`, `"hyde"`, `"step_back"` | Transformation applied to the query before retrieval. |
| `retrieval` | `str` | `"similarity_search"`, `"max_marginal_relevance"`, `"hybrid"`, `"contextual"` | Vector store retrieval algorithm. |
| `reranking` | `str` | `"none"`, `"semantic_similarity"`, `"bm25"`, `"cross_encoder"`, `"pointwise"`, `"llm"` | Reranking algorithm applied to retrieved chunks. |
| `reranking_model` | `str` | `["BAAI/bge-reranker-base", "BAAI/bge-reranker-v2-m3"]` | Hugging Face model IDs. Expands only `cross_encoder` and `pointwise` methods without duplicating other rerankers. |
| `k` | `int` | `[3, 5, 8, 10]` | Number of final candidate document chunks passed to generation. |
| `chunking` | `dict` | `[{"method": "recursive", "size": 500, "overlap": 100}, {"method": "semantic", "size": 600, "overlap": 150}]` | Document splitting strategy and chunk dimensions. |
| `embedding_model` | `str` | `["mohamed2811/Muffakir_Embedding", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"]` | Dense vector embedding models. |
| `vector_db_provider` | `str` | `["chroma", "qdrant", "faiss"]` | Underlying vector database backend. |
| `llm` | `dict` | `[{"provider": "openai", "model": "gpt-4o-mini"}, {"provider": "groq", "model": "llama-3.3-70b-versatile"}]` | Generation LLM provider and model pairing. |

### Smart expansion of reranking models

When `reranking_model` is specified, Composer avoids combinatorial explosion:
- Methods that do not use cross-encoder model checkpoints (`none`, `semantic_similarity`,
  `bm25`, `llm`) are evaluated **once**.
- Methods that load local model weights (`cross_encoder`, `pointwise`) are evaluated
  **once per specified model ID**.

---

## Shared index caching and acceleration

Evaluating dozens of trials on a document collection can be slow if documents are repeatedly
parsed and embedded.

Composer optimizes execution through **Index Key Namespacing** (`compute_index_key`):
1. **Pre-Indexing Once**: Chunks are processed and inserted into the vector database once
   for each unique combination of `(chunking, embedding_model, vector_db_provider)`.
2. **Read-Only Worker Access**: Downstream trials that vary only in `retrieval`, `k`,
   `reranking`, or `query_expansion` reuse the existing indexed database in read-only mode
   (`skip_document_ingestion=True`).
3. **Per-Process Model Caching**: Worker processes cache heavy `EmbeddingProvider` and
   database instances in memory (`_COMPONENT_CACHE`), eliminating repeated model loading
   overhead across parallel trials.

---

## Scoring and composite metrics

Composer ranks trials using a weighted composite score between `0.0` and `1.0`:

```text
Composite Score = Σ (weight_m × score_m)
```

Where `weight_m` is the weight for metric `m` (normalized such that total weights sum to 1.0), and `score_m` is the metric score for that trial.

### LLM Judge Rating normalization

While metrics such as `recall`, `precision`, `faithfulness`, and `answer_correctness` naturally fall between `0.0` and `1.0`, `llm_judge_rating` is evaluated on a 1–5 discrete integer rubric.

In Composer:
- **Composite Score**: `llm_judge_rating` is normalized linearly:
  ```text
  normalized_score = (Rating - 1) / 4
  ```
  (A rating of 1 maps to `0.0`, 3 maps to `0.5`, and 5 maps to `1.0`).
- **Reports and Traces**: The raw, unscaled rating is preserved intact (e.g. `4.82 / 5`) for auditability.

---

## Pipeline execution modes

### 1. Full RAG search (`pipeline_mode="full_rag"`)

The standard mode. Evaluates end-to-end performance across both retrieval accuracy and
generation quality:

```python
report = composer.fit(
    search_space={...},
    eval_dataset="./dataset.jsonl",
    metrics=["recall", "faithfulness", "answer_correctness"],
)
```

### 2. Retrieval-only search (`pipeline_mode="retrieval_only"`)

When tuning index parameters (embeddings, chunk sizes, retrieval algorithms, and rerankers),
calling LLM answer generation on every trial can consume substantial tokens.

Setting `pipeline_mode="retrieval_only"` in the base configuration:
- Evaluates retrieval metrics (`recall`, `precision`, `mrr`, `ndcg`) exclusively.
- Disables LLM answer generation entirely, executing searches in milliseconds.
- Does not require generation LLM API keys.

```python
composer = MuffakirComposer(
    config={
        "data_dir": "./knowledge-base",
        "pipeline_mode": "retrieval_only",
        "embedding_provider": "sentence_transformers",
        "embedding_model": "mohamed2811/Muffakir_Embedding",
    }
)

report = composer.fit(
    search_space={
        "retrieval": ["similarity_search", "max_marginal_relevance", "hybrid"],
        "reranking": ["none", "semantic_similarity", "cross_encoder"],
        "k": [3, 5, 10],
    },
    eval_dataset="./dataset.jsonl",
    metrics=["recall", "mrr", "ndcg"],
)
```

### 3. Web-search-only search (`retrieval_source="web_search_only"`)

When optimizing zero-corpus web search agents via `MuffakirSearch`, set
`retrieval_source="web_search_only"`. `data_dir` is not required, and trials evaluate web
retrieval and generation directly.

---

## Checkpoints, recovery, and resumption

Architecture searches can be long-running. Muffakir includes atomic checkpointing
(`CheckpointManager`) to guarantee zero loss of progress:

- **Atomic Writes**: Checkpoint files are saved using temporary files and atomic
  replacement (`os.replace`), preventing corruption even during abrupt power loss or
  process termination.
- **Resumption**: With `resume=True` (default), re-running `composer.fit()` automatically
  detects completed trials in `checkpoint_dir` and resumes from the exact next unfinished
  trial.
- **Validation**: Checkpoints verify that the search space and configuration match the
  prior run. If incompatible changes are detected, a clear `ConfigurationError` is raised.

To clear a checkpoint and force a fresh run:

```python
composer.clear_checkpoint(checkpoint_dir="./muffakir_checkpoints")
```

---

## Budget and runtime constraints

You can set strict bounds on trial counts or total elapsed search time:

```python
report = composer.fit(
    search_space=large_search_space,
    eval_dataset="./dataset.jsonl",
    max_trials=20,                 # Stop after evaluating 20 trials
    max_runtime_minutes=30.0,      # Stop after 30 minutes of search
    n_jobs=4,
)

if report.stopped_early:
    print(f"Search stopped early. Reason: {report.stop_reason}")
```

---

## Analyzing results and the Pareto frontier

The returned `ComposerReport` provides rich analytical utilities:

```python
# 1. Convert to Pandas DataFrame
df = report.to_dataframe()
print(df[["trial_id", "composite_score", "latency_ms", "cost_usd"]].head())

# 2. Get top 5 performing architectures
top_5 = report.get_top_n_trials(n=5)
for t in top_5:
    print(f"Trial #{t.trial_id} | Score: {t.composite_score:.4f} | Latency: {t.latency_ms:.0f}ms")

# 3. Calculate Pareto frontier (Quality vs. Latency trade-offs)
# Finds architectures that cannot be improved in score without increasing latency
frontier = report.get_pareto_frontier(
    objectives=[("composite_score", "max"), ("latency_ms", "min")]
)
print(f"Found {len(frontier)} Pareto-optimal configurations:")
for t in frontier:
    print(f"  Trial #{t.trial_id}: score={t.composite_score:.3f}, latency={t.latency_ms:.0f}ms")

# 4. Analyze failure clusters if any trials encountered errors
clusters = report.get_failure_clusters()
if clusters:
    print("Failure clusters:", clusters)

# 5. Export results
report.save_json("./search_results.json")
```

---

## Visualizing searches in ComposerUI

Muffakir includes a web-based companion interface for Composer. You can explore search
spaces interactively, monitor live trial progress, view Pareto trade-off curves, and
inspect sample-level generated queries and judge decisions:

```bash
muffakir ui
```

See the [ComposerUI guide](../getting-started/composer-ui.md) for details on launching and
navigating the web interface.

---

## See also

- [Export a trial to Python](python-export.md) — Exporting any saved trial into a standalone Python project.
- [Evaluation guide](evaluation.md) — Metric definitions, LLM judges, and refusal detection.
- [Traces and observability](observability.md) — Detailed trial and sample trace schemas.
- [Cost and pricing](../build/pricing.md) — Configuring token cost calculation and pricing catalogs.
- [ComposerUI guide](../getting-started/composer-ui.md) — Interactive visual architecture optimization.
- [Configuration reference](../reference/configuration.md#composer-muffakircomposer) — Composer configuration schema.
- [Public Python API](../reference/api.md#muffakircomposer) — Signatures for `MuffakirComposer`, `ComposerReport`, and `TrialResult`.
