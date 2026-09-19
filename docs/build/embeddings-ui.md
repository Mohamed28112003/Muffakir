# Embeddings in ComposerUI

In Composer architecture searches, finding the optimal embedding model for an Arabic corpus is often the single most impactful decision for retrieval accuracy. ComposerUI allows you to configure, test, and automatically compare multiple embedding models across automated grid and random search runs.

This page explains how to configure and evaluate embeddings visually in ComposerUI. For code-first configuration and Python SDK examples, see [Embeddings in the Python library](embeddings.md).

---

## Where to configure embeddings

Embedding models and hardware acceleration are configured across two notebook tabs:

1. **Hardware Device Selection**: In **Step 1: Data & Pipeline** (`tab-panel-data`), configure your target compute device under **Device Selection**.
2. **Embedding Search Space**: In **Step 5: Search Space** (`tab-panel-search-space`), configure which embedding models to include in your architecture search.

---

## Configuring the embedding search space

In **Step 5: Search Space**, locate **Card 02: Embedding Models**:

### 1. Interactive chip toggles

ComposerUI presents pre-configured chips for proven Arabic and multilingual embedding models:

- **`mohamed2811/Muffakir_Embedding`**: Default fine-tuned model optimized for Arabic semantic retrieval and question answering.
- **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**: Compact, fast 384-dimension multilingual baseline.
- **Custom Model Input**: Add any model identifier from the Hugging Face Hub (e.g., `intfloat/multilingual-e5-base`, `BAAI/bge-m3`) and click **Add Model**.

### 2. Multi-model comparison

You can activate multiple embedding model chips simultaneously. When multiple models are selected:
- Composer multiplies the search dimensions accordingly (e.g., 2 embedding models × 2 chunking presets × 2 rerankers = 8 architecture combinations).
- Each candidate architecture is systematically evaluated on your evaluation dataset.
- The real-time combination badge (`tab-combo-badge`) updates dynamically to reflect the total trial count.

---

## Hardware acceleration (`Device Selection`)

In **Step 1: Data & Pipeline**, choose how local embedding models run:

| Selection | Hardware Target | Recommended When |
|---|---|---|
| **auto (default)** | CUDA GPU if available, otherwise CPU | Recommended default for all local workstations and cloud VM instances. |
| **cuda** | NVIDIA GPU via PyTorch | High-throughput search runs with large document corpora. |
| **cpu** | Host CPU | Standard developer laptops or environments without dedicated GPU access. |

---

## How caching speeds up Composer runs

Embedding an entire document corpus repeatedly across dozens of trials would be slow and wasteful. Composer leverages Muffakir's persistent disk cache (`.embedding_cache/`):

1. **First Trial**: When an embedding model is evaluated for the first time, all document chunks in your corpus are embedded and atomically cached to disk.
2. **Subsequent Trials**: When future trials evaluate the same embedding model with different retrieval strategies, rerankers, or prompt templates, document vectors are loaded instantly from cache with **zero recomputation overhead**.
3. **Multi-Model Isolation**: Each embedding is hashed using `sha256(provider:model:text)`, ensuring multiple models evaluated in the same run never clobber each other's cache entries.

---

## Monitoring embedding performance in run details

When an experiment is executing or completed, ComposerUI surfaces embedding telemetry across multiple views:

### 1. Run detail KPI and timing cards
On the Run Detail page (`run_detail.html`), the **Stage Timing Breakdown** panel displays the average latency for **Query Embedding (ms)** alongside vector search and reranking times.

### 2. Stage timing trends chart
The locally rendered SVG timing chart plots **Query Embedding Latency** across sequential trials. Hover over a point to see its stage mean, pipeline mean, and sample count; click it to open that trial's configuration, samples, answers, and retrieved chunks. The points also support keyboard focus and Enter/Space. No external chart script is required.

### 3. Live trials telemetry stream
Each trial row in the live results table includes the resolved embedding model and its exact `mean_query_embedding_ms`. Clicking on a trial allows you to inspect its full configuration and composite retrieval score.
