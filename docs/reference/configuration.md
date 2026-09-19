# Configuration reference

Pass configuration as a dictionary or keyword arguments to the public facades. The same
names appear in ComposerUI and in Composer search spaces.

## Core RAG settings

| Key | Purpose | Typical value |
| --- | --- | --- |
| `data_dir` | Directory containing source documents. | `"./knowledge-base"` |
| `language` | Prompt and text-processing language. | `"ar"`, `"en"` |
| `llm_provider` / `llm_model` | Answer-generation provider and model. | `"openai"` / model ID |
| `api_key` | Provider credential. | Read from environment |
| `llm_temperature` / `llm_max_tokens` | Generation sampling and output limit. | `0.2` / `800` |
| `base_url` / `llm_base_url` | Compatible endpoint override or local server URL. | `"http://127.0.0.1:8000/v1"` |
| `embedding_provider` / `embedding_model` | Embedding integration and model. | Sentence-transformer model ID |
| `vector_db_provider` | Vector-store backend. | `"chroma"` |
| `db_path` / `collection_name` | Local storage location and collection. | Local path / name |

## LLM parameters and variants

Use `llm_parameters` for generation. Independent stage dictionaries are
`judge_llm_parameters`, `query_transform_llm_parameters`, `reranker_llm_parameters`,
and `dataset_llm_parameters`. They are accepted by the Composer run API and SDK
configuration; standalone synthetic-data generation accepts `llm_parameters`.

Each entry of `search_space.llm` accepts `{provider, model, parameters}`; `parameters`
is optional for backward compatibility. Explicit generation values override base
`llm_parameters`, then legacy `llm_temperature`/`llm_max_tokens`, then workflow defaults.
Role dictionaries do not merge generation-specific optional sampling settings.

See [portable parameters](../build/llm-providers.md#portable-generation-parameters)
for field types, ranges, provider capabilities, and omission behavior.

## Document parsing and OCR

| Key | Purpose | Typical value |
| --- | --- | --- |
| `document_parser` | Parsing engine for scanned documents, tables, and complex formats. | `"docling"`, `"llama_parse"`, `"azure"`, or `None` (built-in) |
| `document_parser_config` | Dictionary of provider-specific options and credentials. | `{"export_type": "markdown"}` |
| `use_ocr` | Boolean flag enabling OCR processing on supported documents. | `True`, `False` |

### Provider-specific `document_parser_config` options

#### Docling (`"docling"`)

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `export_type` | `str` | `"markdown"` | Output format: `"markdown"` (preserves headers/tables) or `"text"`. |
| `**kwargs` | `Any` | — | Forwarded directly to `DoclingLoader` (`converter_kwargs`, `export_kwargs`). |

#### LlamaParse (`"llama_parse"`)

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `api_key` | `str` | `None` | LlamaCloud API Key. Reads from `LLAMA_CLOUD_API_KEY` if omitted. |
| `language` | `str` | `"ar"` | Language hint optimizing character recognition for Arabic. |
| `result_type` | `str` | `"markdown"` | Output format: `"markdown"` or `"text"`. |
| `verbose` | `bool` | `False` | Enables verbose parsing logs. |

#### Azure Document Intelligence (`"azure"`)

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `endpoint` | `str` | *Required* | Azure Document Intelligence endpoint URL. |
| `api_key` | `str` | *Required* | Azure API service key. |
| `model_id` | `str` | `"prebuilt-layout"` | Prebuilt or custom Azure model ID. |
| `timeout_seconds` | `float` | `300.0` | Maximum seconds to wait per document. |
| `max_retries` | `int` | `2` | Maximum retry attempts for transient errors with exponential backoff. |

## Embeddings

| Key | Purpose | Typical value |
| --- | --- | --- |
| `embedding_provider` | Embedding integration backend. | `"sentence_transformers"`, `"openai"`, `"cohere"`, or `"custom"` |
| `embedding_model` | Model identifier on Hugging Face Hub or cloud provider. | `"mohamed2811/Muffakir_Embedding"`, `"text-embedding-3-small"`, `"embed-multilingual-v3.0"` |
| `device` | Compute device resolution for local models. | `"auto"` (CUDA if available, else CPU), `"cpu"`, `"cuda"` |
| `cache_dir` | Directory path for persistent atomic disk caching. | `".embedding_cache"` |
| `batch_size` | Batch size for bulk document encoding. | `32` |
| `custom_embeddings` | Injected LangChain `Embeddings` instance. | Optional custom instance |

## Retrieval and reranking

### Vector store

| Key | Purpose | Typical value |
| --- | --- | --- |
| `vector_db_provider` | Backend selection. | `"chroma"`, `"faiss"`, `"qdrant"`, `"pinecone"`, `"milvus"` |
| `db_path` | Local path for Chroma, FAISS, or Qdrant local mode. | `"./muffakir_db"` |
| `collection_name` | Collection/index name inside the store. | `"ArabicBooks"` |
| `index_name` | FAISS file prefix or Pinecone index name. | `"index"` / `"muffakir-index"` |
| `url` | Qdrant server URL. | `"http://localhost:6333"` |
| `connection_args` | Milvus connection dict. Shorthand: `db_path` sets `uri`. | `{"uri": "./milvus_local.db"}` |

### Retrieval

| Key | Purpose | Typical value |
| --- | --- | --- |
| `retrieval_method` | Similarity, MMR, hybrid, contextual, or another supported method. | `"similarity"`, `"hybrid"` |
| `k` / `fetch_k` | Returned documents and candidate-pool size. | `5` / `20` |

### Reranking

| Key | Purpose | Typical value |
| --- | --- | --- |
| `reranking_method` | Reranking strategy. | `"semantic_similarity"`, `"bm25"`, `"cross_encoder"`, `"pointwise"`, `"llm"`, `"custom"` |
| `reranking_model` | Hugging Face model for `cross_encoder` or `pointwise`. | `"BAAI/bge-reranker-v2-m3"` |
| `device` | Compute device for local reranking models. | `"auto"`, `"cpu"`, `"cuda"` |

### Remote reranker options (`custom` method)

| Key | Default | Description |
| --- | --- | --- |
| `remote_base_url` | *Required* | Full URL of the Cohere-compatible reranking endpoint. |
| `remote_api_key` | `None` | Bearer token sent in the `Authorization` header. |
| `remote_model` | `None` | Optional model identifier forwarded in the request body. |
| `remote_timeout` | `30.0` | Per-request timeout in seconds. |
| `remote_options` | `{}` | Extra key-value pairs merged into the request body. |

See [Retrieval and vector stores](../build/retrieval.md) for per-backend parameter details, retrieval method comparison, and custom backend patterns. See [Reranking — Python library](../build/reranking.md) for per-strategy recipes and the [Reranking — ComposerUI](../build/reranking-ui.md) guide for search space configuration.

## Query transformation and web search

### Query transformation

| Key | Purpose | Typical value |
| --- | --- | --- |
| `query_transformer` | Enables query transformation before retrieval. | `True`, `False` |
| `query_transformer_strategy` | Transformation strategy to apply. | `"rewrite"`, `"multi_query"`, `"decomposition"`, `"hyde"`, `"step_back"` |

#### Strategy-specific parameters (`strategy_config`)

Pass strategy-specific options as a `strategy_config` dictionary to `QueryTransformer` or via keyword arguments to `create_query_transformer`:

| Strategy | Parameter | Default | Description |
| --- | --- | --- | --- |
| `rewrite` | `prompt_key` | `"query_rewrite"` | Prompt template key resolved from `MuffakirPrompt`. |
| `multi_query` | `prompt_key` | `"multi_query_expansion"` | Prompt template key. |
| `multi_query` | `temperature` | `0.2` | Sampling temperature for synonym variance. |
| `decomposition` | `prompt_key` | `"query_decomposition"` | Prompt template key. |
| `hyde` | `prompt_key` | `"hyde"` | Prompt template key. |
| `hyde` | `temperature` | `0.3` | Sampling temperature for hypothetical document prose. |
| `step_back` | `prompt_key` | `"step_back"` | Prompt template key. |

See [Query transformation — Python library](../build/query-transformers.md) for per-strategy usage and [Query transformation — ComposerUI](../build/query-transformers-ui.md) for the Composer search space dimension.

### Adaptive web search (`MuffakirRAG`)

Configure automatic web search fallback when local corpus retrieval is graded as not relevant:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `adaptive_web_search` | `bool` | `False` | Enables relevance grading and automated web-search fallback. |
| `search_provider` | `str` | `"tavily"` | Web search provider: `"tavily"`, `"firecrawl"`, or `"serpapi"`. |
| `search_provider_config` | `dict` | `{}` | Provider-specific credentials and tuning parameters. |

#### Provider configuration parameters (`search_provider_config`)

| Provider | Parameter | Default | Description |
| --- | --- | --- | --- |
| `tavily` | `api_key` | `env:TAVILY_API_KEY` | Tavily API credential. |
| `tavily` | `max_results` | `5` | Maximum search snippets to retrieve. |
| `firecrawl` | `api_key` | *Required* | Firecrawl API credential (or top-level `fire_crawl_api`). |
| `firecrawl` | `max_depth` | `2` | Maximum crawl depth for deep research. |
| `firecrawl` | `time_limit` | `30` | Maximum crawl duration in seconds. |
| `firecrawl` | `max_urls` | `5` | Maximum unique URLs crawled during session. |
| `serpapi` | `api_key` | `env:SERPAPI_API_KEY` | SerpAPI credential. |
| `serpapi` | `max_results` | `5` | Maximum organic search results to extract. |

### Standalone web search (`MuffakirSearch`)

Pass the configuration dictionary directly to `MuffakirSearch(config)`:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `search_provider` | `str` | `"firecrawl"` | Provider name (`"tavily"`, `"firecrawl"`, `"serpapi"`). |
| `search_provider_config` | `dict` | `{}` | Provider-specific credentials and tuning (see table above). |
| `api_key` | `str` | *Required* | LLM provider credential. |
| `llm_provider` | `str` | *Required* | LLM provider name (`"openai"`, `"anthropic"`, `"groq"`, etc.). |
| `llm_model` | `str` | *Required* | Model identifier. |
| `llm_temperature` | `float` | `0.0` | Sampling temperature for answer generation. |
| `llm_max_tokens` | `int` | `4096` | Max generation tokens. |
| `language` | `str` | `"ar"` | Answer language (`"ar"` or `"en"`). |
| `prompt_overrides` | `dict` | `None` | Custom prompt templates. |

See [Adaptive web search](../build/web-search.md) for complete recipes, architecture diagrams, and evaluation workflows.

## Synthetic data generation

All keys are passed as a dictionary to `MuffakirSyntheticData`.

### Core

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `data_dir` | `str` | *Required* | Directory containing source documents. |
| `api_key` | `str` | *Required* | LLM provider credential. |
| `llm_provider` | `str` | *Required* | Provider name. |
| `llm_model` | `str` | *Required* | Model identifier. |
| `language` | `str` | `"ar"` | Prompt language (`"ar"` or `"en"`). |
| `base_url` | `str` | `None` | Custom OpenAI-compatible endpoint. |
| `llm_temperature` | `float` | `0.3` | Sampling temperature. |
| `llm_max_tokens` | `int` | `2000` | Max tokens per LLM call. |

### Chunking

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `chunking_method` | `str` | `"recursive"` | Chunking strategy. |
| `chunk_size` | `int` | `600` | Target chunk size. |
| `chunk_overlap` | `int` | `200` | Overlap between chunks. |
| `skip_empty_chunks` | `bool` | `True` | Skip chunks below `min_chunk_length`. |
| `min_chunk_length` | `int` | `50` | Minimum character length for a chunk. |

### Validation

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `max_retries` | `int` | `3` | Retry attempts per chunk on failure. |
| `validate_qa_pairs` | `bool` | `True` | Enable Pydantic schema validation. |
| `min_question_length` | `int` | `10` | Minimum question character length. |
| `min_answer_length` | `int` | `15` | Minimum answer character length. |

### Output

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `output_dir` | `str` | `"./muffakir_synthetic_data"` | Directory for datasets and checkpoints. |
| `output_format` | `List[str]` | `["csv", "excel"]` | Export formats: `"csv"`, `"excel"`, `"json"`. |
| `save_frequency` | `int` | `10` | Checkpoint every N successful pairs. |

See the [Synthetic data guide](../guides/synthetic-data.md) for full recipes and the [Public API](api.md#syntheticdata) for class signatures.

## Evaluation and Composer

### Evaluation (`MuffakirEvaluation`)

All keys can be passed in the `config` dictionary to `MuffakirEvaluation`:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `metrics` | `List[str]` | `DEFAULT_METRICS` | List of metrics to compute: `"recall"`, `"precision"`, `"mrr"`, `"ndcg"`, `"faithfulness"`, `"answer_correctness"`, `"llm_judge_rating"`. |
| `k` | `int` | `5` | Retrieval evaluation depth ($k$ chunks). |
| `language` | `str` | `"ar"` | Language for prompt templates (`"ar"` or `"en"`). |
| `api_key` | `str` | `None` | Optional override API key for independent judge LLM calls. |
| `llm_provider` | `str` | `None` | Optional override provider name for judge LLM. |
| `llm_model` | `str` | `None` | Optional override model name for judge LLM. |
| `llm_temperature` | `float` | `0.0` | Sampling temperature for judge models. |
| `llm_max_tokens` | `int` | `4096` | Max token limit for judge responses. |
| `base_url` / `llm_base_url` | `str` | `None` | Custom endpoint for judge LLM. |
| `max_samples` | `int` | `None` | Maximum number of dataset rows to evaluate. |
| `fail_fast_dataset` | `bool` | `True` | Raise immediately on corrupt/missing dataset rows. |
| `output_dir` | `str` | `"./muffakir_eval_results"` | Directory where `summary.json` and `samples.csv` are saved. |
| `prompt_overrides` | `dict` | `None` | Template overrides for `faithfulness`, `answer_correctness`, or `llm_judge_rating`. |

See the [Evaluation guide](../evaluate/evaluation.md) for metric formulas, Arabic text matching, and report interpretation.

### Composer (`MuffakirComposer`)

#### Base configuration keys

Passed to `MuffakirComposer(config)` to establish default parameters shared by all trials:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `data_dir` | `str` | *Required* | Path to document corpus (omitted when `retrieval_source="web_search_only"`). |
| `llm_provider` | `str` | *Required* | Default generation LLM provider (omitted when `pipeline_mode="retrieval_only"`). |
| `llm_model` | `str` | *Required* | Default generation LLM model identifier. |
| `api_key` | `str` | *Required* | LLM API key (omitted for local Ollama/vLLM endpoints or `retrieval_only` mode). |
| `pipeline_mode` | `str` | `"full_rag"` | Search mode: `"full_rag"` (end-to-end) or `"retrieval_only"` (retrieval metrics only). |
| `retrieval_source` | `str` | `"vector_db"` | Retrieval source: `"vector_db"` (local documents) or `"web_search_only"` (live web). |
| `embedding_model` | `str` | `DEFAULT_EMBEDDING_MODEL` | Default dense embedding model for indexing. |
| `embedding_provider` | `str` | `"sentence_transformers"` | Embedding provider backend. |
| `vector_db_provider` | `str` | `"chroma"` | Default vector store backend. |
| `chunk_size` | `int` | `500` | Default document chunk length. |
| `chunk_overlap` | `int` | `100` | Default chunk overlap. |
| `prompt_overrides` | `dict` | `None` | Custom prompt templates used across trials. |

#### `fit()` execution arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `search_space` | `dict` | `DEFAULT_SEARCH_SPACE` | Dictionary of pipeline stages to explore (see table below). |
| `eval_dataset` | `str` / `list` | *Required* | Path to dataset file (`.jsonl`, `.csv`, `.xlsx`, `.json`) or list of `QAPair` objects. |
| `strategy` | `str` | `"grid"` | Search strategy (`"grid"`). |
| `n_jobs` | `int` | `4` | Worker process count for parallel trial execution. |
| `metrics` | `List[str]` | `["recall", "precision", ...]` | Metrics to compute per trial. |
| `metric_weights` | `Dict[str, float]` | Equal weighting | Relative weights used to calculate the composite trial score. |
| `max_eval_samples`| `int` | `50` | Maximum samples from `eval_dataset` to evaluate per trial. |
| `max_trials` | `int` | `None` | Maximum number of trials to run before early termination. |
| `max_runtime_minutes` | `float` | `None` | Maximum elapsed search duration in minutes before stopping. |
| `checkpoint_dir` | `str` | `"./muffakir_checkpoints/"` | Directory for atomic JSON checkpoint state. |
| `resume` | `bool` | `True` | Resume search from checkpoint if a prior run was interrupted. |
| `save_report` | `bool` | `True` | Automatically save summary reports upon completion. |
| `custom_pricing` | `dict` | `None` | Overrides for LLM per-token input and output dollar rates. |
| `enable_trace` | `bool` | `True` | Record fine-grained sample telemetry and error manifests. |
| `trace_dir` | `str` | `None` | Directory for trace files (defaults to `<checkpoint_dir>/trace`). |

#### Search space dimensions (`search_space`)

| Dimension | Values / Examples | Description |
| --- | --- | --- |
| `query_expansion` | `["none", "rewrite", "multi_query", "decomposition", "hyde", "step_back"]` | Pre-retrieval query transformations. |
| `retrieval` | `["similarity_search", "max_marginal_relevance", "hybrid", "contextual"]` | Retrieval algorithms. |
| `reranking` | `["none", "semantic_similarity", "bm25", "cross_encoder", "pointwise", "llm"]` | Reranker strategies. |
| `reranking_model` | `["BAAI/bge-reranker-base", ...]` | Model IDs (evaluated only for `cross_encoder` and `pointwise`). |
| `k` | `[3, 5, 8, 10]` | Candidate chunks passed to generation. |
| `chunking` | `[{"method": "recursive", "size": 500, "overlap": 100}, ...]` | Document segmentation settings. |
| `embedding_model` | `["mohamed2811/Muffakir_Embedding", ...]` | Embedding model names. |
| `vector_db_provider` | `["chroma", "qdrant", "faiss"]` | Storage backends. |
| `llm` | `[{"provider": "openai", "model": "gpt-4o-mini"}, ...]` | Alternative generation LLMs. |

See [Composer search](../evaluate/composer.md) for full usage recipes and Pareto frontier analysis, and see [Export a trial to Python](../evaluate/python-export.md) to generate standalone code.

See the [public API](api.md) for signatures and [Composer search](../evaluate/composer.md)
for complete examples. See [Prompt management](../build/prompts-and-language.md) for prompt keys,
placeholder contracts, and localized overrides. See [Cost and pricing](../build/pricing.md) for custom token rates and
financial tracking. See [Embeddings](../build/embeddings.md) for embedding providers, caching,
and hardware acceleration. See [Document parsers](../build/document-parsers.md) for parsing options,
supported formats, and OCR configuration. See [LLM providers](../build/llm-providers.md) for every supported
provider, custom endpoints, role overrides, and Azure-specific settings. See
[Query transformation](../build/query-transformers.md) for strategy details, conversation history, and custom strategies. See
[Reranking](../build/reranking.md) for all six reranking strategies, remote endpoint configuration, and custom rerankers. See
[Synthetic data](../guides/synthetic-data.md) for dataset generation, validation options, and output formats. See
[Traces and observability](../evaluate/observability.md) for the full trace file schema, per-stage timing fields, and error-rate diagnostics. See
[Retrieval and vector stores](../build/retrieval.md) for backend selection, retrieval methods, and custom vector store patterns. See
[Adaptive web search](../build/web-search.md) for web search backends, provider parameters, fallback mechanics, and zero-corpus QA.
