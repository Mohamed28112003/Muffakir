# Public Python API

This reference covers the stable, user-facing facades and component factories of Muffakir.
See the relevant build guides for end-to-end recipes.

## `MuffakirRAG`

Builds a full RAG pipeline from a configuration. Use `ask`, `get_similar_documents`, and
`add_documents` in application code.

::: Muffakir.Muffakir.MuffakirRAG
    options:
      members:
        - ask
        - get_similar_documents
        - get_similar_documents_with_trace
        - add_documents
        - get_config

## `MuffakirRetrieval`

Runs the retrieval part of a pipeline without answer generation. It is useful for
retrieval-only evaluation and debugging.

::: Muffakir.MuffakirRetrieval.MuffakirRetrieval
    options:
      members:
        - get_similar_documents
        - get_similar_documents_with_trace

## `VectorDB`

Five vector-store backends accessible through a uniform interface.
See the [Retrieval and vector stores guide](../build/retrieval.md) for backend
selection, retrieval methods, and custom backend recipes.

::: VectorDB.factory.create_vector_db

::: VectorDB.base.BaseVectorDBManager
    options:
      members:
        - add_documents
        - similarity_search
        - max_marginal_relevance_search
        - load_all_documents
        - get_all_documents

::: VectorDB.ChromaDBManager.ChromaDBManager
    options:
      members:
        - add_documents
        - load_all_documents
        - get_collection_count

::: VectorDB.FAISSDBManager.FAISSDBManager
    options:
      members:
        - add_documents
        - load_all_documents

::: VectorDB.QdrantDBManager.QdrantDBManager
    options:
      members:
        - add_documents
        - load_all_documents

::: VectorDB.PineconeDBManager.PineconeDBManager
    options:
      members:
        - add_documents
        - load_all_documents

::: VectorDB.MilvusDBManager.MilvusDBManager
    options:
      members:
        - add_documents
        - load_all_documents

## `MuffakirEvaluation`

Evaluates a compatible RAG instance or web search agent against a Q&A dataset. See the
[Evaluation guide](../evaluate/evaluation.md) for metric formulas, Arabic text matching,
and output report analysis.

::: Muffakir.MuffakirEvaluation.MuffakirEvaluation
    options:
      members:
        - evaluate
        - get_config

## `Evaluation`

Evaluation loop execution, report schemas, metric computation, dataset loaders, and
refusal detection.

### Runner and dataset

::: Evaluation.runner.EvalRunner
    options:
      members:
        - run

::: Evaluation.dataset.load_evaluation_dataset

### Models and reports

::: Evaluation.models.EvaluationReport
    options:
      members:
        - summary
        - to_dataframe
        - save

::: Evaluation.models.EvalSampleResult

::: Evaluation.models.RetrievalScores

::: Evaluation.models.GenerationScores

### Metrics and judges

::: Evaluation.metrics.generation.FaithfulnessMetric
    options:
      members:
        - score

::: Evaluation.metrics.generation.AnswerCorrectnessMetric
    options:
      members:
        - score

::: Evaluation.metrics.generation.LLMJudgeRatingMetric
    options:
      members:
        - score

### Observability helpers

::: Evaluation.refusals.detect_answer_refusal

## `MuffakirComposer`

Automated architecture search and hyperparameter optimization for RAG pipelines. See
the [Composer search guide](../evaluate/composer.md) for search space definitions,
Pareto frontier analysis, and checkpoint recovery.

::: Composer.composer.MuffakirComposer
    options:
      members:
        - fit
        - clear_checkpoint

## `Composer`

Architecture search space, checkpoint management, and trial reporting.

### Configuration space and defaults

::: Composer.config_space.ConfigSpace

::: Composer.config_space.DEFAULT_SEARCH_SPACE

### Checkpoint management

::: Composer.checkpoint.CheckpointManager
    options:
      members:
        - load_checkpoint
        - save_trial
        - clear

### Search results and reporting

::: Composer.results.report.ComposerReport
    options:
      members:
        - to_dataframe
        - get_top_n_trials
        - get_pareto_frontier
        - get_failure_clusters
        - save_json

::: Composer.results.trial.TrialResult

To export a winning trial into standalone, executable Python code or a deployable project, see [Export a trial to Python](../evaluate/python-export.md).

## `LLMProvider`

Creates a LangChain chat model from a provider, model, credential, and optional endpoint.
Use it directly when you need an LLM without constructing a RAG pipeline. See the
[LLM provider guide](../build/llm-providers.md) for provider recipes and configuration.

::: LLMProvider.LLMProvider.LLMProvider
    options:
      members:
        - get_llm
        - call
        - get_usage_totals
        - get_per_sample_usage_totals
        - get_sample_usage_totals
        - reset_usage

::: LLMProvider.LLMProvider.create_llm_provider

## `DocumentParser`

Factory and standardized models for document parsing, layout analysis, and OCR. See the
[Document parsers guide](../build/document-parsers.md) for detailed provider recipes and configuration.

::: DocumentParser.create_document_parser

::: DocumentParser.models.ParsedDocument

::: DocumentParser.models.PageContent

::: DocumentParser.base.BaseDocumentParser
    options:
      members:
        - parse_file
        - parse_directory
        - supported_extensions
        - supports

## `Embedding`

Factory and base classes for local, cloud, and custom embedding providers with dual-layer caching and hardware acceleration. See the [Embeddings guide](../build/embeddings.md) for complete recipes.

::: Embedding.create_embedding_provider

::: Embedding.EmbeddingProvider

::: Embedding.base.BaseEmbeddingProvider
    options:
      members:
        - embed_query
        - embed_documents
        - embed_single
        - embed

## `Pricing`

LLM pricing catalog and cost estimation engine. See the [Cost and pricing guide](../build/pricing.md) for usage recipes and custom overrides.

::: Pricing.price_map.PriceMap
    options:
      members:
        - load
        - get_price
        - compute_cost
        - to_dict
        - from_dict

## `PromptManager`

Bilingual prompt template manager and placeholder validation engine. See the [Prompt management guide](../build/prompts-and-language.md) for the complete prompt catalog and override recipes.

::: PromptManager.PromptManager.MuffakirPrompt
    options:
      members:
        - get_prompt
        - update_prompt
        - validate_prompt
        - get_all_prompts
        - print_all_prompts

## `QueryTransformer`

Five pluggable strategies for rewriting or expanding a query before vector retrieval.
See the [Query transformation guide](../build/query-transformers.md) for strategy recipes and comparisons.

::: QueryTransformer.QueryTransformer.QueryTransformer
    options:
      members:
        - transform_query
        - name

::: QueryTransformer.factory.create_query_transformer

::: QueryTransformer.base.BaseQueryTransformer
    options:
      members:
        - transform
        - name

::: QueryTransformer.rewriter.QueryRewriter
    options:
      members:
        - transform

::: QueryTransformer.multi_query.MultiQueryExpansion
    options:
      members:
        - transform

::: QueryTransformer.query_decomposition.QueryDecomposition
    options:
      members:
        - transform

::: QueryTransformer.hyde.HyDEQueryTransformer
    options:
      members:
        - transform

::: QueryTransformer.step_back.StepBackQueryTransformer
    options:
      members:
        - transform

## `Reranker`

Six reranking strategies accessible through a uniform interface and open registry.
See the [Reranking guide](../build/reranking.md) for strategy recipes and comparisons.

::: Reranker.Reranker.Reranker
    options:
      members:
        - rerank

::: Reranker.factory.create_reranker

::: Reranker.factory.register_reranker

::: Reranker.factory.list_reranker_specs

::: Reranker.base.BaseReranker
    options:
      members:
        - score
        - rerank

::: Reranker.semantic_similarity.SemanticSimilarityReranker
    options:
      members:
        - score

::: Reranker.bm25.BM25Reranker
    options:
      members:
        - score

::: Reranker.cross_encoder.CrossEncoderReranker
    options:
      members:
        - score

::: Reranker.pointwise.PointwiseReranker
    options:
      members:
        - score

::: Reranker.llm.LLMReranker
    options:
      members:
        - score

::: Reranker.remote.RemoteReranker
    options:
      members:
        - score

## `MuffakirSearch`

Search RAG facade integrating pluggable web search engines (Tavily, Firecrawl, SerpAPI)
with LLM answer generation for zero-corpus search and QA. See
[Adaptive web search](../build/web-search.md) for full configuration recipes, evaluation,
and telemetry.

::: Muffakir.MuffakirSearch.MuffakirSearch
    options:
      members:
        - search
        - ask
        - get_similar_documents
        - get_config

## `WebSearch`

Pluggable web search engine factory, provider base classes, standardized result models,
and concrete search backends. See the [Adaptive web search guide](../build/web-search.md)
for provider comparison, parameter details, and custom backend development.

::: WebSearch.factory.create_web_search_provider

::: WebSearch.base.BaseWebSearchProvider
    options:
      members:
        - search

::: WebSearch.models.WebSearchResult

::: WebSearch.tavily.TavilyWebSearchProvider
    options:
      members:
        - search

::: WebSearch.firecrawl.FirecrawlWebSearchProvider
    options:
      members:
        - search

::: WebSearch.serpapi.SerpAPIWebSearchProvider
    options:
      members:
        - search

## `SyntheticData`

Generates Q&A evaluation datasets from a document corpus using an LLM.
See the [Synthetic data guide](../guides/synthetic-data.md) for configuration
and workflow details.

::: Muffakir.MuffakirSyntheticData.MuffakirSyntheticData
    options:
      members:
        - generate_dataset

::: SyntheticData.pipeline.SyntheticDataPipeline
    options:
      members:
        - run

::: SyntheticData.models.QAPair

::: SyntheticData.models.GenerationStats

::: SyntheticData.models.SyntheticDataConfig

## `Trace`

Observability helpers, trace schema dataclasses, and the `TraceWriter` that records
per-sample pipeline traces during Composer runs. See
[Traces and observability](../evaluate/observability.md) for the full guide to trace
file layout, field reference, and diagnostic patterns.

### Observability helpers

::: Trace.observability.observation_context

::: Trace.observability.observe_stage

::: Trace.observability.emit_stage_outcome

::: Trace.observability.mark_current_stage_error

::: Trace.observability.record_exception_outcome

::: Trace.observability.sanitize_error_message

### Trace schema

::: Trace.models.RunManifest

::: Trace.models.TrialRecord

::: Trace.models.SampleTraceRecord

### Storage

::: Trace.writer.TraceWriter
    options:
      members:
        - write_manifest
        - write_trial
        - write_trial_started
        - write_sample
        - write_operation
        - close
