"""
MuffakirComposer - Main orchestrator for automated RAG architecture search.

Provides a scikit-learn-like interface (composer.fit()) for finding optimal
RAG pipeline configurations through grid search with parallel execution.
"""

import logging
import time
from copy import deepcopy
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from SyntheticData.models import QAPair
from .config_space import ConfigSpace
from .checkpoint import CheckpointManager
from .report_generator import ReportGenerator
from .results.report import ComposerReport
from .evaluation import evaluate_trial, SUPPORTED_PROVIDERS
from Muffakir.constants import DEFAULT_COMPOSER_CONFIG
from Muffakir.exceptions import ConfigurationError
from PromptManager.PromptManager import MuffakirPrompt
from Evaluation.constants import GENERATION_METRICS

_LAZY_COMPONENTS = {
    "DatasetLoader": ("Composer.dataset_loader", "DatasetLoader"),
    "GridSearch": ("Composer.search.grid_search", "GridSearch"),
}


def _component(name: str):
    value = globals().get(name)
    if value is not None:
        return value
    from importlib import import_module

    module_name, attribute = _LAZY_COMPONENTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __getattr__(name: str):
    if name in _LAZY_COMPONENTS:
        return _component(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

logger = logging.getLogger(__name__)



class MuffakirComposer:
    """
    Automated Architecture Search for RAG Pipelines.
    
    Provides a scikit-learn-like interface for finding optimal RAG pipeline
    configurations by systematically evaluating different combinations of:
    - Query expansion methods
    - Retrieval strategies
    - Reranking approaches
    - Top-k values
    
    Example:
        ```python
        composer = MuffakirComposer(config={
            "data_dir": "/path/to/documents",
            "api_key": "your-api-key",
            "llm_provider": "together",
            "llm_model": "meta-llama/Llama-3-8b-chat-hf",
        })
        
        report = composer.fit(
            search_space={
                "query_expansion": ["none", "multi_query", "hyde"],
                "retrieval": ["similarity_search", "hybrid"],
                "reranking": ["none", "cross_encoder"],
                "k": [3, 5, 10],
            },
            n_jobs=4,
        )
        
        print(report.best_config)
        print(report.best_score)
        ```
    
    Attributes:
        config: Base configuration dictionary
        search_space: ConfigSpace instance defining the search space
    """
    
    # Centralized default configuration values
    DEFAULT_CONFIG = DEFAULT_COMPOSER_CONFIG

    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        """
        Initialize MuffakirComposer.
        
        Args:
            config: Configuration dictionary or kwargs with:
                - data_dir / documents_path (required): Path to documents directory
                - api_key (optional for local endpoints): LLM API key
                - llm_provider (required): Provider name (together, openai, custom, etc.)
                - llm_model (required): Model identifier
                - base_url / llm_base_url (optional): OpenAI-compatible custom endpoint URL
                - embedding_model (optional): Embedding model name
                - vector_db_provider (optional): Vector DB provider
                - chunk_size, chunk_overlap (optional): Chunking settings
                - And other optional settings
        """
        user_config = dict(config) if config else {}
        user_config.update(kwargs)

        if "documents_path" in user_config and "data_dir" not in user_config:
            user_config["data_dir"] = user_config["documents_path"]

        self.config = self._merge_config(user_config)
        self._validate_config()
        MuffakirPrompt(
            language=self.config.get("language", "ar"),
            overrides=self.config.get("prompt_overrides"),
        )
        
        logger.info("🎵 MuffakirComposer initialized")
        logger.info(f"   Data directory: {self.config['data_dir']}")
        logger.info(f"   LLM: {self.config['llm_provider']}/{self.config['llm_model']}")
    
    def _merge_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user config with defaults (deep copy to avoid sharing mutable defaults)."""
        config = deepcopy(self.DEFAULT_CONFIG)
        config.update(user_config)
        return config
    
    def _validate_config(self):
        """Validate required configuration parameters."""
        required = ["data_dir", "llm_provider", "llm_model"]
        
        provider_val = (self.config.get("llm_provider") or "").lower().strip()
        if provider_val not in ("ollama", "vllm") and not self.config.get("base_url") and not self.config.get("llm_base_url"):
            required.append("api_key")

        # web_search_only mode has no local corpus to index -- data_dir is not
        # needed at all (retrieval happens entirely via a live web search
        # provider per trial, see Composer/evaluation.py::evaluate_trial).
        web_search_only = self.config.get("retrieval_source") == "web_search_only"
        if web_search_only:
            required.remove("data_dir")

        # retrieval_only mode never calls a main generation LLM (no answer is
        # ever produced) -- api_key is not required here. llm_provider/
        # llm_model are also not required in this mode when the caller didn't
        # supply them (DEFAULT_COMPOSER_CONFIG defaults them to None, which
        # would otherwise fail the required-param check below). If the
        # caller does supply them anyway (e.g. reusing a full_rag config),
        # they're still validated against SUPPORTED_PROVIDERS further down.
        # A query-transform LLM, if enabled, is validated separately and
        # later, inside MuffakirRetrieval itself, once its own override keys
        # are known per-trial.
        retrieval_only = self.config.get("pipeline_mode") == "retrieval_only"
        if retrieval_only:
            if "api_key" in required:
                required.remove("api_key")
            if not self.config.get("llm_provider") and "llm_provider" in required:
                required.remove("llm_provider")
            if not self.config.get("llm_model") and "llm_model" in required:
                required.remove("llm_model")

        for param in required:
            if not self.config.get(param):
                raise ValueError(f"❌ Required parameter '{param}' is missing")

        # Validate data directory
        if not web_search_only:
            data_path = Path(self.config["data_dir"])
            if not data_path.exists():
                raise FileNotFoundError(f"❌ Data directory does not exist: {self.config['data_dir']}")

        # Validate provider is supported by the downstream trial pipeline (MuffakirRAG).
        # Skipped when retrieval_only left llm_provider unset (nothing to validate).
        if self.config.get("llm_provider") and self.config["llm_provider"] not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"❌ Unsupported llm_provider: '{self.config['llm_provider']}'. "
                f"Supported: {sorted(SUPPORTED_PROVIDERS.keys())}"
            )

    
    def fit(
        self,
        search_space: Optional[Dict[str, List]] = None,
        eval_dataset: Optional[Union[str, List[QAPair]]] = None,
        strategy: str = "grid",
        n_jobs: int = 4,
        metrics: Optional[List[str]] = None,
        metric_weights: Optional[Dict[str, float]] = None,
        max_eval_samples: int = 50,
        save_report: bool = True,
        report_path: str = "./muffakir_report.json",
        checkpoint_dir: str = "./muffakir_checkpoints/",
        resume: bool = True,
        max_trials: Optional[int] = None,
        max_runtime_minutes: Optional[float] = None,
        custom_pricing: Optional[Dict[str, Dict[str, float]]] = None,
        enable_trace: bool = True,
        trace_dir: Optional[str] = None,
        trace_queue: Optional[Any] = None,
    ) -> ComposerReport:
        """Run ``fit`` inside a whole-run observation and writer lifecycle.

        ComposerUI supplies its own cross-process queue so it can begin tracing
        before constructing Composer. Direct SDK callers get an equivalent
        session created here and drained in ``finally``.
        """
        import multiprocessing

        from Trace.models import RunManifest
        from Trace.observability import (
            collect_secret_values,
            observation_context,
            record_exception_outcome,
        )
        from Trace.writer import TraceWriter

        owned_manager = None
        owned_writer = None
        effective_queue = trace_queue if enable_trace else None
        resolved_trace_dir = trace_dir or f"{checkpoint_dir}/trace"
        if enable_trace and effective_queue is None:
            owned_manager = multiprocessing.Manager()
            effective_queue = owned_manager.Queue()
            owned_writer = TraceWriter(resolved_trace_dir, work_queue=effective_queue)

        caught_error: Optional[BaseException] = None
        try:
            with observation_context(
                effective_queue,
                secrets=collect_secret_values(self.config),
            ):
                try:
                    return self._fit_impl(
                        search_space=search_space,
                        eval_dataset=eval_dataset,
                        strategy=strategy,
                        n_jobs=n_jobs,
                        metrics=metrics,
                        metric_weights=metric_weights,
                        max_eval_samples=max_eval_samples,
                        save_report=save_report,
                        report_path=report_path,
                        checkpoint_dir=checkpoint_dir,
                        resume=resume,
                        max_trials=max_trials,
                        max_runtime_minutes=max_runtime_minutes,
                        custom_pricing=custom_pricing,
                        enable_trace=enable_trace,
                        trace_dir=trace_dir,
                        trace_queue=effective_queue,
                    )
                except Exception as error:
                    caught_error = error
                    if enable_trace and effective_queue is not None:
                        record_exception_outcome(
                            error,
                            recovery="unrecovered",
                            fallback_stage="run_execution",
                            fallback_component="runtime",
                        )
                    raise
        finally:
            if owned_writer is not None:
                owned_writer.close()
            if owned_manager is not None:
                owned_manager.shutdown()
            if owned_writer is not None and caught_error is not None:
                import json

                manifest_path = Path(resolved_trace_dir) / "manifest.json"
                manifest_data = RunManifest(
                    status="failed", search_space=search_space or {}
                ).to_dict()
                if manifest_path.exists():
                    try:
                        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                        manifest_data["status"] = "failed"
                    except Exception:
                        pass
                terminal_writer = TraceWriter(resolved_trace_dir)
                terminal_writer.write_manifest(manifest_data)
                terminal_writer.close()

    def _fit_impl(
        self,
        search_space: Optional[Dict[str, List]] = None,
        eval_dataset: Optional[Union[str, List[QAPair]]] = None,
        strategy: str = "grid",
        n_jobs: int = 4,
        metrics: Optional[List[str]] = None,
        metric_weights: Optional[Dict[str, float]] = None,
        max_eval_samples: int = 50,
        save_report: bool = True,
        report_path: str = "./muffakir_report.json",
        checkpoint_dir: str = "./muffakir_checkpoints/",
        resume: bool = True,
        max_trials: Optional[int] = None,
        max_runtime_minutes: Optional[float] = None,
        custom_pricing: Optional[Dict[str, Dict[str, float]]] = None,
        enable_trace: bool = True,
        trace_dir: Optional[str] = None,
        trace_queue: Optional[Any] = None,
    ) -> ComposerReport:
        """
        Run architecture search to find optimal RAG pipeline configuration.
        
        Args:
            search_space: Dictionary defining options for each pipeline stage.
                         If None, uses DEFAULT_SEARCH_SPACE.
                         Example: {"query_expansion": ["none", "hyde"], "k": [3, 5]}
            
            eval_dataset: Evaluation dataset. Can be:
                         - None: Auto-generate synthetic data from documents
                         - str: Path to CSV file with question, context, answer columns
                         - List[QAPair]: Pre-loaded QAPair objects
            
            strategy: Search strategy. Currently only "grid" is supported.
                     Future: "bayesian" for Optuna-based optimization.
            
            n_jobs: Number of parallel workers for grid search.
                   Set to 1 for sequential execution (debugging).
            
            metrics: List of metrics to compute. Options:
                    - "recall": Recall@k
                    - "precision": Precision@k
                    - "mrr": Mean Reciprocal Rank
                    - "ndcg": NDCG@k
                    - "faithfulness": Faithfulness score
                    - "answer_correctness": Answer correctness score
                    - "llm_judge_rating": Semantic correctness rating (1–5,
                      normalized to 0–1 only for the composite score)
                    If None, uses ["recall", "faithfulness", "answer_correctness"]
            
            metric_weights: Optional weights for the composite score (metric name -> weight).
                    Unspecified metrics default to weight 1.0. If None, all metrics are
                    weighted equally.
            
            max_eval_samples: Maximum number of evaluation samples to use.
                            Only applies when auto-generating synthetic data.
            
            save_report: Whether to generate JSON report after search.
            
            report_path: File path for JSON report.
            
            checkpoint_dir: Directory for saving checkpoints.
            
            resume: Whether to resume from checkpoint if one exists.
        
        Returns:
            ComposerReport: Report containing all trial results and best configuration.
        
        Example:
            ```python
            report = composer.fit(
                search_space={
                    "query_expansion": ["none", "multi_query", "hyde"],
                    "retrieval": ["similarity_search", "hybrid"],
                    "reranking": ["none", "cross_encoder"],
                    "k": [3, 5, 10],
                },
                eval_dataset="/path/to/eval.csv",  # or None to auto-generate
                n_jobs=4,
                metrics=["recall", "faithfulness", "answer_correctness"],
            )
            
            print(f"Best config: {report.best_config}")
            print(f"Best score: {report.best_score}")
            report.to_dataframe().to_csv("results.csv")
            ```
        """
        start_time = time.perf_counter()
        
        from Trace.observability import observe_stage

        # Set default metrics
        if metrics is None:
            metrics = ["recall", "faithfulness", "answer_correctness"]

        with observe_stage("configuration", "run_setup"):
            # Validate requested metrics against the Evaluation module's supported set
            self._validate_metrics(metrics)
            self._validate_pipeline_mode(metrics)

            # Initialize search space
            config_space = ConfigSpace(search_space if search_space is not None else None)
            from LLMProvider.parameters import validate_config_parameters
            validate_config_parameters(self.config, config_space.search_space)
            # Validate the entire search-space union before creating checkpoints,
            # indexing documents, or launching any worker process.
            from Muffakir.dependency_validation import validate_composer_dependencies

            validate_composer_dependencies(self.config, config_space.search_space)
        from Trace.models import compute_config_hash
        run_config_hash = compute_config_hash(self.config, config_space.search_space)
        logger.info(f"\n{config_space.summary()}")
        
        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(checkpoint_dir)

        # Load or resume from checkpoint
        completed_trials = []
        if resume and checkpoint_manager.has_checkpoint():
            # Fail fast if the search space changed since this checkpoint was
            # written — trial IDs are positional indices over the search space,
            # so a changed space can silently make a resumed run skip newly
            # added trials whose recycled ID matches a previously completed one.
            checkpoint_manager.validate_search_space(config_space.search_space)
            checkpoint_manager.validate_config_hash(run_config_hash)
            completed_trials = checkpoint_manager.load_completed_trials()
            logger.info(f"📂 Resuming from checkpoint: {len(completed_trials)} trials completed")
        checkpoint_manager.set_search_space(config_space.search_space)
        checkpoint_manager.set_config_hash(run_config_hash)

        # Pricing: fetched at most once per fit() run (never per-trial). On
        # resume, the checkpoint's stored snapshot is reused verbatim instead
        # of re-fetching, so a trial run today and one resumed days later are
        # priced identically — any custom_pricing passed to a resumed run is
        # ignored in favor of the snapshot already on disk.
        from Pricing.price_map import PriceMap

        stored_pricing_snapshot = checkpoint_manager.get_stored_pricing_snapshot()
        if stored_pricing_snapshot is not None:
            price_map = PriceMap.from_dict(stored_pricing_snapshot)
        else:
            price_map = PriceMap(custom_pricing=custom_pricing)
            with observe_stage("pricing", "pricing") as pricing_observation:
                price_map.load()
                if price_map.fetch_failed:
                    pricing_observation.mark_error(
                        RuntimeError("Pricing catalog fetch failed; costs may be unavailable"),
                        "fallback",
                    )
                checkpoint_manager.set_pricing_snapshot(price_map.to_dict())

        # Step 1: Load evaluation dataset
        logger.info("\n📊 Step 1: Loading evaluation dataset...")
        dataset_llm_config = dict(self.config)
        if self.config.get("dataset_llm_parameters") is not None:
            dataset_llm_config["llm_parameters"] = self.config["dataset_llm_parameters"]
        if self.config.get("dataset_llm_provider"):
            dataset_llm_config["llm_provider"] = self.config["dataset_llm_provider"]
        if self.config.get("dataset_llm_model"):
            dataset_llm_config["llm_model"] = self.config["dataset_llm_model"]
        if self.config.get("dataset_api_key"):
            dataset_llm_config["api_key"] = self.config["dataset_api_key"]
        if self.config.get("dataset_base_url"):
            dataset_llm_config["base_url"] = self.config["dataset_base_url"]
        dataset_loader = _component("DatasetLoader")
        with observe_stage("dataset_load", "dataset"):
            eval_pairs = dataset_loader.load_or_generate(
                eval_dataset=eval_dataset,
                data_dir=self.config["data_dir"],
                llm_config=dataset_llm_config,
                max_samples=max_eval_samples,
                language=self.config.get("language", "ar"),
                prompt_overrides=self.config.get("prompt_overrides"),
            )
            if not eval_pairs:
                from Muffakir.exceptions import DatasetError

                raise DatasetError(
                    "The evaluation dataset contains no valid Q&A samples. "
                    "Composer cannot score or complete trials without samples."
                )
        logger.info(f"   Loaded {len(eval_pairs)} evaluation samples")

        # Step 2: Generate combinations
        logger.info("\n🔀 Step 2: Generating pipeline combinations...")
        all_combinations = config_space.generate_combinations_with_ids()
        completed_ids = {t.trial_id for t in completed_trials}
        remaining_combinations = config_space.filter_combinations(completed_ids, all_combinations)
        logger.info(f"   Total: {len(all_combinations)}, Remaining: {len(remaining_combinations)}")

        # Trace: a RunManifest + TraceWriter for this run (see Trace package).
        # The writer's background thread drains a multiprocessing.Manager()
        # Queue so ProcessPoolExecutor workers (evaluate_trial) can push
        # trial/sample records across the process boundary without any
        # per-worker shard files to merge later.
        trace_manager = None
        trace_writer = None
        external_trace_queue = trace_queue
        owns_trace_runtime = False
        if enable_trace:
            import multiprocessing
            from Trace.models import RunManifest
            from Trace.writer import TraceWriter

            resolved_trace_dir = trace_dir or f"{checkpoint_dir}/trace"
            if external_trace_queue is None:
                trace_manager = multiprocessing.Manager()
                trace_queue = trace_manager.Queue()
                trace_writer = TraceWriter(resolved_trace_dir, work_queue=trace_queue)
                owns_trace_runtime = True
            else:
                trace_queue = external_trace_queue
            run_manifest = RunManifest(
                config_hash=run_config_hash,
                status="running",
                total_trials=len(all_combinations),
                search_space=config_space.search_space,
                pricing_source_url=price_map.source_url,
                pricing_fetched_at=price_map.fetched_at,
                pricing_fetch_failed=price_map.fetch_failed,
            )
            if trace_writer is not None:
                trace_writer.write_manifest(run_manifest.to_dict())
            else:
                trace_queue.put(("manifest", run_manifest.to_dict()))

        # Step 3: Index documents into the shared VectorDB(s) (once per distinct
        # index-time combination, in the main process). Skipped entirely in
        # web_search_only mode -- there is no local corpus to index; every
        # trial answers via a live web search provider instead (see
        # Composer/evaluation.py::evaluate_trial).
        from Trace.observability import collect_secret_values, observation_context

        trace_secrets = collect_secret_values(self.config)
        if self.config.get("retrieval_source") == "web_search_only":
            logger.info("\n📚 Step 3: Skipped (retrieval_source=web_search_only, no local corpus)")
        else:
            logger.info("\n📚 Step 3: Indexing documents...")
            with observation_context(trace_queue, secrets=trace_secrets):
                self._index_documents(remaining_combinations)

        # Step 4: Run grid search
        logger.info("\n🚀 Step 4: Running grid search...")
        
        # Picklable shared state passed to every trial (no live DB/LLM objects).
        # Workers open the shared on-disk VectorDB read-only and cache the embedding
        # model once per process (see Composer.evaluation._get_shared_components).
        trial_base_config = deepcopy(self.config)
        prompt_overrides = trial_base_config.pop("prompt_overrides", {}) or {}
        shared_state = {
            "base_config": trial_base_config,
            "prompt_config": {
                "language": self.config.get("language", "ar"),
                "overrides": prompt_overrides,
            },
            "eval_pairs": eval_pairs,
            "metrics": metrics,
            "metric_weights": metric_weights or {},
            "pricing_snapshot": price_map.to_dict(),
            "trace_queue": trace_queue,
            "observation_queue": trace_queue,
        }
        
        # Initialize search strategy
        if strategy == "grid":
            search_class = _component("GridSearch")
            search = search_class(
                n_jobs=n_jobs,
                max_retries=1,
                max_trials=max_trials,
                max_runtime_minutes=max_runtime_minutes,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Available: 'grid'")
        
        # Run search (evaluate_trial is a top-level picklable function)
        try:
            if remaining_combinations:
                new_trials = search.run(
                    combinations=remaining_combinations,
                    eval_fn=evaluate_trial,
                    shared_state=shared_state,
                    checkpoint_manager=checkpoint_manager,
                )
            else:
                logger.info("   All trials already completed (resumed from checkpoint)")
                new_trials = []
        except Exception:
            if enable_trace:
                run_manifest.status = "failed"
                trace_queue.put(("manifest", run_manifest.to_dict()))
                if owns_trace_runtime:
                    trace_writer.close()
                    trace_manager.shutdown()
            raise
        else:
            if enable_trace:
                stopped_early_now = getattr(search, "stopped_early", False) if remaining_combinations else False
                run_manifest.status = "stopped_early" if stopped_early_now else "completed"
                trace_queue.put(("manifest", run_manifest.to_dict()))

        # Step 5: Aggregate results
        logger.info("\n📈 Step 5: Aggregating results...")
        all_trials = completed_trials + new_trials
        
        # Calculate duration
        total_duration_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Create report
        try:
            report = ComposerReport(
                trials=all_trials,
                search_space=config_space.search_space,
                total_duration_ms=total_duration_ms,
                metrics_used=metrics,
                stopped_early=getattr(search, "stopped_early", False) if remaining_combinations else False,
                stop_reason=getattr(search, "stop_reason", None) if remaining_combinations else None,
            )

            # Step 6: Generate report
            if save_report:
                logger.info("\n📄 Step 6: Generating report...")
                with observation_context(trace_queue, secrets=trace_secrets):
                    with observe_stage("report_generation", "reporting"):
                        json_path = str(Path(report_path).with_suffix(".json"))
                        ReportGenerator.generate_json(report, json_path)
        except Exception:
            if enable_trace:
                run_manifest.status = "failed"
                trace_queue.put(("manifest", run_manifest.to_dict()))
                if owns_trace_runtime:
                    trace_writer.close()
                    trace_manager.shutdown()
            raise

        # Finishing the search loop does not mean the run succeeded. Preserve
        # the report and checkpoint for diagnosis, but correct the terminal
        # manifest when every trial failed or no winning result was produced.
        if enable_trace and (
            report.successful_trials == 0 or report.best_trial is None
        ):
            logger.error(
                "All Composer trials failed; marking the run manifest as failed"
            )
            run_manifest.status = "failed"
            trace_queue.put(("manifest", run_manifest.to_dict()))

        if enable_trace and owns_trace_runtime:
            trace_writer.close()
            trace_manager.shutdown()
        
        # Print summary
        logger.info("\n" + report.summary())
        
        return report
    
    def _validate_metrics(self, metrics: List[str]) -> None:
        """Validate that all requested metrics are supported by the Evaluation module."""
        try:
            from Evaluation.models import ALL_METRICS
        except ImportError:
            # Evaluation module not available; skip strict validation
            logger.warning("Evaluation.models not importable; skipping metric validation")
            return
        supported = {m.lower() for m in ALL_METRICS}
        invalid = [m for m in metrics if m.lower() not in supported]
        if invalid:
            raise ValueError(
                f"Unsupported metric(s): {invalid}. "
                f"Supported: {sorted(supported)}"
            )

    def _validate_pipeline_mode(self, metrics: List[str]) -> None:
        """Fail fast, before any trial runs, when pipeline_mode=retrieval_only
        is combined with a config that can't work: generation metrics (no
        answer is ever produced in this mode) or web_search_only (retrieval_only
        needs a local vector index to test; web_search_only has none)."""
        if self.config.get("pipeline_mode") != "retrieval_only":
            return

        generation_metrics_requested = [
            m for m in metrics if m.lower() in GENERATION_METRICS
        ]
        if generation_metrics_requested:
            raise ConfigurationError(
                f"pipeline_mode='retrieval_only' cannot score {generation_metrics_requested} "
                "-- no answer is ever generated in this mode. Use recall/precision/mrr/ndcg."
            )

        if self.config.get("retrieval_source") == "web_search_only":
            raise ConfigurationError(
                "pipeline_mode='retrieval_only' and retrieval_source='web_search_only' "
                "are mutually exclusive -- retrieval_only needs a local vector index "
                "to test, web_search_only has none."
            )

    def _index_documents(self, remaining_combinations: List[tuple]) -> None:
        """
        Index documents into one VectorDB per distinct index-time combination
        (chunking, embedding_model, vector_db_provider) actually present across
        ``remaining_combinations``. Query-time-only search spaces (the common
        case today) resolve to exactly one combination, so this still builds
        exactly one index, matching prior behavior.

        Skips any combination whose target directory already exists on disk
        (resume-safety: re-running the same search space reuses existing
        indices instead of rebuilding them).
        """
        from TextProcessor.ChunkingAndProcessing import ChunkingAndProcessing
        from TextProcessor.MuffakirChunking import MuffakirChunking
        from VectorDB import create_vector_db
        from Embedding.EmbeddingProvider import EmbeddingProvider
        from .config_space import trial_config_to_rag_config
        from .index_key import compute_index_key
        from Trace.observability import observe_stage

        # Deduplicate by index_key — multiple trials with different query-time
        # config (retrieval/reranking/k/llm) still resolve to the same
        # index-time combination and must not re-index the corpus twice.
        seen_keys = set()
        distinct_rag_configs = []
        for _trial_id, trial_config in remaining_combinations:
            rag_config = trial_config_to_rag_config(trial_config, self.config)
            index_key = compute_index_key(rag_config)
            if index_key in seen_keys:
                continue
            seen_keys.add(index_key)
            distinct_rag_configs.append(rag_config)

        already_built = [
            rc for rc in distinct_rag_configs if Path(rc["db_path"]).exists()
        ]
        to_build = [
            rc for rc in distinct_rag_configs if not Path(rc["db_path"]).exists()
        ]
        logger.info(
            f"   {len(distinct_rag_configs)} distinct index configuration(s) needed "
            f"({len(already_built)} already built, {len(to_build)} to build)"
        )

        for rag_config in to_build:
            chunking = MuffakirChunking(
                chunker=rag_config.get("chunking_method", "recursive"),
                chunker_config={
                    "size": rag_config.get("chunk_size", 600),
                    "overlap": rag_config.get("chunk_overlap", 200),
                },
                language=rag_config.get("language", "auto"),
            )

            # Optional OCR/document-parser support (e.g. azure/docling/llama_parse)
            # for scanned or complex PDFs. Only imported/instantiated when a
            # document_parser is actually configured, so the optional parser
            # package is never required unless this is used.
            document_parser = None
            if rag_config.get("document_parser"):
                from DocumentParser import create_document_parser

                document_parser = create_document_parser(
                    rag_config["document_parser"],
                    **(rag_config.get("document_parser_config") or {}),
                )

            processor = ChunkingAndProcessing(
                directory_path=rag_config["data_dir"],
                muffakir_chunking=chunking,
                document_parser=document_parser,
            )
            with observe_stage(
                "document_parse_chunk",
                "document_processing",
                provider=rag_config.get("document_parser") or "builtin",
            ):
                documents = processor.process_all(
                    chunking_method=rag_config.get("chunking_method", "recursive"),
                    use_ocr=bool(rag_config.get("use_ocr")),
                )

            with observe_stage(
                "embedding_initialization",
                "embedding",
                provider=rag_config.get("embedding_provider", "sentence_transformers"),
                model=rag_config.get("embedding_model"),
            ):
                embedding_provider = EmbeddingProvider(
                    model_name=rag_config.get("embedding_model", "mohamed2811/Muffakir_Embedding"),
                    provider=rag_config.get("embedding_provider", "sentence_transformers"),
                    api_key=rag_config.get("api_key"),
                    batch_size=rag_config.get("embedding_batch_size", 16),
                    device=rag_config.get("device", "auto"),
                )

            vector_db_config = dict(rag_config.get("vector_db_config", {}) or {})
            vector_db_config["path"] = rag_config["db_path"]
            vector_db_config["collection_name"] = rag_config["collection_name"]
            vector_db_config.setdefault("model_name", rag_config.get("embedding_model"))

            with observe_stage(
                "vector_db_initialization",
                "vector_db",
                provider=rag_config.get("vector_db_provider", "chroma"),
            ):
                db_manager = create_vector_db(
                    provider=rag_config.get("vector_db_provider", "chroma"),
                    embedding_provider=embedding_provider,
                    **vector_db_config,
                )
            with observe_stage(
                "vector_index_write",
                "vector_db",
                provider=rag_config.get("vector_db_provider", "chroma"),
            ):
                db_manager.add_documents(documents)
            logger.info(
                f"   Indexed {len(documents)} chunks -> {rag_config['db_path']}"
            )
    
    def clear_checkpoint(self, checkpoint_dir: str = "./muffakir_checkpoints/") -> None:
        """
        Clear checkpoint for fresh run.
        
        Args:
            checkpoint_dir: Checkpoint directory to clear
        """
        checkpoint_manager = CheckpointManager(checkpoint_dir)
        checkpoint_manager.clear()
        logger.info("Checkpoint cleared")
    
    def __repr__(self) -> str:
        return f"MuffakirComposer(data_dir={self.config['data_dir']}, llm={self.config['llm_provider']})"
