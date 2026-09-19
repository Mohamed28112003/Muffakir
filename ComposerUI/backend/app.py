"""
FastAPI application and route wiring for ComposerUI.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response

from ComposerUI.backend import dataset_utils, provider_catalog, run_manager, storage, trace_reader
from ComposerUI.backend.logging_config import configure_logging
from ComposerUI.backend.schemas import (
    ActiveTrialRow,
    CreateRunRequest,
    CreateRunResponse,
    GenerateDatasetRequest,
    GenerateDatasetResponse,
    HealthResponse,
    PromptResolveRequest,
    PromptResolveResponse,
    RunDetailResponse,
    RunListResponse,
    RunMetadataResponse,
    RunSummary,
    StorageSettingsResponse,
    StorageSettingsUpdate,
    TrialDetailResponse,
    ErrorObservabilityResponse,
    ErrorRateSummary,
    ErrorEventRow,
    TrialRow,
    TrialsPageResponse,
    ValidateRunResponse,
)
from Evaluation.constants import GENERATION_METRICS, RETRIEVAL_METRICS

configure_logging()

app = FastAPI(
    title="ComposerUI API",
    description="Backend API for MuffakirComposer Architecture Search Web Interface",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _effective_dataset_llm(request: CreateRunRequest) -> Dict[str, Optional[str]]:
    """Resolve the dataset-generation model, defaulting to the answer model."""
    override_requested = any(
        (
            request.dataset_llm_provider,
            request.dataset_llm_model,
            request.dataset_api_key,
            request.dataset_base_url,
        )
    )
    if override_requested:
        if not request.dataset_llm_provider or not request.dataset_llm_model:
            raise ValueError(
                "A dataset-generation LLM override requires dataset_llm_provider "
                "and dataset_llm_model."
            )
        return {
            "provider": request.dataset_llm_provider,
            "model": request.dataset_llm_model,
            "api_key": request.dataset_api_key,
            "base_url": request.dataset_base_url,
        }
    return {
        "provider": request.llm_provider,
        "model": request.llm_model,
        "api_key": request.api_key,
        "base_url": request.base_url,
    }


def _validate_task_llm(role: str, config: Dict[str, Optional[str]]) -> None:
    provider = str(config.get("provider") or "").strip()
    model = str(config.get("model") or "").strip()
    if not provider or not model:
        raise ValueError(f"{role} requires llm_provider and llm_model.")
    from Muffakir.Enums import resolve_provider_name

    resolve_provider_name(provider)
    if provider.lower() not in ("ollama", "vllm") and not config.get("base_url") and not config.get("api_key"):
        raise ValueError(f"{role} requires an API key or a custom base URL for provider '{provider}'.")


def _preflight_check(request: CreateRunRequest) -> int:
    """Synchronously validate config and search space without I/O or network calls.

    Returns total combinations on success, or raises ValueError / FileNotFoundError.
    """
    from Muffakir import MuffakirComposer
    from Composer.config_space import ConfigSpace

    search_space_dict = request.search_space.to_composer_dict() or None
    config_space = ConfigSpace(search_space=search_space_dict)

    from ComposerUI.backend.prompt_service import validate_overrides
    validate_overrides(request)

    from Reranker.factory import get_reranker_spec

    reranking_values = config_space.search_space.get("reranking", [])
    selected_rerankers = {
        get_reranker_spec(value).name
        for value in reranking_values
        if str(value).strip().lower() != "none"
    }

    if request.retrieval_source == "web_search_only" and not request.search_provider:
        raise ValueError(
            "retrieval_source='web_search_only' requires search_provider "
            "(e.g. 'tavily', 'firecrawl', 'serpapi')."
        )
    if request.adaptive_web_search and not request.search_provider:
        raise ValueError(
            "adaptive_web_search=True requires search_provider "
            "(e.g. 'tavily', 'firecrawl', 'serpapi')."
        )
    if request.pipeline_mode == "retrieval_only" and request.retrieval_source == "web_search_only":
        raise ValueError(
            "pipeline_mode='retrieval_only' and retrieval_source='web_search_only' "
            "are mutually exclusive -- retrieval_only needs a local vector index to test."
        )
    if request.retrieval_source == "web_search_only" and request.eval_dataset_mode == "auto":
        raise ValueError(
            "retrieval_source='web_search_only' requires an existing evaluation dataset; "
            "automatic dataset generation requires a local corpus."
        )
    if request.retrieval_source != "web_search_only" and not request.documents_path:
        raise ValueError(
            "documents_path is required for local and adaptive retrieval. "
            "An evaluation dataset does not replace the retrieval corpus."
        )
    if request.eval_dataset_mode != "auto":
        if not request.eval_dataset_path:
            raise ValueError(
                f"eval_dataset_mode='{request.eval_dataset_mode}' requires eval_dataset_path."
            )
        if not Path(request.eval_dataset_path).exists():
            raise FileNotFoundError(f"Eval dataset file not found: {request.eval_dataset_path}")

        from Evaluation.dataset import load_evaluation_dataset

        load_evaluation_dataset(request.eval_dataset_path)

    composer_config: Dict[str, Any] = {
        **request.parameter_config(),
        "data_dir": request.documents_path,
        "language": request.language,
        "prompt_overrides": dict(request.prompt_overrides),
    }

    effective_metrics = request.metrics if request.metrics else [
        "recall",
        "faithfulness",
        "answer_correctness",
    ]
    retrieval_metrics = RETRIEVAL_METRICS

    if request.pipeline_mode == "retrieval_only":
        composer_config["pipeline_mode"] = "retrieval_only"
        # MuffakirComposer._validate_pipeline_mode enforces this too, but only
        # inside fit() -- which runs in run_manager's background thread, long
        # after /api/runs returned 200. Repeat the check here so both
        # /api/runs/validate and /api/runs reject it synchronously instead of
        # failing asynchronously into error.txt. The default list mirrors
        # Composer/composer.py::fit()'s own metrics fallback exactly.
        generation_metrics_requested = [
            m for m in effective_metrics if str(m).lower() in GENERATION_METRICS
        ]
        if generation_metrics_requested:
            raise ValueError(
                f"pipeline_mode='retrieval_only' cannot score {generation_metrics_requested} "
                "-- no answer is ever generated in this mode. Use recall/precision/mrr/ndcg."
            )
    elif not request.llm_provider or not request.llm_model:
        raise ValueError(
            "llm_provider and llm_model are required unless pipeline_mode='retrieval_only'."
        )

    # Auto-generation happens before any trial runs and always consumes the
    # document corpus plus an LLM, including retrieval-only mode. Web-only
    # runs are rejected above. Validate these requirements so a run
    # cannot be accepted and then fail in the background DatasetLoader.
    if request.eval_dataset_mode == "auto":
        if not request.documents_path:
            raise ValueError(
                "eval_dataset_mode='auto' requires documents_path to generate evaluation data."
            )
        dataset_llm = _effective_dataset_llm(request)
        _validate_task_llm("eval_dataset_mode='auto' dataset generation", dataset_llm)
        composer_config.update(
            dataset_llm_provider=dataset_llm["provider"],
            dataset_llm_model=dataset_llm["model"],
        )
        if dataset_llm["api_key"]:
            composer_config["dataset_api_key"] = dataset_llm["api_key"]
        if dataset_llm["base_url"]:
            composer_config["dataset_base_url"] = dataset_llm["base_url"]

    if request.llm_provider:
        composer_config["llm_provider"] = request.llm_provider
    if request.llm_model:
        composer_config["llm_model"] = request.llm_model
    if request.reranking_model:
        composer_config["reranking_model"] = request.reranking_model

    reranker_llm_override = any(
        (
            request.reranker_llm_provider,
            request.reranker_llm_model,
            request.reranker_llm_api_key,
            request.reranker_llm_base_url,
        )
    )
    if reranker_llm_override:
        reranker_llm = {
            "provider": request.reranker_llm_provider,
            "model": request.reranker_llm_model,
            "api_key": request.reranker_llm_api_key,
            "base_url": request.reranker_llm_base_url,
        }
        _validate_task_llm("Dedicated LLM reranker", reranker_llm)
        composer_config["reranker_llm_provider"] = request.reranker_llm_provider
        composer_config["reranker_llm_model"] = request.reranker_llm_model
        if request.reranker_llm_api_key:
            composer_config["reranker_llm_api_key"] = request.reranker_llm_api_key
        if request.reranker_llm_base_url:
            composer_config["reranker_llm_base_url"] = request.reranker_llm_base_url

    if "llm" in selected_rerankers and request.pipeline_mode == "retrieval_only":
        has_reusable_llm = bool(
            request.query_transform_llm_provider
            and request.query_transform_llm_model
        )
        if not reranker_llm_override and not has_reusable_llm:
            raise ValueError(
                "LLM reranking in Retrieval-Only mode requires a dedicated "
                "reranker LLM or a configured Query-Transform LLM to reuse."
            )

    if "custom" in selected_rerankers:
        endpoint = str(request.reranker_base_url or "").strip()
        if not endpoint:
            raise ValueError(
                "Custom reranking requires reranker_base_url."
            )
        if not endpoint.lower().startswith(("http://", "https://")):
            raise ValueError(
                "Custom reranker URL must start with http:// or https://."
            )
        composer_config["reranker_base_url"] = endpoint
        composer_config["reranker_timeout_seconds"] = request.reranker_timeout_seconds
        composer_config["reranker_options"] = dict(request.reranker_options)
        if request.reranker_api_key:
            composer_config["reranker_api_key"] = request.reranker_api_key
        if request.reranker_model:
            composer_config["reranker_model"] = request.reranker_model
    query_transform_requested = any(
        (
            request.query_transform_llm_provider,
            request.query_transform_llm_model,
            request.query_transform_api_key,
            request.query_transform_base_url,
        )
    )
    if query_transform_requested:
        if not request.query_transform_llm_provider or not request.query_transform_llm_model:
            raise ValueError(
                "Query-transform LLM override requires query_transform_llm_provider "
                "and query_transform_llm_model."
            )
        composer_config["query_transform_llm_provider"] = request.query_transform_llm_provider
        composer_config["query_transform_llm_model"] = request.query_transform_llm_model
        if request.query_transform_api_key:
            composer_config["query_transform_api_key"] = request.query_transform_api_key
        if request.query_transform_base_url:
            composer_config["query_transform_base_url"] = request.query_transform_base_url

    query_expansion_values = config_space.search_space.get("query_expansion", [])
    query_transform_enabled = any(value != "none" for value in query_expansion_values)
    if request.pipeline_mode == "retrieval_only" and query_transform_enabled:
        _validate_task_llm(
            "Retrieval-Only query transformation",
            {
                "provider": request.query_transform_llm_provider,
                "model": request.query_transform_llm_model,
                "api_key": request.query_transform_api_key,
                "base_url": request.query_transform_base_url,
            },
        )

    if request.api_key:
        composer_config["api_key"] = request.api_key
    if request.base_url:
        composer_config["base_url"] = request.base_url
    if request.embedding_model:
        composer_config["embedding_model"] = request.embedding_model
    if request.vector_db_provider:
        composer_config["vector_db_provider"] = request.vector_db_provider
    if request.retrieval_source != "web_search_only" and request.document_parser:
        composer_config["document_parser"] = request.document_parser
        composer_config["document_parser_config"] = request.document_parser_config or {}
    if request.retrieval_source != "web_search_only" and request.use_ocr:
        composer_config["use_ocr"] = True
    if request.device:
        composer_config["device"] = request.device
    if request.judge_llm_provider:
        composer_config["judge_llm_provider"] = request.judge_llm_provider
        composer_config["judge_llm_model"] = request.judge_llm_model
        if request.judge_api_key:
            composer_config["judge_api_key"] = request.judge_api_key
        if request.judge_base_url:
            composer_config["judge_base_url"] = request.judge_base_url
    if request.retrieval_source == "web_search_only":
        composer_config["retrieval_source"] = "web_search_only"
        invalid_metrics = sorted(
            str(metric).lower()
            for metric in effective_metrics
            if str(metric).lower() in retrieval_metrics
        )
        if invalid_metrics:
            raise ValueError(
                "retrieval_source='web_search_only' cannot score retrieval metrics "
                f"{invalid_metrics} because it has no local retrieval corpus. "
                "Use faithfulness, answer_correctness, and/or llm_judge_rating."
            )
    if request.adaptive_web_search:
        composer_config["adaptive_web_search"] = True
    if request.search_provider:
        composer_config["search_provider"] = request.search_provider
        composer_config["search_provider_config"] = dict(request.search_provider_config or {})
        if request.search_api_key:
            composer_config["search_provider_config"]["api_key"] = request.search_api_key

    from Muffakir.dependency_validation import validate_composer_dependencies

    validate_composer_dependencies(composer_config, config_space.search_space)

    # Validates data_dir, provider, api_key requirements synchronously
    MuffakirComposer(config=composer_config)

    return config_space.total_combinations


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Backend health-check endpoint checking RUNS_ROOT writability and active runs."""
    writable = run_manager.storage_state()["writable"]
    return HealthResponse(
        status="ok",
        version=app.version,
        runs_root_writable=writable,
        active_runs=run_manager._active_run_count(),
    )


@app.get("/api/settings/storage", response_model=StorageSettingsResponse)
def get_storage_settings() -> StorageSettingsResponse:
    """Return the active ComposerUI runs workspace and recent locations."""
    return StorageSettingsResponse(**run_manager.storage_state())


@app.put("/api/settings/storage", response_model=StorageSettingsResponse)
def update_storage_settings(
    request: StorageSettingsUpdate,
) -> StorageSettingsResponse:
    """Validate, persist, and activate a new runs workspace."""
    try:
        state = run_manager.set_runs_root(request.runs_root)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except storage.StorageConfigurationError as exc:
        status_code = 409 if run_manager.RUNS_ROOT_LOCKED else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return StorageSettingsResponse(**state)


@app.get("/api/catalog")
def get_catalog() -> Dict[str, Any]:
    """Return static catalog data (providers, presets, default search space)."""
    return provider_catalog.get_catalog()


@app.post("/api/prompts/resolve", response_model=PromptResolveResponse)
def resolve_prompts(request: PromptResolveRequest) -> PromptResolveResponse:
    """Return only prompts that any selected run combination can execute."""
    from ComposerUI.backend.prompt_service import resolve_definitions

    try:
        prompts = resolve_definitions(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PromptResolveResponse(
        language=request.language,
        prompts=prompts,
        skipped=not prompts,
    )



@app.post("/api/runs/validate", response_model=ValidateRunResponse)
def validate_run(request: CreateRunRequest) -> ValidateRunResponse:
    """Dry-run validation endpoint. Reports validity and combination count without creating a run."""
    try:
        total_combos = _preflight_check(request)
        return ValidateRunResponse(valid=True, total_combinations=total_combos, error=None)
    except (ValueError, FileNotFoundError) as e:
        return ValidateRunResponse(valid=False, total_combinations=None, error=str(e))
    except Exception as e:
        return ValidateRunResponse(valid=False, total_combinations=None, error=str(e))


@app.post("/api/runs", response_model=CreateRunResponse)
def create_run(request: CreateRunRequest) -> CreateRunResponse:
    """Launch a new architecture search run in a background thread."""
    try:
        total_combos = _preflight_check(request)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    run_info = run_manager.launch_run(request)
    return CreateRunResponse(
        run_id=run_info["run_id"],
        run_name=run_info["run_name"],
        total_combinations=total_combos,
        status="running",
    )


@app.post("/api/synthetic-data/generate", response_model=GenerateDatasetResponse)
def generate_synthetic_data(request: GenerateDatasetRequest) -> GenerateDatasetResponse:
    """Generate a synthetic Q&A dataset now (blocking) and return a preview.

    Not run in a background thread like `/api/runs` — generation over a
    default-sized chunk sample is expected to be short.
    """
    try:
        result = dataset_utils.generate_synthetic_dataset(request)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return GenerateDatasetResponse(**result)


@app.get("/api/runs", response_model=RunListResponse)
def list_runs() -> RunListResponse:
    """List all runs discovered on disk, sorted newest first."""
    runs_data = run_manager.discover_runs()
    return RunListResponse(runs=[RunSummary(**r) for r in runs_data])


@app.get("/api/runs/{run_id}", response_model=RunDetailResponse)
def get_run_detail(run_id: str) -> RunDetailResponse:
    """Fetch live details for a specific run."""
    paths = run_manager.get_run_paths(run_id)
    if paths is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    config: Dict[str, Any] = {}
    if paths["config_file"].exists():
        try:
            with open(paths["config_file"], "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            pass

    manifest = trace_reader.read_manifest(paths["trace_dir"])
    error_msg = trace_reader.read_error_message(paths["root"])
    records, total_lines = trace_reader.read_trials_after(paths["trace_dir"], after=0)
    failure_clusters = trace_reader.cluster_failures_from_records(records)
    stage_timings: Dict[str, Optional[float]] = {}

    # Status resolution
    if manifest and manifest.get("status"):
        status = manifest["status"]
    elif error_msg:
        status = "failed"
    else:
        status = "running"

    total_trials = (
        manifest.get("total_trials", 0) if manifest else (config.get("max_trials") or 0)
    )

    # Compute best score & best config
    best_score: Optional[float] = None
    best_config: Optional[Dict[str, Any]] = None
    best_result: Optional[Dict[str, Any]] = None
    report_best_trial_id: Optional[int] = None
    report_best_result: Optional[Dict[str, Any]] = None
    report_data: Optional[Dict[str, Any]] = None

    report_json_path = paths["report_json"]
    if report_json_path.exists():
        try:
            with open(report_json_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
                best_score = report_data.get("best_score")
                best_config = report_data.get("best_config")
                best_trial = report_data.get("best_trial") or {}
                report_best_trial_id = best_trial.get("trial_id")
                report_best_result = report_data.get("best_result")
        except Exception:
            pass

    # Historical safeguard: old Composer versions could persist a completed
    # manifest solely because report.json existed. An explicit zero-success
    # report is authoritative and must be shown as failed.
    if run_manager.report_successful_trial_count(report_data) == 0:
        status = "failed"
    if status == "completed" and trace_reader.has_zero_evaluation_samples(records):
        status = "failed"
        error_msg = error_msg or (
            "This run evaluated zero samples. Its trials and zero scores are "
            "not valid results; create a new run with a working dataset LLM "
            "or an existing evaluation dataset."
        )

    successful = [r for r in records if r.get("status") == "success"]
    best_rec: Optional[Dict[str, Any]] = None
    if report_best_trial_id is not None:
        best_rec = next(
            (r for r in successful if r.get("trial_id") == report_best_trial_id),
            None,
        )
    if best_rec is None and successful:
        best_rec = max(
            successful,
            key=lambda r: (r.get("composite_score", 0.0), -r.get("trial_id", 0)),
        )

    if best_rec is not None:
        if best_score is None:
            best_score = best_rec.get("composite_score")
        if best_config is None:
            best_config = best_rec.get("resolved_rag_config")
        stage_timings = trace_reader.stage_timings_from_record(best_rec)
        best_result = {
            "trial_id": best_rec.get("trial_id", 0),
            "config": best_rec.get("resolved_rag_config", {}),
            "composite_score": best_rec.get("composite_score", 0.0),
            "metrics": best_rec.get("metrics", {}),
            "mean_pipeline_latency_ms": best_rec.get("mean_pipeline_latency_ms"),
            "mean_evaluation_overhead_ms": best_rec.get("mean_evaluation_overhead_ms"),
            "stage_timings": stage_timings,
        }
    elif report_best_result:
        best_result = report_best_result
        stage_timings = report_best_result.get("stage_timings", {})

    recorded_costs = [r.get("cost_usd") for r in records if r.get("cost_usd") is not None]
    total_cost = sum(recorded_costs) if recorded_costs else None
    total_ws_fallbacks = trace_reader.total_web_search_fallbacks(records)
    refusal_summary = trace_reader.read_error_rates(paths["trace_dir"]) or {}

    created_at = (
        config.get("created_at")
        or (manifest.get("created_at") if manifest else None)
        or ""
    )

    return RunDetailResponse(
        run_id=run_id,
        run_name=config.get("run_name", run_id[:8]),
        status=status,
        created_at=created_at,
        total_trials=total_trials,
        completed_trials=total_lines,
        best_score=round(best_score, 4) if best_score is not None else None,
        best_config=best_config,
        best_result=best_result,
        failure_clusters=failure_clusters,
        error_message=error_msg,
        stage_timings=stage_timings,
        total_cost_usd=round(total_cost, 6) if total_cost is not None else None,
        total_web_search_fallbacks=total_ws_fallbacks if total_ws_fallbacks > 0 else None,
        total_answer_refusals=refusal_summary.get("answer_refusals"),
        answer_refusal_rate=refusal_summary.get("answer_refusal_rate"),
        duration_seconds=None,
    )


@app.get("/api/runs/{run_id}/trials", response_model=TrialsPageResponse)
def get_run_trials(run_id: str, after: int = Query(0, ge=0)) -> TrialsPageResponse:
    """Fetch incremental trial rows for live display."""
    paths = run_manager.get_run_paths(run_id)
    if paths is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    records, next_after = trace_reader.read_trials_after(paths["trace_dir"], after=after)
    trial_rows = [
        TrialRow(
            trial_id=r.get("trial_id", 0),
            resolved_rag_config=r.get("resolved_rag_config", {}),
            composite_score=round(r.get("composite_score", 0.0), 4),
            latency_ms=round(r.get("latency_ms", 0.0), 2),
            metrics=r.get("metrics", {}),
            mean_pipeline_latency_ms=r.get("mean_pipeline_latency_ms"),
            mean_evaluation_overhead_ms=r.get("mean_evaluation_overhead_ms"),
            status=r.get("status", "success"),
            error_code=r.get("error_code"),
            error_type=r.get("error_type"),
            cost_usd=round(r.get("cost_usd"), 6) if r.get("cost_usd") is not None else None,
            mean_query_transform_ms=r.get("mean_query_transform_ms"),
            mean_query_embedding_ms=r.get("mean_query_embedding_ms"),
            mean_vector_search_ms=r.get("mean_vector_search_ms"),
            mean_rerank_ms=r.get("mean_rerank_ms"),
            mean_relevance_check_ms=r.get("mean_relevance_check_ms"),
            mean_web_search_ms=r.get("mean_web_search_ms"),
            mean_generation_ms=r.get("mean_generation_ms"),
            mean_hallucination_check_ms=r.get("mean_hallucination_check_ms"),
            web_search_fallback_count=r.get("web_search_fallback_count"),
            answer_refusal_count=r.get("answer_refusal_count", 0),
            answer_refusal_rate=r.get("answer_refusal_rate"),
            started_at=r.get("started_at"),
            completed_samples=r.get("completed_samples", 0),
            total_samples=r.get("total_samples", 0),
        )
        for r in records
    ]

    manifest = trace_reader.read_manifest(paths["trace_dir"])
    terminal_statuses = {"completed", "failed", "stopped_early"}
    active_records = (
        []
        if manifest and manifest.get("status") in terminal_statuses
        else trace_reader.read_active_trials(paths["trace_dir"])
    )
    active_rows = [
        ActiveTrialRow(
            trial_id=r.get("trial_id", 0),
            status="running",
            resolved_rag_config=r.get("resolved_rag_config", {}),
            started_at=r.get("started_at", ""),
            completed_samples=r.get("completed_samples", 0),
            total_samples=r.get("total_samples", 0),
        )
        for r in active_records
    ]
    return TrialsPageResponse(
        trials=trial_rows,
        active_trials=active_rows,
        next_after=next_after,
    )


@app.get("/api/runs/{run_id}/errors", response_model=ErrorObservabilityResponse)
def get_run_errors(run_id: str, after: int = Query(0, ge=0)) -> ErrorObservabilityResponse:
    """Fetch the full live error-rate summary plus incremental error events."""
    paths = run_manager.get_run_paths(run_id)
    if paths is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    summary_data = trace_reader.read_error_rates(paths["trace_dir"])
    if summary_data is None:
        return ErrorObservabilityResponse(available=False, next_after=after)

    events, next_after = trace_reader.read_errors_after(paths["trace_dir"], after=after)
    return ErrorObservabilityResponse(
        available=True,
        summary=ErrorRateSummary(**summary_data),
        errors=[ErrorEventRow(**event) for event in events],
        next_after=next_after,
    )


@app.get("/api/runs/{run_id}/metadata", response_model=RunMetadataResponse)
def get_run_metadata(run_id: str) -> RunMetadataResponse:
    """Return the persisted redacted configuration and trace manifest."""
    paths = run_manager.get_run_paths(run_id)
    if paths is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    config: Dict[str, Any] = {}
    if paths["config_file"].exists():
        try:
            with open(paths["config_file"], "r", encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            config = {}

    manifest = trace_reader.read_manifest(paths["trace_dir"]) or {}
    from ComposerUI.backend.pricing_service import (
        build_pricing_summary,
        read_pricing_snapshot,
    )

    pricing = build_pricing_summary(
        config,
        read_pricing_snapshot(paths["checkpoint_dir"]),
    )
    # Re-redact on read so runs created by older versions cannot expose a
    # credential that was nested inside a provider/parser config object.
    return RunMetadataResponse(
        config=run_manager.redact_nested_secrets(config),
        manifest=run_manager.redact_nested_secrets(manifest),
        pricing=pricing,
    )


@app.get("/api/runs/{run_id}/trials/{trial_id}", response_model=TrialDetailResponse)
def get_trial_detail(run_id: str, trial_id: int) -> TrialDetailResponse:
    """Return a complete trial record with its question-level sample traces."""
    paths = run_manager.get_run_paths(run_id)
    if paths is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    trial = trace_reader.find_trial_record(paths["trace_dir"], trial_id)
    if trial is None:
        raise HTTPException(status_code=404, detail=f"Trial '{trial_id}' not found")

    samples = trace_reader.read_samples_for_trial(paths["trace_dir"], trial_id)
    return TrialDetailResponse(trial=trial, samples=samples, sample_count=len(samples))


def _saved_export_trial(paths: Dict[str, Path], trial_id: int) -> Optional[Dict[str, Any]]:
    """Use exact trial identity, including checkpoint-only failed trials."""
    trial = trace_reader.find_trial_record(paths["trace_dir"], trial_id)
    if trial and (trial.get("resolved_rag_config") or trial.get("resolved_config")):
        return trial
    for path in (paths["checkpoint_dir"] / "composer_checkpoint.json", paths["report_json"]):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            candidates = payload.get("trials", [])
            for candidate in candidates:
                if candidate.get("trial_id") == trial_id:
                    if candidate.get("resolved_config") or candidate.get("resolved_rag_config"):
                        return candidate
                    trial = trial or candidate
        except (ValueError, OSError, AttributeError, TypeError):
            continue
    return trial


@app.get("/api/runs/{run_id}/trials/{trial_id}/export")
def export_trial_python(run_id: str, trial_id: int,
                        format: str = Query("preview", pattern="^(preview|python|zip)$")):
    """Download one saved pipeline without executing it or recovering credentials."""
    from Composer.python_export import ExportError, export_trial

    paths = run_manager.get_run_paths(run_id)
    if paths is None:
        raise HTTPException(status_code=404, detail="Run not found")
    trial = _saved_export_trial(paths, trial_id)
    if trial is None:
        raise HTTPException(status_code=404, detail="Trial not found")
    try:
        config = json.loads(paths["config_file"].read_text(encoding="utf-8")) if paths["config_file"].is_file() else {}
        exported = export_trial(run_id, trial, config, trace_reader.read_manifest(paths["trace_dir"]) or {})
    except (ExportError, ValueError, TypeError, OSError) as exc:
        detail = str(exc) if isinstance(exc, ExportError) else "Saved configuration cannot be read for export."
        raise HTTPException(status_code=422, detail=detail) from exc
    if format == "preview":
        return exported.preview()
    filename = "rag_app.py" if format == "python" else f"muffakir-trial-{trial_id}.zip"
    return Response(
        content=exported.code.encode("utf-8") if format == "python" else exported.zip_bytes(),
        media_type="text/x-python" if format == "python" else "application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"', "Cache-Control": "no-store"},
    )


# Mount static frontend files
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
