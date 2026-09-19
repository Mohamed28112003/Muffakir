"""
Run lifecycle and execution manager for ComposerUI.

Handles directory creation, background execution thread, crash-safety wrapper,
and run discovery directly from disk.
"""

import json
import logging
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ComposerUI.backend import storage, trace_reader
from ComposerUI.backend.schemas import CreateRunRequest

logger = logging.getLogger("ComposerUI.run_manager")


def _resolve_runs_root() -> Path:
    """Backward-compatible path-only wrapper around the storage resolver."""
    return storage.resolve_runs_root()[0]


RUNS_ROOT, RUNS_ROOT_SOURCE, RUNS_ROOT_LOCKED = storage.resolve_runs_root()
UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$"
)

# In-memory registry of runs launched in the current process
_ACTIVE_RUNS: Dict[str, Dict[str, Any]] = {}
_STORAGE_LOCK = threading.RLock()

_SECRET_CONFIG_KEYS = {
    "api_key",
    "search_api_key",
    "judge_api_key",
    "query_transform_api_key",
    "dataset_api_key",
    "reranker_api_key",
    "reranker_llm_api_key",
    "authorization",
    "password",
    "secret",
    "token",
}


def _active_run_count() -> int:
    """Return only currently alive workers, ignoring completed registry rows."""
    return sum(
        1
        for info in _ACTIVE_RUNS.values()
        if info.get("thread") is not None and info["thread"].is_alive()
    )


def report_successful_trial_count(report: Any) -> Optional[int]:
    """Return the report's explicit successful-trial count when available."""
    if isinstance(report, dict):
        value = report.get("successful_trials")
        if value is None:
            summary = report.get("summary")
            if isinstance(summary, dict):
                value = summary.get("successful_trials")
    else:
        value = getattr(report, "successful_trials", None)

    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def report_has_valid_result(report: Any) -> bool:
    """A finished run is successful only when it has a winning trial/result."""
    successful_trials = report_successful_trial_count(report)
    if isinstance(report, dict):
        best_result = report.get("best_result") or report.get("best_trial")
    else:
        best_result = getattr(report, "best_result", None) or getattr(
            report, "best_trial", None
        )
    return bool(successful_trials and successful_trials > 0 and best_result is not None)


def _persist_manifest_status(
    paths: Dict[str, Path],
    run_id: str,
    search_space: Optional[Dict[str, Any]],
    status: str,
) -> None:
    """Atomically persist a terminal UI run status without deleting artifacts."""
    from Trace.models import RunManifest
    from Trace.writer import TraceWriter

    existing_manifest = trace_reader.read_manifest(paths["trace_dir"])
    if existing_manifest:
        existing_manifest["status"] = status
        manifest_data = existing_manifest
    else:
        manifest_data = RunManifest(
            run_id=run_id,
            status=status,
            search_space=search_space or {},
        ).to_dict()

    writer = TraceWriter(str(paths["trace_dir"]))
    writer.write_manifest(manifest_data)
    writer.close()


def storage_state() -> Dict[str, Any]:
    """Return serializable information about the active runs workspace."""
    root = RUNS_ROOT
    source = RUNS_ROOT_SOURCE
    locked = RUNS_ROOT_LOCKED
    return {
        "runs_root": str(root),
        "source": source,
        "locked": locked,
        "writable": _runs_root_is_writable(root),
        "recent_roots": [
            str(path)
            for path in storage.recent_runs_roots()
            if path.resolve() != root.resolve()
        ],
    }


def _runs_root_is_writable(root: Path) -> bool:
    try:
        storage.validate_runs_root(root)
    except storage.StorageConfigurationError:
        return False
    return True


def set_runs_root(value: str) -> Dict[str, Any]:
    """Validate, persist, and activate a different ComposerUI workspace."""
    global RUNS_ROOT, RUNS_ROOT_SOURCE

    with _STORAGE_LOCK:
        if RUNS_ROOT_LOCKED:
            raise storage.StorageConfigurationError(
                "The runs workspace is locked by MUFFAKIR_RUNS_ROOT or --runs-dir. "
                "Restart the server without that override to change it in the UI."
            )
        if _active_run_count():
            raise RuntimeError(
                "The runs workspace cannot be changed while a run is active."
            )

        new_root = storage.validate_runs_root(value)
        previous_root = RUNS_ROOT
        storage.save_runs_root(new_root, previous_root)
        RUNS_ROOT = new_root
        RUNS_ROOT_SOURCE = "saved"
        return storage_state()


def redact_nested_secrets(value: Any) -> Any:
    """Recursively remove credentials before config persistence or display."""
    if isinstance(value, dict):
        return {
            key: redact_nested_secrets(item)
            for key, item in value.items()
            if str(key).lower() not in _SECRET_CONFIG_KEYS
        }
    if isinstance(value, list):
        return [redact_nested_secrets(item) for item in value]
    return value


def get_run_paths(run_id: str) -> Optional[Dict[str, Path]]:
    """Validate run_id and return standard paths if the run directory exists."""
    if not UUID_PATTERN.match(run_id):
        return None

    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        return None

    return {
        "root": run_dir,
        "trace_dir": run_dir / "trace",
        "checkpoint_dir": run_dir / "checkpoint",
        "config_file": run_dir / "config.json",
        "report_json": run_dir / "report.json",
        "error_file": run_dir / "error.txt",
    }


def create_run_dirs(run_id: str) -> Dict[str, Path]:
    """Create directory structure for a new run."""
    run_dir = RUNS_ROOT / run_id
    trace_dir = run_dir / "trace"
    checkpoint_dir = run_dir / "checkpoint"

    trace_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root": run_dir,
        "trace_dir": trace_dir,
        "checkpoint_dir": checkpoint_dir,
        "config_file": run_dir / "config.json",
        "report_json": run_dir / "report.json",
        "error_file": run_dir / "error.txt",
    }


def write_redacted_config(
    run_dir: Path, request: CreateRunRequest, run_id: str, created_at: str
) -> None:
    """Save run configuration with api_key strictly redacted."""
    from ComposerUI.backend.prompt_service import build_prompt_snapshot
    from ComposerUI.backend.pricing_service import resolve_pricing_models

    prompt_snapshot = build_prompt_snapshot(request)
    redacted_data = {
        **request.parameter_config(),
        "run_id": run_id,
        "run_name": request.run_name,
        "created_at": created_at,
        "language": prompt_snapshot["language"],
        "prompt_overrides": prompt_snapshot["overrides"],
        "resolved_prompts": prompt_snapshot["resolved_prompts"],
        "prompt_hashes": prompt_snapshot["prompt_hashes"],
        "custom_pricing": [entry.model_dump() for entry in request.custom_pricing],
        "pricing_models": resolve_pricing_models(request),
        "documents_path": request.documents_path,
        "llm_provider": request.llm_provider,
        "llm_model": request.llm_model,
        "dataset_llm_provider": request.dataset_llm_provider,
        "dataset_llm_model": request.dataset_llm_model,
        "dataset_base_url": request.dataset_base_url,
        "pipeline_mode": request.pipeline_mode,
        "query_transform_llm_provider": request.query_transform_llm_provider,
        "query_transform_llm_model": request.query_transform_llm_model,
        "query_transform_base_url": request.query_transform_base_url,
        "reranking_model": request.reranking_model,
        "reranker_llm_provider": request.reranker_llm_provider,
        "reranker_llm_model": request.reranker_llm_model,
        "reranker_llm_base_url": request.reranker_llm_base_url,
        "reranker_base_url": request.reranker_base_url,
        "reranker_model": request.reranker_model,
        "reranker_timeout_seconds": request.reranker_timeout_seconds,
        "reranker_options": request.reranker_options,
        "base_url": request.base_url,
        "embedding_model": request.embedding_model,
        "vector_db_provider": request.vector_db_provider,
        "search_space": request.search_space.to_composer_dict(),
        "metrics": request.metrics,
        "max_eval_samples": request.max_eval_samples,
        "n_jobs": request.n_jobs,
        "max_trials": request.max_trials,
        "max_runtime_minutes": request.max_runtime_minutes,
        "eval_dataset_mode": request.eval_dataset_mode,
        "eval_dataset_path": request.eval_dataset_path,
        "document_parser": request.document_parser,
        "document_parser_config": request.document_parser_config,
        "use_ocr": request.use_ocr,
        "train_test_split": (
            request.train_test_split.model_dump() if request.train_test_split else None
        ),
        "device": request.device,
        "judge_llm_provider": request.judge_llm_provider,
        "judge_llm_model": request.judge_llm_model,
        "judge_base_url": request.judge_base_url,
        "retrieval_source": request.retrieval_source,
        "adaptive_web_search": request.adaptive_web_search,
        "search_provider": request.search_provider,
        "search_provider_config": request.search_provider_config,
    }
    config_path = run_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(redact_nested_secrets(redacted_data), f, indent=2, ensure_ascii=False)


def _run_fit_in_background(
    request: CreateRunRequest, run_id: str, paths: Dict[str, Path]
) -> None:
    """Target function for background execution thread with crash safety."""
    search_space_dict = request.search_space.to_composer_dict() or None

    import multiprocessing
    from Trace.observability import (
        observation_context,
        observe_stage,
        record_exception_outcome,
        sanitize_error_message,
        collect_secret_values,
    )
    from Trace.writer import TraceWriter

    trace_manager = multiprocessing.Manager()
    trace_queue = trace_manager.Queue()
    trace_writer = TraceWriter(str(paths["trace_dir"]), work_queue=trace_queue)
    request_values = request.model_dump()
    secrets = collect_secret_values(request_values)
    force_failed_manifest = False

    try:
        from Muffakir import MuffakirComposer
        from Evaluation.dataset import load_evaluation_dataset
        from ComposerUI.backend.dataset_utils import split_train_test

        with observation_context(trace_queue, secrets=secrets):
            # Existing QA data is evaluation ground truth only. The retrieval
            # corpus remains the separate documents_path configured below.
            eval_dataset = None
            if request.eval_dataset_mode != "auto":
                with observe_stage("dataset_load", "dataset"):
                    all_eval_pairs = load_evaluation_dataset(request.eval_dataset_path)
                    if request.train_test_split and request.train_test_split.enabled:
                        _train_pairs, test_pairs = split_train_test(
                            all_eval_pairs,
                            test_size=request.train_test_split.test_size,
                            random_state=request.train_test_split.random_state,
                        )
                        eval_dataset = test_pairs
                    else:
                        eval_dataset = all_eval_pairs

        composer_config: Dict[str, Any] = {
            **request.parameter_config(),
            "data_dir": request.documents_path,
            "language": request.language,
            "prompt_overrides": dict(request.prompt_overrides),
        }
        if request.pipeline_mode == "retrieval_only":
            composer_config["pipeline_mode"] = "retrieval_only"
        if request.llm_provider:
            composer_config["llm_provider"] = request.llm_provider
        if request.llm_model:
            composer_config["llm_model"] = request.llm_model
        if request.reranking_model:
            composer_config["reranking_model"] = request.reranking_model
        if request.reranker_llm_provider:
            composer_config["reranker_llm_provider"] = request.reranker_llm_provider
            composer_config["reranker_llm_model"] = request.reranker_llm_model
            if request.reranker_llm_api_key:
                composer_config["reranker_llm_api_key"] = request.reranker_llm_api_key
            if request.reranker_llm_base_url:
                composer_config["reranker_llm_base_url"] = request.reranker_llm_base_url
        if request.reranker_base_url:
            composer_config["reranker_base_url"] = request.reranker_base_url
            composer_config["reranker_timeout_seconds"] = request.reranker_timeout_seconds
            composer_config["reranker_options"] = dict(request.reranker_options)
            if request.reranker_api_key:
                composer_config["reranker_api_key"] = request.reranker_api_key
            if request.reranker_model:
                composer_config["reranker_model"] = request.reranker_model
        if request.query_transform_llm_provider:
            composer_config["query_transform_llm_provider"] = request.query_transform_llm_provider
            composer_config["query_transform_llm_model"] = request.query_transform_llm_model
            # Held only in this local stack frame, like api_key/search_api_key --
            # never written to the persisted redacted config.
            if request.query_transform_api_key:
                composer_config["query_transform_api_key"] = request.query_transform_api_key
            if request.query_transform_base_url:
                composer_config["query_transform_base_url"] = request.query_transform_base_url
        # api_key is held only in this local stack frame
        if request.api_key:
            composer_config["api_key"] = request.api_key
        if request.base_url:
            composer_config["base_url"] = request.base_url
        if request.eval_dataset_mode == "auto":
            composer_config["dataset_llm_provider"] = request.dataset_llm_provider or request.llm_provider
            composer_config["dataset_llm_model"] = request.dataset_llm_model or request.llm_model
            if request.dataset_api_key or request.api_key:
                composer_config["dataset_api_key"] = request.dataset_api_key or request.api_key
            if request.dataset_base_url or request.base_url:
                composer_config["dataset_base_url"] = request.dataset_base_url or request.base_url
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
        if request.adaptive_web_search:
            composer_config["adaptive_web_search"] = True
        if request.search_provider:
            composer_config["search_provider"] = request.search_provider
            composer_config["search_provider_config"] = dict(request.search_provider_config or {})
            if request.search_api_key:
                composer_config["search_provider_config"]["api_key"] = request.search_api_key

        with observation_context(trace_queue, secrets=secrets):
            with observe_stage("run_setup", "runtime"):
                composer = MuffakirComposer(config=composer_config)
            report = composer.fit(
                search_space=search_space_dict,
                eval_dataset=eval_dataset,
                metrics=request.metrics,
                n_jobs=request.n_jobs,
                max_eval_samples=request.max_eval_samples,
                max_trials=request.max_trials,
                max_runtime_minutes=request.max_runtime_minutes,
                custom_pricing=request.custom_pricing_per_token(),
                save_report=True,
                report_path=str(paths["report_json"]),
                checkpoint_dir=str(paths["checkpoint_dir"]),
                resume=True,
                enable_trace=True,
                trace_dir=str(paths["trace_dir"]),
                trace_queue=trace_queue,
            )
        # Also serialize report JSON if possible
        if hasattr(report, "to_json"):
            try:
                report.to_json(str(paths["report_json"]))
            except Exception:
                pass

        if not report_has_valid_result(report):
            force_failed_manifest = True
            logger.error(
                "run failed: no successful trials or valid result",
                extra={"run_id": run_id},
            )
            return

        logger.info("run completed", extra={"run_id": run_id})
    except Exception as e:
        force_failed_manifest = True
        with observation_context(trace_queue, secrets=secrets):
            record_exception_outcome(
                e,
                recovery="unrecovered",
                fallback_stage="run_execution",
                fallback_component="runtime",
            )
        logger.error(f"Run {run_id} failed during execution: {e}", exc_info=True, extra={"run_id": run_id})
        # Write error sidecar
        try:
            paths["error_file"].write_text(
                sanitize_error_message(e, secrets), encoding="utf-8"
            )
        except Exception as write_err:
            logger.error(f"Failed to write error.txt for {run_id}: {write_err}", extra={"run_id": run_id})

        logger.info("run failed", extra={"run_id": run_id})
    finally:
        trace_writer.close()
        trace_manager.shutdown()
        # Write terminal status only after the shared writer is fully drained;
        # two simultaneous writers must never race on the same atomic files.
        if force_failed_manifest:
            try:
                _persist_manifest_status(paths, run_id, search_space_dict, "failed")
            except Exception as manifest_err:
                logger.error(
                    f"Failed to force-write failed manifest for {run_id}: {manifest_err}",
                    extra={"run_id": run_id},
                )


def _launch_run_locked(request: CreateRunRequest) -> Dict[str, Any]:
    """Create run artifacts on disk and launch background execution thread."""
    run_id = str(uuid.uuid4())
    paths = create_run_dirs(run_id)
    created_at = datetime.now().isoformat()

    write_redacted_config(paths["root"], request, run_id, created_at)

    thread = threading.Thread(
        target=_run_fit_in_background,
        args=(request, run_id, paths),
        daemon=True,
        name=f"ComposerUI-Run-{run_id[:8]}",
    )
    thread.start()
    logger.info("run started", extra={"run_id": run_id})

    run_info = {
        "run_id": run_id,
        "run_name": request.run_name,
        "thread": thread,
        "created_at": created_at,
    }
    _ACTIVE_RUNS[run_id] = run_info
    return run_info


def launch_run(request: CreateRunRequest) -> Dict[str, Any]:
    """Launch a run without racing a concurrent workspace switch."""
    with _STORAGE_LOCK:
        return _launch_run_locked(request)


def discover_runs() -> List[Dict[str, Any]]:
    """Scan RUNS_ROOT for existing runs directly on disk."""
    runs_root = RUNS_ROOT
    if not runs_root.exists():
        return []

    runs: List[Dict[str, Any]] = []

    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        if not UUID_PATTERN.match(run_id):
            continue

        config_path = run_dir / "config.json"
        config: Dict[str, Any] = {}
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception:
                pass

        trace_dir = run_dir / "trace"
        manifest = trace_reader.read_manifest(trace_dir)
        trial_count = trace_reader.count_trial_lines(trace_dir)
        error_msg = trace_reader.read_error_message(run_dir)

        # Determine status
        if manifest and manifest.get("status"):
            status = manifest["status"]
        elif error_msg:
            status = "failed"
        else:
            # Check if active thread is alive
            active_info = _ACTIVE_RUNS.get(run_id)
            if active_info and active_info["thread"].is_alive():
                status = "running"
            else:
                # If thread not alive and not completed/failed, consider stopped or running
                status = "running" if trial_count == 0 else "completed"

        # Best score and config
        best_score = None
        best_config = None

        # Check report.json first. Its explicit success count also protects
        # discovery of older runs whose manifest was incorrectly finalized as
        # completed even though every trial failed.
        report_json_path = run_dir / "report.json"
        report_data: Optional[Dict[str, Any]] = None
        if report_json_path.exists():
            try:
                with open(report_json_path, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
                    best_score = report_data.get("best_score")
                    best_config = report_data.get("best_config")
            except Exception:
                pass

        if report_successful_trial_count(report_data) == 0:
            status = "failed"

        # Older Composer builds could call an empty search successful. Their
        # trace rows explicitly record zero evaluation samples even though
        # report.json may claim every trial succeeded.
        if status == "completed" and trial_count > 0:
            trial_records, _ = trace_reader.read_trials_after(trace_dir, after=0)
            if trace_reader.has_zero_evaluation_samples(trial_records):
                status = "failed"

        # If not found in report, inspect trials.jsonl
        if best_score is None and trial_count > 0:
            records, _ = trace_reader.read_trials_after(trace_dir, after=0)
            successful = [r for r in records if r.get("status") == "success"]
            if successful:
                best_rec = max(successful, key=lambda r: r.get("composite_score", 0.0))
                best_score = best_rec.get("composite_score")
                best_config = best_rec.get("resolved_rag_config")

        created_at = (
            config.get("created_at")
            or (manifest.get("created_at") if manifest else None)
            or datetime.fromtimestamp(run_dir.stat().st_ctime).isoformat()
        )

        total_trials = (
            manifest.get("total_trials", 0) if manifest else 0
        )

        runs.append({
            "run_id": run_id,
            "run_name": config.get("run_name", run_id[:8]),
            "status": status,
            "created_at": created_at,
            "best_score": round(best_score, 4) if best_score is not None else None,
            "best_config": best_config,
            "total_trials": total_trials,
            "completed_trial_count": trial_count,
            "duration_seconds": None,
        })

    runs.sort(key=lambda r: r["created_at"], reverse=True)
    return runs
