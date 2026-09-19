"""
TraceWriter - async queue-drained writer for trial/sample JSONL, manifest.json,
and the atomic active_trials.json live snapshot.

One background thread owns all file I/O for a run's trace directory, so the
caller (Composer.fit(), or a ProcessPoolExecutor worker via a
multiprocessing.Manager().Queue()) never blocks on disk. JSONL files are
append-only (one write + flush per record); manifest.json is atomically
overwritten (temp file + os.replace), matching the pattern already used by
Evaluation.models.EvaluationReport.save() and Composer.checkpoint's
_write_atomic.
"""

import json
import logging
import os
import queue
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# A plain tuple, not a sentinel object() -- a multiprocessing.Manager().Queue()
# always proxies puts/gets through the manager's server process (even when
# both ends are the "same" Python process), which pickles every item. An
# `object()` sentinel loses its identity across that round-trip, so `is`
# comparison would never match; a value comparison on a plain string does.
_STOP = ("__stop__", None)


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".trace_manifest_", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        for attempt in range(6):
            try:
                os.replace(tmp_path, path)
                break
            except PermissionError:
                # Windows can briefly deny replacement while the polling API
                # has the previous snapshot open. Keep the write atomic and
                # retry instead of dropping a live progress update.
                if attempt == 5:
                    raise
                time.sleep(0.01 * (2 ** attempt))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class TraceWriter:
    """
    Drains a queue (plain queue.Queue, or a multiprocessing.Manager().Queue()
    shared with ProcessPoolExecutor workers) on a single background thread.

    Usage:
        writer = TraceWriter(trace_dir, work_queue=optional_manager_queue)
        writer.write_manifest(manifest.to_dict())
        writer.write_trial_started(active_trial_dict)
        writer.write_trial(trial_record.to_dict())      # can also be called
        writer.write_sample(sample_record.to_dict())     # from a worker process
        ...
        writer.close()
    """

    def __init__(self, trace_dir: str, work_queue: Optional[Any] = None):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trials_path = self.trace_dir / "trials.jsonl"
        self.samples_path = self.trace_dir / "samples.jsonl"
        self.manifest_path = self.trace_dir / "manifest.json"
        self.active_trials_path = self.trace_dir / "active_trials.json"
        self.error_rates_path = self.trace_dir / "error_rates.json"
        self.errors_path = self.trace_dir / "errors.jsonl"

        # A writer belongs to one live fit() invocation. Clear a snapshot left
        # behind by an interrupted process before accepting new lifecycle events.
        _atomic_write_json(self.active_trials_path, {"trials": []})

        # Error-rate aggregates survive terminal state and resume.  An empty
        # snapshot is created immediately so readers can distinguish a traced
        # zero-error run from a legacy run that never recorded observability.
        if not self.error_rates_path.exists():
            _atomic_write_json(self.error_rates_path, self._empty_error_rates())

        self._queue = work_queue if work_queue is not None else queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def write_trial(self, record: Dict[str, Any]) -> None:
        self._queue.put(("trial", record))

    def write_trial_started(self, record: Dict[str, Any]) -> None:
        self._queue.put(("trial_started", record))

    def write_sample(self, record: Dict[str, Any]) -> None:
        self._queue.put(("sample", record))

    def write_manifest(self, manifest: Dict[str, Any]) -> None:
        self._queue.put(("manifest", manifest))

    def write_operation(self, record: Dict[str, Any]) -> None:
        self._queue.put(("operation", record))

    @staticmethod
    def _empty_error_rates() -> Dict[str, Any]:
        return {
            "schema_version": "1.1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "attempts": 0,
            "errors": 0,
            "recovered_errors": 0,
            "unrecovered_errors": 0,
            "error_rate": 0.0,
            "health": "HEALTHY",
            # Explicit answer refusals are not provider errors. They live in
            # the same observability snapshot so the UI can update live, while
            # retaining their own count and denominator.
            "generation_samples": 0,
            "answer_refusals": 0,
            "answer_refusal_rate": 0.0,
            "stages": [],
        }

    def _load_error_rates(self) -> Dict[str, Any]:
        try:
            with open(self.error_rates_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload if isinstance(payload, dict) else self._empty_error_rates()
        except Exception:
            return self._empty_error_rates()

    def _run(self) -> None:
        trials_f = open(self.trials_path, "a", encoding="utf-8")
        samples_f = open(self.samples_path, "a", encoding="utf-8")
        errors_f = open(self.errors_path, "a", encoding="utf-8")
        active_trials: Dict[int, Dict[str, Any]] = {}
        error_rates = self._load_error_rates()
        stage_rates = {
            f"{row.get('stage', '')}|{row.get('component', '')}": dict(row)
            for row in error_rates.get("stages", [])
            if isinstance(row, dict)
        }

        def persist_active_trials() -> None:
            rows = sorted(active_trials.values(), key=lambda row: row.get("trial_id", 0))
            _atomic_write_json(self.active_trials_path, {"trials": rows})

        def persist_error_rates() -> None:
            attempts = int(error_rates.get("attempts", 0))
            errors = int(error_rates.get("errors", 0))
            unrecovered = int(error_rates.get("unrecovered_errors", 0))
            error_rates["error_rate"] = round(errors / attempts, 6) if attempts else 0.0
            error_rates["health"] = (
                "ERRORS DETECTED" if unrecovered else "DEGRADED" if errors else "HEALTHY"
            )
            generation_samples = int(error_rates.get("generation_samples", 0))
            answer_refusals = int(error_rates.get("answer_refusals", 0))
            error_rates["answer_refusal_rate"] = (
                round(answer_refusals / generation_samples, 6)
                if generation_samples else 0.0
            )
            for row in stage_rates.values():
                row_attempts = int(row.get("attempts", 0))
                row_errors = int(row.get("errors", 0))
                row["error_rate"] = round(row_errors / row_attempts, 6) if row_attempts else 0.0
            error_rates["stages"] = sorted(
                stage_rates.values(),
                key=lambda row: (
                    -int(row.get("unrecovered_errors", 0)),
                    -float(row.get("error_rate", 0.0)),
                    str(row.get("stage", "")),
                ),
            )
            error_rates["updated_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_write_json(self.error_rates_path, error_rates)

        try:
            while True:
                item = self._queue.get()
                kind, payload = item
                if kind == _STOP[0]:
                    break
                try:
                    if kind == "trial":
                        trial_id = int(payload.get("trial_id", -1))
                        active = active_trials.get(trial_id)
                        if active is not None:
                            # Keep the final record self-contained so the UI can
                            # show sample progress after the active row disappears.
                            payload = dict(payload)
                            payload["started_at"] = active.get("started_at")
                            payload["completed_samples"] = active.get("completed_samples", 0)
                            payload["total_samples"] = active.get("total_samples", 0)
                        trials_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        trials_f.flush()
                        if active_trials.pop(trial_id, None) is not None:
                            persist_active_trials()
                    elif kind == "trial_started":
                        trial_id = int(payload.get("trial_id", -1))
                        active_trials[trial_id] = {
                            **payload,
                            "trial_id": trial_id,
                            "status": "running",
                            "completed_samples": 0,
                        }
                        persist_active_trials()
                    elif kind == "sample":
                        samples_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        samples_f.flush()
                        if payload.get("generation_attempted"):
                            error_rates["generation_samples"] = int(
                                error_rates.get("generation_samples", 0)
                            ) + 1
                            if payload.get("answer_refusal"):
                                error_rates["answer_refusals"] = int(
                                    error_rates.get("answer_refusals", 0)
                                ) + 1
                            persist_error_rates()
                        trial_id = int(payload.get("trial_id", -1))
                        active = active_trials.get(trial_id)
                        if active is not None:
                            completed = int(active.get("completed_samples", 0)) + 1
                            total = int(active.get("total_samples", 0))
                            active["completed_samples"] = min(completed, total) if total > 0 else completed
                            persist_active_trials()
                    elif kind == "manifest":
                        _atomic_write_json(self.manifest_path, payload)
                        if payload.get("status") in {"completed", "failed", "stopped_early"} and active_trials:
                            active_trials.clear()
                            persist_active_trials()
                    elif kind == "operation":
                        from Trace.observability import sanitize_error_message

                        operation = dict(payload or {})
                        stage = str(operation.get("stage") or "unknown")
                        component = str(operation.get("component") or "unknown")
                        outcome = str(operation.get("outcome") or "success")
                        recovery = operation.get("recovery")
                        is_error = outcome == "error"
                        is_recovered = is_error and recovery not in (None, "unrecovered")
                        error_rates["attempts"] = int(error_rates.get("attempts", 0)) + 1
                        row_key = f"{stage}|{component}"
                        row = stage_rates.setdefault(row_key, {
                            "stage": stage,
                            "component": component,
                            "attempts": 0,
                            "errors": 0,
                            "recovered_errors": 0,
                            "unrecovered_errors": 0,
                            "error_rate": 0.0,
                            "last_error_at": None,
                        })
                        row["attempts"] = int(row.get("attempts", 0)) + 1
                        if is_error:
                            operation["message"] = sanitize_error_message(operation.get("message"))
                            errors_f.write(json.dumps(operation, ensure_ascii=False) + "\n")
                            errors_f.flush()
                            error_rates["errors"] = int(error_rates.get("errors", 0)) + 1
                            row["errors"] = int(row.get("errors", 0)) + 1
                            bucket = "recovered_errors" if is_recovered else "unrecovered_errors"
                            error_rates[bucket] = int(error_rates.get(bucket, 0)) + 1
                            row[bucket] = int(row.get(bucket, 0)) + 1
                            row["last_error_at"] = operation.get("occurred_at")
                        persist_error_rates()
                    else:
                        logger.warning(f"TraceWriter: unknown record kind {kind!r}, dropped")
                except Exception as e:
                    logger.error(f"TraceWriter failed to write {kind} record: {e}", exc_info=True)
        finally:
            trials_f.close()
            samples_f.close()
            errors_f.close()
            if active_trials:
                active_trials.clear()
                persist_active_trials()

    def close(self, timeout: float = 30.0) -> None:
        """Drain remaining queued records and stop the background thread."""
        self._queue.put(_STOP)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning(f"TraceWriter background thread did not stop within {timeout}s")
