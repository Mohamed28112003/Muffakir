"""Test suite for Trace.writer.TraceWriter (pytest)."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Trace.writer import TraceWriter


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _wait_for_json(path: Path, predicate, timeout: float = 2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if predicate(payload):
                    return payload
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for {path}")


def test_write_trial_appends_jsonl_line(tmp_path):
    writer = TraceWriter(str(tmp_path))
    writer.write_trial({"trial_id": 0, "composite_score": 0.9})
    writer.close()

    rows = _read_jsonl(tmp_path / "trials.jsonl")
    assert rows == [{"trial_id": 0, "composite_score": 0.9}]


def test_write_sample_appends_jsonl_line(tmp_path):
    writer = TraceWriter(str(tmp_path))
    writer.write_sample({"trial_id": 0, "sample_index": 0, "question": "q?"})
    writer.close()

    rows = _read_jsonl(tmp_path / "samples.jsonl")
    assert rows == [{"trial_id": 0, "sample_index": 0, "question": "q?"}]


def test_multiple_writes_append_in_order(tmp_path):
    writer = TraceWriter(str(tmp_path))
    for i in range(5):
        writer.write_trial({"trial_id": i})
    writer.close()

    rows = _read_jsonl(tmp_path / "trials.jsonl")
    assert [r["trial_id"] for r in rows] == [0, 1, 2, 3, 4]


def test_write_manifest_atomic_overwrite(tmp_path):
    writer = TraceWriter(str(tmp_path))
    writer.write_manifest({"status": "running"})
    writer.write_manifest({"status": "completed"})
    writer.close()

    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == {"status": "completed"}

    # No leftover temp files from the atomic-write mechanism.
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_close_is_idempotent_enough_to_call_once(tmp_path):
    writer = TraceWriter(str(tmp_path))
    writer.write_trial({"trial_id": 0})
    writer.close()
    assert not writer._thread.is_alive()


def test_writer_creates_trace_dir_if_missing(tmp_path):
    nested = tmp_path / "a" / "b" / "trace"
    writer = TraceWriter(str(nested))
    writer.write_trial({"trial_id": 0})
    writer.close()
    assert (nested / "trials.jsonl").exists()


def test_writer_accepts_external_work_queue(tmp_path):
    """A caller-supplied queue (e.g. multiprocessing.Manager().Queue()) must
    be drained the same way as the writer's own default queue."""
    import queue as queue_module

    external_queue = queue_module.Queue()
    writer = TraceWriter(str(tmp_path), work_queue=external_queue)
    external_queue.put(("trial", {"trial_id": 42}))
    writer.close()

    rows = _read_jsonl(tmp_path / "trials.jsonl")
    assert rows == [{"trial_id": 42}]


def test_active_trial_snapshot_tracks_progress_and_finalization(tmp_path):
    writer = TraceWriter(str(tmp_path))
    active_path = tmp_path / "active_trials.json"

    writer.write_trial_started({
        "trial_id": 7,
        "status": "running",
        "resolved_rag_config": {"k": 5},
        "started_at": "2026-09-11T10:00:00+00:00",
        "total_samples": 2,
    })
    first = _wait_for_json(active_path, lambda data: len(data["trials"]) == 1)
    assert first["trials"][0]["completed_samples"] == 0

    writer.write_sample({"trial_id": 7, "sample_index": 0})
    progressed = _wait_for_json(
        active_path,
        lambda data: data["trials"][0].get("completed_samples") == 1,
    )
    assert progressed["trials"][0]["total_samples"] == 2

    writer.write_trial_started({
        "trial_id": 8,
        "resolved_rag_config": {"k": 10},
        "started_at": "2026-09-11T10:00:01+00:00",
        "total_samples": 2,
    })
    _wait_for_json(active_path, lambda data: len(data["trials"]) == 2)

    writer.write_trial({"trial_id": 7, "status": "success", "latency_ms": 1000.0})
    remaining = _wait_for_json(
        active_path,
        lambda data: [row["trial_id"] for row in data["trials"]] == [8],
    )
    assert remaining["trials"][0]["status"] == "running"

    writer.close()
    assert json.loads(active_path.read_text(encoding="utf-8")) == {"trials": []}
    final = _read_jsonl(tmp_path / "trials.jsonl")[0]
    assert final["started_at"] == "2026-09-11T10:00:00+00:00"
    assert final["completed_samples"] == 1
    assert final["total_samples"] == 2


def test_terminal_manifest_clears_active_snapshot(tmp_path):
    writer = TraceWriter(str(tmp_path))
    active_path = tmp_path / "active_trials.json"
    writer.write_trial_started({
        "trial_id": 3,
        "started_at": "2026-09-11T10:00:00+00:00",
        "total_samples": 1,
    })
    _wait_for_json(active_path, lambda data: len(data["trials"]) == 1)
    writer.write_manifest({"status": "failed"})
    _wait_for_json(active_path, lambda data: data == {"trials": []})
    writer.close()


def test_error_rate_aggregation_redaction_and_resume(tmp_path):
    writer = TraceWriter(str(tmp_path))
    initial = json.loads((tmp_path / "error_rates.json").read_text(encoding="utf-8"))
    assert initial["attempts"] == 0
    assert initial["health"] == "HEALTHY"

    writer.write_operation({
        "event_id": "ok-1",
        "occurred_at": "2026-09-11T10:00:00+00:00",
        "stage": "generation",
        "component": "llm",
        "outcome": "success",
    })
    writer.write_operation({
        "event_id": "err-1",
        "occurred_at": "2026-09-11T10:00:01+00:00",
        "stage": "generation",
        "component": "llm",
        "outcome": "error",
        "recovery": "retry",
        "error_code": "PROVIDER_TIMEOUT",
        "message": "authorization=top-secret Bearer another-secret",
    })
    writer.close()

    summary = json.loads((tmp_path / "error_rates.json").read_text(encoding="utf-8"))
    assert summary["attempts"] == 2
    assert summary["errors"] == 1
    assert summary["recovered_errors"] == 1
    assert summary["unrecovered_errors"] == 0
    assert summary["error_rate"] == 0.5
    assert summary["health"] == "DEGRADED"
    assert summary["stages"][0]["attempts"] == 2
    error_text = (tmp_path / "errors.jsonl").read_text(encoding="utf-8")
    assert "top-secret" not in error_text
    assert "another-secret" not in error_text

    resumed = TraceWriter(str(tmp_path))
    resumed.write_operation({
        "event_id": "err-2",
        "occurred_at": "2026-09-11T10:00:02+00:00",
        "stage": "vector_search",
        "component": "vector_db",
        "outcome": "error",
        "recovery": "unrecovered",
        "error_code": "RETRIEVAL_FAILED",
    })
    resumed.close()
    summary = json.loads((tmp_path / "error_rates.json").read_text(encoding="utf-8"))
    assert summary["attempts"] == 3
    assert summary["errors"] == 2
    assert summary["unrecovered_errors"] == 1
    assert summary["health"] == "ERRORS DETECTED"


def test_writer_aggregates_answer_refusals_separately_from_errors(tmp_path):
    writer = TraceWriter(str(tmp_path))
    writer.write_sample({
        "trial_id": 0, "sample_index": 0, "generation_attempted": True,
        "answer_refusal": True, "answer_refusal_reason": "insufficient_information",
    })
    writer.write_sample({
        "trial_id": 0, "sample_index": 1, "generation_attempted": True,
        "answer_refusal": False,
    })
    writer.write_sample({"trial_id": 0, "sample_index": 2, "generation_attempted": False})
    writer.close()

    summary = json.loads((tmp_path / "error_rates.json").read_text(encoding="utf-8"))
    assert summary["generation_samples"] == 2
    assert summary["answer_refusals"] == 1
    assert summary["answer_refusal_rate"] == 0.5
    assert summary["errors"] == 0


def test_observer_records_most_specific_stage_once(tmp_path):
    import queue as queue_module

    from Trace.observability import (
        observation_context,
        observe_stage,
        record_exception_outcome,
    )

    events = queue_module.Queue()
    with observation_context(events, trial_id=7, sample_index=2, attempt=1):
        try:
            with observe_stage("generation", "llm"):
                raise TimeoutError("remote timeout")
        except TimeoutError as error:
            record_exception_outcome(error, recovery="retry")

    kind, event = events.get_nowait()
    assert kind == "operation"
    assert event["stage"] == "generation"
    assert event["trial_id"] == 7
    assert event["sample_index"] == 2
    assert event["attempt"] == 1
    assert event["recovery"] == "retry"
    assert event["error_code"] == "PROVIDER_TIMEOUT"
    assert events.empty()


def test_error_rate_writer_counts_concurrent_worker_events(tmp_path):
    import queue as queue_module
    import threading

    work = queue_module.Queue()
    writer = TraceWriter(str(tmp_path), work_queue=work)

    def push(worker_id):
        for index in range(25):
            work.put(("operation", {
                "event_id": f"{worker_id}-{index}",
                "occurred_at": "2026-09-11T10:00:00+00:00",
                "stage": "vector_search",
                "component": "vector_db",
                "outcome": "error" if index == 0 else "success",
                "recovery": "retry" if index == 0 else None,
            }))

    threads = [threading.Thread(target=push, args=(worker,)) for worker in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    writer.close()

    summary = json.loads((tmp_path / "error_rates.json").read_text(encoding="utf-8"))
    assert summary["attempts"] == 100
    assert summary["errors"] == 4
    assert summary["recovered_errors"] == 4
    assert summary["error_rate"] == 0.04


if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))
