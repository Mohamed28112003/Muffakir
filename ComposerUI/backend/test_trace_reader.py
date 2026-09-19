"""
Tests for ComposerUI trace_reader module.
"""

import json
from pathlib import Path
from ComposerUI.backend import trace_reader


def test_read_manifest(tmp_path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()

    # Non-existent manifest
    assert trace_reader.read_manifest(trace_dir) is None

    # Valid manifest
    manifest_data = {"run_id": "test-run", "status": "running", "total_trials": 10}
    (trace_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    loaded = trace_reader.read_manifest(trace_dir)
    assert loaded is not None
    assert loaded["run_id"] == "test-run"
    assert loaded["status"] == "running"


def test_read_trials_after_pagination(tmp_path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    trials_file = trace_dir / "trials.jsonl"

    records = [
        {"trial_id": 1, "composite_score": 0.85, "status": "success"},
        {"trial_id": 2, "composite_score": 0.90, "status": "success"},
        {"trial_id": 3, "composite_score": 0.0, "status": "failed", "error_code": "timeout"},
    ]

    with open(trials_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Read from 0
    read_1, next_after = trace_reader.read_trials_after(trace_dir, after=0)
    assert len(read_1) == 3
    assert next_after == 3
    assert read_1[0]["trial_id"] == 1

    # Read from 2
    read_2, next_after_2 = trace_reader.read_trials_after(trace_dir, after=2)
    assert len(read_2) == 1
    assert next_after_2 == 3
    assert read_2[0]["trial_id"] == 3

    # Count lines
    assert trace_reader.count_trial_lines(trace_dir) == 3


def test_read_trials_after_partial_line(tmp_path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    trials_file = trace_dir / "trials.jsonl"

    with open(trials_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"trial_id": 1, "status": "success"}) + "\n")
        f.write('{"trial_id": 2, "status": "succ')  # Truncated partial write

    # Truncated trailing line should be skipped gracefully
    records, next_after = trace_reader.read_trials_after(trace_dir, after=0)
    assert len(records) == 1
    assert records[0]["trial_id"] == 1
    assert next_after == 1


def test_read_active_trials_is_sorted_and_legacy_safe(tmp_path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()

    assert trace_reader.read_active_trials(trace_dir) == []
    (trace_dir / "active_trials.json").write_text(
        json.dumps({"trials": [
            {"trial_id": 9, "status": "running"},
            {"trial_id": 2, "status": "running"},
        ]}),
        encoding="utf-8",
    )

    assert [row["trial_id"] for row in trace_reader.read_active_trials(trace_dir)] == [2, 9]


def test_read_active_trials_ignores_malformed_payload(tmp_path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "active_trials.json").write_text(
        json.dumps({"trials": "not-a-list"}),
        encoding="utf-8",
    )
    assert trace_reader.read_active_trials(trace_dir) == []


def test_find_trial_and_read_its_samples(tmp_path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "trials.jsonl").write_text(
        "\n".join([
            json.dumps({"trial_id": 1, "status": "success"}),
            json.dumps({"trial_id": 2, "status": "success"}),
        ]) + "\n",
        encoding="utf-8",
    )
    (trace_dir / "samples.jsonl").write_text(
        "\n".join([
            json.dumps({"trial_id": 2, "sample_index": 1, "question": "second"}),
            json.dumps({"trial_id": 1, "sample_index": 0, "question": "other trial"}),
            json.dumps({"trial_id": 2, "sample_index": 0, "question": "first"}),
        ]) + "\n",
        encoding="utf-8",
    )

    assert trace_reader.find_trial_record(trace_dir, 2)["trial_id"] == 2
    assert trace_reader.find_trial_record(trace_dir, 99) is None
    samples = trace_reader.read_samples_for_trial(trace_dir, 2)
    assert [sample["sample_index"] for sample in samples] == [0, 1]
    assert [sample["question"] for sample in samples] == ["first", "second"]


def test_cluster_failures_and_aggregate_timings():
    records = [
        {
            "trial_id": 1,
            "status": "success",
            "mean_query_transform_ms": 10.0,
            "mean_generation_ms": 100.0,
        },
        {
            "trial_id": 2,
            "status": "failed",
            "error_code": "RATE_LIMIT",
            "mean_query_transform_ms": 20.0,
            "mean_generation_ms": None,
        },
        {
            "trial_id": 3,
            "status": "failed",
            "error_code": "RATE_LIMIT",
            "mean_query_transform_ms": None,
            "mean_generation_ms": None,
        },
        {
            "trial_id": 4,
            "status": "failed",
            "error_type": "ConnectionError",
        },
    ]

    clusters = trace_reader.cluster_failures_from_records(records)
    assert "RATE_LIMIT" in clusters
    assert clusters["RATE_LIMIT"] == [2, 3]
    assert "ConnectionError" in clusters
    assert clusters["ConnectionError"] == [4]

    timings = trace_reader.aggregate_stage_timings(records)
    assert timings["query_transform"] == 15.0  # (10 + 20) / 2
    assert timings["generation"] == 100.0
    assert timings["vector_search"] is None
    assert timings["web_search"] is None

    selected = trace_reader.stage_timings_from_record(records[0])
    assert selected["query_transform"] == 10.0
    assert selected["generation"] == 100.0
    assert selected["rerank"] is None


def test_zero_sample_legacy_detection_requires_explicit_evidence():
    assert not trace_reader.has_zero_evaluation_samples([])
    assert not trace_reader.has_zero_evaluation_samples([{"trial_id": 1}])
    assert not trace_reader.has_zero_evaluation_samples([
        {"trial_id": 1, "total_samples": 0},
        {"trial_id": 2, "total_samples": 1},
    ])
    assert trace_reader.has_zero_evaluation_samples([
        {"trial_id": 1, "total_samples": 0},
        {"trial_id": 2, "total_samples": 0},
    ])


def test_total_web_search_fallbacks_sums_across_records():
    records = [
        {"trial_id": 1, "web_search_fallback_count": 2},
        {"trial_id": 2, "web_search_fallback_count": 0},
        {"trial_id": 3},  # missing key -- must not crash, counts as 0
    ]
    assert trace_reader.total_web_search_fallbacks(records) == 2


def test_read_error_message(tmp_path):
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()

    assert trace_reader.read_error_message(run_dir) is None

    (run_dir / "error.txt").write_text("Fatal error occurred in worker", encoding="utf-8")
    assert trace_reader.read_error_message(run_dir) == "Fatal error occurred in worker"


def test_read_error_rates_and_incremental_events(tmp_path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    assert trace_reader.read_error_rates(trace_dir) is None

    summary = {
        "attempts": 4,
        "errors": 1,
        "recovered_errors": 1,
        "unrecovered_errors": 0,
        "error_rate": 0.25,
        "health": "DEGRADED",
        "stages": [],
    }
    (trace_dir / "error_rates.json").write_text(json.dumps(summary), encoding="utf-8")
    events = [
        {"event_id": "one", "stage": "generation"},
        {"event_id": "two", "stage": "rerank"},
    ]
    (trace_dir / "errors.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n{\"partial\":",
        encoding="utf-8",
    )

    assert trace_reader.read_error_rates(trace_dir)["error_rate"] == 0.25
    first, cursor = trace_reader.read_errors_after(trace_dir, after=0)
    assert [event["event_id"] for event in first] == ["one", "two"]
    assert cursor == 2
    second, next_cursor = trace_reader.read_errors_after(trace_dir, after=1)
    assert [event["event_id"] for event in second] == ["two"]
    assert next_cursor == 2
