"""A propagated exception should be recorded once across run wrappers."""

from queue import Queue

from Trace.observability import observation_context, record_exception_outcome


def test_record_exception_outcome_deduplicates_same_exception():
    queue = Queue()
    error = RuntimeError("dataset failed")
    with observation_context(queue):
        record_exception_outcome(
            error, recovery="unrecovered",
            fallback_stage="dataset_load", fallback_component="dataset",
        )
        record_exception_outcome(
            error, recovery="unrecovered",
            fallback_stage="run_execution", fallback_component="runtime",
        )

    assert queue.qsize() == 1
    kind, record = queue.get_nowait()
    assert kind == "operation"
    assert record["stage"] == "dataset_load"
    assert record["outcome"] == "error"
