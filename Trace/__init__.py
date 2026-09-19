"""Trace - Observability schema, per-sample context tagging, and JSONL storage."""

from .observability import (
    emit_stage_outcome,
    mark_current_stage_error,
    observation_context,
    observe_stage,
    record_exception_outcome,
)

__all__ = [
    "emit_stage_outcome",
    "mark_current_stage_error",
    "observation_context",
    "observe_stage",
    "record_exception_outcome",
]
