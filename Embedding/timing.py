"""
EmbeddingTimingTracker - Captures per-call query-embedding time, bucketed by
the currently-active sample (see Trace.context), so multiple concurrently
running EvalRunner samples sharing one EmbeddingProvider instance each get
their own timing total instead of clobbering a flat running counter.

Mirrors LLMProvider.usage.UsageCallbackHandler's shape (flat cumulative total
+ per-sample bucket + reset), for the same reason: trial-level code (existing,
unchanged) keeps reading the flat total; sample-level trace code reads the
per-sample buckets.
"""

import threading
from typing import Dict, Optional, Tuple

from Trace.context import get_current_sample

# Key used for calls with no active sample tag (e.g. during index-time
# embedding in Composer._index_documents(), which runs outside any sample).
_UNTRACKED_KEY: Tuple[int, int] = (-1, -1)


class EmbeddingTimingTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._flat_total_ms: float = 0.0
        self._flat_count: int = 0
        self._per_sample: Dict[Tuple[int, int], Dict[str, float]] = {}

    def record(self, elapsed_ms: float) -> None:
        sample_key = get_current_sample() or _UNTRACKED_KEY
        with self._lock:
            self._flat_total_ms += elapsed_ms
            self._flat_count += 1
            bucket = self._per_sample.setdefault(sample_key, {"total_ms": 0.0, "count": 0})
            bucket["total_ms"] += elapsed_ms
            bucket["count"] += 1

    def get_totals(self) -> Dict[str, float]:
        """Flat cumulative total across every recorded call so far (any sample)."""
        with self._lock:
            return {"total_ms": self._flat_total_ms, "count": self._flat_count}

    def get_per_sample_totals(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        with self._lock:
            return {k: dict(v) for k, v in self._per_sample.items()}

    def get_sample_total_ms(self, trial_id: int, sample_index: int) -> float:
        """Convenience accessor: total embedding ms recorded for one sample, or 0.0."""
        with self._lock:
            bucket = self._per_sample.get((trial_id, sample_index))
            return bucket["total_ms"] if bucket else 0.0

    def reset(self) -> None:
        with self._lock:
            self._flat_total_ms = 0.0
            self._flat_count = 0
            self._per_sample = {}
