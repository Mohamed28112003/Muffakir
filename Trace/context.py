"""
Thread-local "current sample" tag.

EvalRunner can run samples concurrently (max_workers>1) through the SAME
shared LLMProvider/EmbeddingProvider instances. Attributing a shared
instance's work (token usage, embedding time) to the right sample requires
knowing which sample is "currently running on this thread" -- exactly what
this module tracks. A naive reset-before/diff-after approach would silently
corrupt under concurrency; threading.local isolates correctly per-thread
without that risk.

This is a leaf module (zero dependencies on the rest of the codebase) so it
can be imported from LLMProvider/Embedding without any circular-import risk.
"""

import threading
from typing import Optional, Tuple

_local = threading.local()


def set_current_sample(trial_id: int, sample_index: int) -> None:
    """Tag the calling thread as currently evaluating this (trial, sample)."""
    _local.sample = (trial_id, sample_index)


def get_current_sample() -> Optional[Tuple[int, int]]:
    """Return the (trial_id, sample_index) tag for the calling thread, or None."""
    return getattr(_local, "sample", None)


def clear_current_sample() -> None:
    """Clear the calling thread's sample tag."""
    _local.sample = None
