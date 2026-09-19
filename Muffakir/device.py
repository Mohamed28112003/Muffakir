"""Device resolution for GPU/CPU-backed local models (sentence-transformers/torch).

Muffakir's local embedding/reranker backends previously let sentence-transformers
silently auto-detect CUDA availability with no way to override it. This makes that
choice explicit and user-controllable.
"""

from typing import Optional


def resolve_device(device: Optional[str] = "auto") -> str:
    """Resolve a user-facing device spec to a concrete device string.

    "auto" (the default) picks "cuda" if a CUDA device is available, else "cpu" —
    matching sentence-transformers' own previous implicit behavior, just made
    explicit and overridable. Any other value (e.g. "cpu", "cuda", "cuda:1") is
    passed through unchanged.
    """
    if device is None or device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device
