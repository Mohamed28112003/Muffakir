"""
UsageCallbackHandler - Captures per-call token usage centrally for an LLMProvider.

Attached once to the underlying LangChain chat model (see LLMProvider.__init__),
so every consumer that calls `llm.invoke(...)`, `llm.bind(...).invoke(...)`, or a
`with_structured_output(...)`/prompt-piped chain built from it gets its token
usage recorded automatically -- no changes needed at any call site, since
LangChain's callback manager fires `on_llm_end` on the underlying model call
regardless of how a caller wraps or composes it.
"""

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from Trace.context import get_current_sample

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # pragma: no cover - langchain_core is a hard dependency in practice
    class BaseCallbackHandler:  # type: ignore[no-redef]
        """Minimal stand-in so this module still imports without langchain_core."""

logger = logging.getLogger(__name__)

# Key used for calls with no active sample tag (e.g. a direct MuffakirRAG.ask()
# call outside of EvalRunner/Composer, or index-time LLM use).
_UNTRACKED_KEY: Tuple[int, int] = (-1, -1)


def _sum_calls(calls: List[Dict[str, int]]) -> Dict[str, int]:
    return {
        "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
        "completion_tokens": sum(c["completion_tokens"] for c in calls),
        "total_tokens": sum(c["total_tokens"] for c in calls),
    }


class UsageCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that records prompt/completion token counts
    from every LLM call it observes.

    Token counts are extracted defensively since providers report usage in
    different shapes:
      1. ``response.llm_output["token_usage"]`` -- OpenAI-style dict with
         ``prompt_tokens``/``completion_tokens``/``total_tokens``.
      2. ``generation.message.usage_metadata`` -- LangChain-standardized dict
         with ``input_tokens``/``output_tokens``/``total_tokens`` (present on
         newer chat models regardless of provider).

    If neither shape is present (e.g. some local/Ollama models don't report
    usage), the call is silently skipped -- this is a best-effort capability,
    not a hard requirement, matching the project's pricing design (unknown
    usage/pricing must never break a trial).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: List[Dict[str, int]] = []
        self._per_sample: Dict[Tuple[int, int], List[Dict[str, int]]] = {}

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        usage = self._extract_usage(response)
        if usage is None:
            return
        sample_key = get_current_sample() or _UNTRACKED_KEY
        with self._lock:
            self._calls.append(usage)
            self._per_sample.setdefault(sample_key, []).append(usage)

    @staticmethod
    def _extract_usage(response: Any) -> Optional[Dict[str, int]]:
        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage") if isinstance(llm_output, dict) else None
        if isinstance(token_usage, dict) and token_usage:
            prompt = int(token_usage.get("prompt_tokens", 0) or 0)
            completion = int(token_usage.get("completion_tokens", 0) or 0)
            total = int(token_usage.get("total_tokens", prompt + completion) or (prompt + completion))
            return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}

        try:
            generations = response.generations
            message = generations[0][0].message
            usage_metadata = getattr(message, "usage_metadata", None)
        except (AttributeError, IndexError, TypeError):
            usage_metadata = None

        if isinstance(usage_metadata, dict) and usage_metadata:
            prompt = int(usage_metadata.get("input_tokens", 0) or 0)
            completion = int(usage_metadata.get("output_tokens", 0) or 0)
            total = int(usage_metadata.get("total_tokens", prompt + completion) or (prompt + completion))
            return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}

        return None

    def get_totals(self) -> Dict[str, int]:
        """Sum of prompt/completion/total tokens across every recorded call so far."""
        with self._lock:
            calls = list(self._calls)
        return _sum_calls(calls)

    def get_per_sample_totals(self) -> Dict[Tuple[int, int], Dict[str, int]]:
        """Sum of prompt/completion/total tokens per (trial_id, sample_index)."""
        with self._lock:
            per_sample = {k: list(v) for k, v in self._per_sample.items()}
        return {key: _sum_calls(calls) for key, calls in per_sample.items()}

    def get_sample_totals(self, trial_id: int, sample_index: int) -> Dict[str, int]:
        """Convenience accessor: totals for one sample, or all-zero if unseen."""
        with self._lock:
            calls = list(self._per_sample.get((trial_id, sample_index), []))
        return _sum_calls(calls)

    def reset(self) -> None:
        """Clear all recorded calls."""
        with self._lock:
            self._calls = []
            self._per_sample = {}
