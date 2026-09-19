"""
Test suite for UsageCallbackHandler (pytest).

Feeds synthetic LLMResult-like objects (no real LangChain SDK network calls
needed) through on_llm_end() and checks the extracted/aggregated totals.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from LLMProvider.usage import UsageCallbackHandler
from Trace.context import set_current_sample, clear_current_sample


def _fake_response(llm_output=None, usage_metadata=None):
    """Build a minimal object shaped like langchain_core.outputs.LLMResult."""
    response = types.SimpleNamespace()
    response.llm_output = llm_output or {}

    message = types.SimpleNamespace()
    message.usage_metadata = usage_metadata
    generation = types.SimpleNamespace(message=message)
    response.generations = [[generation]]
    return response


def test_extracts_openai_style_token_usage():
    handler = UsageCallbackHandler()
    response = _fake_response(
        llm_output={"token_usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}}
    )
    handler.on_llm_end(response)
    assert handler.get_totals() == {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}


def test_falls_back_to_usage_metadata():
    handler = UsageCallbackHandler()
    response = _fake_response(usage_metadata={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70})
    handler.on_llm_end(response)
    assert handler.get_totals() == {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}


def test_missing_usage_is_skipped_silently():
    handler = UsageCallbackHandler()
    handler.on_llm_end(_fake_response())
    assert handler.get_totals() == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_totals_accumulate_across_multiple_calls():
    handler = UsageCallbackHandler()
    handler.on_llm_end(_fake_response(llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}))
    handler.on_llm_end(_fake_response(llm_output={"token_usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}}))
    assert handler.get_totals() == {"prompt_tokens": 30, "completion_tokens": 13, "total_tokens": 43}


def test_reset_clears_totals():
    handler = UsageCallbackHandler()
    handler.on_llm_end(_fake_response(llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}))
    handler.reset()
    assert handler.get_totals() == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_token_usage_dict_takes_priority_over_usage_metadata():
    handler = UsageCallbackHandler()
    response = _fake_response(
        llm_output={"token_usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        usage_metadata={"input_tokens": 999, "output_tokens": 999, "total_tokens": 1998},
    )
    handler.on_llm_end(response)
    assert handler.get_totals() == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


# ---------------------------------------------------------------------------
# Per-sample bucketing (see Trace.context)
# ---------------------------------------------------------------------------
def test_per_sample_totals_bucketed_by_current_sample():
    handler = UsageCallbackHandler()
    set_current_sample(1, 0)
    try:
        handler.on_llm_end(_fake_response(llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}))
    finally:
        clear_current_sample()

    set_current_sample(1, 1)
    try:
        handler.on_llm_end(_fake_response(llm_output={"token_usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}}))
    finally:
        clear_current_sample()

    per_sample = handler.get_per_sample_totals()
    assert per_sample[(1, 0)] == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    assert per_sample[(1, 1)] == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
    # Flat cumulative total (existing, trial-level) must still sum everything.
    assert handler.get_totals() == {"prompt_tokens": 30, "completion_tokens": 6, "total_tokens": 36}


def test_get_sample_totals_convenience_accessor():
    handler = UsageCallbackHandler()
    set_current_sample(2, 5)
    try:
        handler.on_llm_end(_fake_response(llm_output={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}))
    finally:
        clear_current_sample()

    assert handler.get_sample_totals(2, 5) == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    assert handler.get_sample_totals(9, 9) == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_no_active_sample_still_recorded_flat():
    """A call with no sample tag set must still count toward get_totals()."""
    clear_current_sample()
    handler = UsageCallbackHandler()
    handler.on_llm_end(_fake_response(llm_output={"token_usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}))
    assert handler.get_totals() == {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}


def test_reset_clears_per_sample_totals_too():
    handler = UsageCallbackHandler()
    set_current_sample(1, 0)
    try:
        handler.on_llm_end(_fake_response(llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}))
    finally:
        clear_current_sample()

    handler.reset()
    assert handler.get_per_sample_totals() == {}
    assert handler.get_totals() == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))
