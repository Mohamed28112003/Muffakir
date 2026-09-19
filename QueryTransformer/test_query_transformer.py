"""
Test suite for the QueryTransformer module (pytest).

The LLM is faked with a LangChain-like mock (bind() returns a delegating
clone; structured output is configurable), so no network or API keys are
needed. Covers all five strategies (structured + parser-fallback paths),
history formatting labels, fail-safe fallbacks, factory/wrapper behavior,
and the P1 no-temperature-mutation regression.
"""

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from QueryTransformer import (
    QueryTransformer,
    QueryRewriter,
    MultiQueryExpansion,
    QueryExpansionOutput,
    QueryDecomposition,
    SubQueryDecomposition,
    HyDEQueryTransformer,
    StepBackQueryTransformer,
    StepBackQueryOutput,
    create_query_transformer,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class MockResp:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """
    LangChain-like mock: ``bind()`` returns a clone carrying bound kwargs
    (mirroring RunnableBinding semantics incl. attribute delegation), and
    ``with_structured_output()`` honors a configurable result/exception.
    """

    last_prompt = None

    def __init__(self, text="ok", exc=None):
        self.temperature = 0.7  # sentinel — must never be mutated by library code
        self._text, self._exc = text, exc
        self.wso_result = None   # Pydantic instance or dict for structured path
        self.wso_exc = None      # exception to force parser fallback

    def bind(self, **kwargs):
        clone = FakeLLM.__new__(FakeLLM)
        clone.__dict__.update({k: v for k, v in self.__dict__.items()})
        return clone

    def _structured(self):
        inner = self

        class _Structured:
            def invoke(self, prompt):
                if inner.wso_exc:
                    raise inner.wso_exc
                if inner.wso_result is None:
                    raise ValueError("no structured result configured")
                return inner.wso_result

        return _Structured()

    def with_structured_output(self, schema):
        return self._structured()

    def invoke(self, prompt):
        FakeLLM.last_prompt = prompt
        if self._exc:
            raise self._exc
        return MockResp(self._text)


class MockProvider:
    def __init__(self, llm):
        self._llm = llm

    def get_llm(self):
        return self._llm


DEFAULT_TEMPLATES = {
    "query_rewrite": "RW: {original_query}",
    "multi_query_expansion": "MQ: {original_query}",
    "query_decomposition": "QD: {original_query}",
    "hyde": "HY: {original_query}",
    "step_back": "SB: {original_query}",
}


class MockPM:
    language = "ar"

    def __init__(self, templates=None):
        self.t = templates or {}

    def get_prompt(self, key):
        merged = dict(DEFAULT_TEMPLATES)
        merged.update(self.t)
        return merged[key]


def capture_bind(raw_llm):
    """Wrap raw.bind to record kwargs while still returning the clone."""
    calls = {}
    orig = raw_llm.bind

    def recording(**kw):
        calls.update(kw)
        return orig(**kw)

    raw_llm.bind = recording
    return calls


HISTORY = [{"role": "user", "content": "hi"},
           {"role": "assistant", "content": "hello"}]


# ---------------------------------------------------------------------------
# P1 regression: no shared-state temperature mutation (all five strategies)
# ---------------------------------------------------------------------------
def test_rewrite_no_temp_mutation():
    raw = FakeLLM(text="rewritten")
    calls = capture_bind(raw)
    t = QueryRewriter(MockProvider(raw), MockPM())
    assert t.transform("q") == "rewritten"
    assert calls == {"temperature": 0.0}
    assert raw.temperature == 0.7


def test_multi_query_no_temp_mutation():
    raw = FakeLLM(text="1. v1")
    calls = capture_bind(raw)
    MultiQueryExpansion(MockProvider(raw), MockPM()).transform("q")
    assert calls == {"temperature": 0.2}
    assert raw.temperature == 0.7


def test_decomposition_no_temp_mutation():
    raw = FakeLLM(text="1. s1")
    calls = capture_bind(raw)
    QueryDecomposition(MockProvider(raw), MockPM()).transform("q")
    assert calls == {"temperature": 0.0}
    assert raw.temperature == 0.7


def test_hyde_no_temp_mutation():
    raw = FakeLLM(text="doc")
    calls = capture_bind(raw)
    HyDEQueryTransformer(MockProvider(raw), MockPM()).transform("q")
    assert calls == {"temperature": 0.3}
    assert raw.temperature == 0.7


def test_step_back_no_temp_mutation():
    raw = FakeLLM(text="broader q")
    calls = capture_bind(raw)
    StepBackQueryTransformer(MockProvider(raw), MockPM()).transform("q")
    assert calls == {"temperature": 0.0}
    assert raw.temperature == 0.7


def test_custom_temperatures_respected_via_bind():
    mq = MultiQueryExpansion(MockProvider(FakeLLM()), MockPM(), temperature=0.9)
    hy = HyDEQueryTransformer(MockProvider(FakeLLM()), MockPM(), temperature=0.05)
    calls_mq = capture_bind(mq.llm_provider.get_llm())
    calls_hy = capture_bind(hy.llm_provider.get_llm())
    mq.transform("q")
    hy.transform("q")
    assert calls_mq == {"temperature": 0.9}
    assert calls_hy == {"temperature": 0.05}


# ---------------------------------------------------------------------------
# Rewriter
# ---------------------------------------------------------------------------
def test_rewriter_strips_quotes_and_whitespace():
    llm = FakeLLM(text='  "the cleaned query"  ')
    t = QueryRewriter(MockProvider(llm), MockPM())
    assert t.transform("q") == "the cleaned query"


def test_rewriter_empty_result_returns_original():
    llm = FakeLLM(text='""')
    t = QueryRewriter(MockProvider(llm), MockPM())
    assert t.transform("original q") == "original q"


def test_rewriter_empty_query_returns_empty():
    assert QueryRewriter(MockProvider(FakeLLM()), MockPM()).transform("   ") == ""


def test_rewriter_failure_returns_original():
    llm = FakeLLM(exc=RuntimeError("down"))
    t = QueryRewriter(MockProvider(llm), MockPM())
    assert t.transform("keep me") == "keep me"


def test_rewriter_history_formatting():
    llm = FakeLLM(text="rw")
    QueryRewriter(MockProvider(llm), MockPM()).transform("latest?", HISTORY)
    p = FakeLLM.last_prompt
    assert "[Context History]:" in p and "user: hi" in p and "assistant: hello" in p
    assert "[Latest Query]: latest?" in p


# ---------------------------------------------------------------------------
# Multi-Query Expansion
# ---------------------------------------------------------------------------
def test_multi_query_structured_pydantic_path():
    llm = FakeLLM()
    llm.wso_result = QueryExpansionOutput(expanded_queries=["v1", "v2"])
    out = MultiQueryExpansion(MockProvider(llm), MockPM()).transform("q")
    # original query is always first
    assert out[0] == "q" and out[1:] == ["v1", "v2"]


def test_multi_query_structured_dict_path():
    llm = FakeLLM()
    llm.wso_result = {"expanded_queries": ["d1"]}
    out = MultiQueryExpansion(MockProvider(llm), MockPM()).transform("q")
    assert out == ["q", "d1"]


def test_multi_query_parser_fallback_on_wso_failure():
    llm = FakeLLM(text="1. one\n- two\n\nthree")
    llm.wso_exc = RuntimeError("no structured support")
    out = MultiQueryExpansion(MockProvider(llm), MockPM()).transform("q")
    assert out == ["q", "one", "two", "three"]


def test_multi_query_original_moved_to_front():
    llm = FakeLLM(text="other\nq")  # parser output contains q later
    out = MultiQueryExpansion(MockProvider(llm), MockPM()).transform("q")
    assert out.count("q") == 1 and out[0] == "q"


def test_multi_query_failure_returns_single_item_list():
    llm = FakeLLM(exc=RuntimeError("down"))
    out = MultiQueryExpansion(MockProvider(llm), MockPM()).transform("orig")
    assert out == ["orig"]


def test_multi_query_empty_query_branches():
    """Designed guard: '' -> [], but non-empty whitespace is preserved as-is."""
    mq = MultiQueryExpansion(MockProvider(FakeLLM()), MockPM())
    assert mq.transform("") == []
    assert mq.transform("  ") == ["  "]


# ---------------------------------------------------------------------------
# Query Decomposition
# ---------------------------------------------------------------------------
def test_decomposition_structured_path():
    llm = FakeLLM()
    llm.wso_result = SubQueryDecomposition(
        sub_queries=["s1", "s2"], rationale="complex question")
    out = QueryDecomposition(MockProvider(llm), MockPM()).transform("q")
    assert out == ["s1", "s2"]


def test_decomposition_parser_skips_rationale_and_braces():
    llm = FakeLLM(text=(
        "Rationale: because\n"
        "التبرير: سبب\n"
        "{\n"
        "1. s1\n"
        "2. s2\n"
        "}"))
    llm.wso_exc = RuntimeError("no structured support")
    out = QueryDecomposition(MockProvider(llm), MockPM()).transform("q")
    assert out == ["s1", "s2"]


def test_decomposition_empty_parse_returns_original():
    llm = FakeLLM(text="Rationale: only\nالتبرير: فقط")
    llm.wso_exc = ValueError("force parser")
    out = QueryDecomposition(MockProvider(llm), MockPM()).transform("orig")
    assert out == ["orig"]


def test_decomposition_history_label():
    llm = FakeLLM(text="1. s1")
    QueryDecomposition(MockProvider(llm), MockPM()).transform("cq?", HISTORY)
    assert "[Complex Query]: cq?" in FakeLLM.last_prompt


# ---------------------------------------------------------------------------
# HyDE
# ---------------------------------------------------------------------------
def test_hyde_returns_document_text():
    llm = FakeLLM(text="  A plausible document about X.  ")
    out = HyDEQueryTransformer(MockProvider(llm), MockPM()).transform("q")
    assert out == "A plausible document about X."


def test_hyde_failure_returns_original():
    llm = FakeLLM(exc=RuntimeError("down"))
    t = HyDEQueryTransformer(MockProvider(llm), MockPM())
    assert t.transform("keep") == "keep"


def test_hyde_history_label():
    llm = FakeLLM(text="doc")
    HyDEQueryTransformer(MockProvider(llm), MockPM()).transform("q", HISTORY)
    assert "[Query]: q" in FakeLLM.last_prompt


# ---------------------------------------------------------------------------
# Step-Back
# ---------------------------------------------------------------------------
def test_step_back_structured_path_returns_both():
    llm = FakeLLM()
    llm.wso_result = StepBackQueryOutput(step_back_query="What is HTTP 413?")
    out = StepBackQueryTransformer(MockProvider(llm), MockPM()).transform(
        "why my webhook 413?")
    assert out == ["why my webhook 413?", "What is HTTP 413?"]


def test_step_back_dict_path():
    llm = FakeLLM()
    llm.wso_result = {"step_back_query": "generic q"}
    out = StepBackQueryTransformer(MockProvider(llm), MockPM()).transform("q")
    assert out == ["q", "generic q"]


def test_step_back_dedupes_identical_case_insensitive():
    llm = FakeLLM()
    llm.wso_result = {"step_back_query": "Q"}  # same as original after strip
    out = StepBackQueryTransformer(MockProvider(llm), MockPM()).transform("q")
    assert out == ["q"]


def test_step_back_parser_fallback_strips_quotes():
    llm = FakeLLM(text='"What does error X mean?"')
    llm.wso_exc = RuntimeError("no structured support")
    out = StepBackQueryTransformer(MockProvider(llm), MockPM()).transform("q")
    assert out == ["q", "What does error X mean?"]


def test_step_back_failure_returns_original():
    llm = FakeLLM(exc=RuntimeError("down"))
    out = StepBackQueryTransformer(MockProvider(llm), MockPM()).transform("orig")
    assert out == ["orig"]


def test_step_back_history_label():
    llm = FakeLLM(text="broader")
    StepBackQueryTransformer(MockProvider(llm), MockPM()).transform("sq", HISTORY)
    assert "[Specific Query]: sq" in FakeLLM.last_prompt


# ---------------------------------------------------------------------------
# Wrapper (QueryTransformer orchestrator)
# ---------------------------------------------------------------------------
def test_wrapper_dispatch_and_delegation():
    w = QueryTransformer(MockProvider(FakeLLM(text="rw")), MockPM(),
                         strategy="rewrite")
    assert w.name == "rewrite"
    assert w.transform_query("hello") == "rw"


def test_wrapper_accepts_transformer_instance():
    custom = MultiQueryExpansion(MockProvider(FakeLLM()), MockPM())
    w = QueryTransformer(MockProvider(FakeLLM()), MockPM(), strategy=custom)
    assert w.transformer is custom
    assert w.name == "multi_query"


def test_wrapper_passes_strategy_config():
    w = QueryTransformer(MockProvider(FakeLLM()), MockPM(),
                         strategy="hyde",
                         strategy_config={"temperature": 0.11})
    assert isinstance(w.transformer, HyDEQueryTransformer)
    assert w.transformer.temperature == 0.11


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def test_factory_alias_groups():
    p = MockProvider(FakeLLM())
    pm = MockPM()
    cases = {
        "rewrite": QueryRewriter, "query_rewrite": QueryRewriter,
        "multi_query": MultiQueryExpansion,
        "query_expansion": MultiQueryExpansion,
        "decomposition": QueryDecomposition, "sub_query": QueryDecomposition,
        "hyde": HyDEQueryTransformer,
        "hypothetical_document": HyDEQueryTransformer,
        "step_back": StepBackQueryTransformer, "stepback": StepBackQueryTransformer,
    }
    for name, cls in cases.items():
        assert isinstance(create_query_transformer(name, p, pm), cls), name


def test_factory_unknown_strategy_lists_options():
    with pytest.raises(ValueError) as ei:
        create_query_transformer("bogus", MockProvider(FakeLLM()), MockPM())
    assert "step_back" in str(ei.value)


def test_factory_requires_llm_provider():
    with pytest.raises(ValueError, match="llm_provider is required"):
        create_query_transformer("rewrite", None)


def test_available_strategies_constant():
    from QueryTransformer.factory import AVAILABLE_STRATEGIES
    assert AVAILABLE_STRATEGIES == ("rewrite", "multi_query", "decomposition",
                                    "hyde", "step_back")


# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------
def test_no_basicconfig_regression():
    root_before = list(logging.getLogger().handlers)
    level_before = logging.getLogger().level
    p = MockProvider(FakeLLM())
    for cls in (QueryRewriter, MultiQueryExpansion, QueryDecomposition,
                HyDEQueryTransformer, StepBackQueryTransformer):
        cls(p, MockPM()).transform("q")
    assert list(logging.getLogger().handlers) == root_before
    assert logging.getLogger().level == level_before


if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))