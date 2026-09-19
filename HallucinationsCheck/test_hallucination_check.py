"""
Test suite for the HallucinationsCheck module (pytest).

No optional SDKs (sentence-transformers/torch) or real LLMs are required:
the CrossEncoder is faked via monkeypatched sys.modules and LLM/embedding
providers are mocked. Validates all four strategies, the factory,
HallucinationResult validation, backward-compatible check_answer(), the
P1 temperature-bind regression, and fail-safe behavior on errors.
"""

import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from HallucinationsCheck.base import BaseHallucinationChecker, HallucinationResult
from HallucinationsCheck.factory import create_hallucination_checker
from HallucinationsCheck.context_grounding import (
    ContextGroundingChecker,
    GroundingVerdict,
)
from HallucinationsCheck.semantic_similarity import SemanticSimilarityChecker
from HallucinationsCheck.text_cleaner import TextCleanerChecker
from Muffakir.exceptions import HallucinationCheckError


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------
class MockResp:
    def __init__(self, content):
        self.content = content


class MockProvider:
    """Wraps a raw LLM mock behind a get_llm() interface."""

    def __init__(self, llm):
        self._llm = llm

    def get_llm(self):
        return self._llm


class RawLLM:
    """Simple LLM mock with bind()/invoke(); tracks its own temperature."""

    def __init__(self, response_text="relevant", exc=None):
        self.temperature = 1.0  # sentinel — must never be mutated by library code
        self.bound_calls = []
        self._resp = MockResp(response_text)
        self._exc = exc

    def bind(self, **kwargs):
        self.bound_calls.append(kwargs)
        clone = RawLLM.__new__(RawLLM)
        clone.__dict__.update({k: v for k, v in self.__dict__.items()
                               if k != "bound_calls"})
        clone.bound_calls = []
        return clone

    def invoke(self, prompt):
        if self._exc:
            raise self._exc
        return self._resp


def _grounding_checker_with_verdict(verdict):
    """Build a ContextGroundingChecker whose structured path returns *verdict*."""
    llm = MagicMock()
    chain = MagicMock()
    chain.return_value = verdict      # direct-call path (RunnableCallable)
    chain.invoke.return_value = verdict
    llm.bind.return_value = llm
    llm.with_structured_output.return_value = chain
    return ContextGroundingChecker(llm_provider=MockProvider(llm)), llm


def _grounding_checker_fallback(response_text, exc=None):
    """Build a checker whose structured-output setup fails → keyword fallback."""
    llm = RawLLM(response_text=response_text, exc=exc)

    def _raise_wso(_schema):
        raise AttributeError("no structured output support")

    llm.with_structured_output = _raise_wso
    return ContextGroundingChecker(llm_provider=MockProvider(llm)), llm


class MockEmbeddingProvider:
    """Returns vec_a on first embed_query call, vec_b on the second."""

    def __init__(self, vec_a, vec_b=None, exc=None):
        self._a, self._b, self._exc = vec_a, vec_b, exc
        self._calls = 0

    def embed_query(self, text):
        if self._exc:
            raise self._exc
        self._calls += 1
        return self._a if self._calls == 1 else self._b


# ---------------------------------------------------------------------------
# HallucinationResult model
# ---------------------------------------------------------------------------
def test_result_validates_confidence_range():
    with pytest.raises(Exception):
        HallucinationResult(is_hallucination=True, confidence=1.5,
                            cleaned_answer="x", reasoning="r")


def test_result_fields():
    r = HallucinationResult(is_hallucination=False, confidence=0.8,
                            cleaned_answer="ans", reasoning="why")
    assert r.is_hallucination is False and r.confidence == 0.8


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def test_factory_unknown_method():
    with pytest.raises(ValueError, match="Unknown hallucination checker"):
        create_hallucination_checker(method="bogus")


def test_factory_empty_method():
    with pytest.raises(ValueError, match="non-empty string"):
        create_hallucination_checker(method="")


def test_factory_aliases_resolve():
    assert callable(create_hallucination_checker)  # smoke
    # Alias groups documented in docstring resolve to the four strategies.
    from HallucinationsCheck.factory import AVAILABLE_METHODS
    assert AVAILABLE_METHODS == ("context_grounding", "nli",
                                 "semantic_similarity", "text_cleaner")


# ---------------------------------------------------------------------------
# ContextGroundingChecker — structured path
# ---------------------------------------------------------------------------
def test_grounding_structured_not_grounded():
    verdict = GroundingVerdict(is_grounded=False, confidence=0.9, reasoning="fabricated")
    checker, llm = _grounding_checker_with_verdict(verdict)
    result = checker.check(answer="ans", context="ctx")
    assert result.is_hallucination is True
    assert result.confidence == 0.9
    llm.bind.assert_called_once_with(temperature=0.0)


def test_grounding_structured_grounded():
    verdict = GroundingVerdict(is_grounded=True, confidence=0.95, reasoning="ok")
    checker, _ = _grounding_checker_with_verdict(verdict)
    result = checker.check(answer="ans", context="ctx")
    assert result.is_hallucination is False


def test_grounding_no_temperature_mutation():
    """P1 regression: shared LLM temperature must never be mutated."""
    llm = MagicMock()
    chain = MagicMock()
    chain.return_value = GroundingVerdict(is_grounded=True, confidence=1.0, reasoning="ok")
    chain.invoke.return_value = chain.return_value
    llm.bind.return_value = llm
    llm.with_structured_output.return_value = chain
    raw = RawLLM()  # has temperature attribute
    llm.get_llm = lambda: raw
    ContextGroundingChecker(llm_provider=MockProvider(llm))
    assert raw.temperature == 1.0  # unchanged


# ---------------------------------------------------------------------------
# ContextGroundingChecker — fallback keyword path
# ---------------------------------------------------------------------------
def test_grounding_fallback_yes_is_grounded():
    checker, _ = _grounding_checker_fallback("yes fully grounded")
    result = checker.check(answer="ans", context="ctx")
    assert result.is_hallucination is False


def test_grounding_fallback_no_hallucinated():
    checker, _ = _grounding_checker_fallback("no, contains hallucinations")
    result = checker.check(answer="ans", context="ctx")
    assert result.is_hallucination is True


def test_grounding_fallback_failure_raises():
    """On LLM failure the fallback raises instead of faking a 'grounded' verdict."""
    checker, _ = _grounding_checker_fallback("irrelevant", exc=RuntimeError("down"))
    with pytest.raises(HallucinationCheckError) as excinfo:
        checker.check(answer="ans", context="ctx")
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_grounding_empty_context():
    checker, _ = _grounding_checker_fallback("yes")
    result = checker.check(answer="ans", context="")
    assert result.is_hallucination is False


# ---------------------------------------------------------------------------
# SemanticSimilarityChecker
# ---------------------------------------------------------------------------
def test_similarity_consistent_answer():
    emb = MockEmbeddingProvider([1.0, 0.0], [1.0, 0.0])
    c = SemanticSimilarityChecker(embedding_provider=emb, similarity_threshold=0.3)
    r = c.check(answer="short answer text here", context="long context " * 100)
    assert r.is_hallucination is False


def test_similarity_distant_answer():
    emb = MockEmbeddingProvider([1.0, 0.0], [0.0, 1.0])
    c = SemanticSimilarityChecker(embedding_provider=emb, similarity_threshold=0.3)
    r = c.check(answer="a" * 600, context="b" * 600)
    assert r.is_hallucination is True


def test_similarity_empty_context():
    emb = MockEmbeddingProvider([1.0], [1.0])
    c = SemanticSimilarityChecker(embedding_provider=emb)
    r = c.check(answer="ans", context="")
    assert r.is_hallucination is False and r.confidence == 0.0


def test_similarity_embedding_failure():
    emb = MockEmbeddingProvider(None, exc=RuntimeError("embed down"))
    c = SemanticSimilarityChecker(embedding_provider=emb)
    with pytest.raises(HallucinationCheckError) as excinfo:
        c.check(answer="ans", context="ctx")
    assert isinstance(excinfo.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# TextCleanerChecker
# ---------------------------------------------------------------------------
class MockCleanerPM:
    def __init__(self):
        self.language = "ar"
        self.prompts = {"hallucination_check_prompt": "Clean: {answer}"}

    def get_prompt(self, key):
        return self.prompts[key]


def test_text_cleaner_success():
    llm = RawLLM(response_text="  dirty\x08 answer  ")
    c = TextCleanerChecker(llm_provider=MockProvider(llm), prompt_manager=MockCleanerPM())
    r = c.check(answer="original", context="ctx")
    assert r.cleaned_answer == "dirty answer"
    assert r.is_hallucination is False


def test_text_cleaner_failure_returns_original():
    llm = RawLLM(exc=RuntimeError("down"))
    c = TextCleanerChecker(llm_provider=MockProvider(llm), prompt_manager=MockCleanerPM())
    r = c.check(answer="  original text ", context="ctx")
    assert r.cleaned_answer == "original text"


def test_text_cleaner_requires_llm_provider():
    with pytest.raises(ValueError, match="llm_provider is required"):
        TextCleanerChecker(llm_provider=None)


def test_text_cleaner_check_answer_backward_compat():
    llm = RawLLM("cleaned output")
    c = TextCleanerChecker(llm_provider=MockProvider(llm), prompt_manager=MockCleanerPM())
    assert c.check_answer("any") == "cleaned output"


# ---------------------------------------------------------------------------
# NLIChecker — requires faked sentence_transformers
# ---------------------------------------------------------------------------
def _install_fake_cross_encoder(monkeypatch, scores, label2id=None):
    st = types.ModuleType("sentence_transformers")

    class FakeModel:
        def __init__(self, model_name=None):
            cfg = types.SimpleNamespace(label2id=label2id) if label2id is not None \
                else types.SimpleNamespace()
            self.model = types.SimpleNamespace(config=cfg)

        def predict(self, pairs, apply_softmax=True):
            return scores

    st.CrossEncoder = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)


def _make_nli(monkeypatch, scores, label2id=None):
    _install_fake_cross_encoder(monkeypatch, scores, label2id)
    from HallucinationsCheck.nli import NLIChecker
    ctx = ("First sentence about the law. Second sentence adds detail. "
           "Third sentence concludes the argument. ") * 3
    return NLIChecker(), ctx


def test_nli_entailment_not_hallucination(monkeypatch):
    checker, ctx = _make_nli(monkeypatch, [[0.05, 0.9, 0.05]] * 3)
    r = checker.check(answer="ans", context=ctx)
    assert r.is_hallucination is False
    assert r.confidence == pytest.approx(0.9)


def test_nli_contradiction_hallucination(monkeypatch):
    checker, ctx = _make_nli(monkeypatch, [[0.85, 0.1, 0.05]] * 3)
    r = checker.check(answer="ans", context=ctx)
    assert r.is_hallucination is True


def test_nli_neutral_fail_safe(monkeypatch):
    checker, ctx = _make_nli(monkeypatch, [[0.2, 0.3, 0.5]] * 3)
    assert checker.check(answer="ans", context=ctx).is_hallucination is False


def test_nli_empty_context_skipped(monkeypatch):
    checker, _ = _make_nli(monkeypatch, [[0.0, 0.0, 1.0]])
    r = checker.check(answer="ans", context="")
    assert r.is_hallucination is False and r.confidence == 0.0


def test_nli_scoring_failure_raises(monkeypatch):
    checker, ctx = _make_nli(monkeypatch, None)
    def _broken(*a, **k):
        raise RuntimeError("gpu down")
    checker._model.predict = _broken
    with pytest.raises(HallucinationCheckError) as excinfo:
        checker.check(answer="ans", context=ctx)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_nli_label_mapping_from_config(monkeypatch):
    """Non-DeBERTa label order resolved via model.config.label2id."""
    # label2id: entailment=0 (swapped vs DeBERTa where entailment=1)
    scores = [[0.9, 0.05, 0.05]] * 3
    checker, ctx = _make_nli(monkeypatch, scores,
                             label2id={"entailment": 0, "contradiction": 1, "neutral": 2})
    r = checker.check(answer="ans", context=ctx)
    assert r.is_hallucination is False
    assert r.confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Wrapper + regressions
# ---------------------------------------------------------------------------
def test_wrapper_optional_llm_for_similarity():
    emb = MockEmbeddingProvider([1.0], [1.0])
    wrapper = create_hallucination_checker(
        method="semantic_similarity", embedding_provider=emb)
    assert wrapper.name == "semantic_similarity"


def test_wrapper_check_and_check_answer():
    verdict = GroundingVerdict(is_grounded=True, confidence=1.0, reasoning="ok")
    checker, _ = _grounding_checker_with_verdict(verdict)
    structured = checker.check(answer="a", context="c")
    assert isinstance(structured, HallucinationResult)
    assert checker.check_answer("a", context="c") == "a"


def test_no_basicconfig_regression():
    root_before = list(logging.getLogger().handlers)
    verdict = GroundingVerdict(is_grounded=True, confidence=1.0, reasoning="ok")
    _grounding_checker_with_verdict(verdict)
    _grounding_checker_fallback("yes")
    emb = MockEmbeddingProvider([1.0], [1.0])
    SemanticSimilarityChecker(embedding_provider=emb)
    TextCleanerChecker(llm_provider=MockProvider(RawLLM()), prompt_manager=MockCleanerPM())
    assert list(logging.getLogger().handlers) == root_before


if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))