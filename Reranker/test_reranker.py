"""
Unit tests for the Reranker module (pytest).

Currently covers LLMReranker's fallback-scoring crash path: a genuine LLM
call failure must raise a typed ProviderError, not silently degrade every
document to a fake, confident 0.0 relevance score. Also covers GPU/CPU
device selection for the local cross-encoder backends.
"""
import sys
import types
from unittest.mock import MagicMock

import pytest

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from Muffakir.exceptions import ConfigurationError, ProviderError
from PromptManager.PromptManager import MuffakirPrompt
from Reranker.llm import LLMReranker


def _make_reranker_with_no_structured_output():
    """LLMReranker whose LLM has no with_structured_output, forcing every
    score() call through the plain-text fallback path."""
    mock_llm_provider = MagicMock()
    mock_llm = MagicMock(spec=["invoke", "temperature"])
    mock_llm_provider.get_llm.return_value = mock_llm

    reranker = LLMReranker(
        llm_provider=mock_llm_provider,
        prompt_manager=MuffakirPrompt(language="ar"),
    )
    assert reranker._structured_chain is None
    return reranker, mock_llm


def test_llm_reranker_fallback_success_parses_score():
    reranker, mock_llm = _make_reranker_with_no_structured_output()
    mock_llm.invoke.return_value = MagicMock(content="0.8")

    results = reranker.score("query", [Document(page_content="doc")])
    assert results[0][1] == 0.8


def test_llm_reranker_fallback_crash_raises():
    reranker, mock_llm = _make_reranker_with_no_structured_output()
    mock_llm.invoke.side_effect = RuntimeError("LLM down")

    with pytest.raises(ProviderError) as excinfo:
        reranker.score("query", [Document(page_content="doc")])
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_llm_reranker_fallback_no_match_returns_zero():
    """Unparseable-but-successful LLM output stays a soft 0.0 (not a crash)."""
    reranker, mock_llm = _make_reranker_with_no_structured_output()
    mock_llm.invoke.return_value = MagicMock(content="I refuse to answer.")

    results = reranker.score("query", [Document(page_content="doc")])
    assert results[0][1] == 0.0


# ---------------------------------------------------------------------------
# Device selection (GPU/CPU) for the local cross-encoder backends
# ---------------------------------------------------------------------------
def _install_fake_sentence_transformers(monkeypatch, capture_device):
    st = types.ModuleType("sentence_transformers")

    class _FakeCrossEncoder:
        def __init__(self, model_name, default_activation_function=None, device=None):
            capture_device["device"] = device
            capture_device["model_name"] = model_name

    st.CrossEncoder = _FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)

    torch_module = types.ModuleType("torch")
    nn_module = types.ModuleType("torch.nn")

    class _FakeSigmoid:
        pass

    nn_module.Sigmoid = _FakeSigmoid
    torch_module.nn = nn_module
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_module)


def test_pointwise_reranker_passes_resolved_device(monkeypatch):
    captured = {}
    _install_fake_sentence_transformers(monkeypatch, captured)

    # resolve_device is imported locally inside __init__ (avoids a circular
    # import with Muffakir's eager __init__ chain) — patch at its source.
    monkeypatch.setattr("Muffakir.device.resolve_device", lambda device: "cpu-resolved")

    from Reranker.pointwise import PointwiseReranker
    PointwiseReranker(model_name="test-model", device="auto")
    assert captured["device"] == "cpu-resolved"
    assert captured["model_name"] == "test-model"


def test_cross_encoder_reranker_passes_resolved_device(monkeypatch):
    captured = {}
    st = types.ModuleType("sentence_transformers")

    class _FakeCrossEncoder:
        def __init__(self, model_name, device=None):
            captured["device"] = device
            captured["model_name"] = model_name

    st.CrossEncoder = _FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)

    monkeypatch.setattr("Muffakir.device.resolve_device", lambda device: "cpu-resolved")

    from Reranker.cross_encoder import CrossEncoderReranker
    CrossEncoderReranker(model_name="test-model", device="auto")
    assert captured["device"] == "cpu-resolved"
    assert captured["model_name"] == "test-model"


def test_cross_encoder_wraps_incompatible_hugging_face_model(monkeypatch):
    st = types.ModuleType("sentence_transformers")

    class _FailingCrossEncoder:
        def __init__(self, model_name, device=None):
            raise OSError("not a sequence-classification model")

    st.CrossEncoder = _FailingCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    monkeypatch.setattr("Muffakir.device.resolve_device", lambda device: "cpu")

    from Reranker.cross_encoder import CrossEncoderReranker
    with pytest.raises(ConfigurationError, match="bad/model"):
        CrossEncoderReranker(model_name="bad/model")


def test_pointwise_wraps_incompatible_hugging_face_model(monkeypatch):
    st = types.ModuleType("sentence_transformers")

    class _FailingCrossEncoder:
        def __init__(
            self, model_name, default_activation_function=None, device=None
        ):
            raise OSError("not a sequence-classification model")

    st.CrossEncoder = _FailingCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)

    torch_module = types.ModuleType("torch")
    nn_module = types.ModuleType("torch.nn")
    nn_module.Sigmoid = type("_FakeSigmoid", (), {})
    torch_module.nn = nn_module
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_module)
    monkeypatch.setattr("Muffakir.device.resolve_device", lambda device: "cpu")

    from Reranker.pointwise import PointwiseReranker
    with pytest.raises(ConfigurationError, match="bad/pointwise-model"):
        PointwiseReranker(model_name="bad/pointwise-model")


def test_create_reranker_forwards_device_to_pointwise(monkeypatch):
    captured = {}
    _install_fake_sentence_transformers(monkeypatch, captured)

    from Reranker.factory import create_reranker
    create_reranker(method="pointwise", device="cuda:0")
    assert captured["device"] == "cuda:0"


@pytest.mark.parametrize("method", ["cross_encoder", "pointwise"])
def test_reranker_wrapper_forwards_selected_local_model(monkeypatch, method):
    captured = {}
    _install_fake_sentence_transformers(monkeypatch, captured)
    monkeypatch.setattr("Muffakir.device.resolve_device", lambda device: device)

    from Reranker.Reranker import Reranker
    Reranker(
        reranking_method=method,
        cross_encoder_model_name="org/arbitrary-reranker",
        device="cpu",
    )

    assert captured["model_name"] == "org/arbitrary-reranker"


def test_semantic_reranker_reuses_embedding_provider_not_local_model_dimension():
    from Reranker.Reranker import Reranker

    embedding_provider = MagicMock()
    reranker = Reranker(
        embedding_provider=embedding_provider,
        model_name="org/selected-embedding",
        reranking_method="semantic_similarity",
        cross_encoder_model_name="org/local-reranker",
    )

    assert reranker._reranker.embedding_provider is embedding_provider


# ---------------------------------------------------------------------------
# Shared rerank_documents() helper function
# ---------------------------------------------------------------------------
def test_rerank_documents_calls_reranker_when_present():
    from Reranker.Reranker import rerank_documents

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = ["reranked_doc"]

    result = rerank_documents(mock_reranker, "query text", ["doc1", "doc2"])

    mock_reranker.rerank.assert_called_once_with("query text", ["doc1", "doc2"])
    assert result == ["reranked_doc"]


def test_rerank_documents_passthrough_when_reranker_is_none():
    from Reranker.Reranker import rerank_documents

    result = rerank_documents(None, "query text", ["doc1", "doc2"])
    assert result == ["doc1", "doc2"]


def test_reranker_registry_exposes_all_builtin_methods():
    from Reranker.factory import get_reranker_spec, list_reranker_specs

    names = {spec.name for spec in list_reranker_specs()}
    assert {
        "semantic_similarity",
        "bm25",
        "cross_encoder",
        "pointwise",
        "llm",
        "custom",
    }.issubset(names)
    assert get_reranker_spec("crossencoder").name == "cross_encoder"
    assert get_reranker_spec("remote").name == "custom"


def test_runtime_registered_reranker_can_be_created():
    from Reranker.base import BaseReranker
    from Reranker.factory import create_reranker, register_reranker

    class _RegisteredReranker(BaseReranker):
        @property
        def name(self):
            return "test_registered"

        def score(self, query, documents):
            return [(document, 1.0) for document in documents]

    register_reranker(
        "test_registered",
        lambda **context: _RegisteredReranker(),
        aliases=("test_alias",),
    )
    assert create_reranker("test_alias").name == "test_registered"


def test_remote_reranker_sends_contract_and_preserves_documents(monkeypatch):
    from Reranker.remote import RemoteReranker

    captured = {}
    response = MagicMock()
    response.json.return_value = {
        "results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.2},
        ]
    }
    requests_module = types.ModuleType("requests")

    def _post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return response

    requests_module.post = _post
    monkeypatch.setitem(sys.modules, "requests", requests_module)
    documents = [Document(page_content="first"), Document(page_content="second")]
    reranker = RemoteReranker(
        "https://example.test/rerank",
        api_key="secret",
        model="rerank-v1",
        timeout_seconds=12,
    )

    results = reranker.score("question", documents)

    assert [document for document, _score in results] == [documents[1], documents[0]]
    assert captured["json"] == {
        "query": "question",
        "documents": ["first", "second"],
        "top_n": 2,
        "model": "rerank-v1",
    }
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["timeout"] == 12


def test_remote_reranker_rejects_incomplete_scores():
    from Reranker.remote import RemoteReranker

    documents = [Document(page_content="one"), Document(page_content="two")]
    with pytest.raises(ProviderError, match="one score per document"):
        RemoteReranker._parse_scores(
            {"results": [{"index": 0, "score": 0.5}]},
            documents,
        )
