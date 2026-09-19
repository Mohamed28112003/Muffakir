"""
Test suite for the Embedding module (pytest).

No optional SDKs (sentence-transformers, langchain-openai, langchain-cohere)
are required: they are mocked via monkeypatching sys.modules. This validates
the two-layer cache (in-memory LRU + disk), batching, length validation,
factory dispatch, API-key fail-fast, and picklability.
"""

import logging
import os
import pickle
import sys
import types
from pathlib import Path

import pytest

# Make the library importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Embedding.base import BaseEmbeddingProvider
from Embedding.factory import create_embedding_provider, available_providers


# ---------------------------------------------------------------------------
# Fake provider (no SDK needed) for base/contract tests
# ---------------------------------------------------------------------------
class _FakeProvider(BaseEmbeddingProvider):
    """Deterministic fake provider: embedding = [len(text)]."""

    def __init__(self, cache_dir, batch_size=2):
        super().__init__(
            model_name="fake_model",
            provider_name="fake",
            cache_dir=cache_dir,
            batch_size=batch_size,
        )
        self.raw_call_count = 0

    def _embed_documents_raw(self, texts):
        self.raw_call_count += 1
        return [[float(len(t))] for t in texts]

    def _embed_query_raw(self, text):
        self.raw_call_count += 1
        return [float(len(text))]


class _FakeEmb:
    """Picklable fake LangChain Embeddings for the custom-wrapper test."""
    def embed_documents(self, texts):
        return [[1.0] for _ in texts]
    def embed_query(self, text):
        return [1.0]


# ---------------------------------------------------------------------------
# Cache key & disk cache
# ---------------------------------------------------------------------------
def test_cache_key_is_sha256(tmp_path):
    p = _FakeProvider(str(tmp_path / "cache"))
    k1 = p._get_cache_key("hello")
    k2 = p._get_cache_key("hello")
    assert k1 == k2
    assert len(k1) == 64  # sha256 hex; md5 would be 32


def test_cache_key_differs_per_provider_model(tmp_path):
    a = _FakeProvider(str(tmp_path / "a"))
    b = _FakeProvider(str(tmp_path / "b"))
    b.provider_name = "other"
    b.model_name = "other_model"
    assert a._get_cache_key("x") != b._get_cache_key("x")


def test_disk_cache_roundtrip(tmp_path):
    p = _FakeProvider(str(tmp_path / "cache"))
    p._save_to_cache("hello", [1.0, 2.0])
    assert p._check_cache("hello") == [1.0, 2.0]


# ---------------------------------------------------------------------------
# In-memory LRU cache
# ---------------------------------------------------------------------------
def test_memory_cache_avoids_raw_call(tmp_path):
    p = _FakeProvider(str(tmp_path / "cache"))
    p.embed_single("hello")
    assert p.raw_call_count == 1
    p.embed_single("hello")  # memory hit
    assert p.raw_call_count == 1


def test_memory_cache_evicts_oldest(tmp_path):
    p = _FakeProvider(str(tmp_path / "cache"))
    p._MEMORY_CACHE_SIZE = 3
    keys = [p._get_cache_key(t) for t in ["a", "b", "c", "d"]]
    for t in ["a", "b", "c", "d"]:
        p.embed_single(t)
    assert keys[0] not in p._memory_cache  # "a" evicted
    assert keys[3] in p._memory_cache      # "d" present
    assert len(p._memory_cache) == 3


# ---------------------------------------------------------------------------
# embed() batching, partial cache, length validation
# ---------------------------------------------------------------------------
def test_partial_cache_only_embeds_uncached(tmp_path):
    p = _FakeProvider(str(tmp_path / "cache"), batch_size=10)
    p.embed_single("cached")  # pre-populate
    assert p.raw_call_count == 1
    results = p.embed(["cached", "new1", "new2"])
    assert results[0] == [6.0]  # len("cached") = 6
    assert results[1] == [4.0] and results[2] == [4.0]
    assert p.raw_call_count == 2  # only one raw batch call


def test_embed_length_validation(tmp_path):
    p = _FakeProvider(str(tmp_path / "cache"))
    p._embed_documents_raw = lambda texts: [[1.0]]  # wrong count
    with pytest.raises(RuntimeError, match="returned 1 embeddings for 2 texts"):
        p.embed(["a", "b"])


def test_embed_empty_list(tmp_path):
    p = _FakeProvider(str(tmp_path / "cache"))
    assert p.embed([]) == []


def test_embed_none_raises(tmp_path):
    p = _FakeProvider(str(tmp_path / "cache"))
    with pytest.raises(ValueError):
        p.embed(None)


# ---------------------------------------------------------------------------
# LangChain interface delegation
# ---------------------------------------------------------------------------
def test_embed_documents_delegates(tmp_path):
    p = _FakeProvider(str(tmp_path / "cache"))
    assert p.embed_documents(["x", "yy"]) == [[1.0], [2.0]]


def test_embed_query_delegates(tmp_path):
    p = _FakeProvider(str(tmp_path / "cache"))
    assert p.embed_query("hello") == [5.0]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def test_factory_unknown_provider(tmp_path):
    with pytest.raises(ValueError):
        create_embedding_provider("not_a_provider", cache_dir=str(tmp_path / "c"))


def test_factory_empty_provider(tmp_path):
    with pytest.raises(ValueError):
        create_embedding_provider("", cache_dir=str(tmp_path / "c"))


def test_available_providers():
    s = available_providers()
    for n in ("sentence_transformers", "openai", "cohere"):
        assert n in s


def test_custom_wrapper_is_picklable(tmp_path):
    p = create_embedding_provider("custom", custom_embeddings=_FakeEmb(), cache_dir=str(tmp_path / "c"))
    restored = pickle.loads(pickle.dumps(p))
    assert restored.embed_query("x") == [1.0]


# ---------------------------------------------------------------------------
# SentenceTransformer provider (mocked)
# ---------------------------------------------------------------------------
def _install_fake_st(monkeypatch, capture_device=None):
    st = types.ModuleType("sentence_transformers")

    class FakeST:
        def __init__(self, name, device=None):
            self.name = name
            if capture_device is not None:
                capture_device["device"] = device

        def encode(self, texts, **kw):
            if isinstance(texts, str):
                return [float(len(texts))]
            return [[float(len(t))] for t in texts]

    st.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)


def _install_fake_torch(monkeypatch, cuda_available):
    torch_module = types.ModuleType("torch")
    cuda_module = types.ModuleType("torch.cuda")
    cuda_module.is_available = lambda: cuda_available
    torch_module.cuda = cuda_module

    class _FakeSigmoid:
        pass

    nn_module = types.ModuleType("torch.nn")
    nn_module.Sigmoid = _FakeSigmoid
    torch_module.nn = nn_module

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.cuda", cuda_module)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_module)


def test_sentence_transformer_provider(tmp_path, monkeypatch):
    _install_fake_st(monkeypatch)
    from Embedding.sentence_transformer import SentenceTransformerEmbeddingProvider
    p = SentenceTransformerEmbeddingProvider(cache_dir=str(tmp_path / "c"))
    assert p.embed_query("hello") == [5.0]
    assert p.embed_documents(["a", "bb"]) == [[1.0], [2.0]]


# ---------------------------------------------------------------------------
# Device resolution (GPU/CPU selection)
# ---------------------------------------------------------------------------
def test_resolve_device_auto_picks_cuda_when_available(monkeypatch):
    _install_fake_torch(monkeypatch, cuda_available=True)
    from Muffakir.device import resolve_device
    assert resolve_device("auto") == "cuda"


def test_resolve_device_auto_picks_cpu_when_unavailable(monkeypatch):
    _install_fake_torch(monkeypatch, cuda_available=False)
    from Muffakir.device import resolve_device
    assert resolve_device("auto") == "cpu"


def test_resolve_device_explicit_value_passes_through():
    from Muffakir.device import resolve_device
    assert resolve_device("cuda:1") == "cuda:1"
    assert resolve_device("cpu") == "cpu"


def test_sentence_transformer_embedding_provider_passes_resolved_device(monkeypatch):
    captured = {}
    _install_fake_st(monkeypatch, capture_device=captured)

    # resolve_device is imported locally inside __init__ (to avoid a circular
    # import with the Muffakir package's eager __init__ chain), so patch it at
    # its source rather than on this module's namespace.
    monkeypatch.setattr("Muffakir.device.resolve_device", lambda device: "cpu-resolved")

    from Embedding.sentence_transformer import SentenceTransformerEmbeddingProvider
    SentenceTransformerEmbeddingProvider(model_name="test-model", device="auto")
    assert captured["device"] == "cpu-resolved"


def test_sentence_transformer_missing_pkg(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    from Embedding.sentence_transformer import SentenceTransformerEmbeddingProvider
    with pytest.raises(ImportError):
        SentenceTransformerEmbeddingProvider(cache_dir=str(tmp_path / "c"))


# ---------------------------------------------------------------------------
# OpenAI / Cohere fail-fast on missing API key
# ---------------------------------------------------------------------------
def test_openai_fail_fast_no_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from Embedding.openai import OpenAIEmbeddingProvider
    with pytest.raises(ValueError):
        OpenAIEmbeddingProvider(cache_dir=str(tmp_path / "c"))


def test_openai_no_silent_model_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake = types.ModuleType("langchain_openai")

    class FakeOE:
        def __init__(self, **kw):
            self.kw = kw

    fake.OpenAIEmbeddings = FakeOE
    monkeypatch.setitem(sys.modules, "langchain_openai", fake)
    from Embedding.openai import OpenAIEmbeddingProvider
    p = OpenAIEmbeddingProvider(model_name="my-custom-model", cache_dir=str(tmp_path / "c"))
    assert p.model_name == "my-custom-model"


def test_cohere_fail_fast_no_key(monkeypatch, tmp_path):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    from Embedding.cohere import CohereEmbeddingProvider
    with pytest.raises(ValueError):
        CohereEmbeddingProvider(cache_dir=str(tmp_path / "c"))


# ---------------------------------------------------------------------------
# Regression: no basicConfig on instantiation
# ---------------------------------------------------------------------------
def test_no_basicconfig_regression(monkeypatch, tmp_path):
    root_before = list(logging.getLogger().handlers)
    _install_fake_st(monkeypatch)
    from Embedding.sentence_transformer import SentenceTransformerEmbeddingProvider
    SentenceTransformerEmbeddingProvider(cache_dir=str(tmp_path / "c"))
    assert list(logging.getLogger().handlers) == root_before


# ---------------------------------------------------------------------------
# Query-embedding timing capture (EmbeddingTimingTracker)
# ---------------------------------------------------------------------------
def test_embed_query_records_timing(tmp_path):
    p = _FakeProvider(cache_dir=str(tmp_path))
    p.embed_query("hello")
    totals = p.timing.get_totals()
    assert totals["count"] == 1
    assert totals["total_ms"] >= 0.0


def test_embed_query_timing_accumulates_flat_total(tmp_path):
    p = _FakeProvider(cache_dir=str(tmp_path))
    p.embed_query("a")
    p.embed_query("bb")
    p.embed_query("ccc")
    assert p.timing.get_totals()["count"] == 3


def test_embed_query_timing_bucketed_by_current_sample(tmp_path):
    from Trace.context import set_current_sample, clear_current_sample

    p = _FakeProvider(cache_dir=str(tmp_path))
    set_current_sample(5, 2)
    try:
        p.embed_query("hello")
    finally:
        clear_current_sample()

    per_sample = p.timing.get_per_sample_totals()
    assert (5, 2) in per_sample
    assert per_sample[(5, 2)]["count"] == 1
    assert p.timing.get_sample_total_ms(5, 2) >= 0.0


def test_embed_query_timing_untracked_when_no_sample_context(tmp_path):
    from Trace.context import clear_current_sample

    clear_current_sample()
    p = _FakeProvider(cache_dir=str(tmp_path))
    p.embed_query("hello")
    # No (trial, sample) tag was set -> lands in the untracked bucket, not a
    # real (trial_id, sample_index) key.
    assert (5, 2) not in p.timing.get_per_sample_totals()
    assert p.timing.get_totals()["count"] == 1


def test_embed_query_timing_reset_clears_everything(tmp_path):
    p = _FakeProvider(cache_dir=str(tmp_path))
    p.embed_query("hello")
    p.timing.reset()
    assert p.timing.get_totals() == {"total_ms": 0.0, "count": 0}
    assert p.timing.get_per_sample_totals() == {}


def test_pickle_resets_timing_but_keeps_working(tmp_path):
    """Timing is process-local; after unpickling it must be a fresh, usable
    tracker (not a shared lock crossing a process boundary)."""
    p = _FakeProvider(cache_dir=str(tmp_path))
    p.embed_query("hello")
    assert p.timing.get_totals()["count"] == 1

    restored = pickle.loads(pickle.dumps(p))
    assert restored.timing.get_totals() == {"total_ms": 0.0, "count": 0}
    restored.embed_query("world")
    assert restored.timing.get_totals()["count"] == 1


if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))