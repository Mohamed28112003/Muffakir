"""Offline regression tests for portable settings and Composer variants."""
import json
from unittest.mock import MagicMock, patch

import pytest

from LLMProvider.LLMProvider import LLMProvider
from LLMProvider.parameters import (
    validate_parameters, resolve_parameters, provider_kwargs, SUPPORT,
    validate_config_parameters, parameter_capabilities,
)
from LLMProvider.test_llm_provider import _install_fake_langchain


@pytest.mark.parametrize("parameters", [
    {"temperature": True}, {"temperature": float("nan")}, {"temperature": float("inf")},
    {"max_tokens": 0}, {"max_tokens": 1.5}, {"max_tokens": "500"},
    {"top_p": 1.1}, {"top_k": -1}, {"seed": False}, {"stop": "STOP"},
    {"stop": [""]}, {"timeout_seconds": 0}, {"max_retries": -1},
    {"api_key": "secret"}, {"frequency_penalty": 3}, {"top_p": None},
])
def test_rejects_invalid_settings(parameters):
    with pytest.raises(ValueError):
        validate_parameters(parameters)


def test_zero_omission_bounds_and_defensive_copy():
    raw = {"temperature": None, "max_retries": 0, "seed": 0, "stop": ["نهاية", "\nEND"]}
    assert validate_parameters(raw, "openai") == raw
    copied = validate_parameters(raw)
    copied["stop"].append("other")
    assert raw["stop"] == ["نهاية", "\nEND"]
    with pytest.raises(ValueError, match="between"):
        validate_parameters({"temperature": 1.5}, "anthropic")
    with pytest.raises(ValueError, match="not supported"):
        validate_parameters({"top_k": 5}, "openai")


@pytest.mark.parametrize("provider", sorted(SUPPORT))
def test_every_adapter_forwards_explicit_settings(monkeypatch, provider):
    _install_fake_langchain(monkeypatch)
    params = {"temperature": None, "max_tokens": 321, "top_p": 0.8,
              "stop": ["END"], "timeout_seconds": 17}
    candidates = {"max_retries": 0, "seed": 0, "top_k": 5,
                  "frequency_penalty": 0.2, "presence_penalty": 0}
    params.update({k: v for k, v in candidates.items() if k in SUPPORT[provider]})
    extra = {"azure_endpoint": "https://example.invalid"} if provider == "azure_openai" else {}
    llm = LLMProvider(provider=provider, api_key="fixture", model="test", parameters=params, **extra)
    actual = type(llm.get_llm()).last_kwargs
    for key, value in provider_kwargs(provider, params).items():
        assert actual[key] == value
    assert "temperature" not in actual
    assert "timeout_seconds" not in actual
    assert "parameters" not in actual


def test_precedence_and_independent_roles():
    config = {"llm_temperature": 0.1, "llm_max_tokens": 600,
              "llm_parameters": {"temperature": 0.9, "seed": 7},
              "judge_llm_parameters": {"temperature": 0, "max_tokens": 100},
              "query_transform_llm_parameters": {"temperature": None}}
    assert resolve_parameters(config)["temperature"] == 0.9
    assert resolve_parameters(config, "judge") == {"temperature": 0, "max_tokens": 100}
    assert resolve_parameters(config, "query_transform") == {"temperature": None, "max_tokens": 600}
    assert resolve_parameters(config, "reranker")["seed"] == 7  # legacy reuse
    assert resolve_parameters({}) == {"temperature": 0, "max_tokens": 4096}


def test_variants_counts_same_index_and_fixed_roles():
    from Composer.config_space import ConfigSpace, trial_config_to_rag_config
    from Composer.index_key import compute_index_key
    variants = [{"provider": "openai", "model": "same", "parameters": {"temperature": t}}
                for t in (0, 0.3, 0.7)]
    space = ConfigSpace({"llm": variants, "k": [3, 5]})
    assert space.total_combinations == 6
    base = {"llm_provider": "openai", "llm_model": "base",
            "llm_parameters": {"max_tokens": 800},
            "judge_llm_parameters": {"temperature": 0, "max_tokens": 100},
            "query_transform_llm_parameters": {"temperature": 0.1, "max_tokens": 200},
            "reranker_llm_parameters": {"temperature": 0.2, "max_tokens": 300}}
    configs = [trial_config_to_rag_config(trial, base) for trial in space.generate_combinations()]
    assert len({compute_index_key(c) for c in configs}) == 1
    assert {c["llm_parameters"]["temperature"] for c in configs} == {0, 0.3, 0.7}
    for role in ("judge", "query_transform", "reranker"):
        assert all(c[f"{role}_llm_parameters"] == base[f"{role}_llm_parameters"] for c in configs)
    assert base["llm_parameters"] == {"max_tokens": 800}
    with pytest.raises(ValueError, match="Duplicate"):
        ConfigSpace({"llm": [variants[0], variants[0]]})
    with pytest.raises(ValueError, match="Duplicate"):
        validate_config_parameters(base, {"llm": [{"provider": "openai", "model": "x"},
            {"provider": "openai", "model": "x", "parameters": {"temperature": 0, "max_tokens": 800}}]})


def test_api_roundtrip_and_retrieval_rejection():
    from ComposerUI.backend.schemas import CreateRunRequest, GenerateDatasetRequest
    from ComposerUI.backend.provider_catalog import get_catalog
    request = CreateRunRequest(run_name="parameters", llm_provider="openai", llm_model="base",
        llm_parameters={"temperature": None}, judge_llm_parameters={"temperature": 0},
        search_space={"llm": [{"provider": "openai", "model": "model", "parameters": {"temperature": None}}]})
    assert request.search_space.to_composer_dict()["llm"][0]["parameters"] == {"temperature": None}
    assert request.parameter_config()["judge_llm_parameters"]["temperature"] == 0
    with pytest.raises(ValueError, match="Retrieval-only"):
        CreateRunRequest(run_name="invalid", pipeline_mode="retrieval_only", search_space=request.search_space)
    with pytest.raises(ValueError, match="not supported"):
        GenerateDatasetRequest(documents_path="docs", llm_provider="ollama", llm_model="x", llm_parameters={"max_retries": 1})
    assert get_catalog()["llm_parameter_capabilities"]["gemini"] == parameter_capabilities("gemini")


@pytest.mark.parametrize("adaptive", [False, True])
def test_rag_parameter_only_overrides_build_isolated_clients(adaptive):
    from Muffakir import MuffakirRAG
    calls = []
    def build(**kwargs):
        calls.append(kwargs)
        return MagicMock()
    with patch("Muffakir.Muffakir.LLMProvider", side_effect=build), \
         patch("Muffakir.Muffakir.RAGPipelineManager"), \
         patch("Muffakir.Muffakir.Reranker"), \
         patch("Muffakir.dependency_validation.validate_rag_dependencies"), \
         patch("WebSearch.create_web_search_provider", return_value=MagicMock()):
        rag = MuffakirRAG(config={"api_key": "fixture", "llm_provider": "openai", "llm_model": "same",
            "skip_document_ingestion": True, "query_transformer": True, "hallucination_check": False,
            "adaptive_web_search": adaptive, "search_provider": "tavily", "search_provider_config": {"api_key": "fake"},
            "reranking": True, "reranking_method": "llm", "llm_parameters": {"temperature": 0.8},
            "query_transform_llm_parameters": {"temperature": 0.2}, "reranker_llm_parameters": {"temperature": 0}},
            embedding_provider=MagicMock(), db_manager=MagicMock())
    assert [c["parameters"]["temperature"] for c in calls] == [0.8, 0.2, 0]
    assert all(c["model"] == "same" for c in calls)
    assert rag.query_transformer.llm_provider is not rag.llm_provider


def test_persistence_and_export_keep_runtime_settings(tmp_path, monkeypatch):
    from Composer.test_python_export import artifact, program
    from ComposerUI.backend.schemas import CreateRunRequest
    from ComposerUI.backend.run_manager import write_redacted_config
    req = CreateRunRequest(run_name="settings", llm_provider="openai", llm_model="test", api_key="fixture-secret",
                           llm_parameters={"temperature": None, "max_tokens": 800})
    write_redacted_config(tmp_path, req, "run", "now")
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["llm_parameters"] == req.llm_parameters
    assert "fixture-secret" not in json.dumps(saved)
    exported = artifact(llm_parameters=req.llm_parameters,
        query_transformer=True, query_transform_llm_parameters={"top_p": 0.8},
        reranking=True, reranking_method="llm", reranker_llm_parameters={"temperature": 0},
        judge_llm_parameters={"max_tokens": 999}, dataset_llm_parameters={"max_tokens": 888})
    config = program(tmp_path, monkeypatch, exported)["CONFIG"]
    assert config["llm_parameters"] == req.llm_parameters
    assert config["query_transform_llm_parameters"] == {"top_p": 0.8}
    assert config["reranker_llm_parameters"] == {"temperature": 0}
    assert "judge_llm_parameters" not in config
    assert "dataset_llm_parameters" not in config


def test_retrieval_parameter_only_reranker_override():
    from Muffakir import MuffakirRetrieval
    calls = []
    def build(**kwargs):
        calls.append(kwargs)
        return MagicMock()
    with patch("Muffakir.MuffakirRetrieval.LLMProvider", side_effect=build), \
         patch("Muffakir.MuffakirRetrieval.RetrieveMethods"), \
         patch("Muffakir.MuffakirRetrieval.Reranker"):
        MuffakirRetrieval({"query_transformer": True, "query_transform_llm_provider": "openai",
            "query_transform_llm_model": "query-model", "query_transform_api_key": "fake",
            "reranking": True, "reranking_method": "llm",
            "query_transform_llm_parameters": {"temperature": 0.4},
            "reranker_llm_parameters": {"temperature": 0}},
            embedding_provider=MagicMock(), db_manager=MagicMock())
    assert [c["parameters"]["temperature"] for c in calls] == [0.4, 0]
    assert all(c["model"] == "query-model" for c in calls)


def test_web_search_runtime_settings():
    from Muffakir import MuffakirSearch
    with patch("Muffakir.MuffakirSearch.LLMProvider") as provider, \
         patch("Muffakir.dependency_validation.validate_search_dependencies"), \
         patch.object(MuffakirSearch, "_setup_pipeline"):
        MuffakirSearch(config={"llm_provider": "openai", "llm_model": "search-model", "api_key": "fake",
            "search_provider": "tavily", "search_provider_config": {"api_key": "fake"},
            "llm_parameters": {"temperature": None, "max_tokens": 654, "top_p": 0.8}})
    assert provider.call_args.kwargs["parameters"] == {"temperature": None, "max_tokens": 654, "top_p": 0.8}


def test_standalone_synthetic_settings_and_retry_separation(tmp_path):
    from SyntheticData.pipeline import SyntheticDataPipeline
    with patch("SyntheticData.pipeline.LLMProvider") as provider, \
         patch("SyntheticData.pipeline.QAGenerator"), \
         patch("SyntheticData.pipeline.DatasetExporter"):
        pipeline = SyntheticDataPipeline(config={"data_dir": str(tmp_path), "llm_provider": "openai",
            "llm_model": "dataset", "api_key": "fake", "chunking": MagicMock(),
            "llm_parameters": {"temperature": 0.6, "max_retries": 0}, "max_retries": 5})
    assert provider.call_args.kwargs["parameters"] == {"temperature": 0.6, "max_retries": 0}
    assert pipeline.config.max_retries == 5


def test_legacy_variant_with_null_parameters():
    from Composer.config_space import trial_config_to_rag_config
    config = trial_config_to_rag_config({"llm": {"provider": "openai", "model": "legacy", "parameters": None}}, {})
    assert config["llm_model"] == "legacy"
    assert "llm_parameters" not in config
