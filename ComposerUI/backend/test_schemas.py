"""
Tests for ComposerUI backend schemas.
"""

import pytest
from Composer.config_space import ConfigSpace
from ComposerUI.backend.schemas import (
    ChunkingEntry,
    CreateRunRequest,
    CustomPricingEntry,
    GenerateDatasetRequest,
    GenerateDatasetResponse,
    LLMEntry,
    SearchSpaceDict,
    TrainTestSplitConfig,
)
from pydantic import ValidationError


def test_search_space_dict_flat_stages():
    ssd = SearchSpaceDict(
        query_expansion=["none", "multi_query"],
        retrieval=["similarity_search"],
        reranking=["none"],
        k=[3, 5],
    )
    d = ssd.to_composer_dict()
    assert d["query_expansion"] == ["none", "multi_query"]
    assert d["retrieval"] == ["similarity_search"]
    assert d["reranking"] == ["none"]
    assert d["k"] == [3, 5]
    assert "llm" not in d
    assert "chunking" not in d

    # Verify ConfigSpace accepts this dictionary without error
    cs = ConfigSpace(d)
    assert cs.total_combinations == 2 * 1 * 1 * 2


def test_search_space_dict_exposes_reranking_model_dimension():
    search = SearchSpaceDict(
        reranking=["cross_encoder", "pointwise", "none"],
        reranking_model=["BAAI/bge-reranker-base", " custom/reranker "],
    )
    data = search.to_composer_dict()
    space = ConfigSpace(data)

    assert data["reranking_model"] == [
        "BAAI/bge-reranker-base", "custom/reranker",
    ]
    assert space.search_space["reranking_model"] == [
        "BAAI/bge-reranker-base", "custom/reranker",
    ]
    assert space.total_combinations == 5


@pytest.mark.parametrize(
    "models",
    [[""], ["org/model", " org/model "]],
)
def test_search_space_dict_rejects_empty_and_duplicate_reranking_models(models):
    with pytest.raises(ValidationError, match="reranking_model"):
        SearchSpaceDict(reranking_model=models)


def test_root_reranking_model_is_trimmed_and_empty_is_rejected():
    request = CreateRunRequest(run_name="model", reranking_model=" org/model ")
    assert request.reranking_model == "org/model"
    with pytest.raises(ValidationError, match="reranking_model"):
        CreateRunRequest(run_name="model", reranking_model="   ")


def test_search_space_dict_compound_stages():
    ssd = SearchSpaceDict(
        llm=[
            LLMEntry(provider="openai", model="gpt-4o-mini"),
            LLMEntry(provider="anthropic", model="claude-3-haiku"),
        ],
        chunking=[
            ChunkingEntry(method="recursive", size=600, overlap=100),
            ChunkingEntry(method="token", size=400, overlap=50),
        ],
    )
    d = ssd.to_composer_dict()
    assert len(d["llm"]) == 2
    assert d["llm"][0] == {"provider": "openai", "model": "gpt-4o-mini"}
    assert len(d["chunking"]) == 2
    assert d["chunking"][0] == {"method": "recursive", "size": 600, "overlap": 100}

    # Verify ConfigSpace accepts compound stages
    cs = ConfigSpace(d)
    assert cs.total_combinations == 2 * 2


def test_create_run_request_valid():
    req_data = {
        "run_name": "test-run",
        "documents_path": "./data",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "sk-test",
        "search_space": {
            "query_expansion": ["none"],
            "k": [5],
        },
    }
    req = CreateRunRequest(**req_data)
    assert req.run_name == "test-run"
    assert req.llm_provider == "openai"
    assert req.api_key == "sk-test"
    assert req.search_space.k == [5]
    # New fields default to the "auto" (unchanged, pre-existing) behavior.
    assert req.eval_dataset_mode == "auto"
    assert req.dataset_llm_provider is None
    assert req.eval_dataset_path is None
    assert req.document_parser is None
    assert req.use_ocr is False
    assert req.train_test_split is None
    assert req.custom_pricing == []


def test_custom_pricing_normalizes_and_converts_per_million_rates():
    req = CreateRunRequest(
        run_name="priced",
        custom_pricing=[
            CustomPricingEntry(
                provider=" OpenAI ",
                model=" gpt-4o-mini ",
                input_usd_per_million_tokens=0.15,
                output_usd_per_million_tokens=0,
            )
        ],
    )
    assert req.custom_pricing[0].provider == "openai"
    assert req.custom_pricing[0].model == "gpt-4o-mini"
    pricing = req.custom_pricing_per_token()["openai/gpt-4o-mini"]
    assert pricing["input_cost_per_token"] == pytest.approx(0.15 / 1_000_000)
    assert pricing["output_cost_per_token"] == 0.0


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_custom_pricing_rejects_invalid_rates(value):
    with pytest.raises(ValidationError):
        CustomPricingEntry(
            provider="openai",
            model="gpt-4o-mini",
            input_usd_per_million_tokens=value,
            output_usd_per_million_tokens=1,
        )


def test_custom_pricing_rejects_duplicate_provider_model_pairs():
    with pytest.raises(ValidationError, match="duplicate provider/model"):
        CreateRunRequest(
            run_name="duplicate",
            custom_pricing=[
                {
                    "provider": "OpenAI",
                    "model": "gpt-4o-mini",
                    "input_usd_per_million_tokens": 1,
                    "output_usd_per_million_tokens": 2,
                },
                {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "input_usd_per_million_tokens": 3,
                    "output_usd_per_million_tokens": 4,
                },
            ],
        )


def test_create_run_request_with_eval_dataset_and_split():
    req = CreateRunRequest(
        run_name="test-run",
        documents_path="./data",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        api_key="sk-test",
        eval_dataset_mode="existing",
        eval_dataset_path="./data/eval.csv",
        document_parser="docling",
        document_parser_config={"export_type": "markdown"},
        use_ocr=True,
        train_test_split=TrainTestSplitConfig(enabled=True, test_size=0.3),
    )
    assert req.eval_dataset_mode == "existing"
    assert req.eval_dataset_path == "./data/eval.csv"
    assert req.document_parser == "docling"
    assert req.use_ocr is True
    assert req.train_test_split.enabled is True
    assert req.train_test_split.test_size == 0.3
    assert req.train_test_split.random_state == 42  # default


def test_create_run_request_pipeline_mode_defaults_to_full_rag():
    req = CreateRunRequest(run_name="r1", documents_path="./data")
    assert req.pipeline_mode == "full_rag"
    assert req.llm_provider is None
    assert req.query_transform_llm_provider is None


def test_create_run_request_accepts_retrieval_only_mode_without_llm():
    req = CreateRunRequest(
        run_name="r1",
        documents_path="./data",
        pipeline_mode="retrieval_only",
        query_transform_llm_provider="openai",
        query_transform_llm_model="gpt-4o-mini",
    )
    assert req.pipeline_mode == "retrieval_only"
    assert req.query_transform_llm_model == "gpt-4o-mini"


def test_create_run_request_accepts_task_specific_llm_urls():
    req = CreateRunRequest(
        run_name="task-llm-run",
        eval_dataset_mode="existing",
        eval_dataset_path="./data/eval.csv",
        dataset_llm_provider="together",
        dataset_llm_model="dataset-model",
        dataset_base_url="https://dataset.example/v1",
        query_transform_base_url="https://query.example/v1",
    )

    assert req.dataset_llm_model == "dataset-model"
    assert req.query_transform_base_url == "https://query.example/v1"


def test_generate_dataset_request_response_shapes():
    req = GenerateDatasetRequest(
        documents_path="./data",
        llm_provider="together",
        llm_model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        api_key="sk-test",
    )
    assert req.language == "ar"
    assert req.use_ocr is False

    resp = GenerateDatasetResponse(dataset_path="./out.csv", preview=[{"question": "q?"}], row_count=1)
    assert resp.row_count == 1
    assert resp.preview[0]["question"] == "q?"
