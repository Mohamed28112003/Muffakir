"""
Tests for ComposerUI provider catalog.
"""

from Evaluation.models import ALL_METRICS
from ComposerUI.backend.provider_catalog import get_catalog


def test_provider_catalog_contents():
    catalog = get_catalog()
    assert "providers" in catalog
    assert "chunking_presets" in catalog
    assert "chunking_methods" in catalog
    assert "embedding_models" in catalog
    assert "reranker_models" in catalog
    assert "vector_dbs" in catalog
    assert "parsers" in catalog
    assert "devices" in catalog
    assert "web_search_providers" in catalog
    assert "embedding_providers" in catalog
    assert "rerankers" in catalog
    assert "capabilities" in catalog
    assert "metrics" in catalog
    assert "default_search_space" in catalog

    provider_names = [p["name"] for p in catalog["providers"]]
    assert len(provider_names) == 10
    assert "openai" in provider_names
    assert "together" in provider_names
    assert "ollama" in provider_names
    for p in catalog["providers"]:
        assert "name" in p and "available" in p and "requires" in p
        assert p["extra"]
        assert p["install_command"] == f'pip install "Muffakir[{p["extra"]}]"'

    for preset in catalog["chunking_presets"]:
        assert "id" in preset
        assert "label" in preset
        assert "value" in preset
        val = preset["value"]
        assert "method" in val
        assert "size" in val
        assert "overlap" in val

    assert set(catalog["chunking_methods"]) == {
        "recursive", "character", "token", "contextual_recursive",
    }

    assert catalog["embedding_models"] == [
        "mohamed2811/Muffakir_Embedding",
        "mohamed2811/Muffakir_Embedding_V2",
    ]
    assert catalog["reranker_models"] == [
        "BAAI/bge-reranker-base",
        "BAAI/bge-reranker-v2-m3",
    ]

    vdb_names = [v["name"] for v in catalog["vector_dbs"]]
    assert vdb_names == ["chroma", "faiss", "qdrant", "milvus", "pinecone"]
    chroma_entry = next(v for v in catalog["vector_dbs"] if v["name"] == "chroma")
    assert chroma_entry["available"] is True

    parser_names = [p["name"] for p in catalog["parsers"]]
    assert parser_names == ["docling", "llama_parse", "azure"]
    for p in catalog["parsers"]:
        assert "fields" in p and isinstance(p["fields"], list)

    device_names = [d["name"] for d in catalog["devices"]]
    assert device_names == ["auto", "cpu", "cuda"]
    by_device = {d["name"]: d for d in catalog["devices"]}
    assert by_device["auto"]["available"] is True
    assert by_device["cpu"]["available"] is True

    web_search_names = [w["name"] for w in catalog["web_search_providers"]]
    assert web_search_names == ["firecrawl", "tavily", "serpapi"]
    for w in catalog["web_search_providers"]:
        assert "fields" in w and isinstance(w["fields"], list)

    reranker_names = {r["name"] for r in catalog["rerankers"]}
    assert {
        "none",
        "semantic_similarity",
        "bm25",
        "cross_encoder",
        "pointwise",
        "llm",
        "custom",
    }.issubset(reranker_names)
    custom_reranker = next(r for r in catalog["rerankers"] if r["name"] == "custom")
    assert custom_reranker["configuration"] == "remote"
    assert custom_reranker["available"] is True

    # Never drift from the real, single source of truth for valid metric names.
    assert catalog["metrics"]["all"] == list(ALL_METRICS)
    assert set(catalog["metrics"]["retrieval"]) == {"recall", "precision", "mrr", "ndcg"}
    assert set(catalog["metrics"]["generation"]) == {
        "faithfulness", "answer_correctness", "llm_judge_rating"
    }
    assert catalog["metrics"]["default"] == ["recall", "precision", "mrr"]


def test_availability_reflects_installed_packages(monkeypatch):
    import ComposerUI.backend.provider_catalog as pc

    def fake_is_importable(module_name: str) -> bool:
        return module_name == "langchain_anthropic"

    monkeypatch.setattr(pc, "_is_importable", fake_is_importable)

    catalog = pc.get_catalog()
    by_name = {p["name"]: p["available"] for p in catalog["providers"]}
    assert by_name["anthropic"] is True
    assert by_name["gemini"] is False
    assert by_name["ollama"] is False
    # Provider SDKs are now targeted extras, not unconditional core deps.
    assert by_name["openai"] is False
    assert by_name["groq"] is False


def test_device_cuda_availability_reflects_torch(monkeypatch):
    import ComposerUI.backend.provider_catalog as pc

    monkeypatch.setattr(pc, "_cuda_available", lambda: True)
    by_device = {d["name"]: d["available"] for d in pc.get_catalog()["devices"]}
    assert by_device["cuda"] is True

    monkeypatch.setattr(pc, "_cuda_available", lambda: False)
    by_device = {d["name"]: d["available"] for d in pc.get_catalog()["devices"]}
    assert by_device["cuda"] is False


def test_web_search_provider_availability_reflects_installed_packages(monkeypatch):
    import ComposerUI.backend.provider_catalog as pc

    def fake_is_importable(module_name: str) -> bool:
        return module_name == "langchain_tavily"

    monkeypatch.setattr(pc, "_is_importable", fake_is_importable)

    by_name = {w["name"]: w["available"] for w in pc.get_catalog()["web_search_providers"]}
    assert by_name["tavily"] is True
    assert by_name["firecrawl"] is False
    assert by_name["serpapi"] is False
