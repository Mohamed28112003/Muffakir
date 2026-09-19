"""Exercise exported programs, not only their source strings."""
import io
import json
import sys
import types
import zipfile

import pytest

from Composer.python_export import ExportError, export_trial


def trial_config(**overrides):
    return {"llm_provider": "openai", "llm_model": "chosen-model",
            "embedding_model": "test/embedding", "embedding_provider": "sentence_transformers",
            "vector_db_provider": "chroma", "chunking_method": "recursive",
            "chunk_size": 800, "chunk_overlap": 100, "k": 7,
            "retrieval_method": "hybrid", "language": "en", **overrides}


def artifact(**overrides):
    return export_trial("run-abc", {"trial_id": 7, "status": "success",
                                   "resolved_rag_config": trial_config(**overrides)})


def program(tmp_path, monkeypatch, exported=None):
    exported = exported or artifact()
    namespace = {"__name__": "export_test", "__file__": str(tmp_path / "rag_app.py")}
    monkeypatch.setitem(sys.modules, "dotenv", types.SimpleNamespace(load_dotenv=lambda *a, **k: None))
    for item in exported.environment:
        if item["required"]:
            monkeypatch.setenv(item["name"], "test-value")
    exec(compile(exported.code, "rag_app.py", "exec"), namespace)
    return namespace


def test_portable_selected_trial_zip_and_import(tmp_path, monkeypatch):
    exported = artifact(api_key="secret-main", data_dir="D:/private/corpus", db_path="D:/private/index",
                        collection_name="private_collection", judge_api_key="secret-judge",
                        judge_llm_model="judge-do-not-export", dataset_llm_model="dataset-do-not-export")
    ns = program(tmp_path, monkeypatch, exported)
    assert ns["CONFIG"]["k"] == 7
    assert ns["CONFIG"]["llm_model"] == "chosen-model"
    assert ns["EXPORT_METADATA"]["trial_id"] == 7
    data = json.dumps(exported.preview())
    for excluded in ("secret-main", "secret-judge", "D:/private", "private_collection", "judge-do-not-export", "dataset-do-not-export"):
        assert excluded not in data
    with zipfile.ZipFile(io.BytesIO(exported.zip_bytes())) as archive:
        assert len(archive.namelist()) == 5
        assert archive.read("muffakir-trial-7/rag_app.py").decode() == exported.code
        assert "secret-main" not in str([archive.read(name) for name in archive.namelist()])
    assert "bm25" in exported.requirements
    assert "datasets" not in exported.requirements


def test_missing_and_failed_records():
    with pytest.raises(ExportError, match="no saved resolved"):
        export_trial("r", {"trial_id": 1})
    with pytest.raises(ExportError, match="llm_model"):
        artifact(llm_model=None)
    exported = export_trial("r", {"trial_id": 2, "status": "failed", "resolved_config": trial_config()})
    assert "may reproduce" in exported.notices[0]


def test_prompts_and_separate_credentials(tmp_path, monkeypatch):
    prompt = 'سؤال {question}\nContext: {context}\n"quoted" and \\ paths'
    config = trial_config(query_transformer=True, query_transformer_strategy="hyde",
                          query_transform_llm_provider="groq", query_transform_llm_model="qt-model",
                          reranking=True, reranking_method="llm", reranker_llm_provider="anthropic",
                          reranker_llm_model="rerank-model", base_url="https://user:secret@private/api",
                          adaptive_web_search=True, search_provider="tavily",
                          search_provider_config={"api_key": "nested-secret"},
                          reranker_options={"headers": {"X-Custom-Auth": "header-secret"}})
    exported = export_trial("r", {"trial_id": 4, "resolved_rag_config": config},
                            {"resolved_prompts": {"generation": prompt, "QA": "dataset", "llm_judge_rating": "judge"}})
    ns = program(tmp_path, monkeypatch, exported)
    loaded, _ = ns["load_config"]()
    assert loaded["prompt_overrides"] == {"generation": prompt}
    assert loaded["query_transform_api_key"] == "test-value"
    assert loaded["reranker_llm_api_key"] == "test-value"
    assert loaded["search_provider_config"]["api_key"] == "test-value"
    for secret in ("nested-secret", "header-secret", "https://user:secret"):
        assert secret not in json.dumps(exported.preview())


@pytest.mark.parametrize("provider", ["chroma", "faiss", "qdrant", "milvus", "pinecone"])
def test_vector_paths_and_dependencies(provider, tmp_path, monkeypatch):
    exported = artifact(vector_db_provider=provider, vector_db_config={
        "path": "C:/original/path", "folder_path": "C:/original/faiss", "collection_name": "old-name",
        "index_name": "old-index", "connection_args": {"uri": "C:/original/milvus.db"}})
    ns = program(tmp_path, monkeypatch, exported)
    config, state = ns["load_config"]()
    assert config["vector_db_provider"] == provider
    assert provider in exported.requirements
    assert "C:/original" not in json.dumps(exported.preview())
    assert state == tmp_path / "index"
    assert config["vector_db_config"]["collection_name"] != "old-name"


@pytest.mark.parametrize("mode", ["full_rag", "retrieval_only", "web_search_only"])
def test_generated_facade_and_no_ingestion(mode, tmp_path, monkeypatch):
    exported = artifact(pipeline_mode="retrieval_only" if mode == "retrieval_only" else "full_rag",
                        retrieval_source="web_search_only" if mode == "web_search_only" else "vector_db",
                        search_provider="tavily")
    ns = program(tmp_path, monkeypatch, exported)
    called = []
    def facade(name):
        def create(*args, **kwargs):
            called.append((name, args, kwargs))
            return name
        return create
    import Muffakir
    for name in ("MuffakirRAG", "MuffakirRetrieval", "MuffakirSearch"):
        # Set module dictionary directly to avoid lazy-loading real model providers.
        monkeypatch.setitem(Muffakir.__dict__, name, facade(name))
    ns["create_components"] = lambda config: ("embedding", "database")
    if mode != "web_search_only":
        with pytest.raises(ValueError, match="--index first"):
            ns["build_pipeline"]()
        config, state = ns["load_config"]()
        state.mkdir()
        (state / "vectors").mkdir()
        (state / "ready.json").write_text(json.dumps({"signature": ns["index_signature"](config)}))
        (state / "chunks.json").write_text('[{"page_content": "corpus", "metadata": {"source": "test"}}]')
    result = ns["build_pipeline"]()
    expected = {"full_rag": "MuffakirRAG", "retrieval_only": "MuffakirRetrieval", "web_search_only": "MuffakirSearch"}[mode]
    assert result == expected
    assert len(called) == 1
    if mode != "web_search_only":
        assert called[0][1][0]["skip_document_ingestion"] is True
        assert called[0][2]["db_manager"].get_all_documents()[0].page_content == "corpus"
    if mode == "retrieval_only":
        assert "llm_model" not in ns["CONFIG"]
        assert "openai" not in exported.requirements


def test_index_once_and_changed_settings(tmp_path, monkeypatch):
    ns = program(tmp_path, monkeypatch)
    (tmp_path / "knowledge-base").mkdir()
    from langchain_core.documents import Document
    calls = []
    class Processor:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def process_all(self, **kwargs):
            calls.append(kwargs)
            return [Document(page_content="chunk", metadata={"source": "file.txt"})]
    monkeypatch.setitem(sys.modules, "TextProcessor.MuffakirChunking", types.SimpleNamespace(MuffakirChunking=lambda **kwargs: kwargs))
    monkeypatch.setitem(sys.modules, "TextProcessor.ChunkingAndProcessing", types.SimpleNamespace(ChunkingAndProcessing=Processor))
    ns["create_components"] = lambda config: (None, types.SimpleNamespace(add_documents=lambda docs: calls.append(docs)))
    ns["index_documents"]()
    assert (tmp_path / "index" / "ready.json").is_file()
    assert calls[0]["muffakir_chunking"]["chunker_config"] == {"size": 800, "overlap": 100}
    with pytest.raises(ValueError, match="already exists"):
        ns["index_documents"]()
    ns["CONFIG"]["chunk_size"] = 900
    with pytest.raises(ValueError, match="settings changed"):
        ns["build_pipeline"]()


def test_missing_environment_and_local_model(tmp_path, monkeypatch):
    ns = program(tmp_path, monkeypatch)
    monkeypatch.delenv("MUFFAKIR_API_KEY")
    with pytest.raises(ValueError, match="MUFFAKIR_API_KEY"):
        ns["load_config"]()
    local = artifact(llm_provider="ollama")
    assert next(item for item in local.environment if item["name"] == "MUFFAKIR_API_KEY")["default"] == "local"


def test_hugging_face_reranker_and_custom_endpoint():
    exported = artifact(reranking=True, reranking_method="cross_encoder", reranking_model="arbitrary/model")
    assert "arbitrary/model" in exported.code
    custom = artifact(reranking=True, reranking_method="custom", reranker_base_url="https://private/rerank")
    assert any(item["name"] == "MUFFAKIR_RERANKER_BASE_URL" and item["required"] for item in custom.environment)
    assert "https://private/rerank" not in custom.code


@pytest.mark.parametrize("provider", ["chroma", "faiss", "qdrant", "milvus", "pinecone"])
def test_component_arguments_match_the_export(provider, tmp_path, monkeypatch):
    ns = program(tmp_path, monkeypatch, artifact(vector_db_provider=provider, device="cuda"))
    calls = {}
    def embedding(**kwargs):
        calls["embedding"] = kwargs
        return "embedding"
    def database(provider, **kwargs):
        calls["database"] = (provider, kwargs)
        return "database"
    monkeypatch.setitem(sys.modules, "Embedding.EmbeddingProvider", types.SimpleNamespace(EmbeddingProvider=embedding))
    monkeypatch.setitem(sys.modules, "VectorDB", types.SimpleNamespace(create_vector_db=database))
    config, _ = ns["load_config"]()
    assert ns["create_components"](config) == ("embedding", "database")
    assert calls["embedding"]["model_name"] == "test/embedding"
    assert calls["embedding"]["device"] == "cuda"
    assert calls["database"][0] == provider
    assert calls["database"][1]["embedding_provider"] == "embedding"
    if provider == "faiss":
        assert calls["database"][1]["folder_path"] == str(tmp_path / "index" / "vectors")
    if provider == "qdrant":
        assert "path" not in calls["database"][1]
        assert calls["database"][1]["location"] == "test-value"


def test_remote_collection_is_checked_before_ingestion(tmp_path, monkeypatch):
    ns = program(tmp_path, monkeypatch, artifact(vector_db_provider="qdrant"))
    (tmp_path / "knowledge-base").mkdir()
    closed = []
    monkeypatch.setitem(sys.modules, "qdrant_client", types.SimpleNamespace(QdrantClient=lambda **kwargs:
        types.SimpleNamespace(collection_exists=lambda name: True, close=lambda: closed.append(True))))
    ns["create_components"] = lambda config: pytest.fail("Must not write to the existing collection")
    with pytest.raises(ValueError, match="Collection already exists"):
        ns["index_documents"]()
    assert closed == [True]
    assert not (tmp_path / "index" / "ready.json").exists()


def test_nonfinite_config_is_not_rendered_as_invalid_python():
    with pytest.raises(ExportError, match="portable Python"):
        artifact(llm_temperature=float("nan"))


def test_failed_search_result_retains_exportable_redacted_configuration():
    from Composer.search.grid_search import GridSearch
    from Composer.results.trial import TrialResult
    def fail(*args):
        raise ValueError("invalid provider configuration")
    result = GridSearch(n_jobs=1)._run_with_retry(
        fail, 12, {"k": 3}, {"base_config": trial_config(api_key="checkpoint-secret")})
    restored = TrialResult.from_dict(result.to_dict())
    assert restored.resolved_config["k"] == 3
    assert "api_key" not in restored.resolved_config
    exported = export_trial("r", restored.to_dict())
    assert exported.metadata["trial_id"] == 12
    assert exported.metadata["status"] == "failed"
    assert "checkpoint-secret" not in exported.code
