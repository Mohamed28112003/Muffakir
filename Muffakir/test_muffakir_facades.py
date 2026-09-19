import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

import Muffakir
from Muffakir import (
    MuffakirRAG,
    MuffakirSearch,
    MuffakirRetrieval,
    MuffakirSyntheticData,
    MuffakirEvaluation,
    MuffakirComposer,
    MuffakirPrompt,
)
from Muffakir.exceptions import ConfigurationError, GenerationError, RetrievalError, DocumentIndexError


# ---------------------------------------------------------------------------
# Lazy Loading and Public API Exports
# ---------------------------------------------------------------------------

def test_muffakir_root_exports():
    assert MuffakirRAG is not None
    assert MuffakirSearch is not None
    assert MuffakirRetrieval is not None
    assert MuffakirSyntheticData is not None
    assert MuffakirEvaluation is not None
    assert MuffakirComposer is not None
    assert MuffakirPrompt is not None


def test_muffakir_invalid_export():
    with pytest.raises(AttributeError):
        _ = getattr(Muffakir, "NonExistentClassXYZ")


# ---------------------------------------------------------------------------
# MuffakirSearch Tests
# ---------------------------------------------------------------------------

def test_muffakir_search_config_validation():
    # Missing required params
    with pytest.raises(ValueError, match="Required parameter 'api_key' is missing"):
        MuffakirSearch(config={})

    # Unsupported provider
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        MuffakirSearch(config={
            "api_key": "key",
            "llm_provider": "unsupported_provider",
            "llm_model": "model",
        })


def test_muffakir_search_execution():
    mock_search_provider = MagicMock()
    mock_search_provider.search.return_value = MagicMock(
        content="محتوى البحث التجريبي",
        sources=[{"title": "مصدر 1", "url": "https://example.com"}]
    )

    with patch("Muffakir.dependency_validation.validate_search_dependencies"), \
         patch("Muffakir.MuffakirSearch.create_web_search_provider", return_value=mock_search_provider), \
         patch("Muffakir.MuffakirSearch.LLMProvider") as mock_llm_cls:

        mock_llm_inst = MagicMock()
        mock_llm_inst.get_llm.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content="الإجابة التوليدية")))
        mock_llm_cls.return_value = mock_llm_inst

        search_agent = MuffakirSearch(config={
            "api_key": "test_key",
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
            "search_provider": "tavily",
            "search_provider_config": {"api_key": "tavily_key"},
        })

        results = search_agent.search("ما هو الذكاء الاصطناعي؟")
        assert "answer" in results
        assert "sources" in results
        assert len(results["sources"]) == 1
        assert results["sources"][0]["title"] == "مصدر 1"
        # Test backward-compatible property alias
        assert search_agent.search_pipline is search_agent.search_pipeline


def test_muffakir_search_propagates_generation_error():
    """search_web() must not swallow a generation failure into a canned string."""
    mock_search_provider = MagicMock()
    mock_search_provider.search.return_value = MagicMock(
        content="محتوى البحث التجريبي",
        sources=[{"title": "مصدر 1", "url": "https://example.com"}]
    )

    with patch("Muffakir.dependency_validation.validate_search_dependencies"), \
         patch("Muffakir.MuffakirSearch.create_web_search_provider", return_value=mock_search_provider), \
         patch("Muffakir.MuffakirSearch.LLMProvider") as mock_llm_cls:

        mock_llm_inst = MagicMock()
        mock_llm_inst.get_llm.return_value = MagicMock(
            invoke=MagicMock(side_effect=RuntimeError("LLM down"))
        )
        mock_llm_cls.return_value = mock_llm_inst

        search_agent = MuffakirSearch(config={
            "api_key": "test_key",
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
            "search_provider": "tavily",
            "search_provider_config": {"api_key": "tavily_key"},
        })

        with pytest.raises(GenerationError) as excinfo:
            search_agent.search("ما هو الذكاء الاصطناعي؟")
        assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_muffakir_search_ask_matches_eval_runner_shape():
    """.ask() is an EvalRunner-compatible adapter for web-search-only mode:
    same call shape (question, **kwargs) and return shape (answer/
    retrieved_documents/source_metadata/stage_timings_ms) as MuffakirRAG.ask()."""
    mock_search_provider = MagicMock()
    mock_search_provider.search.return_value = MagicMock(
        content="محتوى البحث التجريبي",
        sources=[{"title": "مصدر 1", "url": "https://example.com"}],
    )

    with patch("Muffakir.dependency_validation.validate_search_dependencies"), \
         patch("Muffakir.MuffakirSearch.create_web_search_provider", return_value=mock_search_provider), \
         patch("Muffakir.MuffakirSearch.LLMProvider") as mock_llm_cls:

        mock_llm_inst = MagicMock()
        mock_llm_inst.get_llm.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content="الإجابة التوليدية")))
        mock_llm_cls.return_value = mock_llm_inst

        search_agent = MuffakirSearch(config={
            "api_key": "test_key",
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
            "search_provider": "tavily",
            "search_provider_config": {"api_key": "tavily_key"},
        })

        resp = search_agent.ask("ما هو الذكاء الاصطناعي؟", k=5, retrieval_method="similarity_search")

        assert {
            "answer", "retrieved_documents", "source_metadata", "stage_timings_ms",
            "pipeline_latency_ms", "context_source", "used_context", "web_sources",
        } <= set(resp.keys())
        assert resp["answer"] == "الإجابة التوليدية"
        assert resp["retrieved_documents"] == []
        assert resp["source_metadata"] == [{"title": "مصدر 1", "url": "https://example.com"}]
        assert resp["context_source"] == "web_search"
        assert resp["used_context"] == "محتوى البحث التجريبي"
        assert resp["stage_timings_ms"]["web_search_ms"] is not None
        assert resp["stage_timings_ms"]["generation_ms"] is not None
        assert resp["pipeline_latency_ms"] is not None


def test_muffakir_search_get_similar_documents_raises_configuration_error():
    """Web-search-only mode has no local retrieval corpus — must fail loudly
    rather than an ambiguous AttributeError when retrieval metrics are requested."""
    mock_search_provider = MagicMock()

    with patch("Muffakir.dependency_validation.validate_search_dependencies"), \
         patch("Muffakir.MuffakirSearch.create_web_search_provider", return_value=mock_search_provider), \
         patch("Muffakir.MuffakirSearch.LLMProvider") as mock_llm_cls:

        mock_llm_cls.return_value = MagicMock()

        search_agent = MuffakirSearch(config={
            "api_key": "test_key",
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
            "search_provider": "tavily",
            "search_provider_config": {"api_key": "tavily_key"},
        })

        with pytest.raises(ConfigurationError, match="retrieval"):
            search_agent.get_similar_documents("سؤال بحث", k=2)


def test_eval_runner_scores_muffakir_search_on_generation_metrics_only():
    """Integration proof: a MuffakirSearch instance can be scored by the
    existing EvalRunner on generation metrics with zero EvalRunner changes,
    now that it has .ask(). Retrieval metrics are intentionally not
    requested here (get_similar_documents() would raise ConfigurationError)."""
    from Evaluation.runner import EvalRunner
    from SyntheticData.models import QAPair

    class _MockMetric:
        def __init__(self, **kwargs):
            pass

        def score(self, **kwargs):
            return 1.0

    mock_search_provider = MagicMock()
    mock_search_provider.search.return_value = MagicMock(
        content="محتوى البحث التجريبي",
        sources=[{"title": "مصدر 1", "url": "https://example.com"}],
    )

    with patch("Muffakir.dependency_validation.validate_search_dependencies"), \
         patch("Muffakir.MuffakirSearch.create_web_search_provider", return_value=mock_search_provider), \
         patch("Muffakir.MuffakirSearch.LLMProvider") as mock_llm_cls, \
         patch("Evaluation.runner.FaithfulnessMetric", _MockMetric), \
         patch("Evaluation.runner.AnswerCorrectnessMetric", _MockMetric):

        mock_llm_inst = MagicMock()
        mock_llm_inst.get_llm.return_value = MagicMock(invoke=MagicMock(return_value=MagicMock(content="الإجابة التوليدية")))
        mock_llm_cls.return_value = mock_llm_inst

        search_agent = MuffakirSearch(config={
            "api_key": "test_key",
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
            "search_provider": "tavily",
            "search_provider_config": {"api_key": "tavily_key"},
        })

        pairs = [QAPair(question="what is X?", answer="X is Y", context="gold context text here")]

        runner = EvalRunner(
            rag=search_agent,
            llm_provider=None,
            prompt_manager=None,
            metrics=["faithfulness", "answer_correctness"],
            k=5,
        )
        report = runner.run(pairs)

        assert report.generation["faithfulness"] == 1.0
        assert report.generation["answer_correctness"] == 1.0
        assert report.retrieval == {}


# ---------------------------------------------------------------------------
# MuffakirRAG Tests
# ---------------------------------------------------------------------------

def test_muffakir_rag_init_and_ask():
    mock_db_manager = MagicMock()
    mock_db_manager.vector_store.similarity_search.return_value = [
        Document(page_content="محتوى تجريبي عن الذكاء الاصطناعي.", metadata={"source": "test.txt"})
    ]

    mock_emb_provider = MagicMock()

    with patch("Muffakir.Muffakir.LLMProvider") as mock_llm_cls, \
         patch("Muffakir.Muffakir.RAGPipelineManager") as mock_pm_cls:

        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        mock_pm = MagicMock()
        mock_pm.k = 5
        mock_pm.retrieve_method = MagicMock()
        mock_pm.generate_answer.return_value = {
            "answer": "الذكاء الاصطناعي هو محاكاة للذكاء البشري.",
            "retrieved_documents": ["محتوى تجريبي عن الذكاء الاصطناعي."],
            "source_metadata": [{"source": "test.txt"}],
            "context_source": "vector_db",
            "web_sources": []
        }
        mock_pm_cls.return_value = mock_pm

        rag = MuffakirRAG(
            config={
                "api_key": "test_key",
                "llm_provider": "openai",
                "llm_model": "gpt-4o",
                "skip_document_ingestion": True,
            },
            embedding_provider=mock_emb_provider,
            db_manager=mock_db_manager,
        )

        # 1. Ask normal question
        resp = rag.ask("ما هو الذكاء الاصطناعي؟")
        assert "answer" in resp
        assert "الذكاء الاصطناعي" in resp["answer"]

        # 2. Ask with k and retrieval_method overrides
        resp_override = rag.ask("ما هو الذكاء الاصطناعي؟", k=3, retrieval_method="similarity_search")
        assert "answer" in resp_override

        # 3. Empty question check
        empty_resp = rag.ask("")
        assert "error" in empty_resp

        # 4. Get similar documents
        mock_pm.query_similar_documents.return_value = [
            Document(page_content="محتوى وثيقة", metadata={"source": "doc1.txt"})
        ]
        docs = rag.get_similar_documents("سؤال بحث", k=2)
        assert len(docs) == 1

        # 5. Add documents
        assert rag.add_documents([Document(page_content="نص جديد")]) is True
        assert rag.add_documents([]) is False


def test_muffakir_rag_ask_does_not_mutate_shared_rag_manager_state():
    """ask() must pass k/retrieval_method overrides as explicit call arguments,
    never by mutating self.rag_manager.k/.retrieve_method — mutating shared state
    is a data race the moment ask() is ever called concurrently on one instance."""
    mock_db_manager = MagicMock()
    mock_emb_provider = MagicMock()

    with patch("Muffakir.Muffakir.LLMProvider") as mock_llm_cls, \
         patch("Muffakir.Muffakir.RAGPipelineManager") as mock_pm_cls:

        mock_llm_cls.return_value = MagicMock()

        mock_pm = MagicMock()
        mock_pm.k = 5
        original_method = MagicMock(name="original_retrieve_method")
        mock_pm.retrieve_method = original_method
        mock_pm.generate_answer.return_value = {
            "answer": "ok",
            "retrieved_documents": [],
            "source_metadata": [],
            "context_source": "vector_db",
            "web_sources": [],
        }
        mock_pm_cls.return_value = mock_pm

        rag = MuffakirRAG(
            config={
                "api_key": "test_key",
                "llm_provider": "openai",
                "llm_model": "gpt-4o",
                "skip_document_ingestion": True,
            },
            embedding_provider=mock_emb_provider,
            db_manager=mock_db_manager,
        )

        rag.ask("سؤال", k=3, retrieval_method="similarity_search")

        # No shared-attribute mutation left behind.
        assert mock_pm.k == 5
        assert mock_pm.retrieve_method is original_method

        # Overrides were forwarded as explicit call arguments instead.
        _, call_kwargs = mock_pm.generate_answer.call_args
        assert call_kwargs["k"] == 3
        assert call_kwargs["retrieve_method"] is not original_method


def test_muffakir_rag_forwards_device_to_embedding_provider():
    """MuffakirRAG must thread its 'device' config through to EmbeddingProvider,
    not silently drop it and rely on sentence-transformers' own auto-detect."""
    with patch("Muffakir.Muffakir.LLMProvider") as mock_llm_cls, \
         patch("Muffakir.Muffakir.RAGPipelineManager") as mock_pm_cls, \
         patch("Muffakir.Muffakir.EmbeddingProvider") as mock_emb_cls:

        mock_llm_cls.return_value = MagicMock()
        mock_pm_cls.return_value = MagicMock()
        mock_emb_cls.return_value = MagicMock()

        MuffakirRAG(
            config={
                "api_key": "test_key",
                "llm_provider": "openai",
                "llm_model": "gpt-4o",
                "skip_document_ingestion": True,
                "device": "cuda:1",
            },
        )

        _, call_kwargs = mock_emb_cls.call_args
        assert call_kwargs["device"] == "cuda:1"


def test_muffakir_rag_query_transformer_uses_shared_llm_by_default():
    with patch("Muffakir.Muffakir.LLMProvider") as mock_llm_cls, \
         patch("Muffakir.Muffakir.RAGPipelineManager") as mock_pm_cls, \
         patch("Muffakir.Muffakir.EmbeddingProvider") as mock_emb_cls:

        mock_llm_cls.return_value = MagicMock()
        mock_pm_cls.return_value = MagicMock()
        mock_emb_cls.return_value = MagicMock()

        rag = MuffakirRAG(
            config={
                "api_key": "test_key",
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini",
                "skip_document_ingestion": True,
                "query_transformer": True,
                "hallucination_check": False,
            },
        )
        assert rag.query_transformer.llm_provider is rag.llm_provider


def test_muffakir_rag_query_transformer_uses_override_llm_when_set():
    with patch("Muffakir.Muffakir.LLMProvider") as mock_llm_cls, \
         patch("Muffakir.Muffakir.RAGPipelineManager") as mock_pm_cls, \
         patch("Muffakir.Muffakir.EmbeddingProvider") as mock_emb_cls:

        built_configs = []

        def _spy_llm_provider(*args, **kwargs):
            built_configs.append(kwargs)
            return MagicMock()

        mock_llm_cls.side_effect = _spy_llm_provider
        mock_pm_cls.return_value = MagicMock()
        mock_emb_cls.return_value = MagicMock()

        rag = MuffakirRAG(
            config={
                "api_key": "test_key",
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini",
                "skip_document_ingestion": True,
                "query_transformer": True,
                "query_transform_llm_provider": "together",
                "query_transform_llm_model": "meta-llama/Llama-3-8b-chat-hf",
                "query_transform_api_key": "query-key",
                "query_transform_base_url": "https://query.example/v1",
                "hallucination_check": False,
            },
        )
        assert rag.query_transformer.llm_provider is not rag.llm_provider
        assert [config["model"] for config in built_configs] == [
            "gpt-4o-mini",
            "meta-llama/Llama-3-8b-chat-hf",
        ]
        assert built_configs[1]["api_key"] == "query-key"
        assert built_configs[1]["base_url"] == "https://query.example/v1"


def test_muffakir_rag_failures_raise_typed_errors():
    """A failure inside the underlying pipeline manager must raise a typed
    MuffakirError instead of silently returning a fake success value."""
    mock_db_manager = MagicMock()
    mock_emb_provider = MagicMock()

    with patch("Muffakir.Muffakir.LLMProvider") as mock_llm_cls, \
         patch("Muffakir.Muffakir.RAGPipelineManager") as mock_pm_cls:

        mock_llm_cls.return_value = MagicMock()

        mock_pm = MagicMock()
        mock_pm.k = 5
        mock_pm.retrieve_method = MagicMock()
        mock_pm.generate_answer.side_effect = RuntimeError("LLM down")
        mock_pm.query_similar_documents.side_effect = RuntimeError("retrieval down")
        mock_pm.store_documents.side_effect = RuntimeError("index down")
        mock_pm_cls.return_value = mock_pm

        rag = MuffakirRAG(
            config={
                "api_key": "test_key",
                "llm_provider": "openai",
                "llm_model": "gpt-4o",
                "skip_document_ingestion": True,
            },
            embedding_provider=mock_emb_provider,
            db_manager=mock_db_manager,
        )

        with pytest.raises(GenerationError) as excinfo:
            rag.ask("ما هو الذكاء الاصطناعي؟")
        assert isinstance(excinfo.value.__cause__, RuntimeError)

        with pytest.raises(RetrievalError):
            rag.get_similar_documents("سؤال بحث", k=2)

        with pytest.raises(DocumentIndexError):
            rag.add_documents([Document(page_content="نص جديد")])
