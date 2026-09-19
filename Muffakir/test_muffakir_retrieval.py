import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from Muffakir.MuffakirRetrieval import MuffakirRetrieval
from Muffakir.exceptions import ConfigurationError, RetrievalError


def _base_config(**overrides):
    config = {
        "retrieval_method": "similarity_search",
        "k": 5,
        "fetch_k": 7,
        "query_transformer": False,
        "reranking": False,
    }
    config.update(overrides)
    return config


def test_get_similar_documents_plain_retrieval_no_transform_no_rerank():
    mock_embedding_provider = MagicMock()
    mock_db_manager = MagicMock()
    docs = [Document(page_content="a", metadata={"chunk_id": "1"})]

    with patch("Muffakir.MuffakirRetrieval.RetrieveMethods") as mock_retrieve_cls:
        mock_retriever = MagicMock()
        mock_retriever.similarity_search.return_value = docs
        mock_retrieve_cls.return_value = mock_retriever

        mr = MuffakirRetrieval(_base_config(), embedding_provider=mock_embedding_provider, db_manager=mock_db_manager)
        result = mr.get_similar_documents("query", k=5)

        mock_retriever.similarity_search.assert_called_once_with("query", 5)
        assert result == docs
        assert result[0].metadata == {"chunk_id": "1"}


def test_get_similar_documents_applies_query_transform_when_enabled():
    mock_embedding_provider = MagicMock()
    mock_db_manager = MagicMock()
    docs = [Document(page_content="a", metadata={})]

    with patch("Muffakir.MuffakirRetrieval.RetrieveMethods") as mock_retrieve_cls, \
         patch("Muffakir.MuffakirRetrieval.LLMProvider") as mock_llm_cls, \
         patch("Muffakir.MuffakirRetrieval.QueryTransformer") as mock_qt_cls:
        mock_retriever = MagicMock()
        mock_retriever.similarity_search.return_value = docs
        mock_retrieve_cls.return_value = mock_retriever

        mock_qt_instance = MagicMock()
        mock_qt_instance.transform_query.return_value = "transformed query"
        mock_qt_cls.return_value = mock_qt_instance

        config = _base_config(
            query_transformer=True,
            query_transformer_strategy="hyde",
            query_transform_llm_provider="openai",
            query_transform_llm_model="gpt-4o-mini",
            api_key="key",
        )
        mr = MuffakirRetrieval(config, embedding_provider=mock_embedding_provider, db_manager=mock_db_manager)
        mr.get_similar_documents("original query", k=5)

        mock_qt_instance.transform_query.assert_called_once_with("original query")
        mock_retriever.similarity_search.assert_called_once_with("transformed query", 5)


def test_get_similar_documents_retrieves_for_every_sub_query_and_dedupes():
    """A list-returning transform (multi_query/step_back/query_decomposition)
    must retrieve for EVERY sub-query and merge results deduped by
    page_content -- mirroring Generation/DocumentRetriever.retrieve_documents.
    """
    docs_q1 = [
        Document(page_content="shared", metadata={"chunk_id": "1"}),
        Document(page_content="only-in-q1", metadata={"chunk_id": "2"}),
    ]
    docs_q2 = [
        Document(page_content="shared", metadata={"chunk_id": "1"}),
        Document(page_content="only-in-q2", metadata={"chunk_id": "3"}),
    ]

    with patch("Muffakir.MuffakirRetrieval.RetrieveMethods") as mock_retrieve_cls, \
         patch("Muffakir.MuffakirRetrieval.LLMProvider"), \
         patch("Muffakir.MuffakirRetrieval.QueryTransformer") as mock_qt_cls:
        mock_retriever = MagicMock()
        mock_retriever.similarity_search.side_effect = [docs_q1, docs_q2]
        mock_retrieve_cls.return_value = mock_retriever

        mock_qt_instance = MagicMock()
        mock_qt_instance.transform_query.return_value = ["q1", "q2"]
        mock_qt_cls.return_value = mock_qt_instance

        config = _base_config(
            query_transformer=True,
            query_transformer_strategy="multi_query",
            query_transform_llm_provider="openai",
            query_transform_llm_model="gpt-4o-mini",
            api_key="key",
        )
        mr = MuffakirRetrieval(config, embedding_provider=MagicMock(), db_manager=MagicMock())
        result = mr.get_similar_documents("original query", k=5)

        # One retrieval call per sub-query -- not just the first.
        assert mock_retriever.similarity_search.call_count == 2
        assert [c.args[0] for c in mock_retriever.similarity_search.call_args_list] == ["q1", "q2"]

        contents = [d.page_content for d in result]
        assert contents == ["shared", "only-in-q1", "only-in-q2"]
        assert len(contents) == len(set(contents))


def test_multi_query_reranks_once_on_merged_docs_with_first_sub_query():
    docs_q1 = [Document(page_content="a", metadata={})]
    docs_q2 = [Document(page_content="b", metadata={})]
    reranked = [Document(page_content="b", metadata={})]

    with patch("Muffakir.MuffakirRetrieval.RetrieveMethods") as mock_retrieve_cls, \
         patch("Muffakir.MuffakirRetrieval.LLMProvider"), \
         patch("Muffakir.MuffakirRetrieval.QueryTransformer") as mock_qt_cls, \
         patch("Muffakir.MuffakirRetrieval.Reranker"), \
         patch("Muffakir.MuffakirRetrieval.rerank_documents") as mock_rerank_fn:
        mock_retriever = MagicMock()
        mock_retriever.similarity_search.side_effect = [docs_q1, docs_q2]
        mock_retrieve_cls.return_value = mock_retriever
        mock_rerank_fn.return_value = reranked

        mock_qt_instance = MagicMock()
        mock_qt_instance.transform_query.return_value = ["q1", "q2"]
        mock_qt_cls.return_value = mock_qt_instance

        config = _base_config(
            reranking=True,
            query_transformer=True,
            query_transformer_strategy="multi_query",
            query_transform_llm_provider="openai",
            query_transform_llm_model="gpt-4o-mini",
            api_key="key",
        )
        mr = MuffakirRetrieval(config, embedding_provider=MagicMock(), db_manager=MagicMock())
        result = mr.get_similar_documents("original query", k=5)

        mock_rerank_fn.assert_called_once()
        _reranker_arg, rerank_query, rerank_docs = mock_rerank_fn.call_args.args
        assert rerank_query == "q1"
        assert [d.page_content for d in rerank_docs] == ["a", "b"]
        assert result == reranked


def test_get_similar_documents_returns_empty_list_for_blank_query():
    with patch("Muffakir.MuffakirRetrieval.RetrieveMethods") as mock_retrieve_cls:
        mock_retriever = MagicMock()
        mock_retrieve_cls.return_value = mock_retriever

        mr = MuffakirRetrieval(_base_config(), embedding_provider=MagicMock(), db_manager=MagicMock())

        assert mr.get_similar_documents("") == []
        assert mr.get_similar_documents("   ") == []
        mock_retriever.similarity_search.assert_not_called()


def test_unexpected_retrieval_failure_is_wrapped_in_retrieval_error():
    with patch("Muffakir.MuffakirRetrieval.RetrieveMethods") as mock_retrieve_cls:
        mock_retriever = MagicMock()
        mock_retriever.similarity_search.side_effect = Exception("db connection lost")
        mock_retrieve_cls.return_value = mock_retriever

        mr = MuffakirRetrieval(_base_config(), embedding_provider=MagicMock(), db_manager=MagicMock())
        with pytest.raises(RetrievalError, match="Failed to retrieve documents"):
            mr.get_similar_documents("query", k=5)


def test_unrecognized_retrieval_method_raises_configuration_error():
    with patch("Muffakir.MuffakirRetrieval.RetrieveMethods"):
        with pytest.raises(ConfigurationError, match="Unsupported retrieval method"):
            MuffakirRetrieval(
                _base_config(retrieval_method="bogus_method"),
                embedding_provider=MagicMock(),
                db_manager=MagicMock(),
            )


def test_query_transform_api_key_is_used_for_transform_llm():
    with patch("Muffakir.MuffakirRetrieval.RetrieveMethods"), \
         patch("Muffakir.MuffakirRetrieval.LLMProvider") as mock_llm_cls, \
         patch("Muffakir.MuffakirRetrieval.QueryTransformer"):
        config = _base_config(
            query_transformer=True,
            query_transform_llm_provider="openai",
            query_transform_llm_model="gpt-4o-mini",
            query_transform_api_key="qt-key",
            api_key="main-key",
        )
        MuffakirRetrieval(config, embedding_provider=MagicMock(), db_manager=MagicMock())

        assert mock_llm_cls.call_args.kwargs["api_key"] == "qt-key"


def test_query_transform_base_url_is_used_for_transform_llm():
    with patch("Muffakir.MuffakirRetrieval.RetrieveMethods"), \
         patch("Muffakir.MuffakirRetrieval.LLMProvider") as mock_llm_cls, \
         patch("Muffakir.MuffakirRetrieval.QueryTransformer"):
        config = _base_config(
            query_transformer=True,
            query_transform_llm_provider="openai",
            query_transform_llm_model="gpt-4o-mini",
            query_transform_api_key="qt-key",
            query_transform_base_url="https://query.example/v1",
            base_url="https://answer.example/v1",
        )
        MuffakirRetrieval(config, embedding_provider=MagicMock(), db_manager=MagicMock())

        assert mock_llm_cls.call_args.kwargs["base_url"] == "https://query.example/v1"


def test_query_transform_llm_falls_back_to_main_api_key():
    with patch("Muffakir.MuffakirRetrieval.RetrieveMethods"), \
         patch("Muffakir.MuffakirRetrieval.LLMProvider") as mock_llm_cls, \
         patch("Muffakir.MuffakirRetrieval.QueryTransformer"):
        config = _base_config(
            query_transformer=True,
            query_transform_llm_provider="openai",
            query_transform_llm_model="gpt-4o-mini",
            api_key="main-key",
        )
        MuffakirRetrieval(config, embedding_provider=MagicMock(), db_manager=MagicMock())

        assert mock_llm_cls.call_args.kwargs["api_key"] == "main-key"


def test_query_transformer_enabled_without_llm_override_raises_configuration_error():
    config = _base_config(query_transformer=True)
    with pytest.raises(ConfigurationError, match="query_transform_llm_provider"):
        MuffakirRetrieval(config, embedding_provider=MagicMock(), db_manager=MagicMock())


def test_get_similar_documents_applies_reranking_when_enabled():
    mock_embedding_provider = MagicMock()
    mock_db_manager = MagicMock()
    docs = [Document(page_content="a", metadata={})]
    reranked = [Document(page_content="b", metadata={})]

    with patch("Muffakir.MuffakirRetrieval.RetrieveMethods") as mock_retrieve_cls, \
         patch("Muffakir.MuffakirRetrieval.Reranker") as mock_reranker_cls, \
         patch("Muffakir.MuffakirRetrieval.rerank_documents") as mock_rerank_fn:
        mock_retriever = MagicMock()
        mock_retriever.similarity_search.return_value = docs
        mock_retrieve_cls.return_value = mock_retriever
        mock_rerank_fn.return_value = reranked

        config = _base_config(reranking=True, reranking_method="semantic_similarity")
        mr = MuffakirRetrieval(config, embedding_provider=mock_embedding_provider, db_manager=mock_db_manager)
        result = mr.get_similar_documents("query", k=5)

        mock_rerank_fn.assert_called_once()
        assert result == reranked


def test_custom_reranker_configuration_is_forwarded():
    with patch("Muffakir.MuffakirRetrieval.RetrieveMethods"), \
         patch("Muffakir.MuffakirRetrieval.Reranker") as mock_reranker:
        config = _base_config(
            reranking=True,
            reranking_method="custom",
            reranker_base_url="https://reranker.example/v1/rerank",
            reranker_api_key="secret",
            reranker_model="rerank-v2",
            reranker_timeout_seconds=9,
            reranker_options={"tenant": "legal"},
        )
        MuffakirRetrieval(
            config,
            embedding_provider=MagicMock(),
            db_manager=MagicMock(),
        )

        kwargs = mock_reranker.call_args.kwargs
        assert kwargs["remote_base_url"] == "https://reranker.example/v1/rerank"
        assert kwargs["remote_api_key"] == "secret"
        assert kwargs["remote_model"] == "rerank-v2"
        assert kwargs["remote_timeout"] == 9
        assert kwargs["remote_options"] == {"tenant": "legal"}


def test_llm_reranker_builds_dedicated_provider_without_query_transformer():
    with patch("Muffakir.MuffakirRetrieval.RetrieveMethods"), \
         patch("Muffakir.MuffakirRetrieval.LLMProvider") as mock_llm_provider, \
         patch("Muffakir.MuffakirRetrieval.Reranker") as mock_reranker:
        dedicated = MagicMock()
        mock_llm_provider.return_value = dedicated
        config = _base_config(
            reranking=True,
            reranking_method="llm",
            reranker_llm_provider="openai",
            reranker_llm_model="gpt-4o-mini",
            reranker_llm_api_key="reranker-secret",
            reranker_llm_base_url="https://llm.example/v1",
        )
        MuffakirRetrieval(
            config,
            embedding_provider=MagicMock(),
            db_manager=MagicMock(),
        )

        assert mock_llm_provider.call_args.kwargs["api_key"] == "reranker-secret"
        assert mock_llm_provider.call_args.kwargs["model"] == "gpt-4o-mini"
        assert mock_llm_provider.call_args.kwargs["base_url"] == "https://llm.example/v1"
        assert mock_reranker.call_args.kwargs["llm_provider"] is dedicated


def test_llm_reranker_can_reuse_configured_query_llm_without_query_expansion():
    with patch("Muffakir.MuffakirRetrieval.RetrieveMethods"), \
         patch("Muffakir.MuffakirRetrieval.LLMProvider") as mock_llm_provider, \
         patch("Muffakir.MuffakirRetrieval.Reranker") as mock_reranker:
        reusable = MagicMock()
        mock_llm_provider.return_value = reusable
        config = _base_config(
            reranking=True,
            reranking_method="llm",
            query_transform_llm_provider="openai",
            query_transform_llm_model="gpt-4o-mini",
            query_transform_api_key="query-secret",
        )
        MuffakirRetrieval(
            config,
            embedding_provider=MagicMock(),
            db_manager=MagicMock(),
        )

        assert mock_llm_provider.call_args.kwargs["api_key"] == "query-secret"
        assert mock_reranker.call_args.kwargs["llm_provider"] is reusable


def test_ask_raises_configuration_error():
    mr = MuffakirRetrieval(_base_config(), embedding_provider=MagicMock(), db_manager=MagicMock())
    with pytest.raises(ConfigurationError, match="no answer generation"):
        mr.ask("question?")


def test_contextual_retrieval_without_query_transformer_raises_configuration_error():
    config = _base_config(retrieval_method="contextual")
    with patch("Muffakir.MuffakirRetrieval.RetrieveMethods"):
        with pytest.raises(ConfigurationError, match="contextual"):
            MuffakirRetrieval(config, embedding_provider=MagicMock(), db_manager=MagicMock())


def test_contextual_retrieval_with_query_transformer_uses_transform_llm():
    mock_embedding_provider = MagicMock()
    mock_db_manager = MagicMock()
    docs = [Document(page_content="a", metadata={})]

    with patch("Muffakir.MuffakirRetrieval.RetrieveMethods") as mock_retrieve_cls, \
         patch("Muffakir.MuffakirRetrieval.LLMProvider") as mock_llm_cls, \
         patch("Muffakir.MuffakirRetrieval.QueryTransformer") as mock_qt_cls:
        mock_retriever = MagicMock()
        mock_retriever.contextual_search.return_value = docs
        mock_retrieve_cls.return_value = mock_retriever

        mock_llm_instance = MagicMock()
        mock_llm_cls.return_value = mock_llm_instance

        mock_qt_instance = MagicMock()
        mock_qt_instance.llm_provider = mock_llm_instance
        mock_qt_instance.transform_query.return_value = "transformed query"
        mock_qt_cls.return_value = mock_qt_instance

        config = _base_config(
            retrieval_method="contextual",
            query_transformer=True,
            query_transform_llm_provider="openai",
            query_transform_llm_model="gpt-4o-mini",
            api_key="key",
        )
        mr = MuffakirRetrieval(config, embedding_provider=mock_embedding_provider, db_manager=mock_db_manager)
        result = mr.get_similar_documents("original query", k=5)

        mock_retriever.contextual_search.assert_called_once_with(
            query="transformed query", k=5, llm_provider=mock_llm_instance
        )
        assert result == docs
