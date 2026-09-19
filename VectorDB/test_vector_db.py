"""
Unit tests for the VectorDB module.

Covers:
- factory: all providers, aliases, unknown raises ValueError
- BaseVectorDBManager: search guards, get_all_documents delegation
- ChromaDBManager: add_documents, load_all_documents, get_collection_count
- FAISSDBManager: add_documents, load_all_documents, None store guard
- PineconeDBManager: no os.environ mutation
- QdrantDBManager: import guard, load_all_documents fallback
- MilvusDBManager: import guard, load_all_documents
"""
import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from VectorDB.factory import create_vector_db
from VectorDB.ChromaDBManager import ChromaDBManager
from VectorDB.FAISSDBManager import FAISSDBManager
from VectorDB.QdrantDBManager import QdrantDBManager
from VectorDB.base import BaseVectorDBManager
from Muffakir.exceptions import RetrievalError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_embedding_provider():
    """Return a mock EmbeddingProvider that satisfies LangChain's embeddings interface."""
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock


def _make_docs(n: int = 3):
    return [
        Document(page_content=f"Document {i} content", metadata={"source": f"file_{i}.txt"})
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------

class TestVectorDBFactory:

    def test_create_chroma(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("VectorDB.ChromaDBManager.Chroma"):
                db = create_vector_db(
                    provider="chroma",
                    embedding_provider=_mock_embedding_provider(),
                    path=tmpdir,
                    collection_name="test",
                )
                assert isinstance(db, ChromaDBManager)

    def test_create_chroma_alias(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("VectorDB.ChromaDBManager.Chroma"):
                db = create_vector_db(
                    provider="chromadb",
                    embedding_provider=_mock_embedding_provider(),
                    path=tmpdir,
                )
                assert isinstance(db, ChromaDBManager)

    def test_create_faiss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # This is a constructor-wiring test. Dependency availability is
            # covered separately by the typed optional-dependency tests.
            with patch("VectorDB.factory.require_optional_dependency"):
                with patch("langchain_community.vectorstores.FAISS"):
                    db = create_vector_db(
                        provider="faiss",
                        embedding_provider=_mock_embedding_provider(),
                        db_path=tmpdir,
                    )
                    assert isinstance(db, FAISSDBManager)

    def test_create_qdrant_import_error(self):
        """Qdrant should raise ImportError if qdrant-client not installed."""
        with patch.dict("sys.modules", {"qdrant_client": None, "langchain_qdrant": None}):
            with pytest.raises((ImportError, TypeError)):
                create_vector_db(
                    provider="qdrant",
                    embedding_provider=_mock_embedding_provider(),
                )

    def test_create_pinecone_import_error(self):
        """Pinecone should raise ImportError if langchain-pinecone not installed."""
        with patch.dict("sys.modules", {"langchain_pinecone": None}):
            with pytest.raises((ImportError, TypeError)):
                create_vector_db(
                    provider="pinecone",
                    embedding_provider=_mock_embedding_provider(),
                    api_key="fake_key",
                )

    def test_create_milvus_import_error(self):
        """Milvus should raise ImportError if langchain-milvus not installed."""
        with patch.dict("sys.modules", {"langchain_milvus": None}):
            with pytest.raises((ImportError, TypeError)):
                create_vector_db(
                    provider="milvus",
                    embedding_provider=_mock_embedding_provider(),
                )

    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown vector database provider"):
            create_vector_db(provider="unknown_db", embedding_provider=_mock_embedding_provider())

    def test_provider_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("VectorDB.ChromaDBManager.Chroma"):
                db = create_vector_db(
                    provider="CHROMA",
                    embedding_provider=_mock_embedding_provider(),
                    path=tmpdir,
                )
                assert isinstance(db, ChromaDBManager)


# ---------------------------------------------------------------------------
# BaseVectorDBManager: guard tests
# ---------------------------------------------------------------------------

class TestBaseVectorDBManagerGuards:

    def _make_concrete_db(self, store=None):
        """Dynamically create a minimal concrete subclass for testing."""
        class ConcreteDB(BaseVectorDBManager):
            _store = store

            @property
            def vector_store(self):
                return self._store

            def add_documents(self, documents):
                self._all_documents.extend(documents)

            def load_all_documents(self):
                return list(self._all_documents)

        return ConcreteDB(embedding_provider=_mock_embedding_provider())

    def test_similarity_search_raises_when_no_store(self):
        db = self._make_concrete_db(store=None)
        with pytest.raises(RuntimeError, match="No documents have been indexed yet"):
            db.similarity_search("query")

    def test_mmr_search_raises_when_no_store(self):
        db = self._make_concrete_db(store=None)
        with pytest.raises(RuntimeError, match="No documents have been indexed yet"):
            db.max_marginal_relevance_search("query")

    def test_similarity_search_delegates_to_store(self):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [Document(page_content="result")]
        db = self._make_concrete_db(store=mock_store)
        results = db.similarity_search("query", k=1)
        assert len(results) == 1
        mock_store.similarity_search.assert_called_once_with("query", k=1)

    def test_get_all_documents_uses_cache_first(self):
        db = self._make_concrete_db(store=None)
        doc = Document(page_content="cached")
        db._all_documents = [doc]
        result = db.get_all_documents()
        assert result == [doc]

    def test_get_all_documents_calls_load_when_cache_empty(self):
        db = self._make_concrete_db(store=None)
        result = db.get_all_documents()
        assert result == []  # load_all_documents returns [] for empty cache

    def test_get_all_documents_concurrent_calls_load_once(self):
        """Concurrent get_all_documents() calls on an empty cache must build the
        result exactly once — the double-checked lock must prevent every
        racing thread from redundantly calling load_all_documents()."""
        import threading
        import time

        class SlowLoadDB(BaseVectorDBManager):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.load_calls = 0

            @property
            def vector_store(self):
                return None

            def add_documents(self, documents):
                pass

            def load_all_documents(self):
                self.load_calls += 1
                time.sleep(0.05)
                return [Document(page_content="loaded")]

        db = SlowLoadDB(embedding_provider=_mock_embedding_provider())
        results = []
        threads = [
            threading.Thread(target=lambda: results.append(db.get_all_documents()))
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert db.load_calls == 1
        assert all(r == [Document(page_content="loaded")] for r in results)


# ---------------------------------------------------------------------------
# ChromaDBManager Tests
# ---------------------------------------------------------------------------

class TestChromaDBManager:

    def _make_chroma_manager(self, tmpdir):
        mock_chroma = MagicMock()
        mock_chroma._collection = MagicMock()
        mock_chroma._collection.count.return_value = 0
        mock_chroma._collection.get.return_value = {"documents": [], "metadatas": []}

        with patch("VectorDB.ChromaDBManager.Chroma", return_value=mock_chroma):
            manager = ChromaDBManager(
                path=tmpdir,
                collection_name="test",
                embedding_provider=_mock_embedding_provider(),
            )
            manager._vector_store = mock_chroma
            return manager, mock_chroma

    def test_add_documents_updates_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, mock_chroma = self._make_chroma_manager(tmpdir)
            docs = _make_docs(3)
            manager.add_documents(docs)
            assert len(manager._all_documents) == 3
            mock_chroma.add_documents.assert_called_once()

    def test_add_documents_empty_list_is_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, mock_chroma = self._make_chroma_manager(tmpdir)
            manager.add_documents([])
            mock_chroma.add_documents.assert_not_called()

    def test_persist_called_if_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, mock_chroma = self._make_chroma_manager(tmpdir)
            mock_chroma.persist = MagicMock()
            manager.add_documents(_make_docs(1))
            mock_chroma.persist.assert_called_once()

    def test_persist_skipped_if_not_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, mock_chroma = self._make_chroma_manager(tmpdir)
            del mock_chroma.persist  # Simulate Chroma 0.4+
            # Should not raise
            manager.add_documents(_make_docs(1))

    def test_load_all_documents_from_collection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, mock_chroma = self._make_chroma_manager(tmpdir)
            mock_chroma._collection.get.return_value = {
                "documents": ["مرحبا بالعالم", "Hello world"],
                "metadatas": [{"source": "a.txt"}, {"source": "b.txt"}],
            }
            result = manager.load_all_documents()
            assert len(result) == 2
            assert result[0].page_content == "مرحبا بالعالم"
            assert result[0].metadata["source"] == "a.txt"

    def test_load_all_documents_raises_on_collection_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, mock_chroma = self._make_chroma_manager(tmpdir)
            mock_chroma._collection.get.side_effect = Exception("Chroma error")
            with pytest.raises(RetrievalError) as excinfo:
                manager.load_all_documents()
            assert isinstance(excinfo.value.__cause__, Exception)

    def test_get_collection_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, mock_chroma = self._make_chroma_manager(tmpdir)
            mock_chroma._collection.count.return_value = 42
            assert manager.get_collection_count() == 42

    def test_get_collection_count_raises_on_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, mock_chroma = self._make_chroma_manager(tmpdir)
            mock_chroma._collection.count.side_effect = Exception("Chroma error")
            manager._all_documents = _make_docs(5)
            with pytest.raises(RetrievalError) as excinfo:
                manager.get_collection_count()
            assert isinstance(excinfo.value.__cause__, Exception)


# ---------------------------------------------------------------------------
# FAISSDBManager Tests
# ---------------------------------------------------------------------------

class TestFAISSDBManager:

    def _make_faiss_manager(self, tmpdir):
        mock_faiss_class = MagicMock()
        mock_faiss_instance = MagicMock()
        mock_faiss_class.from_documents.return_value = mock_faiss_instance
        mock_faiss_class.load_local.side_effect = Exception("No index")

        with patch("langchain_community.vectorstores.FAISS", mock_faiss_class):
            manager = FAISSDBManager(
                folder_path=tmpdir,
                embedding_provider=_mock_embedding_provider(),
            )
            manager._FAISSClass = mock_faiss_class
            return manager, mock_faiss_class, mock_faiss_instance

    def test_add_documents_creates_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, mock_cls, mock_inst = self._make_faiss_manager(tmpdir)
            docs = _make_docs(2)
            manager.add_documents(docs)
            mock_cls.from_documents.assert_called_once()
            assert len(manager._all_documents) == 2

    def test_add_documents_extends_existing_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, mock_cls, mock_inst = self._make_faiss_manager(tmpdir)
            # Simulate existing index
            manager._vector_store = mock_inst
            docs = _make_docs(2)
            manager.add_documents(docs)
            mock_inst.add_documents.assert_called_once_with(docs)

    def test_add_documents_empty_is_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, mock_cls, _ = self._make_faiss_manager(tmpdir)
            manager.add_documents([])
            mock_cls.from_documents.assert_not_called()

    def test_load_all_documents_returns_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, _, _ = self._make_faiss_manager(tmpdir)
            manager._all_documents = _make_docs(3)
            result = manager.load_all_documents()
            assert len(result) == 3

    def test_vector_store_none_raises_on_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, _, _ = self._make_faiss_manager(tmpdir)
            assert manager.vector_store is None
            with pytest.raises(RuntimeError, match="No documents have been indexed yet"):
                manager.similarity_search("test query")


# ---------------------------------------------------------------------------
# QdrantDBManager Tests
# ---------------------------------------------------------------------------

class TestQdrantDBManager:

    def _make_qdrant_manager(self):
        mock_client_instance = MagicMock()
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        mock_qdrant_client_module = MagicMock()
        mock_qdrant_client_module.QdrantClient = mock_client_cls
        mock_models_module = MagicMock()
        mock_langchain_qdrant_module = MagicMock()

        patcher = patch.dict("sys.modules", {
            "qdrant_client": mock_qdrant_client_module,
            "qdrant_client.models": mock_models_module,
            "langchain_qdrant": mock_langchain_qdrant_module,
        })
        patcher.start()
        manager = QdrantDBManager(
            collection_name="test",
            path="./tmp_qdrant",
            embedding_provider=_mock_embedding_provider(),
        )
        return manager, mock_client_instance, patcher

    def test_load_all_documents_raises_on_scroll_error(self):
        manager, mock_client, patcher = self._make_qdrant_manager()
        try:
            mock_client.scroll.side_effect = Exception("Qdrant connection error")
            with pytest.raises(RetrievalError) as excinfo:
                manager.load_all_documents()
            assert isinstance(excinfo.value.__cause__, Exception)
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# PineconeDBManager: no os.environ mutation
# ---------------------------------------------------------------------------

class TestPineconeSecurity:

    def test_no_os_environ_mutation(self):
        """Confirm PineconeDBManager does NOT write to os.environ."""
        original_env = dict(os.environ)

        mock_pinecone_cls = MagicMock()
        with patch.dict("sys.modules", {"langchain_pinecone": MagicMock(PineconeVectorStore=mock_pinecone_cls)}):
            from VectorDB.PineconeDBManager import PineconeDBManager
            PineconeDBManager(
                index_name="test-index",
                api_key="super_secret_key",
                embedding_provider=_mock_embedding_provider(),
            )

        # os.environ must not have changed
        assert os.environ == original_env
        assert "PINECONE_API_KEY" not in os.environ or os.environ.get("PINECONE_API_KEY") == original_env.get("PINECONE_API_KEY")
