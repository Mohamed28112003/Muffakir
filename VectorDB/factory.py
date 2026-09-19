from typing import Optional, Any
from .base import BaseVectorDBManager
from Embedding.EmbeddingProvider import EmbeddingProvider
from Muffakir.optional_dependencies import require_optional_dependency


def create_vector_db(
    provider: str = "chroma",
    embedding_provider: Optional[EmbeddingProvider] = None,
    **kwargs: Any
) -> BaseVectorDBManager:
    """
    Factory function to instantiate Vector Database providers.

    Args:
        provider (str): 'chroma', 'qdrant', 'pinecone', 'faiss', or 'milvus'.
        embedding_provider (EmbeddingProvider, optional): Injected embedding provider.
        **kwargs: Provider-specific configuration options (path, collection_name, api_key, etc.).

    Returns:
        BaseVectorDBManager: An instance of BaseVectorDBManager.
    """
    provider_name = str(provider).lower().strip()

    if provider_name in ("chroma", "chromadb"):
        require_optional_dependency("chroma")
        from .ChromaDBManager import ChromaDBManager

        return ChromaDBManager(
            embedding_provider=embedding_provider,
            path=kwargs.get("path", kwargs.get("db_path", "./muffakir_db")),
            collection_name=kwargs.get("collection_name", "ArabicBooks"),
            model_name=kwargs.get("model_name", "mohamed2811/Muffakir_Embedding")
        )

    elif provider_name in ("qdrant", "quadrant"):
        require_optional_dependency("qdrant")
        from .QdrantDBManager import QdrantDBManager

        return QdrantDBManager(
            embedding_provider=embedding_provider,
            collection_name=kwargs.get("collection_name", "ArabicBooks"),
            location=kwargs.get("location", kwargs.get("url", "http://localhost:6333")),
            api_key=kwargs.get("api_key"),
            path=kwargs.get("path"),
            model_name=kwargs.get("model_name", "mohamed2811/Muffakir_Embedding")
        )

    elif provider_name in ("pinecone",):
        require_optional_dependency("pinecone")
        from .PineconeDBManager import PineconeDBManager

        return PineconeDBManager(
            embedding_provider=embedding_provider,
            index_name=kwargs.get("index_name", "muffakir-index"),
            api_key=kwargs.get("api_key"),
            model_name=kwargs.get("model_name", "mohamed2811/Muffakir_Embedding")
        )

    elif provider_name in ("faiss",):
        require_optional_dependency("faiss")
        from .FAISSDBManager import FAISSDBManager

        return FAISSDBManager(
            embedding_provider=embedding_provider,
            folder_path=kwargs.get("folder_path", kwargs.get("db_path", "./faiss_index")),
            index_name=kwargs.get("index_name", "index"),
            model_name=kwargs.get("model_name", "mohamed2811/Muffakir_Embedding")
        )

    elif provider_name in ("milvus", "milvus_lite"):
        require_optional_dependency("milvus")
        from .MilvusDBManager import MilvusDBManager

        return MilvusDBManager(
            embedding_provider=embedding_provider,
            collection_name=kwargs.get("collection_name", "ArabicBooks"),
            connection_args=kwargs.get("connection_args", {"uri": kwargs.get("db_path", "./milvus_local.db")}),
            model_name=kwargs.get("model_name", "mohamed2811/Muffakir_Embedding")
        )

    raise ValueError(
        f"Unknown vector database provider: '{provider}'. "
        "Available options: 'chroma', 'qdrant', 'pinecone', 'faiss', 'milvus'."
    )
