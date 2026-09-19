from typing import List, Dict, Any, Optional
import logging

try:
    from langchain_core.documents import Document
except ImportError:
    # pyrefly: ignore [missing-import]
    from langchain.schema import Document

from Muffakir.Enums import RetrievalMethod
from LLMProvider.LLMProvider import LLMProvider
from QueryTransformer.QueryTransformer import QueryTransformer
from PromptManager.PromptManager import MuffakirPrompt
from VectorDB import create_vector_db, BaseVectorDBManager
from Embedding.EmbeddingProvider import EmbeddingProvider
from RAGPipeline.RetrieveMethods import RetrieveMethods
from HallucinationsCheck.HallucinationsCheck import HallucinationsCheck
from Generation.RAGGenerationPipeline import RAGGenerationPipeline
from Reranker.Reranker import Reranker
from WebSearch.base import BaseWebSearchProvider

logger = logging.getLogger(__name__)


class RAGPipelineManager:
    """
    Manages the full RAG pipeline: retrieval, reranking, hallucination checking, generation.
    Supports pluggable VectorDB managers (Chroma, Qdrant, Pinecone, FAISS, Milvus).
    Optional Adaptive RAG (Mode B) via web_search_provider.
    """

    def __init__(
        self,
        db_path: str = "./muffakir_db",
        collection_name: str = 'Book',
        model_name: str = 'mohamed2811/Muffakir_Embedding',
        vector_db_provider: str = "chroma",
        vector_db_config: Optional[Dict[str, Any]] = None,
        db_manager: Optional[BaseVectorDBManager] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
        query_transformer: Optional[QueryTransformer] = None,
        prompt_manager: Optional[MuffakirPrompt] = None,
        hallucination: Optional[HallucinationsCheck] = None,
        reranker: Optional[Reranker] = None,
        web_search_provider: Optional[BaseWebSearchProvider] = None,
        adaptive_web_search: bool = False,
        k: int = 2,
        fetch_k: int = 7,
        retrieve_method: RetrievalMethod = RetrievalMethod.SIMILARITY_SEARCH,
    ):
        self.logger = logging.getLogger(__name__)

        if db_manager is not None:
            self.db_manager = db_manager
            self.embedding_provider = embedding_provider
        else:
            self.embedding_provider = embedding_provider or EmbeddingProvider(model_name=model_name)
            v_config = vector_db_config or {}
            v_config.setdefault("path", db_path)
            v_config.setdefault("collection_name", collection_name)
            v_config.setdefault("model_name", model_name)
            self.db_manager = create_vector_db(
                provider=vector_db_provider,
                embedding_provider=self.embedding_provider,
                **v_config
            )

        self.llm_provider = llm_provider
        self.query_transformer = query_transformer
        self.prompt_manager = prompt_manager or MuffakirPrompt(language="ar")

        self.hallucination = hallucination
        self.reranker = reranker
        self.web_search_provider = web_search_provider
        self.adaptive_web_search = adaptive_web_search

        # RAG parameters
        self.k = k
        self.fetch_k = fetch_k
        self.retrieve_method = retrieve_method

        # Subsystems
        self.retriever = RetrieveMethods(db_manager=self.db_manager)
        self.generation_pipeline = RAGGenerationPipeline(
            pipeline_manager=self,
            llm_provider=self.llm_provider,
            prompt_manager=self.prompt_manager,
            query_transformer=self.query_transformer,
            hallucination=self.hallucination,
            reranker=self.reranker,
            web_search_provider=self.web_search_provider,
            adaptive_web_search=self.adaptive_web_search,
            k=self.k,
        )

    def store_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector database and invalidate retrieval caches.
        """
        self.db_manager.add_documents(documents)
        self.retriever.invalidate_bm25_cache()
        self.logger.info(f"Stored {len(documents)} documents successfully and updated retrieval index.")

    def query_similar_documents(
        self,
        query: str,
        k: Optional[int] = None,
        method: Optional[RetrievalMethod] = None
    ) -> List[Document]:
        """
        Retrieve similar documents based on the selected retrieval strategy.

        :param query: the user’s query
        :param k: override the top-k count (defaults to self.k)
        :param method: override the retrieval method (defaults to self.retrieve_method)
        """
        k = k or self.k
        method = method or self.retrieve_method

        method_name = method.value if hasattr(method, "value") else str(method)
        self.logger.info(f"Retrieving documents using {method_name} (k={k}) for query: {query}")

        if method == RetrievalMethod.MAX_MARGINAL_RELEVANCE:
            return self.retriever.max_marginal_relevance_search(query, k, self.fetch_k)

        if method == RetrievalMethod.SIMILARITY_SEARCH:
            return self.retriever.similarity_search(query, k)

        if method == RetrievalMethod.HYBRID:
            return self.retriever.hybrid_search(query, k)

        if method == RetrievalMethod.CONTEXTUAL:
            return self.retriever.contextual_search(query=query, k=k, llm_provider=self.llm_provider)

        raise ValueError(f"Unsupported retrieval method: {method}")

    def generate_answer(
        self,
        query: str,
        k: Optional[int] = None,
        retrieve_method: Optional[RetrievalMethod] = None,
    ) -> Dict[str, Any]:
        """
        Generate a final answer from retrieved documents and the generation pipeline.

        :param k: override the top-k count for this call only (defaults to the
            pipeline's constructed k when None).
        :param retrieve_method: override the retrieval method for this call only
            (defaults to the pipeline's constructed retrieve_method when None).
        """
        return self.generation_pipeline.generate_response(
            query, k=k, retrieve_method=retrieve_method
        )