"""
Backward-compatible Reranker entry point.
Wraps create_reranker() factory for drop-in replacement.
"""
from typing import Optional, Any
from .base import BaseReranker
from .factory import create_reranker, get_reranker_spec
from Embedding.EmbeddingProvider import EmbeddingProvider


class Reranker:
    """
    Backward-compatible Reranker wrapper.
    Delegates to create_reranker() factory and wraps as BaseReranker instance.
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        model_name: str = 'mohamed2811/Muffakir_Embedding',
        reranking_method: str = 'semantic_similarity',
        cross_encoder_model_name: Optional[str] = None,
        llm_provider: Optional[Any] = None,
        prompt_manager: Optional[Any] = None,
        **kwargs: Any
    ):
        resolved_method = get_reranker_spec(reranking_method).name
        selected_model = (
            cross_encoder_model_name
            if resolved_method in {"cross_encoder", "pointwise"}
            else model_name
        )
        self._reranker: BaseReranker = create_reranker(
            method=resolved_method,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            model_name=selected_model,
            **kwargs
        )

    def rerank(self, query: str, documents, top_k=None):
        return self._reranker.rerank(query, documents, top_k=top_k)


def rerank_documents(reranker: Optional["Reranker"], query: str, documents: list) -> list:
    """Rerank *documents* with an already-constructed reranker instance.

    Shared by `Generation/RAGGenerationPipeline.py` (full-RAG mode) and
    `Muffakir/MuffakirRetrieval.py` (retrieval-only mode) so both call the
    exact same rerank logic instead of drifting apart.
    """
    if reranker:
        return reranker.rerank(query, documents)
    return documents
