import logging
import numpy as np
from typing import List, Tuple, Optional
from .base import BaseReranker
from Embedding.EmbeddingProvider import EmbeddingProvider

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

logger = logging.getLogger(__name__)


class SemanticSimilarityReranker(BaseReranker):
    """
    Semantic Similarity Reranker.
    Scores documents using cosine similarity between query and document embeddings.
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        model_name: str = 'mohamed2811/Muffakir_Embedding'
    ):
        from Embedding import create_embedding_provider
        self.embedding_provider = embedding_provider or create_embedding_provider(
            provider="sentence_transformers",
            model_name=model_name
        )

    @property
    def name(self) -> str:
        return "semantic_similarity"

    def score(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        query_embedding = np.array(self.embedding_provider.embed_query(query))
        doc_embeddings = [
            np.array(self.embedding_provider.embed_documents([doc.page_content])[0])
            for doc in documents
        ]

        scores = []
        for doc, doc_emb in zip(documents, doc_embeddings):
            norm = np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            similarity = float(np.dot(query_embedding, doc_emb) / norm) if norm > 0 else 0.0
            scores.append((doc, similarity))

        return sorted(scores, key=lambda x: x[1], reverse=True)
