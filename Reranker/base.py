from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


class BaseReranker(ABC):
    """
    Abstract Base Class for all Reranking strategies in Muffakir RAG.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        pass

    @abstractmethod
    def score(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Score and rank documents against a query.

        Args:
            query (str): The user query.
            documents (List[Document]): Candidate documents to rank.

        Returns:
            List[Tuple[Document, float]]: Documents with scores, sorted descending.
        """
        pass

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Rerank documents and return the top-k most relevant ones.

        Args:
            query (str): The user query.
            documents (List[Document]): Retrieved documents to rerank.
            top_k (int, optional): Number of documents to return. Defaults to all.

        Returns:
            List[Document]: Reranked documents.
        """
        if not documents:
            return []

        top_k = top_k or len(documents)
        ranked = self.score(query, documents)
        return [doc for doc, _ in ranked[:top_k]]
