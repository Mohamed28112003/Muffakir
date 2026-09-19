from abc import ABC, abstractmethod
from typing import List

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

class BaseChunker(ABC):
    """
    Abstract base class for all text chunking strategies.
    Every custom chunker must inherit from this and implement the interface.
    """

    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents (List[Document]): The input documents.
            
        Returns:
            List[Document]: The resulting chunked documents.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name of the chunking strategy.
        """
        pass

    @abstractmethod
    def get_search_space(self) -> dict:
        """
        Return the hyperparameter search space for Automated Architecture Search (AAS).
        Returns a dictionary mapping parameter names to their types/ranges
        (e.g., categorical choices, int ranges, float ranges) compatible with Optuna.
        """
        pass
