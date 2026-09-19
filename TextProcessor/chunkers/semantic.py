from typing import List, Any
from .base import BaseChunker

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

class SemanticChunker(BaseChunker):
    """
    Semantic Chunking Strategy.
    Uses an embedding model to determine semantic similarity between sentences,
    splitting the text when the semantic topic changes.
    
    Requires: pip install langchain-experimental
    """
    
    def __init__(self, embeddings: Any, breakpoint_threshold_type: str = "percentile", breakpoint_threshold_amount: float = 95.0):
        """
        Args:
            embeddings: LangChain Embeddings model instance.
            breakpoint_threshold_type: "percentile", "standard_deviation", "interquartile", or "gradient".
            breakpoint_threshold_amount: Threshold amount for the chosen type.
        """
        try:
            from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker
        except ImportError:
            raise ImportError(
                "The 'langchain-experimental' package is required to use SemanticChunker. "
                "Install it with: pip install langchain-experimental"
            )
            
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        
        # Instantiate the LangChain SemanticChunker
        self._splitter = LCSemanticChunker(
            embeddings,
            breakpoint_threshold_type=self.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount
        )

    @property
    def name(self) -> str:
        return "semantic"

    def chunk(self, documents: List[Document]) -> List[Document]:
        return self._splitter.split_documents(documents)

    def get_search_space(self) -> dict:
        return {
            "chunker": ["semantic"],
            "semantic_threshold_type": ["percentile", "standard_deviation"],
            "semantic_threshold_amount": {"type": "float", "low": 80.0, "high": 99.0},
        }
