from typing import List, Literal
from .base import BaseChunker

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from langchain_text_splitters import CharacterTextSplitter, TokenTextSplitter

class FixedSizeChunker(BaseChunker):
    """
    Fixed-Size Chunking Strategy.
    Splits text blindly by character or token count, ignoring semantics.
    """
    
    def __init__(
        self, 
        size: int = 600, 
        overlap: int = 200, 
        unit: Literal["character", "token"] = "character"
    ):
        self.size = size
        self.overlap = overlap
        self.unit = unit
        
        if self.unit == "character":
            self._splitter = CharacterTextSplitter(
                separator="",  # Force rigid split if needed
                chunk_size=self.size,
                chunk_overlap=self.overlap
            )
        elif self.unit == "token":
            self._splitter = TokenTextSplitter(
                chunk_size=self.size,
                chunk_overlap=self.overlap
            )
        else:
            raise ValueError(f"Unsupported unit: {unit}")

    @property
    def name(self) -> str:
        return f"fixed_size_{self.unit}"

    def chunk(self, documents: List[Document]) -> List[Document]:
        return self._splitter.split_documents(documents)

    def get_search_space(self) -> dict:
        return {
            "chunker": ["fixed_size"],
            "fixed_size_unit": ["character", "token"],
            "fixed_size_chunk_size": {"type": "int", "low": 200, "high": 1200, "step": 100},
            "fixed_size_chunk_overlap": {"type": "int", "low": 0, "high": 300, "step": 50},
        }
