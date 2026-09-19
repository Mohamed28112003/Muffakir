from typing import List, Optional
from .base import BaseChunker

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

class RecursiveChunker(BaseChunker):
    """
    Recursive Character Chunking Strategy.
    The default and generally most robust chunking method.
    It tries to split on paragraphs, then sentences, then words, to keep 
    semantically related pieces of text together as much as possible.
    """
    
    def __init__(
        self, 
        size: int = 600, 
        overlap: int = 200, 
        separators: Optional[List[str]] = None
    ):
        self.size = size
        self.overlap = overlap
        
        # Default Arabic/English aware separators
        self.separators = separators or [
            "\n\n", 
            "\n", 
            " ", 
            "",
            # Arabic specific sentence enders
            "؟", "!", "؛", "٫"
        ]
        
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.size,
            chunk_overlap=self.overlap,
            separators=self.separators
        )

    @property
    def name(self) -> str:
        return "recursive"

    def chunk(self, documents: List[Document]) -> List[Document]:
        return self._splitter.split_documents(documents)

    def get_search_space(self) -> dict:
        return {
            "chunker": ["recursive"],
            "recursive_chunk_size": {"type": "int", "low": 200, "high": 1200, "step": 100},
            "recursive_chunk_overlap": {"type": "int", "low": 0, "high": 300, "step": 50},
        }
