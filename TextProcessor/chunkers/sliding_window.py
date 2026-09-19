from typing import List
from .base import BaseChunker

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

class SlidingWindowChunker(BaseChunker):
    """
    Sliding Window Chunking Strategy.
    Moves a fixed-size window across the text by a specific step size.
    Good for dense, information-heavy documents where context can span anywhere.
    """
    
    def __init__(self, window_size: int = 500, step_size: int = 250):
        self.window_size = window_size
        self.step_size = step_size

    @property
    def name(self) -> str:
        return "sliding_window"

    def chunk(self, documents: List[Document]) -> List[Document]:
        chunked_docs = []
        for doc in documents:
            text = doc.page_content
            # Basic character-based sliding window
            for i in range(0, len(text), self.step_size):
                chunk_text = text[i:i + self.window_size]
                if not chunk_text.strip():
                    continue
                # Copy metadata
                new_doc = Document(page_content=chunk_text, metadata=doc.metadata.copy())
                chunked_docs.append(new_doc)
                # Break early if we've reached the end
                if i + self.window_size >= len(text):
                    break
                    
        return chunked_docs

    def get_search_space(self) -> dict:
        return {
            "chunker": ["sliding_window"],
            "sliding_window_size": {"type": "int", "low": 200, "high": 1000, "step": 100},
            "sliding_step_size": {"type": "int", "low": 50, "high": 500, "step": 50},
        }
