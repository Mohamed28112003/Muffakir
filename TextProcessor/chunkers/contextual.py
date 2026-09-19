from typing import List
from .base import BaseChunker

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

class ContextualChunker(BaseChunker):
    """
    Contextual Chunking Strategy.
    Wraps another base chunker. After chunking, it prepends contextual 
    information (like the source filename or a sliding window of previous chunks)
    to the beginning of the chunk text, so the LLM doesn't lose global context.
    
    This is an implementation of a 'Pre-Embedding' optimization stage.
    """
    
    def __init__(self, base_chunker: BaseChunker, include_source: bool = True, include_prev_chunk: bool = False):
        self.base_chunker = base_chunker
        self.include_source = include_source
        self.include_prev_chunk = include_prev_chunk

    @property
    def name(self) -> str:
        return f"contextual_{self.base_chunker.name}"

    def chunk(self, documents: List[Document]) -> List[Document]:
        # 1. Chunk using the underlying strategy
        initial_chunks = self.base_chunker.chunk(documents)
        
        if not initial_chunks:
            return []

        # 2. Enrich chunks with context
        enriched_chunks = []
        for i, chunk in enumerate(initial_chunks):
            header_parts = []
            
            # Add Document Source Context
            if self.include_source:
                source = chunk.metadata.get("source", chunk.metadata.get("original_filename", "Unknown Document"))
                header_parts.append(f"[Source: {source}]")
                
            # Add Previous Chunk Context (simplified implementation)
            if self.include_prev_chunk and i > 0:
                # To prevent exponential growth, we just take the last 100 chars of the prev chunk
                prev_text = initial_chunks[i-1].page_content
                prev_snippet = prev_text[-100:] if len(prev_text) > 100 else prev_text
                header_parts.append(f"[Previous Context: ...{prev_snippet}]")
                
            if header_parts:
                header = " ".join(header_parts)
                enriched_text = f"{header}\n{chunk.page_content}"
            else:
                enriched_text = chunk.page_content
                
            enriched_chunks.append(Document(
                page_content=enriched_text,
                metadata=chunk.metadata.copy()
            ))
            
        return enriched_chunks

    def get_search_space(self) -> dict:
        # We merge our search space with the base chunker's space
        space = self.base_chunker.get_search_space()
        space["contextual_include_source"] = [True, False]
        space["contextual_include_prev"] = [True, False]
        return space
