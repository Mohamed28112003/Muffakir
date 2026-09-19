import logging
from typing import List, Union, Optional

from .MuffakirTextCleaner import MuffakirTextCleaner
from .chunkers import BaseChunker, create_chunker

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

logger = logging.getLogger(__name__)


class MuffakirChunking:
    """
    Orchestrator for text cleaning and chunking.
    Combines a pluggable text cleaning pipeline (MuffakirTextCleaner) with a
    pluggable chunking strategy (BaseChunker).

    This is the main entry point for the SDK's chunking phase.
    """

    def __init__(
        self,
        chunker: Union[str, BaseChunker] = "recursive",
        chunker_config: dict = None,
        text_cleaner: Optional[MuffakirTextCleaner] = None,
        language: str = "auto",
    ):
        """
        Initialize the orchestrator.

        Args:
            chunker: Either a string name ("recursive", "semantic", etc.) or a BaseChunker instance.
            chunker_config: If passing a string for `chunker`, these are its kwargs.
            text_cleaner: Optional custom MuffakirTextCleaner instance.
            language: The base language for cleaning if no custom text_cleaner is provided.
        """
        if chunker_config is None:
            chunker_config = {}

        # Instantiate or assign the chunker
        if isinstance(chunker, str):
            self.chunker = create_chunker(chunker, **chunker_config)
        else:
            self.chunker = chunker

        # Instantiate or assign the text cleaner
        self.text_cleaner = text_cleaner or MuffakirTextCleaner(language=language)

    def process(self, documents: List[Document]) -> List[Document]:
        """
        Apply text cleaning and then chunking.

        Args:
            documents: List of raw LangChain Document objects.

        Returns:
            List[Document]: Cleaned and chunked documents.
        """
        logger.info(
            "Cleaning %d documents with language strategy '%s'...",
            len(documents),
            self.text_cleaner.language,
        )

        # 1. Clean the text content
        cleaned_docs = []
        for doc in documents:
            cleaned_content = self.text_cleaner.clean(doc.page_content)

            # Skip chunks that became empty after cleaning
            if not cleaned_content.strip():
                continue

            cleaned_docs.append(Document(
                page_content=cleaned_content,
                metadata=doc.metadata.copy()
            ))

        # 2. Chunk the documents
        logger.info("Chunking with strategy '%s'...", self.chunker.name)
        chunked_docs = self.chunker.chunk(cleaned_docs)

        logger.info("Resulted in %d final chunks.", len(chunked_docs))
        return chunked_docs

    def get_search_space(self) -> dict:
        """
        Merge chunker + cleaner search spaces for Automated Architecture Search (AAS).
        """
        return {
            **self.chunker.get_search_space(),
            **self.text_cleaner.get_search_space(),
        }

    def list_strategies(self) -> None:
        """
        Log information about the current strategies.
        """
        logger.info("Chunking Strategy: %s", self.chunker.name)
        self.text_cleaner.list_steps()
