import os
import logging
from typing import List, Optional, Any
from pydantic import ValidationError

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from TextProcessor.ChunkingAndProcessing import ChunkingAndProcessing
from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import MuffakirPrompt
from DocumentParser import BaseDocumentParser

from .models import QAPair, SyntheticDataConfig
from .parser import QAParser

logger = logging.getLogger(__name__)


class QAGenerator:
    """
    Processes input documents into text chunks and uses LLM + QAParser
    to generate validated Pydantic `QAPair` instances.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: MuffakirPrompt,
        config: SyntheticDataConfig,
        document_parser: Optional[BaseDocumentParser] = None,
        muffakir_chunking: Optional[Any] = None
    ):
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager
        self.config = config
        self.parser = QAParser(prompt_manager=prompt_manager)

        self.chunking_processor = ChunkingAndProcessing(
            directory_path=config.data_dir,
            document_parser=document_parser,
            muffakir_chunking=muffakir_chunking
        )

    def process_and_chunk_data(self) -> List[Document]:
        """Process documents in data_dir into chunked Document objects."""
        try:
            logger.info(f"Processing and chunking documents from: {self.config.data_dir}")
            chunks = self.chunking_processor.process_all(
                use_ocr=self.config.use_ocr,
                ocr_output_dir=os.path.join(self.config.output_dir, "ocr_results") if self.config.use_ocr else None
            )
            logger.info(f"Successfully generated {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return []

    def generate_single_pair(
        self,
        chunk_content: str,
        chunk_id: int = 0,
        source_file: str = "unknown"
    ) -> Optional[QAPair]:
        """
        Generate and validate a single `QAPair` from a chunk string.
        Returns a validated `QAPair` or `None` if validation fails.
        """
        parsed_dict = self.parser.generate_and_parse(
            context=chunk_content,
            llm_provider=self.llm_provider
        )

        q = parsed_dict.get("question", "")
        a = parsed_dict.get("answer", "")

        # Check configurable minimum lengths
        if len(q) < self.config.min_question_length or len(a) < self.config.min_answer_length:
            logger.warning(
                f"Generated Q&A failed length validation (question_len={len(q)}, answer_len={len(a)})."
            )
            return None

        try:
            # Pydantic schema validation
            pair = QAPair(
                question=q,
                answer=a,
                context=chunk_content,
                chunk_id=chunk_id,
                source_file=source_file
            )
            return pair
        except ValidationError as ve:
            logger.warning(f"QAPair Pydantic validation failed: {ve}")
            return None
