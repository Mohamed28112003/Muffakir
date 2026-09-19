import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd

from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import MuffakirPrompt
from Muffakir.Enums import ProviderName, PROVIDER_MAPPING, resolve_provider_name

from .models import QAPair, SyntheticDataConfig, GenerationStats
from .generator import QAGenerator
from .exporter import DatasetExporter

logger = logging.getLogger(__name__)


class SyntheticDataPipeline:
    """
    Main Orchestrator for Synthetic Data Generation in Muffakir RAG.

    Coordinates document parsing, chunking, LLM Q&A generation with retry logic,
    Pydantic schema validation, periodic checkpointing, and dataset export.
    """

    PROVIDER_MAPPING = PROVIDER_MAPPING

    def __init__(
        self,
        config: Dict[str, Any],
        prompt_manager: Optional[MuffakirPrompt] = None
    ):
        # Build validated Pydantic configuration model
        if isinstance(config, SyntheticDataConfig):
            self.config = config
        else:
            self.config = SyntheticDataConfig(**config)

        self._validate_provider()
        self._initialize_components(prompt_manager)
        self.exporter = DatasetExporter(output_dir=self.config.output_dir)

    def _validate_provider(self):
        try:
            resolve_provider_name(self.config.llm_provider)
        except ValueError as e:
            raise ValueError(str(e)) from e

    def _initialize_components(self, prompt_manager: Optional[MuffakirPrompt] = None):
        self.llm_provider = LLMProvider(
            parameters=self.config.llm_parameters,
            api_key=self.config.api_key,
            provider=resolve_provider_name(self.config.llm_provider),
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
            base_url=getattr(self.config, "base_url", None),
        )


        self.prompt_manager = prompt_manager or MuffakirPrompt(
            language=self.config.language,
            overrides=self.config.prompt_overrides,
        )

        # Initialize Document Parser if configured
        doc_parser = None
        if self.config.document_parser:
            from DocumentParser import create_document_parser
            doc_parser = create_document_parser(
                provider=self.config.document_parser,
                **self.config.document_parser_config
            )
        elif self.config.azure_endpoint and self.config.azure_api_key:
            from DocumentParser import create_document_parser
            doc_parser = create_document_parser(
                provider="azure",
                endpoint=self.config.azure_endpoint,
                api_key=self.config.azure_api_key
            )

        # Initialize Chunking Orchestrator
        chunking = self.config.chunking
        if chunking is None:
            from TextProcessor.MuffakirChunking import MuffakirChunking
            chunking = MuffakirChunking(
                chunker=self.config.chunking_method,
                chunker_config={
                    "size": self.config.chunk_size,
                    "overlap": self.config.chunk_overlap
                },
                language=self.config.language
            )

        self.generator = QAGenerator(
            llm_provider=self.llm_provider,
            prompt_manager=self.prompt_manager,
            config=self.config,
            document_parser=doc_parser,
            muffakir_chunking=chunking
        )

    def run(
        self,
        custom_prompt: Optional[str] = None,
        max_chunks: Optional[int] = None
    ) -> Tuple[pd.DataFrame, GenerationStats]:
        """
        Run the complete synthetic dataset generation pipeline.

        Returns:
            Tuple[pd.DataFrame, GenerationStats]: The generated DataFrame and run stats.
        """
        logger.info("🚀 Starting Synthetic Q&A Dataset Generation Pipeline...")

        if custom_prompt:
            self.prompt_manager.add_prompt("QA", custom_prompt)
            logger.info("✅ Custom prompt registered for QA generation.")

        chunks = self.generator.process_and_chunk_data()
        if not chunks:
            logger.warning("⚠️ No chunks were generated from the input directory.")
            return self.exporter.to_dataframe([]), GenerationStats()

        if max_chunks and max_chunks < len(chunks):
            chunks = chunks[:max_chunks]
            logger.info(f"📋 Processing capped to max_chunks={max_chunks}")

        if self.config.skip_empty_chunks:
            original_len = len(chunks)
            chunks = [c for c in chunks if len(c.page_content.strip()) >= self.config.min_chunk_length]
            filtered = original_len - len(chunks)
            if filtered > 0:
                logger.info(f"🔍 Filtered out {filtered} chunks shorter than min_chunk_length={self.config.min_chunk_length}")

        generated_pairs: List[QAPair] = []
        pending_checkpoint_pairs: List[QAPair] = []

        successful = 0
        failed = 0
        total_retries = 0
        sources_set = set()

        for i, chunk in enumerate(chunks, 1):
            logger.info(f"📝 Processing chunk {i}/{len(chunks)}")
            src_file = chunk.metadata.get("source_file", "unknown")
            chunk_id = chunk.metadata.get("chunk_id", i)
            sources_set.add(src_file)

            success = False
            retries = 0

            while not success and retries < self.config.max_retries:
                pair = self.generator.generate_single_pair(
                    chunk_content=chunk.page_content,
                    chunk_id=chunk_id,
                    source_file=src_file
                )

                if pair is not None:
                    generated_pairs.append(pair)
                    pending_checkpoint_pairs.append(pair)
                    successful += 1
                    success = True

                    if len(pending_checkpoint_pairs) >= self.config.save_frequency:
                        self.exporter.append_checkpoint_csv(pending_checkpoint_pairs)
                        pending_checkpoint_pairs = []
                else:
                    retries += 1
                    total_retries += 1
                    if retries < self.config.max_retries:
                        logger.warning(f"⚠️ Chunk {i} validation failed. Retrying ({retries}/{self.config.max_retries})...")

            if not success:
                failed += 1
                logger.warning(f"❌ Chunk {i} failed after {self.config.max_retries} attempts.")

        # Final remaining checkpoint save
        if pending_checkpoint_pairs:
            self.exporter.append_checkpoint_csv(pending_checkpoint_pairs)

        df = self.exporter.to_dataframe(generated_pairs)
        self.exporter.save_dataframe(df, filename_prefix="synthetic_qa_final", formats=self.config.output_format)

        stats = GenerationStats(
            total_chunks=len(chunks),
            successful=successful,
            failed=failed,
            total_retries=total_retries,
            dataset_size=len(generated_pairs),
            sources=list(sources_set)
        )

        logger.info(
            f"🎉 Dataset Generation Completed: {successful} successful, {failed} failed out of {len(chunks)} chunks."
        )
        return df, stats
