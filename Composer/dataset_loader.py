"""
DatasetLoader - Load evaluation datasets from CSV or generate synthetically.

Supports two scenarios:
1. User provides a CSV file with question, context, answer columns
2. User provides raw documents and we auto-generate synthetic eval data
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd

from Muffakir.exceptions import DatasetError

from SyntheticData.models import QAPair

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads evaluation datasets for MuffakirComposer.

    Supports loading from CSV files or generating synthetic data
    from raw documents using the SyntheticDataPipeline.
    """

    # Expected CSV column names
    REQUIRED_COLUMNS = {"question", "context", "answer"}
    OPTIONAL_COLUMNS = {"chunk_id", "source_file"}

    # A single malformed row is tolerated (logged + skipped), but if too many
    # rows fail, the input is probably fundamentally wrong (bad file, wrong
    # column mapping) - raise instead of silently returning a near-empty dataset.
    MAX_ROW_FAILURE_RATE = 0.5
    
    @staticmethod
    def from_csv(csv_path: str) -> List[QAPair]:
        """
        Load QAPairs from a CSV file.
        
        Expected CSV format:
        - question: The question text
        - context: The source context/passage
        - answer: The expected answer
        - chunk_id (optional): Identifier for the source chunk
        - source_file (optional): Path or name of source document
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            List of QAPair objects
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If required columns are missing
        """
        path = Path(csv_path)
        
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        logger.info(f"Loading evaluation dataset from: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        missing_cols = DatasetLoader.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"CSV is missing required columns: {missing_cols}. "
                f"Required: {DatasetLoader.REQUIRED_COLUMNS}"
            )
        
        # Convert to QAPair objects
        pairs = []
        skipped = 0
        for idx, row in df.iterrows():
            try:
                pair = QAPair(
                    question=str(row["question"]),
                    answer=str(row["answer"]),
                    context=str(row["context"]),
                    chunk_id=int(row.get("chunk_id", idx)),
                    source_file=str(row.get("source_file", "unknown")),
                )
                pairs.append(pair)
            except Exception as e:
                skipped += 1
                logger.warning(f"Skipping row {idx}: {e}")

        total = len(df)
        if total and (not pairs or skipped / total > DatasetLoader.MAX_ROW_FAILURE_RATE):
            raise DatasetError(
                f"Failed to parse {skipped}/{total} rows ({skipped / total:.0%}) "
                f"from CSV {csv_path}; exceeds the "
                f"{DatasetLoader.MAX_ROW_FAILURE_RATE:.0%} threshold.",
                context={"csv_path": csv_path, "skipped": skipped, "total": total},
            )

        logger.info(f"Loaded {len(pairs)} QAPairs from CSV")
        return pairs
    
    @staticmethod
    def generate_from_documents(
        data_dir: str,
        llm_config: Dict[str, Any],
        max_samples: int = 50,
        language: str = "ar",
        prompt_overrides: Optional[Dict[str, str]] = None,
    ) -> List[QAPair]:
        """
        Generate synthetic evaluation dataset from raw documents.
        
        Uses the existing SyntheticDataPipeline to generate Q&A pairs
        from document chunks.
        
        Args:
            data_dir: Directory containing source documents
            llm_config: LLM configuration dictionary with keys:
                - api_key: API key for LLM provider
                - llm_provider: Provider name (together, openai, groq, etc.)
                - llm_model: Model identifier
                - chunk_size (optional): Chunk size for processing
                - chunk_overlap (optional): Chunk overlap
            max_samples: Maximum number of Q&A pairs to generate
            language: Language for generation ('ar' or 'en')
            
        Returns:
            List of QAPair objects
        """
        from SyntheticData.pipeline import SyntheticDataPipeline
        from PromptManager.PromptManager import MuffakirPrompt
        
        logger.info(f"Generating synthetic eval dataset from: {data_dir}")
        logger.info(f"Max samples: {max_samples}")
        
        # Build synthetic data config
        synth_config = {
            "data_dir": data_dir,
            "api_key": llm_config["api_key"],
            "llm_provider": llm_config["llm_provider"],
            "llm_model": llm_config["llm_model"],
            "llm_temperature": llm_config.get("llm_temperature", 0.3),
            "llm_max_tokens": llm_config.get("llm_max_tokens", 2000),
            "llm_parameters": llm_config.get("llm_parameters"),
            "chunk_size": llm_config.get("chunk_size", 600),
            "chunk_overlap": llm_config.get("chunk_overlap", 200),
            "chunking_method": llm_config.get("chunking_method", "recursive"),
            "language": language,
            "output_dir": llm_config.get("output_dir", "./muffakir_synthetic_data"),
            "max_retries": llm_config.get("max_retries", 3),
        }
        
        # Add optional configs if present
        if "document_parser" in llm_config:
            synth_config["document_parser"] = llm_config["document_parser"]
            synth_config["document_parser_config"] = llm_config.get("document_parser_config", {})
        
        pipeline = SyntheticDataPipeline(
            config=synth_config,
            prompt_manager=MuffakirPrompt(
                language=language,
                overrides=prompt_overrides,
            ),
        )
        df, stats = pipeline.run(max_chunks=max_samples)
        
        # Convert DataFrame to QAPair list
        pairs = []
        skipped = 0
        for _, row in df.iterrows():
            try:
                pair = QAPair(
                    question=str(row["question"]),
                    answer=str(row["answer"]),
                    context=str(row["context"]),
                    chunk_id=int(row.get("chunk_id", 0)),
                    source_file=str(row.get("source_file", "unknown")),
                )
                pairs.append(pair)
            except Exception as e:
                skipped += 1
                logger.warning(f"Skipping generated pair: {e}")

        total = len(df)
        if total and (not pairs or skipped / total > DatasetLoader.MAX_ROW_FAILURE_RATE):
            raise DatasetError(
                f"Failed to convert {skipped}/{total} generated pairs "
                f"({skipped / total:.0%}) from {data_dir}; exceeds the "
                f"{DatasetLoader.MAX_ROW_FAILURE_RATE:.0%} threshold.",
                context={"data_dir": data_dir, "skipped": skipped, "total": total},
            )

        logger.info(
            f"Generated {len(pairs)} QAPairs "
            f"(successful: {stats.successful}, failed: {stats.failed})"
        )

        if not pairs:
            raise DatasetError(
                "Automatic evaluation-dataset generation produced no valid Q&A "
                f"pairs ({stats.failed}/{stats.total_chunks} chunks failed). "
                "Check the dataset LLM provider, model, credentials, and "
                "generation errors, or supply an existing evaluation dataset.",
                context={
                    "successful_chunks": stats.successful,
                    "failed_chunks": stats.failed,
                    "total_chunks": stats.total_chunks,
                },
            )

        return pairs
    
    @staticmethod
    def load_or_generate(
        eval_dataset: Optional[Union[str, List[QAPair]]],
        data_dir: str,
        llm_config: Dict[str, Any],
        max_samples: int = 50,
        language: str = "ar",
        prompt_overrides: Optional[Dict[str, str]] = None,
    ) -> List[QAPair]:
        """
        Smart loader that handles all three scenarios:
        1. eval_dataset is a CSV path string → load from CSV
        2. eval_dataset is a list of QAPairs → use directly
        3. eval_dataset is None → generate from documents
        
        Args:
            eval_dataset: CSV path, QAPair list, or None
            data_dir: Directory containing source documents (for generation)
            llm_config: LLM configuration (for generation)
            max_samples: Max samples to generate (if generating)
            language: Language for generation
            
        Returns:
            List of QAPair objects
        """
        if eval_dataset is None:
            # Scenario A: Generate synthetic data
            logger.info("No eval dataset provided, generating synthetic data...")
            return DatasetLoader.generate_from_documents(
                data_dir=data_dir,
                llm_config=llm_config,
                max_samples=max_samples,
                language=language,
                prompt_overrides=prompt_overrides,
            )
        
        elif isinstance(eval_dataset, str):
            # Scenario B: Load from CSV
            logger.info(f"Loading eval dataset from CSV: {eval_dataset}")
            return DatasetLoader.from_csv(eval_dataset)
        
        elif isinstance(eval_dataset, list):
            # Scenario C: Use provided QAPairs directly (or convert dicts if provided)
            pairs: List[QAPair] = []
            for item in eval_dataset:
                if isinstance(item, QAPair):
                    pairs.append(item)
                elif isinstance(item, dict):
                    pairs.append(
                        QAPair(
                            question=str(item.get("question", "")),
                            answer=str(item.get("answer") or item.get("ground_truth", "")),
                            context=str(item.get("context", "")),
                            chunk_id=int(item.get("chunk_id", 0)),
                            source_file=str(item.get("source_file", "manual_eval")),
                        )
                    )
                else:
                    pairs.append(item)
            logger.info(f"Using provided eval dataset: {len(pairs)} QAPairs")
            return pairs
        
        else:
            raise ValueError(
                f"Invalid eval_dataset type: {type(eval_dataset)}. "
                "Expected str (CSV path), List[QAPair], List[dict], or None."
            )
