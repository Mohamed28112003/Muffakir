"""
Backward-compatible wrapper for MuffakirSyntheticData.
Delegates generation to SyntheticData.pipeline.SyntheticDataPipeline.
"""
from __future__ import annotations

from typing import Dict, Any, Optional
from PromptManager.PromptManager import MuffakirPrompt


class MuffakirSyntheticData:
    """
    Backward-compatible wrapper for MuffakirSyntheticData.
    Maintains exact original interface while delegating to SyntheticDataPipeline.
    """

    DEFAULT_CONFIG = {
        "data_dir": None,
        "api_key": None,
        "llm_provider": None,
        "llm_model": None,
        "llm_temperature": 0.3,
        "llm_max_tokens": 2000,
        "chunk_size": 600,
        "chunk_overlap": 200,
        "chunking_method": "recursive",
        "chunking": None,
        "use_ocr": False,
        "azure_endpoint": None,
        "azure_api_key": None,
        "document_parser": None,
        "document_parser_config": {},
        "output_dir": "./muffakir_synthetic_data",
        "save_frequency": 10,
        "output_format": ["csv", "excel"],
        "max_retries": 3,
        "skip_empty_chunks": True,
        "min_chunk_length": 50,
        "min_question_length": 10,
        "min_answer_length": 15,
        "validate_qa_pairs": True,
        "language": "ar",
        "prompt_overrides": {},
    }

    def __init__(self, config: Dict[str, Any], prompt_manager: Optional[MuffakirPrompt] = None):
        merged = self.DEFAULT_CONFIG.copy()
        merged.update(user_config if (user_config := config) else {})

        from Muffakir.dependency_validation import validate_synthetic_data_dependencies

        validate_synthetic_data_dependencies(merged)
        from SyntheticData.pipeline import SyntheticDataPipeline

        self.pipeline = SyntheticDataPipeline(
            config=merged,
            prompt_manager=prompt_manager
        )

    def generate_dataset(
        self,
        custom_prompt: Optional[str] = None,
        max_chunks: Optional[int] = None
    ) -> Any:
        """Generate synthetic dataset and return pandas DataFrame."""
        df, _ = self.pipeline.run(
            custom_prompt=custom_prompt,
            max_chunks=max_chunks
        )
        return df
