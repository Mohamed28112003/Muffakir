from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class QAPair(BaseModel):
    """A single validated Q&A pair generated from a document chunk."""

    question: str = Field(
        ...,
        min_length=5,
        description="The generated question in Arabic or English."
    )
    answer: str = Field(
        ...,
        min_length=5,
        description="The generated answer corresponding to the question."
    )
    context: str = Field(
        ...,
        min_length=10,
        description="Source document chunk text."
    )
    chunk_id: int = Field(
        default=0,
        ge=0,
        description="Identifier of the source chunk."
    )
    source_file: str = Field(
        default="unknown",
        description="Path or filename of the source document."
    )
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the Q&A pair was generated."
    )

    @field_validator("question", "answer", "context", mode="before")
    @classmethod
    def strip_text(cls, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        return str(value) if value is not None else ""


class LLMQAOutput(BaseModel):
    """Schema for LLM structured output of Q&A generation."""

    question: str = Field(
        ...,
        description="The generated question based on the provided text."
    )
    answer: str = Field(
        ...,
        description="The detailed and precise answer based strictly on the text."
    )


class GenerationStats(BaseModel):
    """Statistics returned after synthetic dataset generation."""

    total_chunks: int = Field(default=0, ge=0)
    successful: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    total_retries: int = Field(default=0, ge=0)
    dataset_size: int = Field(default=0, ge=0)
    sources: List[str] = Field(default_factory=list)


class SyntheticDataConfig(BaseModel):
    """Validated configuration model for Synthetic Data generation."""

    data_dir: str = Field(..., description="Directory containing input documents.")
    api_key: str = Field(..., description="API key for LLM provider.")
    llm_provider: str = Field(..., description="LLM provider name (e.g. 'together', 'openai', 'groq').")
    llm_model: str = Field(..., description="Model identifier string.")
    base_url: Optional[str] = Field(default=None, description="Custom OpenAI-compatible base_url (for 'custom'/'custom_openai' providers).")
    llm_temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=2000, gt=0)
    llm_parameters: Optional[Dict[str, Any]] = None
    prompt_overrides: Dict[str, str] = Field(default_factory=dict)
    chunk_size: int = Field(default=600, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    chunking_method: str = Field(default="recursive")
    chunking: Optional[Any] = Field(default=None, description="Optional injected MuffakirChunking instance.")
    use_ocr: bool = Field(default=False)
    azure_endpoint: Optional[str] = Field(default=None)
    azure_api_key: Optional[str] = Field(default=None)
    document_parser: Optional[str] = Field(default=None)
    document_parser_config: Dict[str, Any] = Field(default_factory=dict)
    output_dir: str = Field(default="./muffakir_synthetic_data")
    save_frequency: int = Field(default=10, gt=0)
    output_format: List[str] = Field(default_factory=lambda: ["csv", "excel"])
    max_retries: int = Field(default=3, gt=0)
    skip_empty_chunks: bool = Field(default=True)
    min_chunk_length: int = Field(default=50, ge=0)
    min_question_length: int = Field(default=10, ge=0)
    min_answer_length: int = Field(default=15, ge=0)
    validate_qa_pairs: bool = Field(default=True)
    language: str = Field(default="ar")

    @field_validator("data_dir")
    @classmethod
    def validate_data_dir(cls, v: str) -> str:
        path = Path(v)
        if not path.exists():
            raise FileNotFoundError(f"Data directory does not exist: {v}")
        return v

    @field_validator("output_format")
    @classmethod
    def validate_formats(cls, v: List[str]) -> List[str]:
        supported = {"csv", "excel", "json"}
        invalid = [fmt for fmt in v if fmt not in supported]
        if invalid:
            raise ValueError(f"Unsupported output format(s): {invalid}. Supported: {supported}")
        return v
