"""
Pydantic schemas for ComposerUI request and response contracts.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from LLMProvider.parameters import validate_parameters, validate_config_parameters, config_parameter_fields


class LLMEntry(BaseModel):
    provider: str
    model: str
    parameters: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_settings(self):
        if self.parameters is not None:
            self.parameters = validate_parameters(self.parameters, self.provider)
        return self


class ChunkingEntry(BaseModel):
    method: str
    size: int
    overlap: int


class SearchSpaceDict(BaseModel):
    query_expansion: Optional[List[str]] = None
    retrieval: Optional[List[str]] = None
    reranking: Optional[List[str]] = None
    reranking_model: Optional[List[str]] = None
    k: Optional[List[int]] = None
    llm: Optional[List[LLMEntry]] = None
    chunking: Optional[List[ChunkingEntry]] = None
    embedding_model: Optional[List[str]] = None
    vector_db_provider: Optional[List[str]] = None

    @field_validator("reranking_model")
    @classmethod
    def normalize_reranking_models(
        cls, value: Optional[List[str]]
    ) -> Optional[List[str]]:
        if value is None:
            return None

        normalized: List[str] = []
        seen = set()
        for model_id in value:
            cleaned = model_id.strip()
            if not cleaned:
                raise ValueError("reranking_model entries cannot be empty")
            if cleaned in seen:
                raise ValueError(
                    f"reranking_model contains duplicate model ID: {cleaned}"
                )
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized

    def to_composer_dict(self) -> Dict[str, Any]:
        """Convert to dict shape expected by ConfigSpace (plain dicts for compound stages,
        dropping unset/None fields)."""
        result: Dict[str, Any] = {}
        if self.query_expansion is not None:
            result["query_expansion"] = self.query_expansion
        if self.retrieval is not None:
            result["retrieval"] = self.retrieval
        if self.reranking is not None:
            result["reranking"] = self.reranking
        if self.reranking_model is not None:
            result["reranking_model"] = self.reranking_model
        if self.k is not None:
            result["k"] = self.k
        if self.llm is not None:
            result["llm"] = [e.model_dump(exclude_none=True) for e in self.llm]
        if self.chunking is not None:
            result["chunking"] = [
                {"method": e.method, "size": e.size, "overlap": e.overlap}
                for e in self.chunking
            ]
        if self.embedding_model is not None:
            result["embedding_model"] = self.embedding_model
        if self.vector_db_provider is not None:
            result["vector_db_provider"] = self.vector_db_provider
        return result


class TrainTestSplitConfig(BaseModel):
    enabled: bool = False
    test_size: float = 0.2
    random_state: int = 42


class CustomPricingEntry(BaseModel):
    """One per-run LLM price override, expressed in UI-friendly units."""

    provider: str
    model: str
    input_usd_per_million_tokens: float = Field(ge=0, allow_inf_nan=False)
    output_usd_per_million_tokens: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("provider cannot be empty")
        return normalized

    @field_validator("model")
    @classmethod
    def normalize_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("model cannot be empty")
        return normalized

    @property
    def canonical_key(self) -> str:
        return f"{self.provider}/{self.model}"

    def to_per_token_rate(self) -> Dict[str, float]:
        return {
            "input_cost_per_token": self.input_usd_per_million_tokens / 1_000_000,
            "output_cost_per_token": self.output_usd_per_million_tokens / 1_000_000,
        }


class CreateRunRequest(BaseModel):
    llm_parameters: Optional[Dict[str, Any]] = None
    judge_llm_parameters: Optional[Dict[str, Any]] = None
    query_transform_llm_parameters: Optional[Dict[str, Any]] = None
    reranker_llm_parameters: Optional[Dict[str, Any]] = None
    dataset_llm_parameters: Optional[Dict[str, Any]] = None
    llm_temperature: Optional[float] = None
    llm_max_tokens: Optional[int] = None

    def parameter_config(self):
        return config_parameter_fields(self.model_dump())

    @model_validator(mode="after")
    def validate_llm_settings(self):
        values = self.model_dump(exclude_none=True)
        validate_config_parameters(values, self.search_space.to_composer_dict())
        return self

    run_name: str
    language: Literal["ar", "en"] = "ar"
    prompt_overrides: Dict[str, str] = Field(default_factory=dict)
    custom_pricing: List[CustomPricingEntry] = Field(default_factory=list)
    # Local retrieval corpus. Required for vector/adaptive retrieval and not
    # used by web-search-only runs.
    documents_path: Optional[str] = None
    # Optional: not required when pipeline_mode="retrieval_only" (no main
    # LLM is ever called in that mode). Required otherwise -- enforced in
    # ComposerUI/backend/app.py::_preflight_check, not here, since that
    # requirement depends on another field's value.
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    # Optional dataset-generation LLM override. When omitted, automatic
    # dataset generation reuses the answer LLM configuration above.
    dataset_llm_provider: Optional[str] = None
    dataset_llm_model: Optional[str] = None
    dataset_api_key: Optional[str] = None
    dataset_base_url: Optional[str] = None
    embedding_model: Optional[str] = None
    vector_db_provider: Optional[str] = None
    search_space: SearchSpaceDict = Field(default_factory=SearchSpaceDict)
    metrics: Optional[List[str]] = None
    max_eval_samples: int = 50
    n_jobs: int = 4
    max_trials: Optional[int] = None
    max_runtime_minutes: Optional[float] = None
    # Eval dataset: "auto" (Composer generates it internally, current default
    # behavior), "existing" (eval_dataset_path points at a real file), or
    # "generated" (eval_dataset_path was produced by /api/synthetic-data/generate).
    eval_dataset_mode: Literal["auto", "existing", "generated"] = "auto"
    eval_dataset_path: Optional[str] = None
    # Document parser (OCR) — threaded into both synthetic-data generation and,
    # via Composer._index_documents, the actual retrieval-corpus indexing.
    document_parser: Optional[str] = None
    document_parser_config: Optional[Dict[str, Any]] = None
    use_ocr: bool = False
    train_test_split: Optional[TrainTestSplitConfig] = None
    # "auto"/"cpu"/"cuda" — forwarded to the embedding/reranker components.
    # None means the core library's own default ("auto") applies unchanged.
    device: Optional[str] = None
    # Judge LLM override: by default (all None) the evaluation judge reuses
    # the same llm_provider/llm_model/api_key/base_url as generation. Set
    # judge_llm_provider to use a separate model/account for scoring.
    judge_llm_provider: Optional[str] = None
    judge_llm_model: Optional[str] = None
    judge_api_key: Optional[str] = None
    judge_base_url: Optional[str] = None
    # "vector_db" (default, current behavior) indexes documents_path and
    # retrieves locally for every trial. "web_search_only" skips indexing
    # entirely -- every trial answers purely via a live web search provider
    # (Muffakir/MuffakirSearch.py), scored on generation metrics only (no
    # local corpus means retrieval metrics are unavailable in this mode).
    retrieval_source: Literal["vector_db", "web_search_only"] = "vector_db"
    # Adaptive fallback: only meaningful when retrieval_source="vector_db".
    # When true, a trial falls back to a live web search if local retrieval
    # isn't relevant enough (Generation/RAGGenerationPipeline.py, already
    # built into MuffakirRAG -- this just exposes the existing flag).
    adaptive_web_search: bool = False
    # Shared by both adaptive_web_search and retrieval_source="web_search_only".
    search_provider: Optional[str] = None
    search_provider_config: Optional[Dict[str, Any]] = None
    search_api_key: Optional[str] = None
    # "full_rag" (default) generates answers as today. "retrieval_only"
    # tests retrieval quality in isolation -- embedding/chunking/retrieval
    # method/reranking/query-transformation all still apply, but no answer
    # is ever generated (no main LLM is built). Mutually exclusive with
    # retrieval_source="web_search_only" (enforced server-side).
    pipeline_mode: Literal["full_rag", "retrieval_only"] = "full_rag"
    # Query-transform's own LLM. Required when query_expansion is swept to
    # any non-"none" value AND pipeline_mode="retrieval_only" -- that mode
    # has no main LLM to silently reuse for the transform call, unlike
    # full_rag mode.
    query_transform_llm_provider: Optional[str] = None
    query_transform_llm_model: Optional[str] = None
    query_transform_base_url: Optional[str] = None
    # Credential for the query-transform LLM above. Needed because
    # retrieval_only mode hides the main api_key field entirely (no main LLM
    # exists there), leaving no other way to supply a key for the transform
    # call. Never persisted to config.json -- same handling as search_api_key.
    query_transform_api_key: Optional[str] = None
    # Reranker configuration is run-wide while the selected method remains a
    # search-space dimension. Local model methods use reranking_model. The LLM
    # strategy can reuse the answer/query-transform LLM or use this override.
    reranking_model: Optional[str] = None
    reranker_llm_provider: Optional[str] = None
    reranker_llm_model: Optional[str] = None
    reranker_llm_api_key: Optional[str] = None
    reranker_llm_base_url: Optional[str] = None
    # The custom strategy calls an HTTP endpoint and keeps its credential out
    # of config.json and trace records.
    reranker_base_url: Optional[str] = None
    reranker_api_key: Optional[str] = None
    reranker_model: Optional[str] = None
    reranker_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    reranker_options: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("reranking_model")
    @classmethod
    def normalize_root_reranking_model(cls, value: Optional[str]) -> Optional[str]:
        """Preserve the legacy one-model API while normalizing user input."""
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("reranking_model cannot be empty when provided")
        return normalized

    @model_validator(mode="after")
    def reject_duplicate_custom_pricing(self) -> "CreateRunRequest":
        seen = set()
        duplicates = set()
        for entry in self.custom_pricing:
            if entry.canonical_key in seen:
                duplicates.add(entry.canonical_key)
            seen.add(entry.canonical_key)
        if duplicates:
            raise ValueError(
                "custom_pricing contains duplicate provider/model entries: "
                + ", ".join(sorted(duplicates))
            )
        return self

    def custom_pricing_per_token(self) -> Dict[str, Dict[str, float]]:
        """Convert API/UI per-million rates to PriceMap's per-token format."""
        return {
            entry.canonical_key: entry.to_per_token_rate()
            for entry in self.custom_pricing
        }


class PromptResolveRequest(BaseModel):
    language: Literal["ar", "en"] = "ar"
    pipeline_mode: Literal["full_rag", "retrieval_only"] = "full_rag"
    retrieval_source: Literal["vector_db", "web_search_only"] = "vector_db"
    adaptive_web_search: bool = False
    eval_dataset_mode: Literal["auto", "existing", "generated"] = "auto"
    search_space: SearchSpaceDict = Field(default_factory=SearchSpaceDict)
    metrics: Optional[List[str]] = None
    hallucination_check: bool = True
    hallucination_method: str = "text_cleaner"


class PromptDefinition(BaseModel):
    key: str
    label: str
    category: str
    description: str
    stage: str
    required_variables: List[str] = Field(default_factory=list)
    allowed_variables: List[str] = Field(default_factory=list)
    default_template: str


class PromptResolveResponse(BaseModel):
    language: Literal["ar", "en"]
    prompts: List[PromptDefinition] = Field(default_factory=list)
    skipped: bool = False


class GenerateDatasetRequest(BaseModel):
    llm_parameters: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_llm_settings(self):
        validate_parameters(self.llm_parameters, self.llm_provider)
        return self

    documents_path: str
    llm_provider: str
    llm_model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    language: str = "ar"
    document_parser: Optional[str] = None
    document_parser_config: Optional[Dict[str, Any]] = None
    use_ocr: bool = False
    max_chunks: Optional[int] = None


class GenerateDatasetResponse(BaseModel):
    dataset_path: str
    preview: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0


class ValidateRunResponse(BaseModel):
    valid: bool
    total_combinations: Optional[int] = None
    error: Optional[str] = None


class CreateRunResponse(BaseModel):
    run_id: str
    run_name: str
    total_combinations: int
    status: str


class RunSummary(BaseModel):
    run_id: str
    run_name: str
    status: str
    created_at: str
    best_score: Optional[float] = None
    best_config: Optional[Dict[str, Any]] = None
    total_trials: int = 0
    completed_trial_count: int = 0
    duration_seconds: Optional[float] = None


class RunListResponse(BaseModel):
    runs: List[RunSummary] = Field(default_factory=list)


class BestResult(BaseModel):
    trial_id: int
    config: Dict[str, Any] = Field(default_factory=dict)
    composite_score: float = 0.0
    metrics: Dict[str, float] = Field(default_factory=dict)
    mean_pipeline_latency_ms: Optional[float] = None
    mean_evaluation_overhead_ms: Optional[float] = None
    stage_timings: Dict[str, Optional[float]] = Field(default_factory=dict)


class RunDetailResponse(BaseModel):
    run_id: str
    run_name: str
    status: str
    created_at: str
    total_trials: int = 0
    completed_trials: int = 0
    best_score: Optional[float] = None
    best_config: Optional[Dict[str, Any]] = None
    best_result: Optional[BestResult] = None
    failure_clusters: Dict[str, List[int]] = Field(default_factory=dict)
    error_message: Optional[str] = None
    stage_timings: Dict[str, Optional[float]] = Field(default_factory=dict)
    total_cost_usd: Optional[float] = None
    total_web_search_fallbacks: Optional[int] = None
    total_answer_refusals: Optional[int] = None
    answer_refusal_rate: Optional[float] = None
    duration_seconds: Optional[float] = None


class TrialRow(BaseModel):
    trial_id: int
    resolved_rag_config: Dict[str, Any] = Field(default_factory=dict)
    composite_score: float = 0.0
    latency_ms: float = 0.0
    metrics: Dict[str, float] = Field(default_factory=dict)
    mean_pipeline_latency_ms: Optional[float] = None
    mean_evaluation_overhead_ms: Optional[float] = None
    status: str = "success"
    error_code: Optional[str] = None
    error_type: Optional[str] = None
    cost_usd: Optional[float] = None
    mean_query_transform_ms: Optional[float] = None
    mean_query_embedding_ms: Optional[float] = None
    mean_vector_search_ms: Optional[float] = None
    mean_rerank_ms: Optional[float] = None
    mean_relevance_check_ms: Optional[float] = None
    mean_web_search_ms: Optional[float] = None
    mean_generation_ms: Optional[float] = None
    mean_hallucination_check_ms: Optional[float] = None
    web_search_fallback_count: Optional[int] = None
    answer_refusal_count: int = 0
    answer_refusal_rate: Optional[float] = None
    started_at: Optional[str] = None
    completed_samples: int = 0
    total_samples: int = 0


class ActiveTrialRow(BaseModel):
    trial_id: int
    status: str = "running"
    resolved_rag_config: Dict[str, Any] = Field(default_factory=dict)
    started_at: str
    completed_samples: int = 0
    total_samples: int = 0


class TrialsPageResponse(BaseModel):
    trials: List[TrialRow] = Field(default_factory=list)
    active_trials: List[ActiveTrialRow] = Field(default_factory=list)
    next_after: int = 0


class StageErrorRate(BaseModel):
    stage: str
    component: str
    attempts: int = 0
    errors: int = 0
    recovered_errors: int = 0
    unrecovered_errors: int = 0
    error_rate: float = 0.0
    last_error_at: Optional[str] = None


class ErrorRateSummary(BaseModel):
    attempts: int = 0
    errors: int = 0
    recovered_errors: int = 0
    unrecovered_errors: int = 0
    error_rate: float = 0.0
    health: str = "HEALTHY"
    generation_samples: int = 0
    answer_refusals: int = 0
    answer_refusal_rate: float = 0.0
    updated_at: Optional[str] = None
    stages: List[StageErrorRate] = Field(default_factory=list)


class ErrorEventRow(BaseModel):
    event_id: str
    occurred_at: str
    stage: str
    component: str
    trial_id: Optional[int] = None
    sample_index: Optional[int] = None
    attempt: Optional[int] = None
    duration_ms: Optional[float] = None
    recovery: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    error_code: Optional[str] = None
    error_type: Optional[str] = None
    retryable: bool = False
    message: Optional[str] = None


class ErrorObservabilityResponse(BaseModel):
    available: bool = False
    summary: Optional[ErrorRateSummary] = None
    errors: List[ErrorEventRow] = Field(default_factory=list)
    next_after: int = 0


class RunMetadataResponse(BaseModel):
    config: Dict[str, Any] = Field(default_factory=dict)
    manifest: Dict[str, Any] = Field(default_factory=dict)
    pricing: Dict[str, Any] = Field(default_factory=dict)


class TrialDetailResponse(BaseModel):
    trial: Dict[str, Any] = Field(default_factory=dict)
    samples: List[Dict[str, Any]] = Field(default_factory=list)
    sample_count: int = 0


class HealthResponse(BaseModel):
    status: str
    version: str
    runs_root_writable: bool
    active_runs: int


class StorageSettingsUpdate(BaseModel):
    runs_root: str


class StorageSettingsResponse(BaseModel):
    runs_root: str
    source: Literal["environment", "saved", "legacy", "default"]
    locked: bool
    writable: bool
    recent_roots: List[str] = Field(default_factory=list)
