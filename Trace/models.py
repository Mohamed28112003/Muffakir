"""
Trace schema: RunManifest -> TrialRecord -> SampleTraceRecord.

Three JSON-serializable levels, matching TrialResult's existing to_dict()/
from_dict() convention:
  - RunManifest: one per Composer.fit() run.
  - TrialRecord: one per trial (mean per-stage timings across its samples).
  - SampleTraceRecord: one per evaluated QA sample (per-sample timings/cost).
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.3"

# Config keys never written into a manifest's config_hash input or its stored
# base_config -- redacted so a shared/committed manifest never leaks secrets.
_REDACTED_CONFIG_KEYS = {
    "api_key",
    "search_api_key",
    "judge_api_key",
    "query_transform_api_key",
    "dataset_api_key",
    "reranker_api_key",
    "reranker_llm_api_key",
    "authorization",
    "password",
    "secret",
    "token",
    "llm_base_url",
    "base_url",
}


def redact_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively redact credentials from manifest config and hash inputs."""
    def _redact(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: _redact(item)
                for key, item in value.items()
                if str(key).lower() not in _REDACTED_CONFIG_KEYS
            }
        if isinstance(value, list):
            return [_redact(item) for item in value]
        return value

    return _redact(config)


def compute_config_hash(base_config: Dict[str, Any], search_space: Dict[str, Any]) -> str:
    """Deterministic SHA-256 over the redacted base_config + search_space.

    Mirrors Composer.index_key.compute_index_key's approach (sorted-key JSON
    encode, hash, truncate) so two fit() calls with identical search
    parameters (but different api_key) hash identically.
    """
    payload = {
        "base_config": redact_config(base_config),
        "search_space": search_space,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class RunManifest:
    schema_version: str = SCHEMA_VERSION
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "running"  # running | completed | stopped_early | failed | pruned (reserved)
    total_trials: int = 0
    search_space: Dict[str, Any] = field(default_factory=dict)
    pricing_source_url: Optional[str] = None
    pricing_fetched_at: Optional[str] = None
    pricing_fetch_failed: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunManifest":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class TrialRecord:
    trial_id: int
    resolved_rag_config: Dict[str, Any] = field(default_factory=dict)
    composite_score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    mean_pipeline_latency_ms: Optional[float] = None
    mean_evaluation_overhead_ms: Optional[float] = None
    token_usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: Optional[float] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    error_type: Optional[str] = None
    status: str = "success"  # success | failed | pruned (reserved)
    mean_query_transform_ms: Optional[float] = None
    mean_query_embedding_ms: Optional[float] = None
    mean_vector_search_ms: Optional[float] = None
    mean_rerank_ms: Optional[float] = None
    mean_relevance_check_ms: Optional[float] = None
    mean_web_search_ms: Optional[float] = None
    mean_generation_ms: Optional[float] = None
    mean_hallucination_check_ms: Optional[float] = None
    web_search_fallback_count: int = 0
    started_at: Optional[str] = None
    completed_samples: int = 0
    total_samples: int = 0
    answer_refusal_count: int = 0
    answer_refusal_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrialRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class SampleTraceRecord:
    trial_id: int
    sample_index: int
    question: str = ""
    transformed_query: Optional[Any] = None
    query_transform_strategy: Optional[str] = None
    gold_answer: str = ""
    gold_context: str = ""
    predicted_answer: str = ""
    context_source: str = "vector_db"
    latency_ms: float = 0.0
    pipeline_latency_ms: Optional[float] = None
    evaluation_overhead_ms: Optional[float] = None
    query_transform_ms: Optional[float] = None
    query_embedding_ms: Optional[float] = None
    vector_search_ms: Optional[float] = None
    rerank_ms: Optional[float] = None
    relevance_check_ms: Optional[float] = None
    web_search_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    hallucination_check_ms: Optional[float] = None
    retrieved_candidates: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: Optional[float] = None
    error: Optional[str] = None
    web_search_used: bool = False
    generation_attempted: bool = False
    answer_refusal: bool = False
    answer_refusal_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleTraceRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})
