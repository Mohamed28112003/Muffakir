"""
TrialResult - Data class for storing individual trial results.

Each trial represents one complete RAG pipeline configuration evaluated
against the evaluation dataset.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import json


@dataclass
class TrialResult:
    """
    Stores the results of a single pipeline configuration trial.
    
    Attributes:
        trial_id: Unique identifier for this trial
        config: Pipeline configuration dictionary
        metrics: Computed evaluation metrics (recall, faithfulness, etc.)
        composite_score: Weighted average of all metrics
        latency_ms: Total execution time in milliseconds
        error: Error message if trial failed, None otherwise
        error_code: Stable machine-readable failure code (e.g. "PROVIDER_TIMEOUT"),
            set from a MuffakirError's error_code when the trial fails
        error_type: The failing exception's class name (e.g. "ProviderTimeoutError")
        completed_at: Timestamp when trial completed
        token_usage: Aggregated prompt/completion/total token counts across every
            distinct LLMProvider used by this trial (generation, query-transform
            override if any, and the eval judge). Always populated when usage was
            observable, even if pricing for it is unknown.
        cost_usd: Best-effort dollar cost for this trial's LLM calls, priced via
            Pricing.price_map.PriceMap. None when no component's (provider, model)
            could be priced at all — see Composer/evaluation.py.
    """
    trial_id: int
    config: Dict[str, Any]
    metrics: Dict[str, float] = field(default_factory=dict)
    composite_score: float = 0.0
    latency_ms: float = 0.0
    error: Optional[str] = None
    error_code: Optional[str] = None
    error_type: Optional[str] = None
    completed_at: datetime = field(default_factory=datetime.now)
    token_usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: Optional[float] = None
    resolved_config: Optional[Dict[str, Any]] = None
    mean_pipeline_latency_ms: Optional[float] = None
    mean_evaluation_overhead_ms: Optional[float] = None
    stage_timings_ms: Dict[str, Optional[float]] = field(default_factory=dict)
    answer_refusal_count: int = 0
    answer_refusal_rate: Optional[float] = None
    
    @property
    def is_successful(self) -> bool:
        """Check if trial completed successfully (no error)."""
        return self.error is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trial_id": self.trial_id,
            "config": self.config,
            "metrics": self.metrics,
            "composite_score": self.composite_score,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "error_code": self.error_code,
            "error_type": self.error_type,
            "completed_at": self.completed_at.isoformat(),
            "token_usage": self.token_usage,
            "cost_usd": self.cost_usd,
            "resolved_config": self.resolved_config,
            "mean_pipeline_latency_ms": self.mean_pipeline_latency_ms,
            "mean_evaluation_overhead_ms": self.mean_evaluation_overhead_ms,
            "stage_timings_ms": self.stage_timings_ms,
            "answer_refusal_count": self.answer_refusal_count,
            "answer_refusal_rate": self.answer_refusal_rate,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrialResult":
        """Create TrialResult from dictionary."""
        completed_at = data.get("completed_at")
        if isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at)
        elif completed_at is None:
            completed_at = datetime.now()
        
        return cls(
            trial_id=data["trial_id"],
            config=data["config"],
            metrics=data.get("metrics", {}),
            composite_score=data.get("composite_score", 0.0),
            latency_ms=data.get("latency_ms", 0.0),
            error=data.get("error"),
            error_code=data.get("error_code"),
            error_type=data.get("error_type"),
            completed_at=completed_at,
            token_usage=data.get("token_usage", {}),
            cost_usd=data.get("cost_usd"),
            resolved_config=data.get("resolved_config"),
            mean_pipeline_latency_ms=data.get("mean_pipeline_latency_ms"),
            mean_evaluation_overhead_ms=data.get("mean_evaluation_overhead_ms"),
            stage_timings_ms=data.get("stage_timings_ms", {}),
            answer_refusal_count=data.get("answer_refusal_count", 0),
            answer_refusal_rate=data.get("answer_refusal_rate"),
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "TrialResult":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def __repr__(self) -> str:
        status = "✓" if self.is_successful else "✗"
        return f"TrialResult(id={self.trial_id}, score={self.composite_score:.3f}, status={status})"
