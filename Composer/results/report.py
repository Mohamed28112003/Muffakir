"""
ComposerReport - Aggregated results from architecture search.

Contains all trial results, best configuration, and export methods
for DataFrame, JSON, and dictionary formats.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

from .trial import TrialResult

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class ComposerReport:
    """
    Aggregated report from MuffakirComposer architecture search.
    
    Attributes:
        trials: List of all trial results
        best_trial: The trial with highest composite score
        search_space: The search space that was explored
        total_duration_ms: Total execution time in milliseconds
        metrics_used: List of metrics that were computed
        created_at: Timestamp when report was created
    """
    trials: List[TrialResult] = field(default_factory=list)
    best_trial: Optional[TrialResult] = None
    search_space: Dict[str, List] = field(default_factory=dict)
    total_duration_ms: float = 0.0
    metrics_used: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    stopped_early: bool = False
    stop_reason: Optional[str] = None

    def __post_init__(self):
        """Compute best trial from trials (kept consistent after load/merge)."""
        if self.trials:
            self._compute_best_trial()
    
    def _compute_best_trial(self):
        """Find the trial with highest composite score."""
        successful_trials = [t for t in self.trials if t.is_successful]
        if successful_trials:
            self.best_trial = max(
                successful_trials,
                key=lambda t: (t.composite_score, -t.trial_id),
            )
    
    @property
    def best_config(self) -> Optional[Dict[str, Any]]:
        """Get the best configuration dictionary."""
        return self.best_trial.config if self.best_trial else None
    
    @property
    def best_score(self) -> float:
        """Get the best composite score."""
        return self.best_trial.composite_score if self.best_trial else 0.0

    @property
    def best_result(self) -> Optional[Dict[str, Any]]:
        """Return the winning architecture together with its evaluation data."""
        if self.best_trial is None:
            return None
        trial = self.best_trial
        return {
            "trial_id": trial.trial_id,
            "config": trial.resolved_config or trial.config,
            "composite_score": trial.composite_score,
            "metrics": trial.metrics,
            "mean_pipeline_latency_ms": trial.mean_pipeline_latency_ms,
            "mean_evaluation_overhead_ms": trial.mean_evaluation_overhead_ms,
            "stage_timings": trial.stage_timings_ms,
        }
    
    @property
    def total_trials(self) -> int:
        """Total number of trials."""
        return len(self.trials)
    
    @property
    def successful_trials(self) -> int:
        """Number of successful trials."""
        return sum(1 for t in self.trials if t.is_successful)
    
    @property
    def failed_trials(self) -> int:
        """Number of failed trials."""
        return sum(1 for t in self.trials if not t.is_successful)

    @property
    def total_cost_usd(self) -> Optional[float]:
        """
        Sum of cost_usd across successful trials where it was priceable.

        None only when not one successful trial has a known cost (e.g. the
        price fetch failed for this run and no custom_pricing covered any
        model used).
        """
        priced = [t.cost_usd for t in self.trials if t.is_successful and t.cost_usd is not None]
        return sum(priced) if priced else None
    
    def get_trials_sorted_by_score(self, descending: bool = True) -> List[TrialResult]:
        """Get trials sorted by composite score."""
        successful = [t for t in self.trials if t.is_successful]
        return sorted(successful, key=lambda t: t.composite_score, reverse=descending)
    
    def get_top_n_trials(self, n: int = 5) -> List[TrialResult]:
        """Get top N trials by composite score."""
        return self.get_trials_sorted_by_score(descending=True)[:n]

    def get_failure_clusters(self) -> Dict[str, List[int]]:
        """Group failed trials' trial_ids by error_code (or error_type, or
        "unknown"). See Trace.clustering.cluster_failures."""
        from Trace.clustering import cluster_failures
        return cluster_failures(self.trials)

    def get_pareto_frontier(
        self,
        objectives: Optional[List[Tuple[str, str]]] = None,
    ) -> List[TrialResult]:
        """
        Return the non-dominated set of successful trials across the given objectives.

        Each objective is a (field_name, direction) pair, direction is "max" or "min".
        Default objectives are quality (composite_score, maximize) vs latency
        (latency_ms, minimize) — both fields exist on every TrialResult today. A
        cost-based objective can be added later as an extra tuple without changing
        this method's signature, once per-trial cost exists.

        A trial is on the frontier unless some other trial is at least as good on
        every objective and strictly better on at least one (i.e. it is not
        dominated by any other trial).
        """
        if objectives is None:
            objectives = [("composite_score", "max"), ("latency_ms", "min")]

        candidates = [t for t in self.trials if t.is_successful]

        def _dominates(a: TrialResult, b: TrialResult) -> bool:
            at_least_as_good_everywhere = True
            strictly_better_somewhere = False
            for field_name, direction in objectives:
                a_val = getattr(a, field_name)
                b_val = getattr(b, field_name)
                if direction == "max":
                    if a_val < b_val:
                        at_least_as_good_everywhere = False
                    if a_val > b_val:
                        strictly_better_somewhere = True
                else:
                    if a_val > b_val:
                        at_least_as_good_everywhere = False
                    if a_val < b_val:
                        strictly_better_somewhere = True
            return at_least_as_good_everywhere and strictly_better_somewhere

        frontier = [
            t for t in candidates
            if not any(_dominates(other, t) for other in candidates if other is not t)
        ]
        return frontier
    
    def to_dataframe(self) -> "pd.DataFrame":
        """
        Convert results to pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with all trial results
        """
        import pandas as pd
        
        rows = []
        for trial in self.trials:
            row = {
                "trial_id": trial.trial_id,
                "composite_score": trial.composite_score,
                "latency_ms": trial.latency_ms,
                "mean_pipeline_latency_ms": trial.mean_pipeline_latency_ms,
                "mean_evaluation_overhead_ms": trial.mean_evaluation_overhead_ms,
                "is_successful": trial.is_successful,
                "error": trial.error,
                "error_code": trial.error_code,
                "error_type": trial.error_type,
            }
            # Add search config values plus the resolved local reranker details.
            export_config = dict(trial.config)
            resolved = trial.resolved_config or {}
            reranking_method = str(
                resolved.get("reranking_method") or ""
            ).strip().lower()
            if resolved.get("reranking") and reranking_method:
                export_config.setdefault("reranking_method", reranking_method)
            if (
                reranking_method in {"cross_encoder", "pointwise"}
                and resolved.get("reranking_model")
            ):
                export_config.setdefault(
                    "reranking_model", resolved["reranking_model"]
                )
            for key, value in export_config.items():
                row[f"config_{key}"] = value
            # Add metric values
            for key, value in trial.metrics.items():
                row[f"metric_{key}"] = value
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by composite score descending. kind="mergesort" (stable) matches
        # get_trials_sorted_by_score()/get_top_n_trials()'s use of Python's
        # stable sorted() — pandas' default quicksort is not stable, which
        # could otherwise order tied-score trials differently between this
        # DataFrame export and those methods.
        if not df.empty:
            df = df.sort_values(
                "composite_score", ascending=False, kind="mergesort"
            ).reset_index(drop=True)
        
        return df
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trials": [t.to_dict() for t in self.trials],
            "best_trial": self.best_trial.to_dict() if self.best_trial else None,
            "best_result": self.best_result,
            "search_space": self.search_space,
            "total_duration_ms": self.total_duration_ms,
            "metrics_used": self.metrics_used,
            "created_at": self.created_at.isoformat(),
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "summary": {
                "total_trials": self.total_trials,
                "successful_trials": self.successful_trials,
                "failed_trials": self.failed_trials,
                "best_score": self.best_score,
                "best_config": self.best_config,
                "best_metrics": self.best_trial.metrics if self.best_trial else {},
                "best_result": self.best_result,
            }
        }
    
    def to_json(self, path: Optional[str] = None) -> str:
        """
        Export report as JSON.
        
        Args:
            path: Optional file path to save JSON. If None, returns JSON string.
            
        Returns:
            JSON string representation
        """
        json_str = json.dumps(self.to_dict(), indent=2)
        
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
        
        return json_str
    
    @classmethod
    def from_json(cls, json_str: str) -> "ComposerReport":
        """Create ComposerReport from JSON string."""
        data = json.loads(json_str)
        
        trials = [TrialResult.from_dict(t) for t in data.get("trials", [])]
        best_trial = TrialResult.from_dict(data["best_trial"]) if data.get("best_trial") else None
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()
        
        return cls(
            trials=trials,
            best_trial=best_trial,
            search_space=data.get("search_space", {}),
            total_duration_ms=data.get("total_duration_ms", 0.0),
            metrics_used=data.get("metrics_used", []),
            created_at=created_at,
            stopped_early=data.get("stopped_early", False),
            stop_reason=data.get("stop_reason"),
        )
    
    def to_html(self, path: str) -> None:
        """
        Deprecated: HTML report generation has been removed.

        Saves a structured JSON report to the equivalent path instead.
        """
        import warnings
        warnings.warn(
            "ComposerReport.to_html() is deprecated; saving JSON report instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..report_generator import ReportGenerator
        ReportGenerator.generate_html(self, path)
    
    def summary(self) -> str:
        """Get a text summary of the report."""
        lines = [
            "=" * 60,
            "MUFFAKIR COMPOSER - ARCHITECTURE SEARCH REPORT",
            "=" * 60,
            "",
            f"Total Trials: {self.total_trials}",
            f"Successful: {self.successful_trials}",
            f"Failed: {self.failed_trials}",
            f"Duration: {self.total_duration_ms / 1000:.1f} seconds",
            "",
        ]
        
        if self.best_trial:
            lines.extend([
                "🏆 BEST CONFIGURATION",
                "-" * 40,
            ])
            summary_config = dict(self.best_trial.config)
            resolved = self.best_trial.resolved_config or {}
            reranking_method = str(
                resolved.get("reranking_method") or ""
            ).strip().lower()
            if resolved.get("reranking") and reranking_method:
                summary_config.setdefault("reranking_method", reranking_method)
            if (
                reranking_method in {"cross_encoder", "pointwise"}
                and resolved.get("reranking_model")
            ):
                summary_config.setdefault(
                    "reranking_model", resolved["reranking_model"]
                )
            for key, value in summary_config.items():
                lines.append(f"  {key}: {value}")
            lines.append(f"  Composite Score: {self.best_trial.composite_score:.4f}")
            lines.append("")
            
            if self.best_trial.metrics:
                lines.append("  Metrics:")
                for metric, value in self.best_trial.metrics.items():
                    if metric == "llm_judge_rating":
                        lines.append(f"    LLM Judge Rating: {value:.2f} / 5")
                    else:
                        lines.append(f"    {metric}: {value:.4f}")
        
        lines.extend(["", "=" * 60])
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (
            f"ComposerReport(trials={self.total_trials}, "
            f"successful={self.successful_trials}, "
            f"best_score={self.best_score:.3f})"
        )
