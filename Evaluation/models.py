from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .constants import ALL_METRICS, DEFAULT_METRICS, GENERATION_METRICS, RETRIEVAL_METRICS

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class RetrievalScores:
    recall_at_k: Optional[float] = None
    precision_at_k: Optional[float] = None
    mrr: Optional[float] = None
    ndcg_at_k: Optional[float] = None


@dataclass
class GenerationScores:
    faithfulness: Optional[float] = None
    answer_correctness: Optional[float] = None
    llm_judge_rating: Optional[int] = None


# Column order for EvalSampleResult.to_flat_dict() — kept in sync with that
# method and used as the explicit schema for an empty EvaluationReport's
# to_dataframe().
_EVAL_SAMPLE_COLUMNS = (
    "question",
    "transformed_query",
    "query_transform_strategy",
    "gold_answer",
    "gold_context",
    "predicted_answer",
    "retrieved_previews",
    "latency_ms",
    "pipeline_latency_ms",
    "evaluation_overhead_ms",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "faithfulness",
    "answer_correctness",
    "llm_judge_rating",
    "answer_refusal",
    "answer_refusal_reason",
    "error",
)


@dataclass
class EvalSampleResult:
    question: str
    gold_answer: str
    gold_context: str = ""
    transformed_query: Optional[Any] = None
    query_transform_strategy: Optional[str] = None
    answer_refusal: bool = False
    answer_refusal_reason: Optional[str] = None
    predicted_answer: str = ""
    retrieved_previews: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    pipeline_latency_ms: Optional[float] = None
    evaluation_overhead_ms: Optional[float] = None
    stage_timings_ms: Dict[str, Optional[float]] = field(default_factory=dict)
    retrieval: RetrievalScores = field(default_factory=RetrievalScores)
    generation: GenerationScores = field(default_factory=GenerationScores)
    error: Optional[str] = None

    def to_flat_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "transformed_query": (
                " || ".join(str(query) for query in self.transformed_query)
                if isinstance(self.transformed_query, list)
                else self.transformed_query
            ),
            "query_transform_strategy": self.query_transform_strategy,
            "gold_answer": self.gold_answer,
            "gold_context": self.gold_context,
            "predicted_answer": self.predicted_answer,
            "retrieved_previews": " || ".join(self.retrieved_previews),
            "latency_ms": self.latency_ms,
            "pipeline_latency_ms": self.pipeline_latency_ms,
            "evaluation_overhead_ms": self.evaluation_overhead_ms,
            "recall_at_k": self.retrieval.recall_at_k,
            "precision_at_k": self.retrieval.precision_at_k,
            "mrr": self.retrieval.mrr,
            "ndcg_at_k": self.retrieval.ndcg_at_k,
            "faithfulness": self.generation.faithfulness,
            "answer_correctness": self.generation.answer_correctness,
            "llm_judge_rating": self.generation.llm_judge_rating,
            "answer_refusal": self.answer_refusal,
            "answer_refusal_reason": self.answer_refusal_reason,
            "error": self.error,
        }


@dataclass
class EvaluationReport:
    """
    Sklearn-like evaluation results object.
    """

    n_samples: int
    k: int
    metrics_enabled: List[str]
    retrieval: Dict[str, float] = field(default_factory=dict)
    generation: Dict[str, float] = field(default_factory=dict)
    samples: List[EvalSampleResult] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        """Flat aggregate metrics dict (like sklearn classification_report averages)."""
        out: Dict[str, float] = {}
        out.update(self.retrieval)
        out.update(self.generation)
        return out

    def to_dataframe(self) -> "pd.DataFrame":
        from Muffakir.optional_dependencies import require_optional_dependency

        require_optional_dependency("datasets")
        import pandas as pd

        rows = [s.to_flat_dict() for s in self.samples]
        if not rows:
            # pd.DataFrame([]) would otherwise produce a DataFrame with zero
            # columns (not the expected schema with zero rows), breaking any
            # caller that indexes a column (e.g. report.to_dataframe()["question"])
            # on an empty evaluation run.
            df = pd.DataFrame(columns=list(_EVAL_SAMPLE_COLUMNS))
        else:
            df = pd.DataFrame(rows)

        # Preserve the per-sample integer contract even when errored samples
        # introduce missing values (plain pandas integer columns would
        # otherwise be widened to floats such as 4.0).
        df["llm_judge_rating"] = pd.to_numeric(
            df["llm_judge_rating"], errors="coerce"
        ).astype("Int64")
        return df

    def save(self, output_dir: str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Atomic write for summary.json (temp file + os.replace).
        summary_path = path / "summary.json"
        payload = json.dumps(
            {
                "n_samples": self.n_samples,
                "k": self.k,
                "metrics_enabled": self.metrics_enabled,
                "retrieval": self.retrieval,
                "generation": self.generation,
                "summary": self.summary(),
            },
            ensure_ascii=False,
            indent=2,
        )
        fd, tmp = tempfile.mkstemp(dir=str(path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, str(summary_path))
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

        # Atomic write for samples.csv too (temp file + os.replace) — matches
        # summary.json's guarantee so a crash mid-write never leaves one
        # artifact valid and the other truncated/corrupt.
        df = self.to_dataframe()
        samples_path = path / "samples.csv"
        csv_fd, csv_tmp = tempfile.mkstemp(dir=str(path), suffix=".tmp")
        try:
            os.close(csv_fd)
            df.to_csv(csv_tmp, index=False, encoding="utf-8-sig")
            os.replace(csv_tmp, str(samples_path))
        except Exception:
            try:
                os.remove(csv_tmp)
            except OSError:
                pass
            raise

    def __repr__(self) -> str:
        parts = [f"EvaluationReport(n={self.n_samples}, k={self.k})"]
        for key, val in self.summary().items():
            if key == "llm_judge_rating":
                parts.append(f"  LLM Judge Rating={val:.2f} / 5")
            else:
                parts.append(f"  {key}={val:.4f}")
        return "\n".join(parts)
