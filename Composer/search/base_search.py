"""
BaseSearch - Abstract base class for search strategies.

Defines the interface that all search strategies must implement.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Dict, Any, Callable, Optional

from Muffakir.exceptions import MuffakirError

from ..results.trial import TrialResult
from ..checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class BaseSearch(ABC):
    """
    Abstract base class for architecture search strategies.
    
    All search strategies (Grid, Bayesian, etc.) must inherit from this
    and implement the run() method.
    
    Attributes:
        n_jobs: Number of parallel workers
        max_retries: Maximum retry attempts per trial
    """
    
    def __init__(
        self,
        n_jobs: int = 4,
        max_retries: int = 1,
        max_trials: Optional[int] = None,
        max_runtime_minutes: Optional[float] = None,
    ):
        """
        Initialize search strategy.

        Args:
            n_jobs: Number of parallel workers (default: 4)
            max_retries: Max retry attempts per failed trial (default: 1)
            max_trials: Stop dispatching new trials once this many have been
                        evaluated in this run (default: None = no cap)
            max_runtime_minutes: Stop dispatching/collecting new trials once
                        this many minutes have elapsed (default: None = no cap)
        """
        self.n_jobs = n_jobs
        self.max_retries = max_retries
        self.max_trials = max_trials
        self.max_runtime_minutes = max_runtime_minutes
        self.stopped_early: bool = False
        self.stop_reason: Optional[str] = None

    def _budget_exceeded(self, completed_count: int, start_time: float) -> Optional[str]:
        """Return a stop reason string if a budget cap has been hit, else None."""
        import time

        if self.max_trials is not None and completed_count >= self.max_trials:
            return "max_trials"
        if self.max_runtime_minutes is not None:
            elapsed_minutes = (time.perf_counter() - start_time) / 60.0
            if elapsed_minutes >= self.max_runtime_minutes:
                return "max_runtime_minutes"
        return None
    
    @abstractmethod
    def run(
        self,
        combinations: List[tuple],
        eval_fn: Callable,
        shared_state: Dict[str, Any],
        checkpoint_manager: CheckpointManager,
        progress_callback: Optional[Callable] = None,
    ) -> List[TrialResult]:
        """
        Execute the search strategy.
        
        Args:
            combinations: List of (trial_id, config_dict) tuples to evaluate
            eval_fn: Top-level (picklable) callable with signature
                     eval_fn(trial_id, config, shared_state) -> TrialResult
            shared_state: Picklable dict of shared state passed to every trial
                          (e.g. base_config, eval_pairs, metrics).
            checkpoint_manager: Manager for saving/loading checkpoints
            progress_callback: Optional callback for progress updates
                              Called with (completed_count, total_count, latest_result)
            
        Returns:
            List of TrialResult objects for all evaluated combinations
        """
        pass
    
    def _run_with_retry(
        self,
        eval_fn: Callable,
        trial_id: int,
        config: Dict[str, Any],
        shared_state: Dict[str, Any],
    ) -> TrialResult:
        """
        Run a single trial with type-aware retry logic.

        Retries only failures that a MuffakirError marks as `retryable`
        (timeouts, rate limits, transient provider unavailability). Failures
        that will never succeed on retry (auth failures, bad configuration,
        or any unclassified/unexpected exception) fail fast on the first
        attempt instead of wasting time on exponential backoff.

        Args:
            eval_fn: Evaluation function (top-level / picklable)
            trial_id: Trial identifier
            config: Trial configuration
            shared_state: Picklable shared state dict

        Returns:
            TrialResult (may contain error/error_code/error_type if every
            attempt failed). This method never lets an exception escape -
            eval_fn either returns a successful TrialResult or raises, and
            every raise is converted into a failed TrialResult here.
        """
        import time

        start_time = time.perf_counter()
        self._trace_trial_started(trial_id, config, shared_state)
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            from Trace.observability import (
                collect_secret_values,
                observation_context,
                record_exception_outcome,
            )

            attempt_state = dict(shared_state)
            attempt_state["_trace_attempt"] = attempt + 1
            base_config = shared_state.get("base_config", {}) or {}
            secrets = collect_secret_values(base_config)
            with observation_context(
                shared_state.get("observation_queue"),
                trial_id=trial_id,
                attempt=attempt + 1,
                secrets=secrets,
            ):
                try:
                    return eval_fn(trial_id, config, attempt_state)
                except MuffakirError as e:
                    last_exc = e
                    retryable = e.retryable
                except Exception as e:
                    # eval_fn didn't raise a typed MuffakirError (an unexpected bug,
                    # or a custom eval_fn). Treat as non-retryable: retrying an
                    # unclassified failure rarely helps and just wastes time/quota.
                    logger.error(
                        f"Trial {trial_id} raised an unclassified error: {e}",
                        exc_info=True,
                    )
                    last_exc = e
                    retryable = False

                will_retry = retryable and attempt < self.max_retries
                record_exception_outcome(
                    last_exc,
                    recovery="retry" if will_retry else "unrecovered",
                )

            if not retryable or attempt >= self.max_retries:
                break
            logger.warning(
                f"Trial {trial_id} failed with retryable {type(last_exc).__name__} "
                f"(attempt {attempt + 1}/{self.max_retries + 1}); "
                f"retrying in {2 ** attempt}s: {last_exc}"
            )
            time.sleep(2 ** attempt)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        result = TrialResult(
            trial_id=trial_id,
            config=config,
            resolved_config=self._resolved_trace_config(config, shared_state),
            error=str(last_exc),
            error_code=getattr(last_exc, "error_code", type(last_exc).__name__),
            error_type=type(last_exc).__name__,
            latency_ms=latency_ms,
        )
        self._trace_failed_result(result, config, shared_state)
        return result

    @staticmethod
    def _resolved_trace_config(
        config: Dict[str, Any],
        shared_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve and redact a trial config without making tracing fatal."""
        from Trace.models import redact_config

        try:
            from ..config_space import trial_config_to_rag_config

            return redact_config(
                trial_config_to_rag_config(config, shared_state.get("base_config", {}))
            )
        except Exception:
            # Configuration resolution may itself be what the trial is about to
            # fail on. The raw search-space values are still useful and redacted.
            return redact_config(config)

    @classmethod
    def _trace_trial_started(
        cls,
        trial_id: int,
        config: Dict[str, Any],
        shared_state: Dict[str, Any],
    ) -> None:
        trace_queue = shared_state.get("trace_queue")
        if trace_queue is None:
            return

        trace_queue.put(("trial_started", {
            "trial_id": trial_id,
            "status": "running",
            "resolved_rag_config": cls._resolved_trace_config(config, shared_state),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_samples": 0,
            "total_samples": len(shared_state.get("eval_pairs") or []),
        }))

    @classmethod
    def _trace_failed_result(
        cls,
        result: TrialResult,
        config: Dict[str, Any],
        shared_state: Dict[str, Any],
    ) -> None:
        """Persist a failed result and close its active telemetry row."""
        trace_queue = shared_state.get("trace_queue")
        if trace_queue is None or result.is_successful:
            return

        from Trace.models import TrialRecord

        record = TrialRecord(
            trial_id=result.trial_id,
            resolved_rag_config=cls._resolved_trace_config(config, shared_state),
            latency_ms=result.latency_ms,
            token_usage=result.token_usage,
            cost_usd=result.cost_usd,
            error=result.error,
            error_code=result.error_code,
            error_type=result.error_type,
            status="failed",
            total_samples=len(shared_state.get("eval_pairs") or []),
        )
        trace_queue.put(("trial", record.to_dict()))
