"""
GridSearch - Brute-force search with parallel execution.

Evaluates all possible pipeline combinations using ProcessPoolExecutor
for true parallelism across multiple CPU cores.

The evaluation function must be a top-level (picklable) callable with signature
``eval_fn(trial_id, config, shared_state) -> TrialResult`` and ``shared_state``
must be a plain picklable dict (no live DB/LLM objects). This keeps the parallel
path working: closures capturing non-picklable objects cannot be sent to workers.
"""

import logging
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .base_search import BaseSearch
from ..results.trial import TrialResult
from ..checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class GridSearch(BaseSearch):
    """
    Brute-force grid search with parallel execution.
    
    Evaluates all combinations in the search space using multiple
    worker processes for parallelism.
    
    Features:
    - ProcessPoolExecutor for true parallelism (bypasses GIL)
    - Retry logic with exponential backoff
    - Checkpoint saving after each trial (in the parent process; no races)
    - Progress bar with tqdm
    
    Example:
        search = GridSearch(n_jobs=4, max_retries=1)
        results = search.run(combinations, eval_fn, shared_state, checkpoint_manager)
    """
    
    def __init__(
        self,
        n_jobs: int = 4,
        max_retries: int = 1,
        max_trials: Optional[int] = None,
        max_runtime_minutes: Optional[float] = None,
    ):
        """
        Initialize GridSearch.

        Args:
            n_jobs: Number of parallel workers (default: 4)
                   Set to 1 for sequential execution (debugging)
            max_retries: Max retry attempts per failed trial (default: 1)
            max_trials: Stop dispatching new trials once this many have been
                        evaluated in this run (default: None = no cap)
            max_runtime_minutes: Stop dispatching/collecting new trials once
                        this many minutes have elapsed (default: None = no cap)
        """
        super().__init__(
            n_jobs=n_jobs,
            max_retries=max_retries,
            max_trials=max_trials,
            max_runtime_minutes=max_runtime_minutes,
        )
    
    def run(
        self,
        combinations: List[tuple],
        eval_fn: Callable,
        shared_state: Dict[str, Any],
        checkpoint_manager: CheckpointManager,
        progress_callback: Optional[Callable] = None,
    ) -> List[TrialResult]:
        """
        Execute grid search over all combinations.
        
        Args:
            combinations: List of (trial_id, config_dict) tuples
            eval_fn: Top-level (picklable) callable:
                     eval_fn(trial_id, config, shared_state) -> TrialResult
            shared_state: Picklable dict of shared state passed to every trial
            checkpoint_manager: Manager for checkpoint operations
            progress_callback: Optional callback(completed, total, result)
            
        Returns:
            List of TrialResult objects
        """
        if not combinations:
            logger.warning("No combinations to evaluate")
            return []
        
        total = len(combinations)
        logger.info(f"Starting grid search with {total} combinations, {self.n_jobs} workers")
        
        if self.n_jobs <= 1:
            results = self._run_sequential(
                combinations, eval_fn, shared_state, checkpoint_manager, progress_callback
            )
        else:
            results = self._run_parallel(
                combinations, eval_fn, shared_state, checkpoint_manager, progress_callback
            )
        
        logger.info(f"Grid search completed: {len(results)} trials evaluated")
        return results
    
    def _run_sequential(
        self,
        combinations: List[tuple],
        eval_fn: Callable,
        shared_state: Dict[str, Any],
        checkpoint_manager: CheckpointManager,
        progress_callback: Optional[Callable],
    ) -> List[TrialResult]:
        """Run trials sequentially (useful for debugging)."""
        import time

        results: List[TrialResult] = []
        total = len(combinations)
        start_time = time.perf_counter()

        for trial_id, config in tqdm(combinations, desc="Grid Search", unit="trial"):
            reason = self._budget_exceeded(len(results), start_time)
            if reason:
                self.stopped_early = True
                self.stop_reason = reason
                logger.warning(f"Stopping grid search early: {reason}")
                break
            result = self._run_with_retry(eval_fn, trial_id, config, shared_state)
            results.append(result)
            from Trace.observability import observation_context, observe_stage

            with observation_context(
                shared_state.get("observation_queue"), trial_id=trial_id
            ):
                with observe_stage(
                    "checkpoint_write", "checkpoint", defer_propagated_error=False
                ):
                    checkpoint_manager.save_trial(result)
            if progress_callback:
                progress_callback(len(results), total, result)

        return results
    
    def _run_parallel(
        self,
        combinations: List[tuple],
        eval_fn: Callable,
        shared_state: Dict[str, Any],
        checkpoint_manager: CheckpointManager,
        progress_callback: Optional[Callable],
    ) -> List[TrialResult]:
        """Run trials in parallel using ProcessPoolExecutor.

        ``eval_fn`` must be a top-level (picklable) function and ``shared_state`` a
        picklable dict, so they can be shipped to worker processes.
        """
        import time

        results: List[TrialResult] = []
        total = len(combinations)
        start_time = time.perf_counter()

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_trial = {}
            for trial_id, config in combinations:
                future = executor.submit(
                    self._run_with_retry,
                    eval_fn,
                    trial_id,
                    config,
                    shared_state,
                )
                future_to_trial[future] = (trial_id, config)
            
            with tqdm(total=total, desc="Grid Search", unit="trial") as pbar:
                for future in as_completed(future_to_trial):
                    trial_id, config = future_to_trial[future]
                    
                    try:
                        result = future.result()
                    except Exception as e:
                        # Handle unexpected executor-level errors (e.g. worker died,
                        # BrokenProcessPool, pickling failure). _run_with_retry never
                        # lets a trial-evaluation exception escape, so anything caught
                        # here is a process/executor-level failure, not a trial one.
                        logger.error(
                            f"Trial {trial_id} failed at executor level: {e}",
                            exc_info=True,
                        )
                        result = TrialResult(
                            trial_id=trial_id,
                            config=config,
                            error=f"Executor error: {e}",
                            error_code="EXECUTOR_ERROR",
                            error_type=type(e).__name__,
                        )
                        self._trace_failed_result(result, config, shared_state)
                        from Trace.observability import observation_context, emit_stage_outcome

                        with observation_context(
                            shared_state.get("observation_queue"), trial_id=trial_id
                        ):
                            emit_stage_outcome(
                                "trial_execution",
                                "executor",
                                outcome="error",
                                recovery="unrecovered",
                                error=e,
                            )
                    
                    results.append(result)
                    # Checkpoint writes happen in the parent process -> no races.
                    from Trace.observability import observation_context, observe_stage

                    with observation_context(
                        shared_state.get("observation_queue"), trial_id=trial_id
                    ):
                        with observe_stage(
                            "checkpoint_write", "checkpoint", defer_propagated_error=False
                        ):
                            checkpoint_manager.save_trial(result)

                    pbar.update(1)
                    if progress_callback:
                        progress_callback(len(results), total, result)

                    reason = self._budget_exceeded(len(results), start_time)
                    if reason:
                        self.stopped_early = True
                        self.stop_reason = reason
                        logger.warning(f"Stopping grid search early: {reason}")
                        # Best-effort: only cancels futures that haven't started
                        # running yet (mirrors EvalRunner's concurrent abort
                        # semantics) — already in-flight trials still complete
                        # and their results are still saved via the loop above
                        # on their next iteration.
                        for f in future_to_trial:
                            f.cancel()
                        break

        # as_completed() yields in completion order, not submission order —
        # sort by trial_id so results (and hence ComposerReport's best_trial
        # tie-break) are reproducible across runs regardless of which worker
        # finished first.
        results.sort(key=lambda t: t.trial_id)
        return results
