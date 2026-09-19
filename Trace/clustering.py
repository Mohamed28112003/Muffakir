"""
Rule-based failure clustering: groups failed trials by their error_code (or
error_type, or "unknown" if neither is set). Pure post-hoc function over
already-collected TrialResults -- no new capture needed, no embedding model,
no new dependency.
"""

from typing import Any, Dict, List


def cluster_failures(trials: List[Any]) -> Dict[str, List[int]]:
    """
    Group failed trials' trial_ids by (error_code or error_type or "unknown").

    Args:
        trials: TrialResult-like objects exposing .is_successful, .trial_id,
            .error_code, .error_type.

    Returns:
        Dict mapping a cluster key to the sorted list of trial_ids in it.
        Successful trials are excluded entirely. Empty dict if none failed.
    """
    clusters: Dict[str, List[int]] = {}
    for trial in trials:
        if trial.is_successful:
            continue
        key = trial.error_code or trial.error_type or "unknown"
        clusters.setdefault(key, []).append(trial.trial_id)

    for key in clusters:
        clusters[key].sort()
    return clusters
