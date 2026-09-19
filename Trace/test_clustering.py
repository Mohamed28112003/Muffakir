"""Test suite for Trace.clustering (pytest)."""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Trace.clustering import cluster_failures


@dataclass
class _FakeTrial:
    trial_id: int
    error: Optional[str] = None
    error_code: Optional[str] = None
    error_type: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        return self.error is None


def test_groups_by_error_code():
    trials = [
        _FakeTrial(0, error="x", error_code="PROVIDER_TIMEOUT"),
        _FakeTrial(1, error="y", error_code="PROVIDER_TIMEOUT"),
        _FakeTrial(2, error="z", error_code="PROVIDER_AUTH_FAILED"),
    ]
    clusters = cluster_failures(trials)
    assert clusters == {"PROVIDER_TIMEOUT": [0, 1], "PROVIDER_AUTH_FAILED": [2]}


def test_falls_back_to_error_type_when_no_error_code():
    trials = [_FakeTrial(0, error="x", error_code=None, error_type="ValueError")]
    clusters = cluster_failures(trials)
    assert clusters == {"ValueError": [0]}


def test_falls_back_to_unknown_when_neither_set():
    trials = [_FakeTrial(0, error="x", error_code=None, error_type=None)]
    clusters = cluster_failures(trials)
    assert clusters == {"unknown": [0]}


def test_successful_trials_excluded():
    trials = [_FakeTrial(0, error=None), _FakeTrial(1, error="x", error_code="E")]
    clusters = cluster_failures(trials)
    assert clusters == {"E": [1]}


def test_no_failures_returns_empty_dict():
    trials = [_FakeTrial(0, error=None), _FakeTrial(1, error=None)]
    assert cluster_failures(trials) == {}


def test_cluster_ids_sorted():
    trials = [
        _FakeTrial(5, error="x", error_code="E"),
        _FakeTrial(1, error="y", error_code="E"),
        _FakeTrial(3, error="z", error_code="E"),
    ]
    clusters = cluster_failures(trials)
    assert clusters == {"E": [1, 3, 5]}


if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))
