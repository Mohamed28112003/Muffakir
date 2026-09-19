"""Test suite for Trace.context (pytest)."""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Trace.context import set_current_sample, get_current_sample, clear_current_sample


def test_default_is_none():
    clear_current_sample()
    assert get_current_sample() is None


def test_set_and_get_roundtrip():
    set_current_sample(3, 7)
    assert get_current_sample() == (3, 7)
    clear_current_sample()


def test_clear_resets_to_none():
    set_current_sample(1, 1)
    clear_current_sample()
    assert get_current_sample() is None


def test_isolated_per_thread():
    """A sample tag set on one thread must not leak into another thread."""
    results = {}

    def worker(tid, sid):
        set_current_sample(tid, sid)
        results[threading.current_thread().name] = get_current_sample()

    t1 = threading.Thread(target=worker, args=(1, 10), name="t1")
    t2 = threading.Thread(target=worker, args=(2, 20), name="t2")
    t1.start(); t1.join()
    t2.start(); t2.join()

    assert results["t1"] == (1, 10)
    assert results["t2"] == (2, 20)
    # Main thread's own tag (never set here) stays unaffected.
    assert get_current_sample() is None


if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))
