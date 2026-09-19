"""
Test suite for the MuffakirComposer module (pytest).

Covers: pure data classes, CheckpointManager (atomic/dedup), ConfigSpace
(validation, deep-copy), ReportGenerator (XSS, None metrics), weighted composite,
GridSearch (sequential + parallel) with a picklable mock worker (regression test
for the parallel pickling fix), and MuffakirComposer provider/metric validation.
"""

import json
import sys
from pathlib import Path

import pytest

# Make the library importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Composer.results.trial import TrialResult
from Composer.results.report import ComposerReport
from Composer.checkpoint import CheckpointManager
from Composer.config_space import ConfigSpace, trial_config_to_rag_config
from Composer.report_generator import ReportGenerator, _fmt_metric
from Composer.search.grid_search import GridSearch
from Composer.search.base_search import BaseSearch
from Composer.dataset_loader import DatasetLoader
from SyntheticData.models import QAPair
from Muffakir.exceptions import (
    CorruptCheckpointError,
    DatasetError,
    ProviderAuthenticationError,
    ProviderTimeoutError,
)


# Top-level (picklable) mock evaluation workers for GridSearch tests.
# Must be top-level so ProcessPoolExecutor can ship them to workers.
def _mock_eval_success(trial_id, config, shared_state):
    return TrialResult(
        trial_id=trial_id,
        config=config,
        metrics={"recall": 0.5 + (trial_id % 3) * 0.1},
        composite_score=0.5 + (trial_id % 3) * 0.1,
        latency_ms=10.0,
    )


def _mock_eval_failure(trial_id, config, shared_state):
    return TrialResult(trial_id=trial_id, config=config, error="intentional failure", latency_ms=1.0)


def _mock_eval_reverse_delay(trial_id, config, shared_state):
    """Sleeps longer for smaller trial_id, so higher trial_ids finish first —
    forces out-of-completion-order results to actually exercise the sort fix."""
    import time
    time.sleep(0.05 * (4 - trial_id))
    return TrialResult(trial_id=trial_id, config=config, composite_score=0.5, latency_ms=1.0)


def test_trial_result_serialization_roundtrip():
    trial = TrialResult(
        trial_id=1, config={"query_expansion": "hyde", "k": 5},
        metrics={"recall": 0.85}, composite_score=0.85, latency_ms=1500.0,
    )
    assert trial.is_successful
    restored = TrialResult.from_dict(trial.to_dict())
    assert restored.trial_id == trial.trial_id
    assert restored.composite_score == trial.composite_score
    assert restored.config == trial.config


def test_trial_result_error_is_not_successful():
    assert not TrialResult(trial_id=2, config={}, error="boom").is_successful


def test_trial_result_error_code_roundtrip():
    trial = TrialResult(
        trial_id=3, config={"k": 5}, error="Provider timed out",
        error_code="PROVIDER_TIMEOUT", error_type="ProviderTimeoutError",
    )
    restored = TrialResult.from_dict(trial.to_dict())
    assert restored.error_code == "PROVIDER_TIMEOUT"
    assert restored.error_type == "ProviderTimeoutError"
    # Old checkpoints without these keys must still load (backward compatible).
    legacy = TrialResult.from_dict({"trial_id": 4, "config": {}, "error": "boom"})
    assert legacy.error_code is None and legacy.error_type is None


# ---------------------------------------------------------------------------
# ComposerReport
# ---------------------------------------------------------------------------
def test_composer_report_best_trial_and_top_n():
    trials = [
        TrialResult(trial_id=1, config={"k": 3}, metrics={"recall": 0.7}, composite_score=0.7),
        TrialResult(trial_id=2, config={"k": 5}, metrics={"recall": 0.85}, composite_score=0.85),
        TrialResult(trial_id=3, config={"k": 10}, metrics={"recall": 0.8}, composite_score=0.8),
        TrialResult(trial_id=4, config={"k": 5}, error="fail"),
    ]
    report = ComposerReport(
        trials=trials, search_space={"k": [3, 5, 10]},
        total_duration_ms=1000.0, metrics_used=["recall"],
    )
    assert report.total_trials == 4
    assert report.successful_trials == 3
    assert report.failed_trials == 1
    assert report.best_score == pytest.approx(0.85)
    assert report.best_trial.trial_id == 2
    assert [t.trial_id for t in report.get_top_n_trials(2)] == [2, 3]
    assert "BEST CONFIGURATION" in report.summary()


def test_composer_report_best_result_contains_metrics_resolved_config_and_timings():
    trial = TrialResult(
        trial_id=4,
        config={"k": 3},
        resolved_config={"k": 3, "retrieval_method": "hybrid"},
        metrics={"recall@3": 0.8},
        composite_score=0.8,
        mean_pipeline_latency_ms=12.0,
        mean_evaluation_overhead_ms=2.0,
        stage_timings_ms={"vector_search": 7.0, "rerank": 3.0},
    )
    report = ComposerReport(trials=[trial])

    assert report.best_result["trial_id"] == 4
    assert report.best_result["config"]["retrieval_method"] == "hybrid"
    assert report.best_result["metrics"] == {"recall@3": 0.8}
    assert report.to_dict()["summary"]["best_metrics"] == {"recall@3": 0.8}


def test_composer_report_breaks_equal_score_ties_by_lowest_trial_id():
    report = ComposerReport(trials=[
        TrialResult(trial_id=9, config={}, composite_score=0.8),
        TrialResult(trial_id=2, config={}, composite_score=0.8),
    ])
    assert report.best_trial.trial_id == 2


def test_composer_report_to_dataframe_stable_sort_matches_top_n():
    """to_dataframe()'s composite_score sort must be stable (kind='mergesort'),
    matching get_top_n_trials()'s stable sorted() — otherwise tied-score
    trials could come out in a different order between the two."""
    trials = [
        TrialResult(trial_id=0, config={}, composite_score=0.5),
        TrialResult(trial_id=1, config={}, composite_score=0.9),
        TrialResult(trial_id=2, config={}, composite_score=0.5),
        TrialResult(trial_id=3, config={}, composite_score=0.5),
    ]
    report = ComposerReport(trials=trials, search_space={}, total_duration_ms=0.0, metrics_used=[])

    df_order = list(report.to_dataframe()["trial_id"])
    top_n_order = [t.trial_id for t in report.get_top_n_trials(n=10)]
    assert df_order == top_n_order


def test_composer_report_best_recomputes_after_merge():
    report = ComposerReport(
        trials=[TrialResult(trial_id=1, config={"k": 3}, metrics={"recall": 0.6}, composite_score=0.6)],
        search_space={"k": [3]}, metrics_used=["recall"],
    )
    assert report.best_trial.trial_id == 1
    report.trials.append(TrialResult(trial_id=2, config={"k": 5}, metrics={"recall": 0.9}, composite_score=0.9))
    report._compute_best_trial()
    assert report.best_trial.trial_id == 2


# ---------------------------------------------------------------------------
# CheckpointManager (atomic writes + dedup)
# ---------------------------------------------------------------------------
def test_checkpoint_save_load_dedup(tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    assert not mgr.has_checkpoint()
    mgr.save_trial(TrialResult(trial_id=1, config={"k": 5}, metrics={"recall": 0.8}, composite_score=0.8))
    assert mgr.get_trial_count() == 1
    # Saving the same trial_id again must replace, not duplicate (idempotent)
    mgr.save_trial(TrialResult(trial_id=1, config={"k": 5}, metrics={"recall": 0.9}, composite_score=0.9))
    assert mgr.get_trial_count() == 1
    loaded = mgr.load_completed_trials()
    assert len(loaded) == 1 and loaded[0].composite_score == 0.9


def test_checkpoint_no_temp_left_behind(tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    for i in range(5):
        mgr.save_trial(TrialResult(trial_id=i, config={"k": i}, metrics={}, composite_score=float(i)))
    leftovers = [p for p in Path(tmp_path).iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
    assert (tmp_path / "composer_checkpoint.json").exists()


def test_checkpoint_resume_ids(tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    for i in range(3):
        mgr.save_trial(TrialResult(trial_id=i, config={"k": i}, metrics={}, composite_score=float(i)))
    assert mgr.get_completed_ids() == {0, 1, 2}
    mgr.clear()
    assert not mgr.has_checkpoint()


def test_checkpoint_corrupt_file_raises(tmp_path):
    """A corrupt checkpoint must not be silently treated as 'no checkpoint yet' -
    that would risk discarding prior trial results by overwriting them."""
    mgr = CheckpointManager(str(tmp_path))
    mgr.checkpoint_file.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(CorruptCheckpointError):
        mgr.load_completed_trials()
    with pytest.raises(CorruptCheckpointError):
        mgr.get_last_updated()

    # Recovery path: clear() removes the corrupt file, resuming normal behavior.
    mgr.clear()
    assert mgr.load_completed_trials() == []
    assert mgr.get_last_updated() is None


def test_checkpoint_save_trial_does_not_reread_disk_every_call(tmp_path, monkeypatch):
    """save_trial() must not re-parse the whole checkpoint file on every call —
    that was the source of the O(n^2) total cost across a run. The in-memory
    cache should mean at most one json.load() for the whole test, regardless
    of how many trials are saved."""
    import json as json_module

    load_calls = {"n": 0}
    real_load = json_module.load

    def counting_load(*a, **kw):
        load_calls["n"] += 1
        return real_load(*a, **kw)

    monkeypatch.setattr(json_module, "load", counting_load)

    mgr = CheckpointManager(str(tmp_path))
    for i in range(10):
        mgr.save_trial(TrialResult(trial_id=i, config={"k": i}, metrics={}, composite_score=float(i)))

    assert mgr.get_trial_count() == 10
    # Zero because the checkpoint file didn't exist yet on the very first
    # save (nothing to load), and every subsequent save/read used the cache.
    assert load_calls["n"] == 0


def test_checkpoint_search_space_persisted_and_validated(tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    assert mgr.get_stored_search_space() is None
    mgr.validate_search_space({"k": [3, 5]})  # nothing stored yet -> no-op

    mgr.set_search_space({"k": [3, 5]})
    assert mgr.get_stored_search_space() == {"k": [3, 5]}

    # Same search space on "resume" -> fine.
    mgr.validate_search_space({"k": [3, 5]})

    # Changed search space -> must raise, not silently skip newly-added trials.
    from Muffakir.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError, match="[Ss]earch space"):
        mgr.validate_search_space({"k": [3, 5, 10]})


def test_checkpoint_search_space_persists_across_reload(tmp_path):
    """A fresh CheckpointManager instance pointed at the same directory must
    still see the previously-stored search space (round-trips through disk,
    not just kept in the first instance's memory)."""
    mgr1 = CheckpointManager(str(tmp_path))
    mgr1.set_search_space({"k": [3, 5]})

    mgr2 = CheckpointManager(str(tmp_path))
    assert mgr2.get_stored_search_space() == {"k": [3, 5]}


def test_checkpoint_pricing_snapshot_persisted(tmp_path):
    mgr = CheckpointManager(str(tmp_path))
    assert mgr.get_stored_pricing_snapshot() is None

    snapshot = {"raw_map": {"gpt-4o": {"input_cost_per_token": 0.1}}, "custom_pricing": {}, "fetch_failed": False, "fetched_at": "t"}
    mgr.set_pricing_snapshot(snapshot)
    assert mgr.get_stored_pricing_snapshot() == snapshot


def test_checkpoint_pricing_snapshot_persists_across_reload(tmp_path):
    """A fresh CheckpointManager instance pointed at the same directory must
    still see the previously-stored pricing snapshot (round-trips through
    disk), so a resumed fit() run never re-fetches prices."""
    snapshot = {"raw_map": {"gpt-4o": {"input_cost_per_token": 0.1}}, "custom_pricing": {}, "fetch_failed": False, "fetched_at": "t"}
    mgr1 = CheckpointManager(str(tmp_path))
    mgr1.set_pricing_snapshot(snapshot)

    mgr2 = CheckpointManager(str(tmp_path))
    assert mgr2.get_stored_pricing_snapshot() == snapshot


def test_checkpoint_pricing_snapshot_survives_alongside_search_space(tmp_path):
    """Both cached blocks must coexist in the same checkpoint file."""
    mgr = CheckpointManager(str(tmp_path))
    mgr.set_search_space({"k": [3, 5]})
    mgr.set_pricing_snapshot({"raw_map": {}, "custom_pricing": {}, "fetch_failed": True, "fetched_at": "t"})

    mgr2 = CheckpointManager(str(tmp_path))
    assert mgr2.get_stored_search_space() == {"k": [3, 5]}
    assert mgr2.get_stored_pricing_snapshot()["fetch_failed"] is True


# ---------------------------------------------------------------------------
# ConfigSpace
# ---------------------------------------------------------------------------
def test_config_space_default_combinations():
    cs = ConfigSpace()
    assert cs.total_combinations == 108
    assert len(cs.generate_combinations_with_ids()) == 108
    assert cs.generate_combinations_with_ids()[0][0] == 0


def test_config_space_rejects_empty_and_unknown():
    with pytest.raises(ValueError):
        ConfigSpace({})
    with pytest.raises(ValueError):
        ConfigSpace({"unknown_stage": ["a", "b"]})


def test_config_space_deepcopy_isolation():
    custom = {"query_expansion": ["none", "hyde"], "k": [3, 5]}
    cs = ConfigSpace(custom)
    # Mutate the caller's list AFTER construction -> ConfigSpace must be unaffected (deep copy)
    custom["query_expansion"].append("step_back")
    assert cs.search_space["query_expansion"] == ["none", "hyde"]
    # The module-level default must never be mutated
    from Composer.config_space import DEFAULT_SEARCH_SPACE
    assert DEFAULT_SEARCH_SPACE["query_expansion"] == ["none", "multi_query", "hyde", "step_back"]


def test_reranking_model_is_a_conditional_search_dimension():
    models = ["org/reranker-a", "org/reranker-b", "org/reranker-c"]
    space = ConfigSpace({
        "reranking": [
            "cross_encoder", "pointwise", "none", "bm25",
            "semantic_similarity", "llm", "custom",
        ],
        "reranking_model": models,
    })

    combinations = space.generate_combinations()
    assert space.total_combinations == 11  # 2 local methods × 3 models + 5 others
    local = [
        combo for combo in combinations
        if combo["reranking"] in {"cross_encoder", "pointwise"}
    ]
    non_local = [combo for combo in combinations if combo not in local]
    assert len(local) == 6
    assert {combo["reranking_model"] for combo in local} == set(models)
    assert all("reranking_model" not in combo for combo in non_local)
    assert len({combo["reranking"] for combo in non_local}) == 5


@pytest.mark.parametrize("models", [[""], ["  "], ["same/model", "same/model"]])
def test_config_space_rejects_invalid_reranking_model_lists(models):
    with pytest.raises(ValueError, match="reranking model|reranking_model"):
        ConfigSpace({"reranking": ["cross_encoder"], "reranking_model": models})


def test_config_space_rejects_reranking_models_without_a_compatible_method():
    with pytest.raises(ValueError, match="cross_encoder or pointwise"):
        ConfigSpace({"reranking": ["none", "llm"], "reranking_model": ["org/model"]})


def test_trial_config_to_rag_config_emits_strategy_key():
    rag_cfg = trial_config_to_rag_config(
        {"query_expansion": "hyde", "retrieval": "hybrid", "reranking": "cross_encoder", "k": 5},
        {"api_key": "k", "llm_provider": "together", "llm_model": "m", "data_dir": "d"},
    )
    assert rag_cfg["query_transformer"] is True
    assert rag_cfg["query_transformer_strategy"] == "hyde"          # real downstream key
    assert rag_cfg["retrieval_method"] == "hybrid"
    assert rag_cfg["reranking"] is True
    assert rag_cfg["reranking_method"] == "cross_encoder"
    assert rag_cfg["k"] == 5
    # 'none' must disable, not emit a bogus strategy
    off = trial_config_to_rag_config({"query_expansion": "none"}, {})
    assert off["query_transformer"] is False


def test_trial_config_maps_reranking_model_to_rag_config():
    rag_config = trial_config_to_rag_config(
        {"reranking": "cross_encoder", "reranking_model": " org/my-reranker "},
        {},
    )
    assert rag_config["reranking_model"] == "org/my-reranker"


def test_trial_config_does_not_mutate_base():
    base = {"api_key": "k", "llm_provider": "together", "llm_model": "m", "k": 3}
    trial_config_to_rag_config({"k": 5}, base)
    assert base["k"] == 3  # base untouched (deep copy)


# ---------------------------------------------------------------------------
# DatasetLoader row-failure threshold
# ---------------------------------------------------------------------------
def test_dataset_loader_csv_mostly_valid_rows_ok(tmp_path):
    csv_path = tmp_path / "eval.csv"
    csv_path.write_text(
        "question,context,answer\n"
        "what is the capital of Egypt?,Cairo is the capital of Egypt.,Cairo\n"
        "what is the capital of France?,Paris is the capital of France.,Paris\n",
        encoding="utf-8",
    )
    pairs = DatasetLoader.from_csv(str(csv_path))
    assert len(pairs) == 2


def test_dataset_loader_csv_exceeds_failure_threshold_raises(tmp_path):
    """A CSV where most rows fail to parse into a QAPair should raise
    DatasetError instead of silently returning a near-empty dataset.
    A non-numeric chunk_id makes int(row["chunk_id"]) raise, so that row
    is skipped - a realistic malformed-row scenario, no mocking needed."""
    csv_path = tmp_path / "eval.csv"
    csv_path.write_text(
        "question,context,answer,chunk_id\n"
        "q1,c1,a1,not-a-number\n"
        "q2,c2,a2,also-bad\n"
        "q3,c3,a3,still-bad\n"
        "q4,c4,a4,4\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError):
        DatasetLoader.from_csv(str(csv_path))


def test_composer_rejects_empty_evaluation_dataset_before_trials(tmp_path, monkeypatch):
    import Composer.composer as composer_module

    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    monkeypatch.setattr(
        composer_module.DatasetLoader, "load_or_generate",
        staticmethod(lambda **kwargs: []),
    )
    monkeypatch.setattr(
        composer_module.MuffakirComposer, "_index_documents",
        lambda self, combos: pytest.fail("empty dataset must not index documents"),
    )

    composer = composer_module.MuffakirComposer(config={
        "data_dir": str(data_dir),
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
    })
    with pytest.raises(DatasetError, match="no valid Q&A samples"):
        composer.fit(
            search_space={"k": [3]},
            save_report=False,
            enable_trace=False,
            checkpoint_dir=str(tmp_path / "checkpoint"),
        )


def test_auto_generated_dataset_with_all_failed_chunks_raises(tmp_path, monkeypatch):
    import pandas as pd
    from types import SimpleNamespace
    import SyntheticData.pipeline as pipeline_module

    class _EmptySyntheticPipeline:
        def __init__(self, config, prompt_manager):
            pass

        def run(self, max_chunks):
            return pd.DataFrame(), SimpleNamespace(
                successful=0, failed=3, total_chunks=3,
            )

    monkeypatch.setattr(
        pipeline_module, "SyntheticDataPipeline", _EmptySyntheticPipeline
    )

    documents = tmp_path / "documents"
    documents.mkdir()
    with pytest.raises(DatasetError, match="no valid Q&A pairs") as error:
        DatasetLoader.generate_from_documents(
            data_dir=str(documents),
            llm_config={
                "api_key": "test-key", "llm_provider": "openai",
                "llm_model": "test-model",
            },
            max_samples=3,
            language="en",
        )
    assert error.value.context["failed_chunks"] == 3


# ---------------------------------------------------------------------------
# ReportGenerator & JSON export
# ---------------------------------------------------------------------------
def _make_report(trials):
    return ComposerReport(
        trials=trials, search_space={"k": [3, 5, 10]},
        total_duration_ms=1000.0, metrics_used=["recall"],
    )


def test_reports_display_resolved_local_reranker_model(tmp_path):
    trial = TrialResult(
        trial_id=0,
        config={"reranking": "cross_encoder"},
        resolved_config={
            "reranking": True,
            "reranking_method": "cross_encoder",
            "reranking_model": "BAAI/bge-reranker-base",
        },
        metrics={"recall": 0.8},
        composite_score=0.8,
    )
    report = _make_report([trial])
    out = tmp_path / "reranker-report.json"

    ReportGenerator.generate_json(report, str(out))

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["trials"][0]["resolved_config"]["reranking_model"] == "BAAI/bge-reranker-base"
    assert report.to_dataframe().loc[0, "config_reranking_model"] == (
        "BAAI/bge-reranker-base"
    )
    assert "reranking_model: BAAI/bge-reranker-base" in report.summary()


def test_fmt_metric_none_and_value():
    assert _fmt_metric(None) == "N/A"
    assert _fmt_metric(0.123456) == "0.1235"
    assert _fmt_metric(0.5, precision=3) == "0.500"

    from Composer.report_generator import _fmt_named_metric

    assert _fmt_named_metric("llm_judge_rating", 4.25) == "4.25 / 5"


# ---------------------------------------------------------------------------
# Weighted composite scoring
# ---------------------------------------------------------------------------
def test_weighted_composite():
    from Composer.evaluation import _weighted_composite
    # equal weights
    assert _weighted_composite({"a": 0.4, "b": 0.8}, {}) == pytest.approx(0.6)
    # explicit weights
    assert _weighted_composite({"a": 0.4, "b": 0.8}, {"a": 1.0, "b": 3.0}) == pytest.approx(0.7)
    # None ignored
    assert _weighted_composite({"a": None, "b": 0.8}, {}) == pytest.approx(0.8)
    # all None -> 0.0
    assert _weighted_composite({"a": None}, {}) == 0.0
    # empty -> 0.0
    assert _weighted_composite({}, {}) == 0.0


def test_weighted_composite_normalizes_llm_judge_rating_only_for_ranking():
    from Composer.evaluation import _weighted_composite

    assert _weighted_composite({"llm_judge_rating": 1.0}, {}) == pytest.approx(0.0)
    assert _weighted_composite({"llm_judge_rating": 3.0}, {}) == pytest.approx(0.5)
    assert _weighted_composite({"llm_judge_rating": 5.0}, {}) == pytest.approx(1.0)
    assert _weighted_composite(
        {"answer_correctness": 0.75, "llm_judge_rating": 3.0},
        {"answer_correctness": 1.0, "llm_judge_rating": 1.0},
    ) == pytest.approx(0.625)


# ---------------------------------------------------------------------------
# GridSearch (sequential + parallel) with a picklable mock worker
# ---------------------------------------------------------------------------
class _NullCheckpoint:
    """In-memory checkpoint stub for GridSearch tests."""
    def __init__(self):
        self.saved = []

    def save_trial(self, trial):
        self.saved.append(trial)


def test_grid_search_sequential_runs_and_checkpoints():
    combos = [(0, {"k": 3}), (1, {"k": 5}), (2, {"k": 10})]
    search = GridSearch(n_jobs=1)
    ckpt = _NullCheckpoint()
    results = search.run(combos, _mock_eval_success, {"marker": "ok"}, ckpt)
    assert len(results) == 3
    assert all(r.is_successful for r in results)
    assert len(ckpt.saved) == 3


def test_grid_search_parallel_runs_without_pickling_error():
    """Regression test: the default parallel path must NOT raise PicklingError.
    Requires a top-level eval_fn + picklable shared_state (no closures over DBs)."""
    combos = [(0, {"k": 3}), (1, {"k": 5}), (2, {"k": 10}), (3, {"k": 5})]
    search = GridSearch(n_jobs=2)
    ckpt = _NullCheckpoint()
    results = search.run(combos, _mock_eval_success, {"marker": "ok"}, ckpt)
    assert len(results) == 4
    assert all(r.is_successful for r in results)


def test_grid_search_parallel_results_sorted_by_trial_id():
    """as_completed() yields in completion order, not submission order — results
    must be sorted by trial_id afterward so they're reproducible run-to-run
    regardless of which worker happens to finish first."""
    combos = [(0, {"k": 3}), (1, {"k": 5}), (2, {"k": 10}), (3, {"k": 5})]
    search = GridSearch(n_jobs=4)
    ckpt = _NullCheckpoint()
    results = search.run(combos, _mock_eval_reverse_delay, {}, ckpt)
    assert [r.trial_id for r in results] == [0, 1, 2, 3]


def test_grid_search_handles_failed_trials():
    combos = [(0, {"k": 3}), (1, {"k": 5})]
    search = GridSearch(n_jobs=1)
    ckpt = _NullCheckpoint()
    results = search.run(combos, _mock_eval_failure, {}, ckpt)
    assert len(results) == 2
    assert all(not r.is_successful for r in results)


def test_evaluate_trial_rejects_unsupported_provider(monkeypatch):
    """evaluate_trial() can be invoked directly (not just via composer.py, which
    validates the provider at construction time) — an unrecognized provider
    must raise loudly, not silently misclassify as 'custom'."""
    from unittest.mock import MagicMock
    from Composer.evaluation import evaluate_trial
    from Muffakir.exceptions import ConfigurationError

    monkeypatch.setattr(
        "Composer.evaluation._get_shared_components",
        lambda base_config: (MagicMock(), MagicMock()),
    )
    monkeypatch.setattr("Muffakir.Muffakir.MuffakirRAG", MagicMock())

    shared_state = {
        "base_config": {
            "llm_provider": "not_a_real_provider",
            "llm_model": "m",
            "data_dir": "d",
        },
        "eval_pairs": [],
        # Must include a generation metric -- the judge LLM (and its provider
        # validation) is only built when a generation metric is requested;
        # see test_evaluate_trial_builds_judge_llm_when_generation_metrics_requested.
        "metrics": ["faithfulness"],
    }
    with pytest.raises(ConfigurationError, match="Unsupported judge_llm_provider/llm_provider"):
        evaluate_trial(0, {}, shared_state)


def test_grid_search_empty_combinations():
    search = GridSearch(n_jobs=2)
    assert search.run([], _mock_eval_success, {}, _NullCheckpoint()) == []


# ---------------------------------------------------------------------------
# BaseSearch._run_with_retry - type-aware retry classification
# ---------------------------------------------------------------------------
def test_run_with_retry_retries_retryable_error_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)  # skip real backoff delay in tests
    calls = {"n": 0}

    def flaky_then_ok(trial_id, config, shared_state):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderTimeoutError("timed out")
        return TrialResult(trial_id=trial_id, config=config, composite_score=0.9)

    search = GridSearch(n_jobs=1, max_retries=3)
    result = search._run_with_retry(flaky_then_ok, 0, {"k": 5}, {})

    assert calls["n"] == 3
    assert result.is_successful
    assert result.composite_score == 0.9


def test_run_with_retry_fails_fast_on_non_retryable_error():
    calls = {"n": 0}

    def always_auth_failure(trial_id, config, shared_state):
        calls["n"] += 1
        raise ProviderAuthenticationError("bad api key")

    search = GridSearch(n_jobs=1, max_retries=3)
    result = search._run_with_retry(always_auth_failure, 0, {"k": 5}, {})

    assert calls["n"] == 1  # no retries wasted on a non-retryable failure
    assert not result.is_successful
    assert result.error_code == "PROVIDER_AUTH_FAILED"
    assert result.error_type == "ProviderAuthenticationError"


def test_run_with_retry_emits_redacted_start_and_failed_terminal_trace():
    import queue

    trace_queue = queue.Queue()
    shared_state = {
        "base_config": {
            "api_key": "must-not-leak",
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "db_path": "./db",
        },
        "eval_pairs": [object(), object()],
        "trace_queue": trace_queue,
    }

    def fail(trial_id, config, state):
        raise ProviderAuthenticationError("bad api key")

    result = GridSearch(n_jobs=1)._run_with_retry(fail, 4, {"k": 5}, shared_state)
    assert not result.is_successful

    start_kind, start = trace_queue.get_nowait()
    final_kind, final = trace_queue.get_nowait()
    assert start_kind == "trial_started"
    assert start["trial_id"] == 4
    assert start["status"] == "running"
    assert start["total_samples"] == 2
    assert start["resolved_rag_config"]["k"] == 5
    assert "api_key" not in start["resolved_rag_config"]
    assert final_kind == "trial"
    assert final["status"] == "failed"
    assert final["error_code"] == "PROVIDER_AUTH_FAILED"


def test_parallel_executor_failure_emits_terminal_trace(monkeypatch):
    import queue
    import Composer.search.grid_search as grid_search_module

    class _BrokenFuture:
        def result(self):
            raise RuntimeError("worker disappeared")

        def cancel(self):
            return False

    class _BrokenExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, *args, **kwargs):
            return _BrokenFuture()

    monkeypatch.setattr(grid_search_module, "ProcessPoolExecutor", _BrokenExecutor)
    monkeypatch.setattr(grid_search_module, "as_completed", lambda futures: list(futures))

    trace_queue = queue.Queue()
    results = GridSearch(n_jobs=2).run(
        [(6, {"k": 3})],
        _mock_eval_success,
        {"base_config": {}, "eval_pairs": [], "trace_queue": trace_queue},
        _NullCheckpoint(),
    )

    assert results[0].error_code == "EXECUTOR_ERROR"
    kind, record = trace_queue.get_nowait()
    assert kind == "trial"
    assert record["trial_id"] == 6
    assert record["status"] == "failed"


# ---------------------------------------------------------------------------
# MuffakirComposer validation (no LLM/DB required)
# ---------------------------------------------------------------------------
def _composer_kwargs(tmp_path):
    return {
        "data_dir": str(tmp_path),
        "api_key": "test-key",
        "llm_provider": "together",
        "llm_model": "some-model",
    }


def test_composer_rejects_unsupported_provider(tmp_path):
    from Composer.composer import MuffakirComposer
    kw = _composer_kwargs(tmp_path)
    kw["llm_provider"] = "nonexistent_provider_xyz"  # truly invalid provider
    with pytest.raises(ValueError):
        MuffakirComposer(kw)



def test_composer_accepts_supported_provider(tmp_path):
    from Composer.composer import MuffakirComposer
    MuffakirComposer(_composer_kwargs(tmp_path))  # must not raise


def test_composer_metric_validation(tmp_path):
    from Composer.composer import MuffakirComposer
    composer = MuffakirComposer(_composer_kwargs(tmp_path))
    # valid metrics pass
    composer._validate_metrics(["recall", "faithfulness", "answer_correctness", "llm_judge_rating"])
    # invalid metric raises
    with pytest.raises(ValueError):
        composer._validate_metrics(["recal"])  # typo


# ---------------------------------------------------------------------------
# pipeline_mode="retrieval_only" -- no main generation LLM required/called
# ---------------------------------------------------------------------------
def test_retrieval_only_mode_does_not_require_api_key():
    from Composer.composer import MuffakirComposer

    # Would normally raise "Required parameter 'api_key' is missing" for a
    # non-ollama/vllm provider with no base_url -- retrieval_only mode never
    # calls the main LLM, so api_key is not required.
    composer = MuffakirComposer(config={
        "data_dir": ".",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "pipeline_mode": "retrieval_only",
    })
    assert composer.config["pipeline_mode"] == "retrieval_only"


def test_fit_rejects_generation_metrics_with_retrieval_only_mode(tmp_path):
    from Composer.composer import MuffakirComposer
    from Muffakir.exceptions import ConfigurationError

    composer = MuffakirComposer(config={
        "data_dir": str(tmp_path),
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "pipeline_mode": "retrieval_only",
    })
    with pytest.raises(ConfigurationError, match="retrieval_only"):
        composer._validate_pipeline_mode(["recall", "faithfulness"])

    with pytest.raises(ConfigurationError, match="llm_judge_rating"):
        composer._validate_pipeline_mode(["llm_judge_rating"])


def test_fit_rejects_retrieval_only_combined_with_web_search_only():
    from Composer.composer import MuffakirComposer
    from Muffakir.exceptions import ConfigurationError

    composer = MuffakirComposer(config={
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "pipeline_mode": "retrieval_only",
        "retrieval_source": "web_search_only",
    })
    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        composer._validate_pipeline_mode(["recall"])


def test_validate_pipeline_mode_allows_retrieval_metrics_only():
    from Composer.composer import MuffakirComposer

    composer = MuffakirComposer(config={
        "data_dir": ".",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "pipeline_mode": "retrieval_only",
    })
    composer._validate_pipeline_mode(["recall", "mrr"])  # must not raise


# ---------------------------------------------------------------------------
# retrieval_source="web_search_only" -- no local corpus needed
# ---------------------------------------------------------------------------
def test_composer_web_search_only_skips_data_dir_requirement():
    from Composer.composer import MuffakirComposer

    # No data_dir at all, and it would not exist on disk even if given --
    # must not raise, since web_search_only mode has no local corpus.
    composer = MuffakirComposer({
        "api_key": "test-key",
        "llm_provider": "together",
        "llm_model": "some-model",
        "retrieval_source": "web_search_only",
    })
    assert composer.config.get("data_dir") is None


def test_composer_vector_db_mode_still_requires_data_dir():
    """Regression: omitting retrieval_source (default vector_db mode) must
    still require data_dir exactly as before."""
    from Composer.composer import MuffakirComposer

    with pytest.raises(ValueError, match="data_dir"):
        MuffakirComposer({
            "api_key": "test-key",
            "llm_provider": "together",
            "llm_model": "some-model",
        })


def test_composer_fit_skips_indexing_for_web_search_only(monkeypatch):
    from Composer.composer import MuffakirComposer
    import Composer.composer as composer_module

    class _FakeSearch:
        def __init__(self, **kwargs):
            self.stopped_early = False
            self.stop_reason = None

        def run(self, **kwargs):
            return []

    index_calls = []
    monkeypatch.setattr(composer_module, "GridSearch", _FakeSearch)
    monkeypatch.setattr(
        composer_module.MuffakirComposer, "_index_documents",
        lambda self, combos: index_calls.append(combos),
    )
    monkeypatch.setattr(
        composer_module.DatasetLoader, "load_or_generate", staticmethod(lambda **kwargs: [QAPair(question="What happened?", answer="A valid answer", context="Reference context")])
    )

    composer = MuffakirComposer({
        "api_key": "test-key",
        "llm_provider": "together",
        "llm_model": "some-model",
        "retrieval_source": "web_search_only",
    })
    composer.fit(search_space={"k": [3]}, enable_trace=False, save_report=False)

    assert index_calls == []


# ---------------------------------------------------------------------------
# llm search-space dimension
# ---------------------------------------------------------------------------
def test_config_space_accepts_llm_dimension():
    from Composer.config_space import ConfigSpace

    space = ConfigSpace({
        "llm": [
            {"provider": "openai", "model": "gpt-4o-mini"},
            {"provider": "together", "model": "meta-llama/Llama-3-8b-chat-hf"},
        ],
    })
    assert space.total_combinations == 2


def test_config_space_rejects_llm_entry_missing_model():
    from Composer.config_space import ConfigSpace

    with pytest.raises(ValueError, match="model"):
        ConfigSpace({"llm": [{"provider": "openai"}]})


def test_config_space_rejects_llm_entry_not_a_dict():
    from Composer.config_space import ConfigSpace

    with pytest.raises(ValueError, match="dict"):
        ConfigSpace({"llm": ["openai/gpt-4o-mini"]})


def test_trial_config_to_rag_config_maps_llm_dimension():
    from Composer.config_space import trial_config_to_rag_config

    rag_config = trial_config_to_rag_config(
        {"llm": {"provider": "together", "model": "meta-llama/Llama-3-8b-chat-hf"}},
        {"llm_provider": "openai", "llm_model": "gpt-4o-mini"},
    )
    assert rag_config["llm_provider"] == "together"
    assert rag_config["llm_model"] == "meta-llama/Llama-3-8b-chat-hf"


# ---------------------------------------------------------------------------
# Judge LLM must stay fixed regardless of trial_config["llm"]
# ---------------------------------------------------------------------------
def test_evaluate_trial_judge_llm_ignores_trial_config_llm_override(monkeypatch):
    """The eval judge must always come from base_config, never trial_config['llm'] —
    comparing trials judged by different models would silently corrupt every result."""
    import Composer.evaluation as evaluation_module
    from Evaluation.models import EvaluationReport

    captured_models = []

    class _SpyLLMProvider:
        def __init__(self, *args, **kwargs):
            captured_models.append(kwargs.get("model"))

    class _FakeRunner:
        def __init__(self, **kwargs):
            pass

        def run(self, pairs):
            return EvaluationReport(n_samples=0, k=5, metrics_enabled=[])

    class _FakeRag:
        pass

    import importlib

    llm_provider_module = importlib.import_module("LLMProvider.LLMProvider")
    runner_module = importlib.import_module("Evaluation.runner")
    muffakir_module = importlib.import_module("Muffakir.Muffakir")

    monkeypatch.setattr(llm_provider_module, "LLMProvider", _SpyLLMProvider)
    monkeypatch.setattr(runner_module, "EvalRunner", _FakeRunner)
    monkeypatch.setattr(muffakir_module, "MuffakirRAG", lambda *a, **k: _FakeRag())
    monkeypatch.setattr(evaluation_module, "_get_shared_components", lambda base_config: (None, None))

    base_config = {
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
        "db_path": "./test_db",
    }
    trial_config = {
        "llm": {"provider": "together", "model": "meta-llama/Llama-3-8b-chat-hf"},
    }
    # Must include a generation metric -- the judge LLM is only built when one
    # is requested (see test_evaluate_trial_builds_judge_llm_when_generation_metrics_requested).
    shared_state = {"base_config": base_config, "eval_pairs": [], "metrics": ["faithfulness"]}

    evaluation_module.evaluate_trial(0, trial_config, shared_state)

    # The judge must have been built with the base config's model, never the
    # trial's overridden one.
    assert captured_models == ["gpt-4o-mini"]


def _run_evaluate_trial_for_judge_llm_capture(monkeypatch, base_config):
    """Shared setup for the judge-LLM-override tests below: same monkeypatch
    style as test_evaluate_trial_judge_llm_ignores_trial_config_llm_override,
    returns the kwargs the judge LLMProvider was constructed with."""
    import importlib

    import Composer.evaluation as evaluation_module
    from Evaluation.models import EvaluationReport

    captured_kwargs = {}

    class _SpyLLMProvider:
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)

    class _FakeRunner:
        def __init__(self, **kwargs):
            pass

        def run(self, pairs):
            return EvaluationReport(n_samples=0, k=5, metrics_enabled=[])

    class _FakeRag:
        pass

    llm_provider_module = importlib.import_module("LLMProvider.LLMProvider")
    runner_module = importlib.import_module("Evaluation.runner")
    muffakir_module = importlib.import_module("Muffakir.Muffakir")

    monkeypatch.setattr(llm_provider_module, "LLMProvider", _SpyLLMProvider)
    monkeypatch.setattr(runner_module, "EvalRunner", _FakeRunner)
    monkeypatch.setattr(muffakir_module, "MuffakirRAG", lambda *a, **k: _FakeRag())
    monkeypatch.setattr(evaluation_module, "_get_shared_components", lambda base_config: (None, None))

    # Must include a generation metric -- the judge LLM is only built when one
    # is requested (see test_evaluate_trial_builds_judge_llm_when_generation_metrics_requested).
    shared_state = {"base_config": base_config, "eval_pairs": [], "metrics": ["faithfulness"]}
    evaluation_module.evaluate_trial(0, {}, shared_state)
    return captured_kwargs


def test_evaluate_trial_uses_judge_llm_override_when_present(monkeypatch):
    """judge_llm_provider/judge_llm_model/judge_api_key/judge_base_url, when set
    on base_config, override the generation LLM's config for the eval judge."""
    base_config = {
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "gen-key",
        "base_url": "https://gen.example/v1",
        "db_path": "./test_db",
        "judge_llm_provider": "anthropic",
        "judge_llm_model": "claude-3-haiku",
        "judge_api_key": "judge-key",
        "judge_base_url": "https://judge.example/v1",
    }
    captured = _run_evaluate_trial_for_judge_llm_capture(monkeypatch, base_config)

    assert captured["provider"] == "anthropic"
    assert captured["model"] == "claude-3-haiku"
    assert captured["api_key"] == "judge-key"
    assert captured["base_url"] == "https://judge.example/v1"


def test_evaluate_trial_uses_muffakir_search_for_web_search_only(monkeypatch):
    """retrieval_source='web_search_only' must route through MuffakirSearch,
    never build the vector-db shared components (embedding/vector db)."""
    import importlib
    from unittest.mock import MagicMock

    import Composer.evaluation as evaluation_module
    from Evaluation.models import EvaluationReport

    captured_configs = []
    captured_runner_kwargs = {}

    class _SpyMuffakirSearch:
        def __init__(self, config, prompt_manager=None):
            captured_configs.append(config)
            self.llm_provider = None

    class _FakeRunner:
        def __init__(self, **kwargs):
            captured_runner_kwargs.update(kwargs)

        def run(self, pairs):
            return EvaluationReport(n_samples=0, k=5, metrics_enabled=[])

    def _boom(rag_config):
        raise AssertionError("_get_shared_components must not be called for web_search_only")

    llm_provider_module = importlib.import_module("LLMProvider.LLMProvider")
    runner_module = importlib.import_module("Evaluation.runner")
    muffakir_search_module = importlib.import_module("Muffakir.MuffakirSearch")

    monkeypatch.setattr(muffakir_search_module, "MuffakirSearch", _SpyMuffakirSearch)
    monkeypatch.setattr(evaluation_module, "_get_shared_components", _boom)
    monkeypatch.setattr(llm_provider_module, "LLMProvider", lambda *a, **k: MagicMock())
    monkeypatch.setattr(runner_module, "EvalRunner", _FakeRunner)

    base_config = {
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
        "search_provider": "tavily",
        "search_provider_config": {"api_key": "tavily-key"},
        "retrieval_source": "web_search_only",
    }
    shared_state = {"base_config": base_config, "eval_pairs": [], "metrics": []}

    evaluation_module.evaluate_trial(0, {}, shared_state)

    assert len(captured_configs) == 1
    assert captured_configs[0]["retrieval_source"] == "web_search_only"
    assert captured_runner_kwargs["execute_generation"] is True
    assert captured_runner_kwargs["llm_provider"] is None


def test_evaluate_trial_uses_vector_db_path_when_no_retrieval_source(monkeypatch):
    """Regression: omitting retrieval_source (default) must still build the
    vector-db MuffakirRAG path unchanged, never MuffakirSearch."""
    import importlib
    from unittest.mock import MagicMock

    import Composer.evaluation as evaluation_module
    from Evaluation.models import EvaluationReport

    rag_calls = []
    captured_runner_kwargs = {}

    class _FakeRunner:
        def __init__(self, **kwargs):
            captured_runner_kwargs.update(kwargs)

        def run(self, pairs):
            return EvaluationReport(n_samples=0, k=5, metrics_enabled=[])

    def _boom_search(config):
        raise AssertionError("MuffakirSearch must not be constructed for the default vector_db path")

    llm_provider_module = importlib.import_module("LLMProvider.LLMProvider")
    runner_module = importlib.import_module("Evaluation.runner")
    muffakir_module = importlib.import_module("Muffakir.Muffakir")
    muffakir_search_module = importlib.import_module("Muffakir.MuffakirSearch")

    monkeypatch.setattr(muffakir_search_module, "MuffakirSearch", _boom_search)
    monkeypatch.setattr(evaluation_module, "_get_shared_components", lambda rag_config: (None, None))
    monkeypatch.setattr(llm_provider_module, "LLMProvider", lambda *a, **k: MagicMock())
    monkeypatch.setattr(runner_module, "EvalRunner", _FakeRunner)
    monkeypatch.setattr(muffakir_module, "MuffakirRAG", lambda *a, **k: rag_calls.append(1) or MagicMock())

    base_config = {
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
        "db_path": "./test_db",
    }
    shared_state = {"base_config": base_config, "eval_pairs": [], "metrics": []}

    evaluation_module.evaluate_trial(0, {}, shared_state)

    assert rag_calls == [1]
    assert captured_runner_kwargs["execute_generation"] is True
    assert captured_runner_kwargs["llm_provider"] is None


def test_evaluate_trial_judge_llm_falls_back_when_no_override(monkeypatch):
    """Omitting judge_* keys reproduces today's exact behavior: judge reuses
    the generation LLM's own provider/model/api_key/base_url unchanged."""
    base_config = {
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "gen-key",
        "base_url": "https://gen.example/v1",
        "db_path": "./test_db",
    }
    captured = _run_evaluate_trial_for_judge_llm_capture(monkeypatch, base_config)

    assert captured["provider"] == "openai"
    assert captured["model"] == "gpt-4o-mini"
    assert captured["api_key"] == "gen-key"
    assert captured["base_url"] == "https://gen.example/v1"


# ---------------------------------------------------------------------------
# evaluate_trial() -> TrialRecord push (Trace wiring)
# ---------------------------------------------------------------------------
def test_evaluate_trial_pushes_trial_record_when_trace_queue_set(monkeypatch):
    import importlib
    import queue as queue_module

    import Composer.evaluation as evaluation_module
    from Evaluation.models import EvaluationReport

    class _FakeRunner:
        def __init__(self, **kwargs):
            self.collected_sample_traces = [
                {"pipeline_latency_ms": 55.0, "evaluation_overhead_ms": 5.0, "query_embedding_ms": 2.0, "vector_search_ms": 8.0, "generation_ms": 40.0},
                {"pipeline_latency_ms": 45.0, "evaluation_overhead_ms": 3.0, "query_embedding_ms": 4.0, "vector_search_ms": 6.0, "generation_ms": 20.0},
            ]

        def run(self, pairs):
            return EvaluationReport(n_samples=2, k=5, metrics_enabled=["recall"], retrieval={"recall@5": 1.0})

    class _FakeRag:
        pass

    llm_provider_module = importlib.import_module("LLMProvider.LLMProvider")
    runner_module = importlib.import_module("Evaluation.runner")
    muffakir_module = importlib.import_module("Muffakir.Muffakir")

    class _StubLLMProvider:
        def __init__(self, *a, **k):
            pass

        def get_usage_totals(self):
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    monkeypatch.setattr(llm_provider_module, "LLMProvider", _StubLLMProvider)
    monkeypatch.setattr(runner_module, "EvalRunner", _FakeRunner)
    monkeypatch.setattr(muffakir_module, "MuffakirRAG", lambda *a, **k: _FakeRag())
    monkeypatch.setattr(evaluation_module, "_get_shared_components", lambda rag_config: (None, None))

    base_config = {
        "llm_provider": "openai", "llm_model": "gpt-4o-mini", "api_key": "test-key", "db_path": "./test_db",
    }
    trace_queue = queue_module.Queue()
    shared_state = {"base_config": base_config, "eval_pairs": [], "metrics": [], "trace_queue": trace_queue}

    evaluation_module.evaluate_trial(7, {"k": 5}, shared_state)

    kind, record = trace_queue.get_nowait()
    assert kind == "trial"
    assert record["trial_id"] == 7
    assert record["mean_query_embedding_ms"] == 3.0
    assert record["mean_vector_search_ms"] == 7.0
    assert record["mean_generation_ms"] == 30.0
    assert record["mean_pipeline_latency_ms"] == 50.0
    assert record["mean_evaluation_overhead_ms"] == 4.0
    assert "api_key" not in record["resolved_rag_config"]


def test_evaluate_trial_does_not_push_when_trace_queue_absent(monkeypatch):
    import importlib

    import Composer.evaluation as evaluation_module
    from Evaluation.models import EvaluationReport

    class _FakeRunner:
        def __init__(self, **kwargs):
            self.collected_sample_traces = []

        def run(self, pairs):
            return EvaluationReport(n_samples=0, k=5, metrics_enabled=[])

    class _FakeRag:
        pass

    llm_provider_module = importlib.import_module("LLMProvider.LLMProvider")
    runner_module = importlib.import_module("Evaluation.runner")
    muffakir_module = importlib.import_module("Muffakir.Muffakir")

    monkeypatch.setattr(llm_provider_module, "LLMProvider", lambda *a, **k: object())
    monkeypatch.setattr(runner_module, "EvalRunner", _FakeRunner)
    monkeypatch.setattr(muffakir_module, "MuffakirRAG", lambda *a, **k: _FakeRag())
    monkeypatch.setattr(evaluation_module, "_get_shared_components", lambda rag_config: (None, None))

    base_config = {
        "llm_provider": "openai", "llm_model": "gpt-4o-mini", "api_key": "test-key", "db_path": "./test_db",
    }
    shared_state = {"base_config": base_config, "eval_pairs": [], "metrics": []}  # no trace_queue key

    result = evaluation_module.evaluate_trial(0, {}, shared_state)
    assert result.is_successful


# ---------------------------------------------------------------------------
# evaluate_trial() -- pipeline_mode="retrieval_only" branch + conditional judge LLM
#
# Note on mocking style: evaluate_trial() imports MuffakirRetrieval/MuffakirRAG/
# MuffakirSearch/LLMProvider/EvalRunner *inside the function body* (fresh on every
# call), so Composer.evaluation itself never holds these as module-level
# attributes -- patching "Composer.evaluation.MuffakirRetrieval" (etc.) would raise
# AttributeError. Like every other evaluate_trial test in this file, these two
# monkeypatch the *origin* module's attribute instead (e.g.
# Muffakir.MuffakirRetrieval.MuffakirRetrieval), which the function's own
# `from ... import ...` picks up fresh on each call.
# ---------------------------------------------------------------------------
def test_evaluate_trial_builds_muffakir_retrieval_for_retrieval_only_mode(monkeypatch):
    """pipeline_mode='retrieval_only' must build MuffakirRetrieval (never
    MuffakirRAG), and the judge LLM/prompt_manager must NOT be constructed when
    no generation metric is requested -- retrieval-only trials never need a judge LLM."""
    import importlib
    import Composer.evaluation as evaluation_module
    from Evaluation.models import EvaluationReport

    captured_retrieval_init = []
    captured_runner_kwargs = {}

    class _SpyMuffakirRetrieval:
        def __init__(self, rag_config, embedding_provider, db_manager, prompt_manager=None):
            captured_retrieval_init.append(rag_config)

    class _FakeRunner:
        def __init__(self, **kwargs):
            captured_runner_kwargs.update(kwargs)
            self.collected_sample_traces = []

        def run(self, pairs):
            return EvaluationReport(
                n_samples=0, k=5, metrics_enabled=["recall"], retrieval={"recall@5": 1.0}
            )

    retrieval_module = importlib.import_module("Muffakir.MuffakirRetrieval")
    runner_module = importlib.import_module("Evaluation.runner")

    monkeypatch.setattr(retrieval_module, "MuffakirRetrieval", _SpyMuffakirRetrieval)
    monkeypatch.setattr(runner_module, "EvalRunner", _FakeRunner)
    monkeypatch.setattr(evaluation_module, "_get_shared_components", lambda rag_config: (None, None))

    base_config = {
        "data_dir": ".",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "pipeline_mode": "retrieval_only",
        "db_path": "./test_db",
        "collection_name": "test",
        "embedding_model": "mohamed2811/Muffakir_Embedding",
    }
    shared_state = {
        "base_config": base_config,
        "eval_pairs": [],
        "metrics": ["recall"],
        "pricing_snapshot": {},
    }

    evaluation_module.evaluate_trial(trial_id=1, trial_config={}, shared_state=shared_state)

    assert len(captured_retrieval_init) == 1
    assert captured_retrieval_init[0]["pipeline_mode"] == "retrieval_only"
    # Judge LLM must NOT be constructed -- no generation metric requested.
    assert captured_runner_kwargs["llm_provider"] is None
    assert captured_runner_kwargs["prompt_manager"] is not None
    assert captured_runner_kwargs["execute_generation"] is False


def test_evaluate_trial_builds_judge_llm_when_generation_metrics_requested(monkeypatch):
    """A requested generation metric (e.g. faithfulness) must still trigger judge
    LLM/prompt_manager construction for a full_rag trial, passed into EvalRunner."""
    import importlib
    import Composer.evaluation as evaluation_module
    from Evaluation.models import EvaluationReport

    captured_runner_kwargs = {}
    judge_settings = []

    class _FakeRag:
        pass

    class _SpyLLMProvider:
        def __init__(self, *a, **k):
            judge_settings.append(k["parameters"])

    class _FakeRunner:
        def __init__(self, **kwargs):
            captured_runner_kwargs.update(kwargs)
            self.collected_sample_traces = []

        def run(self, pairs):
            return EvaluationReport(
                n_samples=0, k=5, metrics_enabled=["faithfulness"], generation={"faithfulness": 1.0}
            )

    llm_provider_module = importlib.import_module("LLMProvider.LLMProvider")
    runner_module = importlib.import_module("Evaluation.runner")
    muffakir_module = importlib.import_module("Muffakir.Muffakir")

    monkeypatch.setattr(llm_provider_module, "LLMProvider", _SpyLLMProvider)
    monkeypatch.setattr(runner_module, "EvalRunner", _FakeRunner)
    monkeypatch.setattr(muffakir_module, "MuffakirRAG", lambda *a, **k: _FakeRag())
    monkeypatch.setattr(evaluation_module, "_get_shared_components", lambda rag_config: (None, None))

    base_config = {
        "judge_llm_parameters": {"temperature": 0, "max_tokens": 333},
        "data_dir": ".",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
        "db_path": "./test_db",
        "collection_name": "test",
        "embedding_model": "mohamed2811/Muffakir_Embedding",
    }
    shared_state = {
        "base_config": base_config,
        "eval_pairs": [],
        "metrics": ["faithfulness"],
        "pricing_snapshot": {},
    }

    evaluation_module.evaluate_trial(trial_id=1, trial_config={}, shared_state=shared_state)
    evaluation_module.evaluate_trial(trial_id=2, trial_config={"llm": {
        "provider": "openai", "model": "gpt-4o-mini", "parameters": {"temperature": 0.9}
    }}, shared_state=shared_state)
    assert judge_settings == [{"temperature": 0, "max_tokens": 333}] * 2

    assert captured_runner_kwargs["llm_provider"] is not None
    assert captured_runner_kwargs["prompt_manager"] is not None
    assert captured_runner_kwargs["execute_generation"] is True


# ---------------------------------------------------------------------------
# Trial-count / wall-clock budget cap
# ---------------------------------------------------------------------------
def _module_level_capped_eval_fn(trial_id, config, shared_state):
    return TrialResult(trial_id=trial_id, config=config, composite_score=1.0)


def test_grid_search_sequential_respects_max_trials(tmp_path):
    from Composer.search.grid_search import GridSearch

    combinations = [(i, {"k": i}) for i in range(5)]
    checkpoint_manager = CheckpointManager(str(tmp_path / "ckpt"))

    search = GridSearch(n_jobs=1, max_trials=2)
    results = search.run(combinations, _module_level_capped_eval_fn, {}, checkpoint_manager)

    assert len(results) == 2
    assert search.stopped_early is True
    assert search.stop_reason == "max_trials"


def test_grid_search_parallel_respects_max_trials(tmp_path):
    from Composer.search.grid_search import GridSearch

    combinations = [(i, {"k": i}) for i in range(6)]
    checkpoint_manager = CheckpointManager(str(tmp_path / "ckpt"))

    search = GridSearch(n_jobs=2, max_trials=3)
    results = search.run(combinations, _module_level_capped_eval_fn, {}, checkpoint_manager)

    assert len(results) <= 3
    assert search.stopped_early is True
    assert search.stop_reason == "max_trials"


def test_grid_search_no_cap_runs_all_trials(tmp_path):
    from Composer.search.grid_search import GridSearch

    combinations = [(i, {"k": i}) for i in range(4)]
    checkpoint_manager = CheckpointManager(str(tmp_path / "ckpt"))

    search = GridSearch(n_jobs=1)
    results = search.run(combinations, _module_level_capped_eval_fn, {}, checkpoint_manager)

    assert len(results) == 4
    assert search.stopped_early is False


# ---------------------------------------------------------------------------
# ComposerReport.stopped_early / stop_reason
# ---------------------------------------------------------------------------
def test_composer_report_defaults_not_stopped_early():
    report = ComposerReport()
    assert report.stopped_early is False
    assert report.stop_reason is None


def test_composer_report_stopped_early_roundtrips_through_json():
    report = ComposerReport(stopped_early=True, stop_reason="max_trials")
    restored = ComposerReport.from_json(report.to_json())
    assert restored.stopped_early is True
    assert restored.stop_reason == "max_trials"


def test_fit_populates_stopped_early_from_search(monkeypatch, tmp_path):
    import Composer.composer as composer_module

    class _FakeSearch:
        def __init__(self, **kwargs):
            self.stopped_early = True
            self.stop_reason = "max_trials"

        def run(self, **kwargs):
            return []

    monkeypatch.setattr(composer_module, "GridSearch", _FakeSearch)
    monkeypatch.setattr(
        composer_module.MuffakirComposer, "_index_documents", lambda self, combos: None
    )
    monkeypatch.setattr(
        composer_module.DatasetLoader, "load_or_generate", staticmethod(lambda **kwargs: [QAPair(question="What happened?", answer="A valid answer", context="Reference context")])
    )

    data_dir = tmp_path / "docs"
    data_dir.mkdir()

    composer = composer_module.MuffakirComposer(config={
        "data_dir": str(data_dir),
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
    })
    report = composer.fit(
        save_report=False,
        checkpoint_dir=str(tmp_path / "ckpt"),
        resume=False,
    )
    assert report.stopped_early is True
    assert report.stop_reason == "max_trials"


# ---------------------------------------------------------------------------
# fit(enable_trace=...) -> RunManifest + trials.jsonl + samples.jsonl
# ---------------------------------------------------------------------------
def test_fit_enable_trace_writes_manifest_and_jsonl(monkeypatch, tmp_path):
    import json
    import Composer.composer as composer_module

    class _FakeSearch:
        """Simulates two completed trials, each pushing one trial record and
        two sample records to shared_state["trace_queue"] -- exactly what real
        worker processes do via evaluate_trial()/EvalRunner."""

        def __init__(self, **kwargs):
            self.stopped_early = False
            self.stop_reason = None

        def run(self, combinations, eval_fn, shared_state, checkpoint_manager):
            trace_queue = shared_state["trace_queue"]
            results = []
            for trial_id, _config in combinations:
                for sample_index in range(2):
                    trace_queue.put(("sample", {"trial_id": trial_id, "sample_index": sample_index}))
                trace_queue.put(("trial", {"trial_id": trial_id, "composite_score": 0.5}))
                results.append(TrialResult(trial_id=trial_id, config={}, composite_score=0.5))
            return results

    monkeypatch.setattr(composer_module, "GridSearch", _FakeSearch)
    monkeypatch.setattr(composer_module.MuffakirComposer, "_index_documents", lambda self, combos: None)
    monkeypatch.setattr(composer_module.DatasetLoader, "load_or_generate", staticmethod(lambda **kwargs: [QAPair(question="What happened?", answer="A valid answer", context="Reference context")]))

    data_dir = tmp_path / "docs"
    data_dir.mkdir()

    composer = composer_module.MuffakirComposer(config={
        "data_dir": str(data_dir), "llm_provider": "openai", "llm_model": "gpt-4o-mini", "api_key": "test-key",
    })
    composer.fit(
        search_space={"k": [3, 5]},
        save_report=False,
        checkpoint_dir=str(tmp_path / "ckpt"),
        resume=False,
        enable_trace=True,
        trace_dir=str(tmp_path / "trace"),
    )

    manifest_path = tmp_path / "trace" / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["status"] == "completed"
    assert manifest["total_trials"] == 2
    assert manifest["config_hash"]

    with open(tmp_path / "trace" / "trials.jsonl", "r", encoding="utf-8") as f:
        trial_lines = [json.loads(line) for line in f if line.strip()]
    assert len(trial_lines) == 2

    with open(tmp_path / "trace" / "samples.jsonl", "r", encoding="utf-8") as f:
        sample_lines = [json.loads(line) for line in f if line.strip()]
    assert len(sample_lines) == 4  # 2 trials x 2 samples each


def test_fit_all_failed_trials_persists_failed_manifest_and_report(
    monkeypatch, tmp_path
):
    import json

    import Composer.composer as composer_module

    class _AllFailedSearch:
        def __init__(self, **kwargs):
            self.stopped_early = False
            self.stop_reason = None

        def run(self, **kwargs):
            return [
                TrialResult(
                    trial_id=0,
                    config={"k": 3},
                    error="simulated trial failure",
                    error_code="TEST_FAILURE",
                )
            ]

    monkeypatch.setattr(composer_module, "GridSearch", _AllFailedSearch)
    monkeypatch.setattr(
        composer_module.MuffakirComposer,
        "_index_documents",
        lambda self, combos: None,
    )
    monkeypatch.setattr(
        composer_module.DatasetLoader,
        "load_or_generate",
        staticmethod(lambda **kwargs: [QAPair(question="What happened?", answer="A valid answer", context="Reference context")]),
    )

    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    composer = composer_module.MuffakirComposer(
        config={
            "data_dir": str(data_dir),
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "api_key": "test-key",
        }
    )
    report_path = tmp_path / "report.json"
    report = composer.fit(
        search_space={"k": [3]},
        save_report=True,
        report_path=str(report_path),
        checkpoint_dir=str(tmp_path / "ckpt"),
        resume=False,
        enable_trace=True,
        trace_dir=str(tmp_path / "trace"),
    )

    assert report.successful_trials == 0
    assert report.best_trial is None
    manifest = json.loads(
        (tmp_path / "trace" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["summary"]["successful_trials"] == 0
    assert saved_report["trials"][0]["error"] == "simulated trial failure"


def test_fit_enable_trace_false_creates_no_trace_dir(monkeypatch, tmp_path):
    import Composer.composer as composer_module

    class _FakeSearch:
        def __init__(self, **kwargs):
            self.stopped_early = False
            self.stop_reason = None

        def run(self, **kwargs):
            assert kwargs["shared_state"]["trace_queue"] is None
            return []

    monkeypatch.setattr(composer_module, "GridSearch", _FakeSearch)
    monkeypatch.setattr(composer_module.MuffakirComposer, "_index_documents", lambda self, combos: None)
    monkeypatch.setattr(composer_module.DatasetLoader, "load_or_generate", staticmethod(lambda **kwargs: [QAPair(question="What happened?", answer="A valid answer", context="Reference context")]))

    data_dir = tmp_path / "docs"
    data_dir.mkdir()

    composer = composer_module.MuffakirComposer(config={
        "data_dir": str(data_dir), "llm_provider": "openai", "llm_model": "gpt-4o-mini", "api_key": "test-key",
    })
    composer.fit(
        save_report=False,
        checkpoint_dir=str(tmp_path / "ckpt"),
        resume=False,
        enable_trace=False,
    )
    assert not (tmp_path / "ckpt" / "trace").exists()


# ---------------------------------------------------------------------------
# ComposerReport.get_pareto_frontier()
# ---------------------------------------------------------------------------
def test_get_pareto_frontier_returns_non_dominated_trials():
    # A: best quality, slowest. B: balanced. C: cheapest/fastest but worst quality.
    # D: dominated by B on both axes (worse score AND worse latency) — must be excluded.
    trials = [
        TrialResult(trial_id=0, config={}, composite_score=0.95, latency_ms=500.0),
        TrialResult(trial_id=1, config={}, composite_score=0.85, latency_ms=200.0),
        TrialResult(trial_id=2, config={}, composite_score=0.70, latency_ms=100.0),
        TrialResult(trial_id=3, config={}, composite_score=0.80, latency_ms=300.0),
    ]
    report = ComposerReport(trials=trials)

    frontier = report.get_pareto_frontier()
    frontier_ids = {t.trial_id for t in frontier}

    assert frontier_ids == {0, 1, 2}


def test_get_pareto_frontier_excludes_failed_trials():
    trials = [
        TrialResult(trial_id=0, config={}, composite_score=0.9, latency_ms=100.0),
        TrialResult(trial_id=1, config={}, error="boom", error_code="X", error_type="Y"),
    ]
    report = ComposerReport(trials=trials)

    frontier = report.get_pareto_frontier()
    assert {t.trial_id for t in frontier} == {0}


# ---------------------------------------------------------------------------
# ComposerReport.get_failure_clusters()
# ---------------------------------------------------------------------------
def test_get_failure_clusters_groups_by_error_code():
    trials = [
        TrialResult(trial_id=0, config={}, composite_score=0.9),
        TrialResult(trial_id=1, config={}, error="a", error_code="PROVIDER_TIMEOUT"),
        TrialResult(trial_id=2, config={}, error="b", error_code="PROVIDER_TIMEOUT"),
    ]
    report = ComposerReport(trials=trials)
    assert report.get_failure_clusters() == {"PROVIDER_TIMEOUT": [1, 2]}


def test_get_failure_clusters_empty_when_all_succeed():
    report = ComposerReport(trials=[TrialResult(trial_id=0, config={}, composite_score=0.5)])
    assert report.get_failure_clusters() == {}


# ---------------------------------------------------------------------------
# chunking / embedding_model / vector_db_provider search-space dimensions
# ---------------------------------------------------------------------------
def test_trial_config_to_rag_config_maps_chunking_dimension():
    from Composer.config_space import trial_config_to_rag_config

    rag_config = trial_config_to_rag_config(
        {"chunking": {"method": "semantic", "size": 512, "overlap": 64}},
        {"chunking_method": "recursive", "chunk_size": 600, "chunk_overlap": 200},
    )
    assert rag_config["chunking_method"] == "semantic"
    assert rag_config["chunk_size"] == 512
    assert rag_config["chunk_overlap"] == 64


def test_trial_config_to_rag_config_maps_embedding_and_vector_db_dimensions():
    from Composer.config_space import trial_config_to_rag_config

    rag_config = trial_config_to_rag_config(
        {"embedding_model": "text-embedding-3-large", "vector_db_provider": "faiss"},
        {"embedding_model": "mohamed2811/Muffakir_Embedding", "vector_db_provider": "chroma"},
    )
    assert rag_config["embedding_model"] == "text-embedding-3-large"
    assert rag_config["vector_db_provider"] == "faiss"


def test_config_space_rejects_chunking_entry_missing_overlap():
    from Composer.config_space import ConfigSpace

    with pytest.raises(ValueError, match="overlap"):
        ConfigSpace({"chunking": [{"method": "recursive", "size": 512}]})


def test_config_space_accepts_chunking_embedding_vector_db_dimensions():
    from Composer.config_space import ConfigSpace

    space = ConfigSpace({
        "chunking": [{"method": "recursive", "size": 512, "overlap": 64}],
        "embedding_model": ["model-a", "model-b"],
        "vector_db_provider": ["chroma", "faiss"],
    })
    assert space.total_combinations == 4


# ---------------------------------------------------------------------------
# Deterministic index key + db_path/collection_name namespacing
# ---------------------------------------------------------------------------
def test_compute_index_key_is_deterministic():
    from Composer.index_key import compute_index_key

    rag_config = {
        "chunking_method": "recursive", "chunk_size": 600, "chunk_overlap": 200,
        "embedding_model": "model-a", "vector_db_provider": "chroma",
    }
    assert compute_index_key(rag_config) == compute_index_key(dict(rag_config))


def test_compute_index_key_differs_for_different_embedding_model():
    from Composer.index_key import compute_index_key

    base = {
        "chunking_method": "recursive", "chunk_size": 600, "chunk_overlap": 200,
        "vector_db_provider": "chroma",
    }
    key_a = compute_index_key({**base, "embedding_model": "model-a"})
    key_b = compute_index_key({**base, "embedding_model": "model-b"})
    assert key_a != key_b


def test_trial_config_to_rag_config_namespaces_db_path_by_index_key():
    from Composer.config_space import trial_config_to_rag_config
    from Composer.index_key import compute_index_key

    base_config = {
        "db_path": "./muffakir_db", "collection_name": "MuffakirComposer",
        "chunking_method": "recursive", "chunk_size": 600, "chunk_overlap": 200,
        "embedding_model": "model-a", "vector_db_provider": "chroma",
    }
    rag_config = trial_config_to_rag_config({"k": 5}, base_config)
    expected_key = compute_index_key(rag_config)

    assert rag_config["db_path"] == f"./muffakir_db/{expected_key}"
    assert rag_config["collection_name"] == f"MuffakirComposer_{expected_key}"


def test_trial_config_to_rag_config_same_index_key_when_index_dims_unchanged():
    """Two trials that don't vary any index-time dimension must resolve to the
    exact same db_path/collection_name — i.e. share one index, as today."""
    from Composer.config_space import trial_config_to_rag_config

    base_config = {
        "db_path": "./muffakir_db", "collection_name": "MuffakirComposer",
        "chunking_method": "recursive", "chunk_size": 600, "chunk_overlap": 200,
        "embedding_model": "model-a", "vector_db_provider": "chroma",
    }
    rag_config_a = trial_config_to_rag_config({"k": 3}, base_config)
    rag_config_b = trial_config_to_rag_config({"k": 10}, base_config)

    assert rag_config_a["db_path"] == rag_config_b["db_path"]
    assert rag_config_a["collection_name"] == rag_config_b["collection_name"]


# ---------------------------------------------------------------------------
# _index_documents(): one index per distinct index-time combination
# ---------------------------------------------------------------------------
def test_index_documents_builds_one_index_per_distinct_combo(tmp_path, monkeypatch):
    import importlib
    import Composer.composer as composer_module

    built_configs = []

    class _FakeDBManager:
        def add_documents(self, documents):
            pass

    def _fake_create_vector_db(provider, embedding_provider, **kwargs):
        built_configs.append((provider, kwargs.get("path"), kwargs.get("collection_name")))
        return _FakeDBManager()

    class _FakeChunking:
        def __init__(self, *a, **k):
            pass

    class _FakeProcessor:
        def __init__(self, *a, **k):
            pass

        def process_all(self, chunking_method=None, use_ocr=False):
            return [f"doc-for-{chunking_method}"]

    # Patch each fake at its defining submodule (not via a dotted string) —
    # every one of these packages re-exports its class/function at the
    # package level too (e.g. `from .EmbeddingProvider import EmbeddingProvider`
    # in Embedding/__init__.py), which shadows the submodule name with the
    # class/function itself and breaks a 3-segment dotted monkeypatch path.
    vectordb_pkg = importlib.import_module("VectorDB")
    monkeypatch.setattr(vectordb_pkg, "create_vector_db", _fake_create_vector_db)
    muffakir_chunking_mod = importlib.import_module("TextProcessor.MuffakirChunking")
    monkeypatch.setattr(muffakir_chunking_mod, "MuffakirChunking", _FakeChunking)
    chunking_processing_mod = importlib.import_module("TextProcessor.ChunkingAndProcessing")
    monkeypatch.setattr(chunking_processing_mod, "ChunkingAndProcessing", _FakeProcessor)
    embedding_provider_mod = importlib.import_module("Embedding.EmbeddingProvider")
    monkeypatch.setattr(embedding_provider_mod, "EmbeddingProvider", lambda *a, **k: object())

    data_dir = tmp_path / "docs"
    data_dir.mkdir()

    composer = composer_module.MuffakirComposer(config={
        "data_dir": str(data_dir),
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
        "db_path": str(tmp_path / "muffakir_db"),
        "collection_name": "MuffakirComposer",
    })

    combos = [
        (0, {"embedding_model": "model-a"}),
        (1, {"embedding_model": "model-a"}),
        (2, {"embedding_model": "model-b"}),
    ]
    composer._index_documents(combos)

    distinct_paths = {cfg[1] for cfg in built_configs}
    assert len(distinct_paths) == 2  # one per distinct embedding_model, not per trial


def test_index_documents_skips_already_built_index(tmp_path, monkeypatch):
    import importlib
    import Composer.composer as composer_module

    build_calls = []

    class _FakeDBManager:
        def add_documents(self, documents):
            pass

    def _fake_create_vector_db(provider, embedding_provider, **kwargs):
        build_calls.append(kwargs.get("path"))
        # Real vector DB providers create their persistence directory on
        # disk as a side effect of construction — replicate that here so the
        # resume-skip logic (Path(db_path).exists()) has something to detect.
        Path(kwargs["path"]).mkdir(parents=True, exist_ok=True)
        return _FakeDBManager()

    class _FakeChunking:
        def __init__(self, *a, **k):
            pass

    class _FakeProcessor:
        def __init__(self, *a, **k):
            pass

        def process_all(self, chunking_method=None, use_ocr=False):
            return ["doc"]

    vectordb_pkg = importlib.import_module("VectorDB")
    monkeypatch.setattr(vectordb_pkg, "create_vector_db", _fake_create_vector_db)
    muffakir_chunking_mod = importlib.import_module("TextProcessor.MuffakirChunking")
    monkeypatch.setattr(muffakir_chunking_mod, "MuffakirChunking", _FakeChunking)
    chunking_processing_mod = importlib.import_module("TextProcessor.ChunkingAndProcessing")
    monkeypatch.setattr(chunking_processing_mod, "ChunkingAndProcessing", _FakeProcessor)
    embedding_provider_mod = importlib.import_module("Embedding.EmbeddingProvider")
    monkeypatch.setattr(embedding_provider_mod, "EmbeddingProvider", lambda *a, **k: object())

    data_dir = tmp_path / "docs"
    data_dir.mkdir()

    composer = composer_module.MuffakirComposer(config={
        "data_dir": str(data_dir),
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
        "db_path": str(tmp_path / "muffakir_db"),
        "collection_name": "MuffakirComposer",
    })

    combos = [(0, {"embedding_model": "model-a"})]
    composer._index_documents(combos)
    assert len(build_calls) == 1

    # Re-running the same combination must not rebuild — the on-disk directory
    # from the first call already exists.
    composer._index_documents(combos)
    assert len(build_calls) == 1


def test_index_documents_passes_document_parser_and_use_ocr_when_configured(tmp_path, monkeypatch):
    import importlib
    import Composer.composer as composer_module

    captured_processor_kwargs = []
    captured_process_all_kwargs = []

    class _FakeDBManager:
        def add_documents(self, documents):
            pass

    def _fake_create_vector_db(provider, embedding_provider, **kwargs):
        return _FakeDBManager()

    class _FakeChunking:
        def __init__(self, *a, **k):
            pass

    class _FakeProcessor:
        def __init__(self, *a, **k):
            captured_processor_kwargs.append(k)

        def process_all(self, chunking_method=None, use_ocr=False):
            captured_process_all_kwargs.append({"chunking_method": chunking_method, "use_ocr": use_ocr})
            return ["doc"]

    class _FakeDocumentParser:
        pass

    def _fake_create_document_parser(provider, **kwargs):
        captured_processor_kwargs.append({"parser_provider": provider, "parser_kwargs": kwargs})
        return _FakeDocumentParser()

    vectordb_pkg = importlib.import_module("VectorDB")
    monkeypatch.setattr(vectordb_pkg, "create_vector_db", _fake_create_vector_db)
    muffakir_chunking_mod = importlib.import_module("TextProcessor.MuffakirChunking")
    monkeypatch.setattr(muffakir_chunking_mod, "MuffakirChunking", _FakeChunking)
    chunking_processing_mod = importlib.import_module("TextProcessor.ChunkingAndProcessing")
    monkeypatch.setattr(chunking_processing_mod, "ChunkingAndProcessing", _FakeProcessor)
    embedding_provider_mod = importlib.import_module("Embedding.EmbeddingProvider")
    monkeypatch.setattr(embedding_provider_mod, "EmbeddingProvider", lambda *a, **k: object())
    document_parser_pkg = importlib.import_module("DocumentParser")
    monkeypatch.setattr(document_parser_pkg, "create_document_parser", _fake_create_document_parser)

    data_dir = tmp_path / "docs"
    data_dir.mkdir()

    composer = composer_module.MuffakirComposer(config={
        "data_dir": str(data_dir),
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
        "db_path": str(tmp_path / "muffakir_db"),
        "collection_name": "MuffakirComposer",
        "document_parser": "docling",
        "document_parser_config": {"export_type": "markdown"},
        "use_ocr": True,
    })

    composer._index_documents([(0, {})])

    assert captured_process_all_kwargs[0]["use_ocr"] is True
    parser_calls = [k for k in captured_processor_kwargs if "parser_provider" in k]
    assert len(parser_calls) == 1
    assert parser_calls[0]["parser_provider"] == "docling"
    assert parser_calls[0]["parser_kwargs"] == {"export_type": "markdown"}
    processor_init_calls = [k for k in captured_processor_kwargs if "document_parser" in k]
    assert len(processor_init_calls) == 1
    assert isinstance(processor_init_calls[0]["document_parser"], _FakeDocumentParser)


def test_index_documents_no_parser_reproduces_prior_behavior(tmp_path, monkeypatch):
    """Regression guard: omitting document_parser/use_ocr from the base config
    keeps the exact pre-patch behavior — document_parser=None, use_ocr=False."""
    import importlib
    import Composer.composer as composer_module

    captured_processor_kwargs = []
    captured_process_all_kwargs = []

    class _FakeDBManager:
        def add_documents(self, documents):
            pass

    def _fake_create_vector_db(provider, embedding_provider, **kwargs):
        return _FakeDBManager()

    class _FakeChunking:
        def __init__(self, *a, **k):
            pass

    class _FakeProcessor:
        def __init__(self, *a, **k):
            captured_processor_kwargs.append(k)

        def process_all(self, chunking_method=None, use_ocr=False):
            captured_process_all_kwargs.append(use_ocr)
            return ["doc"]

    vectordb_pkg = importlib.import_module("VectorDB")
    monkeypatch.setattr(vectordb_pkg, "create_vector_db", _fake_create_vector_db)
    muffakir_chunking_mod = importlib.import_module("TextProcessor.MuffakirChunking")
    monkeypatch.setattr(muffakir_chunking_mod, "MuffakirChunking", _FakeChunking)
    chunking_processing_mod = importlib.import_module("TextProcessor.ChunkingAndProcessing")
    monkeypatch.setattr(chunking_processing_mod, "ChunkingAndProcessing", _FakeProcessor)
    embedding_provider_mod = importlib.import_module("Embedding.EmbeddingProvider")
    monkeypatch.setattr(embedding_provider_mod, "EmbeddingProvider", lambda *a, **k: object())

    data_dir = tmp_path / "docs"
    data_dir.mkdir()

    composer = composer_module.MuffakirComposer(config={
        "data_dir": str(data_dir),
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
        "db_path": str(tmp_path / "muffakir_db"),
        "collection_name": "MuffakirComposer",
    })

    composer._index_documents([(0, {})])

    assert captured_process_all_kwargs == [False]
    assert captured_processor_kwargs[0]["document_parser"] is None


# ---------------------------------------------------------------------------
# _get_shared_components() routed by resolved per-trial index path
# ---------------------------------------------------------------------------
def test_get_shared_components_cache_keyed_by_resolved_db_path(monkeypatch):
    import importlib
    import Composer.evaluation as evaluation_module

    build_calls = []

    def _fake_embedding_provider(*args, **kwargs):
        build_calls.append(kwargs.get("model_name"))
        return object()

    embedding_provider_mod = importlib.import_module("Embedding.EmbeddingProvider")
    monkeypatch.setattr(embedding_provider_mod, "EmbeddingProvider", _fake_embedding_provider)
    vectordb_pkg = importlib.import_module("VectorDB")
    monkeypatch.setattr(vectordb_pkg, "create_vector_db", lambda **kwargs: object())
    monkeypatch.setattr(evaluation_module, "_COMPONENT_CACHE", {})

    rag_config_a = {
        "db_path": "./muffakir_db/keyA", "collection_name": "MuffakirComposer_keyA",
        "embedding_model": "model-a", "vector_db_provider": "chroma",
    }
    rag_config_b = {
        "db_path": "./muffakir_db/keyB", "collection_name": "MuffakirComposer_keyB",
        "embedding_model": "model-b", "vector_db_provider": "chroma",
    }

    evaluation_module._get_shared_components(rag_config_a)
    evaluation_module._get_shared_components(rag_config_a)  # cached, no rebuild
    evaluation_module._get_shared_components(rag_config_b)  # different index, rebuild

    assert build_calls == ["model-a", "model-b"]


# ---------------------------------------------------------------------------
# LLM cost tracking (litellm price map + token usage)
# ---------------------------------------------------------------------------
class _FakeLLMProviderForCost:
    """Stands in for a real LLMProvider: exposes .provider/.model/.get_usage_totals()."""

    def __init__(self, provider, model, usage):
        self.provider = provider
        self.model = model
        self._usage = usage

    def get_usage_totals(self):
        return self._usage


def test_compute_trial_cost_sums_distinct_providers():
    from Composer.evaluation import _compute_trial_cost
    from Pricing.price_map import PriceMap

    class _Provider:
        value = "openai"

    gen_provider = _FakeLLMProviderForCost(
        _Provider(), "gpt-4o-mini", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    )
    judge_provider = _FakeLLMProviderForCost(
        _Provider(), "gpt-4o", {"prompt_tokens": 200, "completion_tokens": 20, "total_tokens": 220}
    )

    class _Rag:
        llm_provider = gen_provider
        query_transformer = None

    price_map = PriceMap()
    price_map.raw_map = {
        "gpt-4o-mini": {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002},
        "gpt-4o": {"input_cost_per_token": 0.01, "output_cost_per_token": 0.02},
    }

    token_usage, cost_usd = _compute_trial_cost(_Rag(), judge_provider, price_map)

    assert token_usage == {"prompt_tokens": 300, "completion_tokens": 70, "total_tokens": 370}
    expected = (100 * 0.001 + 50 * 0.002) + (200 * 0.01 + 20 * 0.02)
    assert cost_usd == pytest.approx(expected)


def test_compute_trial_cost_dedupes_shared_query_transform_provider():
    """When no per-task LLM override is set, rag.query_transformer.llm_provider is
    the SAME object as rag.llm_provider — its usage must not be counted twice."""
    from Composer.evaluation import _compute_trial_cost
    from Pricing.price_map import PriceMap

    class _Provider:
        value = "openai"

    shared_provider = _FakeLLMProviderForCost(
        _Provider(), "gpt-4o-mini", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    )

    class _QueryTransformer:
        llm_provider = shared_provider

    class _Rag:
        llm_provider = shared_provider
        query_transformer = _QueryTransformer()

    price_map = PriceMap()
    price_map.raw_map = {"gpt-4o-mini": {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}}

    token_usage, cost_usd = _compute_trial_cost(_Rag(), None, price_map)

    assert token_usage == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    assert cost_usd == pytest.approx(100 * 0.001 + 50 * 0.002)


def test_compute_trial_cost_distinct_query_transform_provider_counted_separately():
    """When a per-task override IS set, rag.query_transformer.llm_provider is a
    distinct object from rag.llm_provider — its usage must be included."""
    from Composer.evaluation import _compute_trial_cost
    from Pricing.price_map import PriceMap

    class _Provider:
        value = "together"

    gen_provider = _FakeLLMProviderForCost(
        _Provider(), "model-a", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    qt_provider = _FakeLLMProviderForCost(
        _Provider(), "model-b", {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
    )

    class _QueryTransformer:
        llm_provider = qt_provider

    class _Rag:
        llm_provider = gen_provider
        query_transformer = _QueryTransformer()

    price_map = PriceMap()
    token_usage, cost_usd = _compute_trial_cost(_Rag(), None, price_map)

    assert token_usage == {"prompt_tokens": 30, "completion_tokens": 13, "total_tokens": 43}
    assert cost_usd is None  # nothing in raw_map -> unpriceable, but usage still populated


def test_compute_trial_cost_all_unpriced_returns_none_cost_but_populated_usage():
    from Composer.evaluation import _compute_trial_cost
    from Pricing.price_map import PriceMap

    class _Provider:
        value = "custom"

    provider = _FakeLLMProviderForCost(
        _Provider(), "unknown-local-model", {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50}
    )

    class _Rag:
        llm_provider = provider
        query_transformer = None

    price_map = PriceMap()  # empty raw_map, no custom_pricing -> nothing priceable

    token_usage, cost_usd = _compute_trial_cost(_Rag(), None, price_map)

    assert token_usage == {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50}
    assert cost_usd is None


def test_push_trial_trace_counts_web_search_fallbacks():
    """web_search_fallback_count on the TrialRecord must equal the number of
    sample_traces with web_search_used=True — missing/False entries excluded."""
    import queue as queue_module

    from Composer.evaluation import _push_trial_trace

    trace_queue = queue_module.Queue()
    sample_traces = [
        {"web_search_used": True},
        {"web_search_used": False},
        {"web_search_used": True},
        {},  # no key at all -- must not count or crash
    ]

    _push_trial_trace(
        trace_queue, trial_id=1, rag_config={}, composite_score=0.5, metrics={},
        latency_ms=10.0, token_usage={}, cost_usd=None, sample_traces=sample_traces,
    )

    _, record = trace_queue.get_nowait()
    assert record["web_search_fallback_count"] == 2


def test_fit_custom_pricing_flows_into_trial_pricing_snapshot(monkeypatch, tmp_path):
    """fit(custom_pricing=...) must reach evaluate_trial() via shared_state's
    pricing_snapshot (end-to-end wiring, network mocked out)."""
    import Composer.composer as composer_module

    monkeypatch.setattr("requests.get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network in test")))

    captured_shared_state = {}

    class _FakeSearch:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            captured_shared_state.update(kwargs["shared_state"])
            return []

    monkeypatch.setattr(composer_module, "GridSearch", _FakeSearch)
    monkeypatch.setattr(composer_module.MuffakirComposer, "_index_documents", lambda self, combos: None)
    monkeypatch.setattr(composer_module.DatasetLoader, "load_or_generate", staticmethod(lambda **kwargs: [QAPair(question="What happened?", answer="A valid answer", context="Reference context")]))

    data_dir = tmp_path / "docs"
    data_dir.mkdir()

    composer = composer_module.MuffakirComposer(config={
        "data_dir": str(data_dir),
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
    })
    custom_pricing = {"gpt-4o-mini": {"input_cost_per_token": 0.5, "output_cost_per_token": 0.5}}
    composer.fit(
        save_report=False,
        checkpoint_dir=str(tmp_path / "ckpt"),
        resume=False,
        custom_pricing=custom_pricing,
    )

    assert captured_shared_state["pricing_snapshot"]["custom_pricing"] == custom_pricing
    assert captured_shared_state["pricing_snapshot"]["fetch_failed"] is True  # network mocked to fail


def test_fit_uses_independent_dataset_generation_llm_config(monkeypatch, tmp_path):
    """Dataset generation can use a model/endpoint distinct from Answer LLM."""
    import Composer.composer as composer_module

    captured = {}

    class _FakeSearch:
        def __init__(self, **kwargs):
            self.stopped_early = False
            self.stop_reason = None

        def run(self, **kwargs):
            return []

    def _capture_loader(**kwargs):
        captured.update(kwargs["llm_config"])
        return [QAPair(question="What happened?", answer="A valid answer", context="Reference context")]

    monkeypatch.setattr("requests.get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(composer_module, "GridSearch", _FakeSearch)
    monkeypatch.setattr(composer_module.MuffakirComposer, "_index_documents", lambda self, combos: None)
    monkeypatch.setattr(composer_module.DatasetLoader, "load_or_generate", staticmethod(_capture_loader))

    docs = tmp_path / "docs"
    docs.mkdir()
    composer = composer_module.MuffakirComposer(config={
        "data_dir": str(docs),
        "llm_provider": "openai",
        "llm_model": "answer-model",
        "api_key": "answer-key",
        "base_url": "https://answer.example/v1",
        "dataset_llm_provider": "together",
        "dataset_llm_model": "dataset-model",
        "dataset_api_key": "dataset-key",
        "dataset_base_url": "https://dataset.example/v1",
        "dataset_llm_parameters": {"temperature": 0.4, "max_tokens": 987},
    })

    composer.fit(
        search_space={"query_expansion": ["none"], "k": [3]},
        save_report=False,
        checkpoint_dir=str(tmp_path / "ckpt"),
        resume=False,
        enable_trace=False,
    )

    assert captured["llm_provider"] == "together"
    assert captured["llm_model"] == "dataset-model"
    assert captured["api_key"] == "dataset-key"
    assert captured["base_url"] == "https://dataset.example/v1"
    assert captured["llm_parameters"] == {"temperature": 0.4, "max_tokens": 987}


def test_fit_resume_reuses_stored_pricing_snapshot_without_refetching(monkeypatch, tmp_path):
    """A resumed run must reuse the checkpoint's stored pricing snapshot verbatim,
    never calling requests.get again."""
    import Composer.composer as composer_module
    from Composer.checkpoint import CheckpointManager

    ckpt_dir = tmp_path / "ckpt"
    stored_snapshot = {
        "raw_map": {"gpt-4o-mini": {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}},
        "custom_pricing": {},
        "fetch_failed": False,
        "fetched_at": "2026-08-30T00:00:00",
        "source_url": "https://example.invalid",
    }
    mgr = CheckpointManager(str(ckpt_dir))
    mgr.set_search_space({"k": [3]})
    mgr.set_pricing_snapshot(stored_snapshot)

    monkeypatch.setattr("requests.get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not re-fetch on resume")))

    captured_shared_state = {}

    class _FakeSearch:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            captured_shared_state.update(kwargs["shared_state"])
            return []

    monkeypatch.setattr(composer_module, "GridSearch", _FakeSearch)
    monkeypatch.setattr(composer_module.MuffakirComposer, "_index_documents", lambda self, combos: None)
    monkeypatch.setattr(composer_module.DatasetLoader, "load_or_generate", staticmethod(lambda **kwargs: [QAPair(question="What happened?", answer="A valid answer", context="Reference context")]))

    data_dir = tmp_path / "docs"
    data_dir.mkdir()

    composer = composer_module.MuffakirComposer(config={
        "data_dir": str(data_dir),
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
    })
    composer.fit(
        search_space={"k": [3]},
        save_report=False,
        checkpoint_dir=str(ckpt_dir),
        resume=True,
    )

    assert captured_shared_state["pricing_snapshot"]["raw_map"] == stored_snapshot["raw_map"]


def test_fit_whole_run_observer_records_early_sdk_failure(monkeypatch, tmp_path):
    import json

    from Composer.composer import MuffakirComposer

    composer = object.__new__(MuffakirComposer)
    composer.config = {"api_key": "must-not-leak"}

    def fail_before_dataset(self, **kwargs):
        raise RuntimeError("setup failed with must-not-leak")

    monkeypatch.setattr(MuffakirComposer, "_fit_impl", fail_before_dataset)
    trace_dir = tmp_path / "trace"

    with pytest.raises(RuntimeError, match="setup failed"):
        composer.fit(enable_trace=True, trace_dir=str(trace_dir))

    summary = json.loads((trace_dir / "error_rates.json").read_text(encoding="utf-8"))
    assert summary["attempts"] == 1
    assert summary["unrecovered_errors"] == 1
    assert summary["health"] == "ERRORS DETECTED"
    manifest = json.loads((trace_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    errors = (trace_dir / "errors.jsonl").read_text(encoding="utf-8")
    assert "must-not-leak" not in errors
    assert "***REDACTED***" in errors


if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))
