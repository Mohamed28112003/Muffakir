"""
Integration tests for ComposerUI FastAPI endpoints.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from ComposerUI.backend.app import app
from ComposerUI.backend import run_manager
from SyntheticData.models import QAPair


def _write_eval_csv(path: Path) -> Path:
    path.write_text(
        'question,answer,context\n"What?","Answer","Reference context long enough"\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(run_manager, "RUNS_ROOT", tmp_path)
    return TestClient(app)


def test_get_catalog(client):
    response = client.get("/api/catalog")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    assert "chunking_presets" in data
    provider_names = [p["name"] for p in data["providers"]]
    assert "openai" in provider_names


def test_storage_settings_get_and_update(client, tmp_path, monkeypatch):
    settings_path = tmp_path / "settings" / "settings.json"
    monkeypatch.setattr(run_manager, "RUNS_ROOT_SOURCE", "default")
    monkeypatch.setattr(run_manager, "RUNS_ROOT_LOCKED", False)
    monkeypatch.setattr(run_manager, "_ACTIVE_RUNS", {})
    monkeypatch.setattr(run_manager.storage, "settings_file_path", lambda: settings_path)
    selected = tmp_path / "selected-runs"

    initial = client.get("/api/settings/storage")
    assert initial.status_code == 200
    assert initial.json()["runs_root"] == str(tmp_path)

    response = client.put("/api/settings/storage", json={"runs_root": str(selected)})
    assert response.status_code == 200
    assert response.json()["runs_root"] == str(selected.resolve())
    assert response.json()["recent_roots"] == [str(tmp_path.resolve())]


def test_storage_settings_rejects_locked_session(client, tmp_path, monkeypatch):
    monkeypatch.setattr(run_manager, "RUNS_ROOT_LOCKED", True)
    response = client.put(
        "/api/settings/storage", json={"runs_root": str(tmp_path / "selected")}
    )
    assert response.status_code == 409
    assert "locked" in response.json()["detail"]


def test_storage_settings_rejects_file_without_switching(client, tmp_path, monkeypatch):
    current = run_manager.RUNS_ROOT
    monkeypatch.setattr(run_manager, "RUNS_ROOT_LOCKED", False)
    monkeypatch.setattr(run_manager, "_ACTIVE_RUNS", {})
    invalid = tmp_path / "workspace-file"
    invalid.write_text("not a directory", encoding="utf-8")

    response = client.put(
        "/api/settings/storage", json={"runs_root": str(invalid)}
    )

    assert response.status_code == 400
    assert "points to a file" in response.json()["detail"]
    assert run_manager.RUNS_ROOT == current


def test_validate_run_valid(client, tmp_path):
    data_dir = tmp_path / "docs"
    data_dir.mkdir()

    payload = {
        "run_name": "valid-run",
        "documents_path": str(data_dir),
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
        "search_space": {
            "query_expansion": ["none", "multi_query"],
            "k": [3, 5],
        },
    }
    response = client.post("/api/runs/validate", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["valid"] is True
    assert res["total_combinations"] == 4
    assert res["error"] is None


def test_validate_run_counts_reranker_models_only_for_local_methods(client, tmp_path):
    data_dir = tmp_path / "reranker-docs"
    data_dir.mkdir()
    payload = {
        "run_name": "reranker-model-search",
        "documents_path": str(data_dir),
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
        "search_space": {
            "reranking": ["cross_encoder", "pointwise", "none"],
            "reranking_model": [
                "BAAI/bge-reranker-base",
                "BAAI/bge-reranker-v2-m3",
            ],
        },
    }

    response = client.post("/api/runs/validate", json=payload)

    assert response.status_code == 200
    assert response.json()["valid"] is True
    assert response.json()["total_combinations"] == 5


def test_validate_run_invalid_dir(client):
    payload = {
        "run_name": "bad-dir-run",
        "documents_path": "non_existent_path_xyz_123",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
    }
    response = client.post("/api/runs/validate", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["valid"] is False
    assert res["total_combinations"] is None
    assert res["error"] is not None


def test_create_run_preflight_rejection_leaves_no_dirs(client, tmp_path):
    payload = {
        "run_name": "rejected-run",
        "documents_path": "non_existent_directory",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
    }
    response = client.post("/api/runs", json=payload)
    assert response.status_code == 400
    # No run directories should be created
    assert list(tmp_path.iterdir()) == []


def test_create_run_and_query_details(client, tmp_path, monkeypatch):
    data_dir = tmp_path / "docs"
    data_dir.mkdir()

    import Composer.composer as composer_module

    class _FakeSearch:
        def __init__(self, **kwargs):
            self.stopped_early = False
            self.stop_reason = None

        def run(self, **kwargs):
            return []

    monkeypatch.setattr(composer_module, "GridSearch", _FakeSearch)
    monkeypatch.setattr(
        composer_module.MuffakirComposer, "_index_documents", lambda self, combos: None
    )
    monkeypatch.setattr(
        composer_module.DatasetLoader, "load_or_generate", staticmethod(lambda **kwargs: [QAPair(question="What happened?", answer="A valid answer", context="Reference context")])
    )

    payload = {
        "run_name": "api-run",
        "documents_path": str(data_dir),
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
        "search_space": {"k": [3]},
    }

    create_res = client.post("/api/runs", json=payload)
    assert create_res.status_code == 200
    run_id = create_res.json()["run_id"]
    assert create_res.json()["status"] == "running"

    # Fetch list
    list_res = client.get("/api/runs")
    assert list_res.status_code == 200
    runs = list_res.json()["runs"]
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id

    # Fetch detail
    detail_res = client.get(f"/api/runs/{run_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["run_id"] == run_id
    assert detail["run_name"] == "api-run"

    # Fetch trials pagination
    trials_res = client.get(f"/api/runs/{run_id}/trials?after=0")
    assert trials_res.status_code == 200
    assert trials_res.json()["trials"] == []
    assert trials_res.json()["next_after"] == 0

    # 404 on nonexistent run
    bad_res = client.get("/api/runs/00000000-0000-0000-0000-000000000000")
    assert bad_res.status_code == 404


def test_error_observability_endpoint_live_and_legacy(client, tmp_path):
    legacy_run_id = "10000000-0000-0000-0000-000000000000"
    legacy_paths = run_manager.create_run_dirs(legacy_run_id)
    legacy = client.get(f"/api/runs/{legacy_run_id}/errors?after=0")
    assert legacy.status_code == 200
    assert legacy.json() == {
        "available": False,
        "summary": None,
        "errors": [],
        "next_after": 0,
    }

    run_id = "20000000-0000-0000-0000-000000000000"
    paths = run_manager.create_run_dirs(run_id)
    summary = {
        "attempts": 5,
        "errors": 1,
        "recovered_errors": 1,
        "unrecovered_errors": 0,
        "error_rate": 0.2,
        "health": "DEGRADED",
        "updated_at": "2026-09-11T10:00:00+00:00",
        "stages": [{
            "stage": "generation",
            "component": "llm",
            "attempts": 5,
            "errors": 1,
            "recovered_errors": 1,
            "unrecovered_errors": 0,
            "error_rate": 0.2,
            "last_error_at": "2026-09-11T10:00:00+00:00",
        }],
    }
    (paths["trace_dir"] / "error_rates.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    event = {
        "event_id": "event-1",
        "occurred_at": "2026-09-11T10:00:00+00:00",
        "stage": "generation",
        "component": "llm",
        "trial_id": 3,
        "sample_index": 1,
        "attempt": 2,
        "recovery": "retry",
        "error_code": "PROVIDER_TIMEOUT",
        "error_type": "TimeoutError",
        "retryable": True,
        "message": "timed out",
    }
    (paths["trace_dir"] / "errors.jsonl").write_text(
        json.dumps(event) + "\n", encoding="utf-8"
    )

    response = client.get(f"/api/runs/{run_id}/errors?after=0")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["summary"]["health"] == "DEGRADED"
    assert body["summary"]["stages"][0]["stage"] == "generation"
    assert body["errors"][0]["error_code"] == "PROVIDER_TIMEOUT"
    assert body["next_after"] == 1

    incremental = client.get(f"/api/runs/{run_id}/errors?after=1").json()
    assert incremental["errors"] == []
    assert incremental["next_after"] == 1


def test_create_run_with_existing_dataset_and_train_test_split(client, tmp_path, monkeypatch):
    data_dir = tmp_path / "docs"
    data_dir.mkdir()

    # A 10-row eval dataset fixture (question/answer/context required columns).
    eval_csv = tmp_path / "eval.csv"
    rows = ["question,answer,context"]
    for i in range(10):
        rows.append(
            f'"what is question number {i}?","this is answer number {i}",'
            f'"context text number {i} padded to be long enough"'
        )
    eval_csv.write_text("\n".join(rows), encoding="utf-8")

    import Composer.composer as composer_module

    captured_fit_kwargs = {}

    class _FakeReport:
        pass

    def _fake_fit(self, **kwargs):
        captured_fit_kwargs.update(kwargs)
        return _FakeReport()

    monkeypatch.setattr(composer_module.MuffakirComposer, "fit", _fake_fit)

    payload = {
        "run_name": "existing-dataset-run",
        "documents_path": str(data_dir),
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
        "search_space": {"k": [3]},
        "eval_dataset_mode": "existing",
        "eval_dataset_path": str(eval_csv),
        "train_test_split": {"enabled": True, "test_size": 0.3, "random_state": 1},
        "custom_pricing": [{
            "provider": "openai",
            "model": "gpt-4o-mini",
            "input_usd_per_million_tokens": 0.15,
            "output_usd_per_million_tokens": 0,
        }],
    }

    create_res = client.post("/api/runs", json=payload)
    assert create_res.status_code == 200
    run_id = create_res.json()["run_id"]

    run_manager._ACTIVE_RUNS[run_id]["thread"].join(timeout=10)

    assert "eval_dataset" in captured_fit_kwargs
    eval_dataset = captured_fit_kwargs["eval_dataset"]
    assert eval_dataset is not None
    # 30% test split of 10 rows -> 3 items, not the full 10.
    assert len(eval_dataset) == 3
    assert captured_fit_kwargs["custom_pricing"] == {
        "openai/gpt-4o-mini": {
            "input_cost_per_token": pytest.approx(0.15 / 1_000_000),
            "output_cost_per_token": 0.0,
        }
    }


def test_preflight_threads_device_and_judge_llm_into_composer_config(client, tmp_path, monkeypatch):
    data_dir = tmp_path / "docs"
    data_dir.mkdir()

    import Composer.composer as composer_module

    captured_configs = []
    original_init = composer_module.MuffakirComposer.__init__

    def _spy_init(self, config=None, *args, **kwargs):
        captured_configs.append(dict(config or {}))
        return original_init(self, config, *args, **kwargs)

    monkeypatch.setattr(composer_module.MuffakirComposer, "__init__", _spy_init)

    payload = {
        "run_name": "device-judge-run",
        "documents_path": str(data_dir),
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
        "search_space": {"k": [3]},
        "device": "cpu",
        "judge_llm_provider": "anthropic",
        "judge_llm_model": "claude-3-haiku",
        "judge_api_key": "judge-key",
        "judge_base_url": "https://judge.example/v1",
    }

    res = client.post("/api/runs/validate", json=payload)
    assert res.status_code == 200
    assert res.json()["valid"] is True

    assert captured_configs, "MuffakirComposer.__init__ was never called"
    cfg = captured_configs[0]
    assert cfg["device"] == "cpu"
    assert cfg["judge_llm_provider"] == "anthropic"
    assert cfg["judge_llm_model"] == "claude-3-haiku"
    assert cfg["judge_api_key"] == "judge-key"
    assert cfg["judge_base_url"] == "https://judge.example/v1"


def test_preflight_omits_device_and_judge_llm_when_not_provided(client, tmp_path, monkeypatch):
    data_dir = tmp_path / "docs"
    data_dir.mkdir()

    import Composer.composer as composer_module

    captured_configs = []
    original_init = composer_module.MuffakirComposer.__init__

    def _spy_init(self, config=None, *args, **kwargs):
        captured_configs.append(dict(config or {}))
        return original_init(self, config, *args, **kwargs)

    monkeypatch.setattr(composer_module.MuffakirComposer, "__init__", _spy_init)

    payload = {
        "run_name": "no-device-judge-run",
        "documents_path": str(data_dir),
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
        "search_space": {"k": [3]},
    }

    res = client.post("/api/runs/validate", json=payload)
    assert res.status_code == 200
    assert res.json()["valid"] is True

    cfg = captured_configs[0]
    assert "device" not in cfg
    assert "judge_llm_provider" not in cfg
    assert "judge_llm_model" not in cfg
    assert "judge_api_key" not in cfg
    assert "judge_base_url" not in cfg


def test_preflight_check_allows_retrieval_only_without_llm_provider(tmp_path):
    from ComposerUI.backend.app import _preflight_check
    from ComposerUI.backend.schemas import CreateRunRequest

    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    eval_csv = _write_eval_csv(tmp_path / "eval.csv")

    req = CreateRunRequest(
        run_name="r1",
        documents_path=str(data_dir),
        pipeline_mode="retrieval_only",
        eval_dataset_mode="existing",
        eval_dataset_path=str(eval_csv),
        # Explicit retrieval-only metrics: leaving metrics unset falls back to
        # fit()'s ["recall", "faithfulness", "answer_correctness"] default,
        # which retrieval_only mode legitimately rejects (see
        # test_preflight_check_rejects_generation_metrics_with_retrieval_only).
        metrics=["recall"],
        search_space={"query_expansion": ["none"], "k": [3]},
    )
    # Must not raise for missing llm_provider/llm_model in this mode.
    _preflight_check(req)


def test_preflight_check_rejects_generation_metrics_with_retrieval_only(tmp_path):
    from ComposerUI.backend.app import _preflight_check
    from ComposerUI.backend.schemas import CreateRunRequest

    data_dir = tmp_path / "docs"
    data_dir.mkdir()

    req = CreateRunRequest(
        run_name="r1",
        documents_path=str(data_dir),
        pipeline_mode="retrieval_only",
        metrics=["faithfulness"],
    )
    with pytest.raises(ValueError, match="faithfulness"):
        _preflight_check(req)


def test_preflight_check_allows_retrieval_metrics_with_retrieval_only(tmp_path):
    from ComposerUI.backend.app import _preflight_check
    from ComposerUI.backend.schemas import CreateRunRequest

    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    eval_csv = _write_eval_csv(tmp_path / "eval.csv")

    req = CreateRunRequest(
        run_name="r1",
        documents_path=str(data_dir),
        pipeline_mode="retrieval_only",
        metrics=["recall"],
        eval_dataset_mode="existing",
        eval_dataset_path=str(eval_csv),
        search_space={"query_expansion": ["none"], "k": [3]},
    )
    _preflight_check(req)  # must not raise


def test_preflight_retrieval_only_requires_a_real_corpus(tmp_path):
    from ComposerUI.backend.app import _preflight_check
    from ComposerUI.backend.schemas import CreateRunRequest

    eval_csv = _write_eval_csv(tmp_path / "eval.csv")
    req = CreateRunRequest(
        run_name="retrieval-without-corpus",
        pipeline_mode="retrieval_only",
        metrics=["recall"],
        eval_dataset_mode="existing",
        eval_dataset_path=str(eval_csv),
        search_space={"query_expansion": ["none"], "k": [3]},
    )

    with pytest.raises(ValueError, match="documents_path is required"):
        _preflight_check(req)


def test_preflight_threads_query_transform_api_key_into_composer_config(tmp_path, monkeypatch):
    from ComposerUI.backend.app import _preflight_check
    from ComposerUI.backend.schemas import CreateRunRequest
    import Composer.composer as composer_module

    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    eval_csv = _write_eval_csv(tmp_path / "eval.csv")

    captured_configs = []
    original_init = composer_module.MuffakirComposer.__init__

    def _spy_init(self, config=None, *args, **kwargs):
        captured_configs.append(dict(config or {}))
        return original_init(self, config, *args, **kwargs)

    monkeypatch.setattr(composer_module.MuffakirComposer, "__init__", _spy_init)

    req = CreateRunRequest(
        run_name="qt-key-run",
        documents_path=str(data_dir),
        pipeline_mode="retrieval_only",
        metrics=["recall"],
        eval_dataset_mode="existing",
        eval_dataset_path=str(eval_csv),
        query_transform_llm_provider="openai",
        query_transform_llm_model="gpt-4o-mini",
        query_transform_api_key="qt-secret",
    )
    _preflight_check(req)

    assert captured_configs, "MuffakirComposer.__init__ was never called"
    cfg = captured_configs[0]
    assert cfg["query_transform_llm_provider"] == "openai"
    assert cfg["query_transform_api_key"] == "qt-secret"


def test_preflight_retrieval_only_query_transform_requires_its_own_llm(tmp_path):
    from ComposerUI.backend.app import _preflight_check
    from ComposerUI.backend.schemas import CreateRunRequest

    docs = tmp_path / "docs"
    docs.mkdir()
    eval_csv = _write_eval_csv(tmp_path / "eval.csv")
    req = CreateRunRequest(
        run_name="retrieval-query-transform",
        documents_path=str(docs),
        pipeline_mode="retrieval_only",
        metrics=["recall"],
        eval_dataset_mode="existing",
        eval_dataset_path=str(eval_csv),
        search_space={"query_expansion": ["multi_query"], "k": [3]},
    )

    with pytest.raises(ValueError, match="Retrieval-Only query transformation"):
        _preflight_check(req)


def test_preflight_rejects_unknown_reranker(tmp_path):
    from ComposerUI.backend.app import _preflight_check
    from ComposerUI.backend.schemas import CreateRunRequest

    docs = tmp_path / "docs"
    docs.mkdir()
    eval_csv = _write_eval_csv(tmp_path / "eval.csv")
    request = CreateRunRequest(
        run_name="unknown-reranker",
        documents_path=str(docs),
        pipeline_mode="retrieval_only",
        metrics=["recall"],
        eval_dataset_mode="existing",
        eval_dataset_path=str(eval_csv),
        search_space={"query_expansion": ["none"], "reranking": ["not-real"]},
    )

    with pytest.raises(ValueError, match="Unknown reranking method"):
        _preflight_check(request)


def test_preflight_custom_reranker_requires_endpoint(tmp_path):
    from ComposerUI.backend.app import _preflight_check
    from ComposerUI.backend.schemas import CreateRunRequest

    docs = tmp_path / "docs"
    docs.mkdir()
    eval_csv = _write_eval_csv(tmp_path / "eval.csv")
    request = CreateRunRequest(
        run_name="custom-reranker",
        documents_path=str(docs),
        pipeline_mode="retrieval_only",
        metrics=["recall"],
        eval_dataset_mode="existing",
        eval_dataset_path=str(eval_csv),
        search_space={"query_expansion": ["none"], "reranking": ["custom"]},
    )

    with pytest.raises(ValueError, match="reranker_base_url"):
        _preflight_check(request)


def test_preflight_retrieval_only_llm_reranker_requires_reusable_or_dedicated_llm(tmp_path):
    from ComposerUI.backend.app import _preflight_check
    from ComposerUI.backend.schemas import CreateRunRequest

    docs = tmp_path / "docs"
    docs.mkdir()
    eval_csv = _write_eval_csv(tmp_path / "eval.csv")
    request = CreateRunRequest(
        run_name="llm-reranker",
        documents_path=str(docs),
        pipeline_mode="retrieval_only",
        metrics=["recall"],
        eval_dataset_mode="existing",
        eval_dataset_path=str(eval_csv),
        search_space={"query_expansion": ["none"], "reranking": ["llm"]},
    )

    with pytest.raises(ValueError, match="dedicated reranker LLM"):
        _preflight_check(request)


def test_preflight_threads_dedicated_reranker_llm_and_custom_endpoint(tmp_path, monkeypatch):
    from ComposerUI.backend.app import _preflight_check
    from ComposerUI.backend.schemas import CreateRunRequest
    import Composer.composer as composer_module

    docs = tmp_path / "docs"
    docs.mkdir()
    eval_csv = _write_eval_csv(tmp_path / "eval.csv")
    captured = []
    original_init = composer_module.MuffakirComposer.__init__

    def _spy_init(self, config=None, *args, **kwargs):
        captured.append(dict(config or {}))
        return original_init(self, config, *args, **kwargs)

    monkeypatch.setattr(composer_module.MuffakirComposer, "__init__", _spy_init)
    request = CreateRunRequest(
        run_name="configured-rerankers",
        documents_path=str(docs),
        pipeline_mode="retrieval_only",
        metrics=["recall"],
        eval_dataset_mode="existing",
        eval_dataset_path=str(eval_csv),
        search_space={
            "query_expansion": ["none"],
            "reranking": ["llm", "custom"],
        },
        reranker_llm_provider="openai",
        reranker_llm_model="gpt-4o-mini",
        reranker_llm_api_key="llm-secret",
        reranker_base_url="https://reranker.example/v1/rerank",
        reranker_api_key="remote-secret",
        reranker_model="rerank-v2",
        reranker_timeout_seconds=15,
    )

    _preflight_check(request)

    assert captured
    config = captured[0]
    assert config["reranker_llm_provider"] == "openai"
    assert config["reranker_llm_api_key"] == "llm-secret"
    assert config["reranker_base_url"] == "https://reranker.example/v1/rerank"
    assert config["reranker_api_key"] == "remote-secret"
    assert config["reranker_timeout_seconds"] == 15


def test_preflight_retrieval_only_auto_uses_dataset_generation_llm(tmp_path, monkeypatch):
    from ComposerUI.backend.app import _preflight_check
    from ComposerUI.backend.schemas import CreateRunRequest
    import Composer.composer as composer_module

    docs = tmp_path / "docs"
    docs.mkdir()
    captured = []
    original_init = composer_module.MuffakirComposer.__init__

    def _spy_init(self, config=None, *args, **kwargs):
        captured.append(dict(config or {}))
        return original_init(self, config, *args, **kwargs)

    monkeypatch.setattr(composer_module.MuffakirComposer, "__init__", _spy_init)
    req = CreateRunRequest(
        run_name="retrieval-auto-dataset",
        documents_path=str(docs),
        pipeline_mode="retrieval_only",
        metrics=["recall"],
        search_space={"query_expansion": ["none"], "k": [3]},
        dataset_llm_provider="openai",
        dataset_llm_model="gpt-4o-mini",
        dataset_api_key="dataset-secret",
    )

    _preflight_check(req)
    assert captured[0]["dataset_llm_provider"] == "openai"
    assert captured[0]["dataset_llm_model"] == "gpt-4o-mini"
    assert captured[0]["dataset_api_key"] == "dataset-secret"


def test_query_transform_api_key_never_written_to_redacted_config(tmp_path):
    from ComposerUI.backend.run_manager import write_redacted_config
    from ComposerUI.backend.schemas import CreateRunRequest
    import json as _json

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    req = CreateRunRequest(
        run_name="qt-key-run",
        documents_path=str(tmp_path),
        pipeline_mode="retrieval_only",
        metrics=["recall"],
        query_transform_llm_provider="openai",
        query_transform_llm_model="gpt-4o-mini",
        query_transform_api_key="qt-secret",
        api_key="main-secret",
    )
    write_redacted_config(run_dir, req, run_id="run-1", created_at="2026-09-06T00:00:00")

    raw = (run_dir / "config.json").read_text(encoding="utf-8")
    saved = _json.loads(raw)
    assert "query_transform_api_key" not in saved
    # Same exclusion rule the pre-existing secret fields follow.
    assert "api_key" not in saved
    assert "search_api_key" not in saved
    assert "qt-secret" not in raw
    assert saved["query_transform_llm_provider"] == "openai"


def test_preflight_check_requires_llm_provider_for_full_rag():
    from ComposerUI.backend.app import _preflight_check
    from ComposerUI.backend.schemas import CreateRunRequest

    req = CreateRunRequest(run_name="r1", documents_path="./data")
    with pytest.raises(ValueError, match="llm_provider"):
        _preflight_check(req)


def test_preflight_check_rejects_retrieval_only_with_web_search_only():
    from ComposerUI.backend.app import _preflight_check
    from ComposerUI.backend.schemas import CreateRunRequest

    req = CreateRunRequest(
        run_name="r1",
        pipeline_mode="retrieval_only",
        retrieval_source="web_search_only",
        search_provider="tavily",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _preflight_check(req)


def test_validate_run_rejects_malformed_eval_dataset_content(client, tmp_path):
    data_dir = tmp_path / "docs"
    data_dir.mkdir()

    # Missing the required "context" column.
    bad_csv = tmp_path / "bad_eval.csv"
    bad_csv.write_text("question,answer\n\"q1?\",\"a1\"\n", encoding="utf-8")

    payload = {
        "run_name": "bad-eval-dataset-run",
        "documents_path": str(data_dir),
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
        "search_space": {"k": [3]},
        "eval_dataset_mode": "existing",
        "eval_dataset_path": str(bad_csv),
    }

    res = client.post("/api/runs/validate", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["valid"] is False
    assert "context" in body["error"]


def test_create_run_rejects_malformed_eval_dataset_content(client, tmp_path):
    data_dir = tmp_path / "docs"
    data_dir.mkdir()

    bad_csv = tmp_path / "bad_eval.csv"
    bad_csv.write_text("question,answer\n\"q1?\",\"a1\"\n", encoding="utf-8")

    payload = {
        "run_name": "bad-eval-dataset-run",
        "documents_path": str(data_dir),
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
        "search_space": {"k": [3]},
        "eval_dataset_mode": "existing",
        "eval_dataset_path": str(bad_csv),
    }

    entries_before = set(run_manager.RUNS_ROOT.iterdir())

    res = client.post("/api/runs", json=payload)
    assert res.status_code == 400
    # No run directory should be created — same fail-fast guarantee as a bad documents_path.
    assert set(run_manager.RUNS_ROOT.iterdir()) == entries_before


def test_validate_run_web_search_only_requires_search_provider(client, tmp_path):
    payload = {
        "run_name": "web-search-only-run",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
        "search_space": {"k": [3]},
        "retrieval_source": "web_search_only",
        "eval_dataset_mode": "existing",
        "eval_dataset_path": str(tmp_path / "does_not_matter.csv"),
    }
    res = client.post("/api/runs/validate", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["valid"] is False
    assert "search_provider" in body["error"]


def test_preflight_auto_dataset_requires_documents_and_llm(tmp_path):
    from ComposerUI.backend.app import _preflight_check
    from ComposerUI.backend.schemas import CreateRunRequest

    missing_docs = CreateRunRequest(
        run_name="auto-no-docs",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        api_key="test-key",
    )
    with pytest.raises(ValueError, match="documents_path"):
        _preflight_check(missing_docs)

    docs = tmp_path / "docs"
    docs.mkdir()
    missing_llm = CreateRunRequest(
        run_name="auto-no-llm",
        documents_path=str(docs),
        pipeline_mode="retrieval_only",
        metrics=["recall"],
    )
    with pytest.raises(ValueError, match="eval_dataset_mode='auto'.*llm_provider"):
        _preflight_check(missing_llm)


def test_preflight_web_search_only_requires_existing_evaluation_dataset():
    from ComposerUI.backend.app import _preflight_check
    from ComposerUI.backend.schemas import CreateRunRequest

    req = CreateRunRequest(
        run_name="web-auto-dataset",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        api_key="test-key",
        retrieval_source="web_search_only",
        search_provider="tavily",
        metrics=["faithfulness"],
    )
    with pytest.raises(ValueError, match="requires an existing evaluation dataset"):
        _preflight_check(req)


def test_preflight_adaptive_web_search_requires_provider(tmp_path):
    from ComposerUI.backend.app import _preflight_check
    from ComposerUI.backend.schemas import CreateRunRequest

    docs = tmp_path / "docs"
    docs.mkdir()
    req = CreateRunRequest(
        run_name="adaptive-without-provider",
        documents_path=str(docs),
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        api_key="test-key",
        adaptive_web_search=True,
        metrics=["recall"],
    )
    with pytest.raises(ValueError, match="adaptive_web_search=True requires search_provider"):
        _preflight_check(req)


def test_preflight_web_search_only_rejects_retrieval_metrics(tmp_path):
    from ComposerUI.backend.app import _preflight_check
    from ComposerUI.backend.schemas import CreateRunRequest

    eval_csv = _write_eval_csv(tmp_path / "eval.csv")
    req = CreateRunRequest(
        run_name="web-with-retrieval-metric",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        api_key="test-key",
        retrieval_source="web_search_only",
        search_provider="tavily",
        search_api_key="search-key",
        eval_dataset_mode="existing",
        eval_dataset_path=str(eval_csv),
        metrics=["recall", "faithfulness"],
        search_space={"k": [3]},
    )
    with pytest.raises(ValueError, match="cannot score retrieval metrics.*recall"):
        _preflight_check(req)


def test_create_run_web_search_only_launches_without_documents_path(client, tmp_path, monkeypatch):
    """retrieval_source='web_search_only' must not require documents_path at
    all -- no local corpus is ever indexed for this mode."""
    eval_csv = tmp_path / "eval.csv"
    rows = ["question,answer,context"]
    for i in range(3):
        rows.append(
            f'"what is question number {i}?","this is answer number {i}",'
            f'"context text number {i} padded to be long enough"'
        )
    eval_csv.write_text("\n".join(rows), encoding="utf-8")

    import Composer.composer as composer_module

    captured_fit_kwargs = {}

    class _FakeReport:
        pass

    def _fake_fit(self, **kwargs):
        captured_fit_kwargs.update(kwargs)
        return _FakeReport()

    monkeypatch.setattr(composer_module.MuffakirComposer, "fit", _fake_fit)
    monkeypatch.setattr(
        "Muffakir.dependency_validation.validate_composer_dependencies",
        lambda *_args, **_kwargs: None,
    )

    payload = {
        "run_name": "web-search-only-run",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "api_key": "test-key",
        "search_space": {"k": [3]},
        "retrieval_source": "web_search_only",
        "search_provider": "tavily",
        "search_provider_config": {"max_results": 3},
        "search_api_key": "tavily-key",
        "metrics": ["faithfulness", "answer_correctness", "llm_judge_rating"],
        "eval_dataset_mode": "existing",
        "eval_dataset_path": str(eval_csv),
    }

    create_res = client.post("/api/runs", json=payload)
    assert create_res.status_code == 200
    run_id = create_res.json()["run_id"]

    run_manager._ACTIVE_RUNS[run_id]["thread"].join(timeout=10)

    assert "eval_dataset" in captured_fit_kwargs
    assert len(captured_fit_kwargs["eval_dataset"]) == 3


def test_static_frontend_routes(client):
    # Root index.html
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "Muffakir Composer // New Search Run" in res_root.text
    assert "6. Prompt Manager" in res_root.text
    assert "7. Review &amp; Launch" in res_root.text
    assert "Models &amp; Providers" in res_root.text
    assert 'id="custom-pricing-card"' in res_root.text
    assert "Custom Pricing" in res_root.text
    assert 'id="reranker-llm-card"' in res_root.text
    assert 'id="custom-reranker-fields"' in res_root.text
    assert 'id="reranker-base-url-input"' in res_root.text
    assert 'id="btn-add-reranking-model"' in res_root.text
    assert 'id="reranking-model-suggestions"' in res_root.text
    assert "Dataset Contexts" not in res_root.text
    assert 'id="btn-launch-run"' in res_root.text
    assert res_root.text.count('id="btn-launch-run"') == 1
    assert "Generate a synthetic dataset now" not in res_root.text
    assert '<link href="/static/favicon.png" rel="icon" type="image/png"/>' in res_root.text

    # runs.html
    res_runs = client.get("/runs.html")
    assert res_runs.status_code == 200
    assert "Pipeline Search Runs" in res_runs.text
    assert "/static/favicon.png" in res_runs.text
    assert 'id="chart-cross-run"' in res_runs.text
    assert "cdnjs.cloudflare.com/ajax/libs/Chart.js" not in res_runs.text

    # run_detail.html
    res_detail = client.get("/run_detail.html")
    assert res_detail.status_code == 200
    assert "Run Telemetry" in res_detail.text
    assert "/static/favicon.png" in res_detail.text
    assert 'id="run-completion-popup"' in res_detail.text
    assert 'id="completion-popup-view-result"' in res_detail.text
    assert 'id="chart-cost"' in res_detail.text
    assert 'id="chart-stage-timings"' in res_detail.text
    assert "Hover points · click for details" in res_detail.text
    assert "cdnjs.cloudflare.com/ajax/libs/Chart.js" not in res_detail.text

    # Branded favicon asset
    res_favicon = client.get("/static/favicon.png")
    assert res_favicon.status_code == 200
    assert res_favicon.headers["content-type"] == "image/png"
    assert "Sample Progress" in res_detail.text
    assert "Elapsed" in res_detail.text
    assert "Error Rate Observability" in res_detail.text
    assert 'id="answer-refusal-count-value"' in res_detail.text
    assert 'id="answer-refusal-rate-value"' in res_detail.text
    assert 'id="error-stage-rates"' in res_detail.text
    assert 'id="error-events-container"' in res_detail.text
    for sort_key in (
        "trial_id",
        "status",
        "composite_score",
        "mean_pipeline_latency_ms",
        "cost_usd",
        "web_search_fallback_count",
        "resolved_rag_config",
    ):
        assert f'data-trial-sort="{sort_key}"' in res_detail.text

    # static js
    res_js = client.get("/static/js/common.js")
    assert res_js.status_code == 200
    assert "apiGet" in res_js.text

    res_new_run_js = client.get("/static/js/new_run.js")
    assert res_new_run_js.status_code == 200
    assert "populateRerankerChips" in res_new_run_js.text
    assert "reranker_llm_provider" in res_new_run_js.text
    assert "reranker_base_url" in res_new_run_js.text
    assert "reranking_model" in res_new_run_js.text
    assert "populateRerankerModelSuggestions" in res_new_run_js.text

    res_detail_js = client.get("/static/js/run_detail.js")
    assert res_detail_js.status_code == 200
    assert "showRunCompletionPopup" in res_detail_js.text
    assert "active_trials" in res_detail_js.text
    assert "data-trial-elapsed" in res_detail_js.text
    assert "pollErrors" in res_detail_js.text
    assert "/errors?after=" in res_detail_js.text
    assert "Transformed Query" in res_detail_js.text
    assert "answer_refusals" in res_detail_js.text
    assert "renderTelemetryChart" in res_detail_js.text
    assert "createElementNS" in res_detail_js.text
    assert "Chart data is available, but it could not be displayed." in res_detail_js.text
    assert "addTimingPointInteraction" in res_detail_js.text
    assert 'openTrialInspector(point.trialId, "overview")' in res_detail_js.text
    assert "showTimingTooltip" in res_detail_js.text
    assert "hit.addEventListener(\"keydown\"" in res_detail_js.text
    assert "hit.click()" not in res_detail_js.text
    assert "window.Chart" not in res_detail_js.text

    res_runs_js = client.get("/static/js/runs_list.js")
    assert res_runs_js.status_code == 200
    assert "renderComparisonChart" in res_runs_js.text
    assert "createElementNS" in res_runs_js.text
    assert "window.Chart" not in res_runs_js.text


def test_concurrency_stream_trials(client, tmp_path):
    run_id = "11112222-3333-4444-5555-666677778888"
    paths = run_manager.create_run_dirs(run_id)

    trials_file = paths["trace_dir"] / "trials.jsonl"
    with open(trials_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"trial_id": 1, "composite_score": 0.88, "status": "success"}) + "\n")
        f.flush()

    res1 = client.get(f"/api/runs/{run_id}/trials?after=0")
    assert res1.status_code == 200
    assert len(res1.json()["trials"]) == 1
    assert res1.json()["next_after"] == 1

    # Append second line
    with open(trials_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"trial_id": 2, "composite_score": 0.95, "status": "success"}) + "\n")
        f.flush()

    res2 = client.get(f"/api/runs/{run_id}/trials?after=1")
    assert res2.status_code == 200
    assert len(res2.json()["trials"]) == 1
    assert res2.json()["trials"][0]["trial_id"] == 2
    assert res2.json()["next_after"] == 2


def test_trials_endpoint_returns_active_snapshot_without_counting_it_completed(client):
    run_id = "12112222-3333-4444-5555-666677778888"
    paths = run_manager.create_run_dirs(run_id)
    (paths["trace_dir"] / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "status": "running", "total_trials": 2}),
        encoding="utf-8",
    )
    (paths["trace_dir"] / "trials.jsonl").write_text(
        json.dumps({
            "trial_id": 1,
            "status": "success",
            "completed_samples": 5,
            "total_samples": 5,
        }) + "\n",
        encoding="utf-8",
    )
    (paths["trace_dir"] / "active_trials.json").write_text(
        json.dumps({"trials": [{
            "trial_id": 2,
            "status": "running",
            "resolved_rag_config": {"k": 10},
            "started_at": "2026-09-11T10:00:00+00:00",
            "completed_samples": 3,
            "total_samples": 5,
        }]}),
        encoding="utf-8",
    )

    response = client.get(f"/api/runs/{run_id}/trials?after=0")
    assert response.status_code == 200
    payload = response.json()
    assert [trial["trial_id"] for trial in payload["trials"]] == [1]
    assert payload["active_trials"] == [{
        "trial_id": 2,
        "status": "running",
        "resolved_rag_config": {"k": 10},
        "started_at": "2026-09-11T10:00:00+00:00",
        "completed_samples": 3,
        "total_samples": 5,
    }]

    detail = client.get(f"/api/runs/{run_id}")
    assert detail.status_code == 200
    assert detail.json()["completed_trials"] == 1


def test_trials_endpoint_ignores_stale_active_snapshot_for_terminal_run(client):
    run_id = "13112222-3333-4444-5555-666677778888"
    paths = run_manager.create_run_dirs(run_id)
    (paths["trace_dir"] / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "status": "completed", "total_trials": 1}),
        encoding="utf-8",
    )
    (paths["trace_dir"] / "active_trials.json").write_text(
        json.dumps({"trials": [{
            "trial_id": 0,
            "started_at": "2026-09-11T10:00:00+00:00",
        }]}),
        encoding="utf-8",
    )

    response = client.get(f"/api/runs/{run_id}/trials?after=0")
    assert response.status_code == 200
    assert response.json()["active_trials"] == []


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert data["runs_root_writable"] is True
    assert isinstance(data["active_runs"], int)


def test_trials_endpoint_includes_stage_timings(client):
    run_id = "22223333-4444-5555-6666-777788889999"
    paths = run_manager.create_run_dirs(run_id)

    trials_file = paths["trace_dir"] / "trials.jsonl"
    trial_data = {
        "trial_id": 1,
        "composite_score": 0.85,
        "latency_ms": 120.0,
        "status": "success",
        "mean_query_transform_ms": 12.5,
        "mean_query_embedding_ms": 15.0,
        "mean_vector_search_ms": 25.5,
        "mean_rerank_ms": 18.0,
        "mean_generation_ms": 45.0,
        "mean_hallucination_check_ms": 4.0,
    }
    with open(trials_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(trial_data) + "\n")

    res = client.get(f"/api/runs/{run_id}/trials?after=0")
    assert res.status_code == 200
    trials = res.json()["trials"]
    assert len(trials) == 1
    t = trials[0]
    assert t["trial_id"] == 1
    assert t["mean_query_transform_ms"] == 12.5
    assert t["mean_query_embedding_ms"] == 15.0
    assert t["mean_vector_search_ms"] == 25.5
    assert t["mean_rerank_ms"] == 18.0
    assert t["mean_generation_ms"] == 45.0
    assert t["mean_hallucination_check_ms"] == 4.0
    assert t["web_search_fallback_count"] is None


def test_trials_endpoint_includes_web_search_fallback_count(client):
    run_id = "aaaa1111-2222-3333-4444-555566667777"
    paths = run_manager.create_run_dirs(run_id)

    trials_file = paths["trace_dir"] / "trials.jsonl"
    trial_data = {
        "trial_id": 1,
        "composite_score": 0.85,
        "status": "success",
        "web_search_fallback_count": 3,
    }
    with open(trials_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(trial_data) + "\n")

    res = client.get(f"/api/runs/{run_id}/trials?after=0")
    assert res.status_code == 200
    trials = res.json()["trials"]
    assert trials[0]["web_search_fallback_count"] == 3


def test_run_detail_preserves_recorded_zero_cost(client):
    run_id = "bbbb1111-2222-3333-4444-555566667777"
    paths = run_manager.create_run_dirs(run_id)
    paths["config_file"].write_text(
        json.dumps({"run_name": "free-model", "created_at": "2026-09-10T00:00:00"}),
        encoding="utf-8",
    )
    (paths["trace_dir"] / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "status": "completed", "total_trials": 1}),
        encoding="utf-8",
    )
    (paths["trace_dir"] / "trials.jsonl").write_text(
        json.dumps({"trial_id": 0, "status": "success", "cost_usd": 0.0}) + "\n",
        encoding="utf-8",
    )
    response = client.get(f"/api/runs/{run_id}")
    assert response.status_code == 200
    assert response.json()["total_cost_usd"] == 0.0


def test_run_detail_corrects_completed_manifest_when_report_has_no_successes(client):
    run_id = "eeee1111-2222-3333-4444-555566667777"
    paths = run_manager.create_run_dirs(run_id)
    paths["config_file"].write_text(
        json.dumps({"run_name": "all-failed", "created_at": "2026-09-10T00:00:00"}),
        encoding="utf-8",
    )
    (paths["trace_dir"] / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "status": "completed", "total_trials": 1}),
        encoding="utf-8",
    )
    paths["report_json"].write_text(
        json.dumps(
            {
                "trials": [{"trial_id": 0, "error": "trial failed"}],
                "best_trial": None,
                "best_result": None,
                "summary": {
                    "total_trials": 1,
                    "successful_trials": 0,
                    "failed_trials": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    response = client.get(f"/api/runs/{run_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "failed"


def test_run_detail_corrects_legacy_zero_sample_completed_run(client):
    run_id = "eeee1111-2222-3333-4444-555566667778"
    paths = run_manager.create_run_dirs(run_id)
    paths["config_file"].write_text(
        json.dumps({"run_name": "zero-sample", "created_at": "2026-09-15T00:00:00"}),
        encoding="utf-8",
    )
    (paths["trace_dir"] / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "status": "completed", "total_trials": 1}),
        encoding="utf-8",
    )
    (paths["trace_dir"] / "trials.jsonl").write_text(
        json.dumps({
            "trial_id": 0, "status": "success", "total_samples": 0,
            "composite_score": 0.0,
        }) + "\n",
        encoding="utf-8",
    )
    paths["report_json"].write_text(
        json.dumps({
            "best_trial": {"trial_id": 0},
            "summary": {"successful_trials": 1, "failed_trials": 0},
        }),
        encoding="utf-8",
    )

    detail = client.get(f"/api/runs/{run_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "failed"
    assert "zero samples" in detail.json()["error_message"]
    listed = client.get("/api/runs")
    assert listed.json()["runs"][0]["status"] == "failed"


def test_run_metadata_and_trial_inspector_endpoints(client):
    run_id = "dddd1111-2222-3333-4444-555566667777"
    paths = run_manager.create_run_dirs(run_id)
    config = {
        "run_name": "inspectable",
        "llm_model": "test-model",
        "api_key": "must-not-exist",
        "document_parser_config": {"endpoint": "https://parser.test", "api_key": "nested-secret"},
    }
    # The endpoint reads the already-redacted persisted artifact. This fixture
    # intentionally mirrors the fields that write_redacted_config emits.
    config.pop("api_key")
    paths["config_file"].write_text(json.dumps(config), encoding="utf-8")
    (paths["trace_dir"] / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "status": "completed", "config_hash": "abc123"}),
        encoding="utf-8",
    )
    (paths["trace_dir"] / "trials.jsonl").write_text(
        json.dumps({
            "trial_id": 7,
            "status": "success",
            "metrics": {"recall_at_k": 1.0},
            "resolved_rag_config": {"k": 5},
        }) + "\n",
        encoding="utf-8",
    )
    (paths["trace_dir"] / "samples.jsonl").write_text(
        json.dumps({
            "trial_id": 7,
            "sample_index": 0,
            "question": "What is RAG?",
            "gold_answer": "Retrieval augmented generation",
            "gold_context": "Reference context",
            "predicted_answer": "Retrieval-augmented generation",
            "retrieved_candidates": [{"rank": 1, "page_content": "Full chunk", "metadata": {"chunk_id": 4}}],
        }) + "\n",
        encoding="utf-8",
    )

    metadata_res = client.get(f"/api/runs/{run_id}/metadata")
    assert metadata_res.status_code == 200
    assert metadata_res.json()["config"]["llm_model"] == "test-model"
    assert "api_key" not in metadata_res.json()["config"]
    assert "api_key" not in metadata_res.json()["config"]["document_parser_config"]
    assert metadata_res.json()["manifest"]["config_hash"] == "abc123"
    assert metadata_res.json()["pricing"]["mode"] == "default"

    trial_res = client.get(f"/api/runs/{run_id}/trials/7")
    assert trial_res.status_code == 200
    detail = trial_res.json()
    assert detail["trial"]["resolved_rag_config"] == {"k": 5}
    assert detail["sample_count"] == 1
    assert detail["samples"][0]["question"] == "What is RAG?"
    assert detail["samples"][0]["retrieved_candidates"][0]["page_content"] == "Full chunk"

    assert client.get(f"/api/runs/{run_id}/trials/999").status_code == 404


def test_run_detail_includes_total_web_search_fallbacks(client):
    run_id = "bbbb1111-2222-3333-4444-555566667777"
    paths = run_manager.create_run_dirs(run_id)

    trials_file = paths["trace_dir"] / "trials.jsonl"
    with open(trials_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"trial_id": 1, "composite_score": 0.5, "status": "success", "web_search_fallback_count": 2}) + "\n")
        f.write(json.dumps({"trial_id": 2, "composite_score": 0.6, "status": "success", "web_search_fallback_count": 1}) + "\n")

    res = client.get(f"/api/runs/{run_id}")
    assert res.status_code == 200
    assert res.json()["total_web_search_fallbacks"] == 3


def test_run_detail_returns_deterministic_best_result_with_metrics_and_timings(client):
    run_id = "eeee1111-2222-3333-4444-555566667777"
    paths = run_manager.create_run_dirs(run_id)
    paths["config_file"].write_text(
        json.dumps({"run_name": "best-result-test"}), encoding="utf-8"
    )
    paths["trace_dir"].joinpath("manifest.json").write_text(
        json.dumps({"run_id": run_id, "status": "completed", "total_trials": 2}),
        encoding="utf-8",
    )
    records = [
        {
            "trial_id": 8,
            "status": "success",
            "composite_score": 0.9,
            "metrics": {"recall@3": 0.9},
            "resolved_rag_config": {"k": 3, "retrieval_method": "hybrid"},
            "mean_pipeline_latency_ms": 15.0,
            "mean_evaluation_overhead_ms": 2.0,
            "mean_vector_search_ms": 8.0,
            "mean_rerank_ms": 4.0,
        },
        {
            "trial_id": 2,
            "status": "success",
            "composite_score": 0.9,
            "metrics": {"recall@5": 0.95},
            "resolved_rag_config": {"k": 5, "retrieval_method": "similarity_search"},
            "mean_pipeline_latency_ms": 10.0,
            "mean_evaluation_overhead_ms": 1.0,
            "mean_vector_search_ms": 7.0,
        },
    ]
    paths["trace_dir"].joinpath("trials.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    response = client.get(f"/api/runs/{run_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["best_result"]["trial_id"] == 2
    assert data["best_result"]["metrics"] == {"recall@5": 0.95}
    assert data["best_result"]["config"]["k"] == 5
    assert data["stage_timings"]["vector_search"] == 7.0
    assert data["stage_timings"]["rerank"] is None


def test_run_detail_omits_total_web_search_fallbacks_when_zero(client):
    run_id = "cccc1111-2222-3333-4444-555566667777"
    paths = run_manager.create_run_dirs(run_id)

    trials_file = paths["trace_dir"] / "trials.jsonl"
    with open(trials_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"trial_id": 1, "composite_score": 0.5, "status": "success"}) + "\n")

    res = client.get(f"/api/runs/{run_id}")
    assert res.status_code == 200
    assert res.json()["total_web_search_fallbacks"] is None


def test_prompt_resolver_returns_only_applicable_full_rag_prompts(client):
    response = client.post("/api/prompts/resolve", json={
        "language": "en",
        "pipeline_mode": "full_rag",
        "retrieval_source": "vector_db",
        "adaptive_web_search": True,
        "eval_dataset_mode": "auto",
        "search_space": {
            "query_expansion": ["rewrite", "hyde"],
            "reranking": ["llm"],
        },
        "metrics": ["faithfulness", "answer_correctness", "llm_judge_rating"],
    })
    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "en"
    assert {prompt["key"] for prompt in body["prompts"]} == {
        "QA", "query_rewrite", "hyde", "reranker_scoring", "generation",
        "context_relevance", "hallucination_check_prompt",
        "answer_correctness", "llm_judge_rating", "context_grounding",
    }
    generation = next(p for p in body["prompts"] if p["key"] == "generation")
    assert generation["required_variables"] == ["context", "question"]
    assert "{context}" in generation["default_template"]
    rating = next(p for p in body["prompts"] if p["key"] == "llm_judge_rating")
    assert rating["required_variables"] == ["gold_answer", "predicted_answer"]
    assert "{gold_answer}" in rating["default_template"]
    assert "{predicted_answer}" in rating["default_template"]


def test_prompt_resolver_marks_exact_retrieval_only_case_skipped(client):
    response = client.post("/api/prompts/resolve", json={
        "language": "ar",
        "pipeline_mode": "retrieval_only",
        "retrieval_source": "vector_db",
        "eval_dataset_mode": "existing",
        "search_space": {"query_expansion": ["none"], "reranking": ["none"]},
        "metrics": ["recall"],
    })
    assert response.status_code == 200
    assert response.json()["prompts"] == []
    assert response.json()["skipped"] is True


def test_preflight_rejects_irrelevant_prompt_override_before_run_creation():
    from ComposerUI.backend.app import _preflight_check
    from ComposerUI.backend.schemas import CreateRunRequest

    request = CreateRunRequest(
        run_name="prompt-check",
        pipeline_mode="retrieval_only",
        eval_dataset_mode="existing",
        prompt_overrides={"generation": "Answer {question} from {context}."},
        search_space={"query_expansion": ["none"], "reranking": ["none"]},
        metrics=["recall"],
    )
    with pytest.raises(ValueError, match="not applicable"):
        _preflight_check(request)
