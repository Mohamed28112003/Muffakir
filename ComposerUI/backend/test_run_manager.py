"""
Tests for ComposerUI run_manager module.
"""

import json
from pathlib import Path
import pytest
from ComposerUI.backend import run_manager, trace_reader
from ComposerUI.backend.schemas import CreateRunRequest, SearchSpaceDict
from SyntheticData.models import QAPair


def test_resolve_runs_root_default(monkeypatch):
    monkeypatch.delenv("MUFFAKIR_RUNS_ROOT", raising=False)
    expected = Path(run_manager.__file__).resolve().parent.parent / "safe-user-data-runs"
    monkeypatch.setattr(run_manager.storage, "_read_settings", lambda path=None: {})
    monkeypatch.setattr(run_manager.storage, "has_legacy_user_data", lambda path=None: False)
    monkeypatch.setattr(run_manager.storage, "default_runs_root", lambda: expected)
    assert run_manager._resolve_runs_root() == expected


def test_resolve_runs_root_env_override(monkeypatch, tmp_path):
    override_dir = tmp_path / "custom-runs"
    monkeypatch.setenv("MUFFAKIR_RUNS_ROOT", str(override_dir))
    assert run_manager._resolve_runs_root() == override_dir.resolve()


def test_get_run_paths_security_and_discovery(tmp_path, monkeypatch):
    monkeypatch.setattr(run_manager, "RUNS_ROOT", tmp_path)

    # Invalid path traversal strings must return None
    assert run_manager.get_run_paths("../../../etc/passwd") is None
    assert run_manager.get_run_paths("non-uuid-string") is None

    # Valid UUID but non-existent dir returns None
    valid_uuid = "12345678-1234-5678-1234-567812345678"
    assert run_manager.get_run_paths(valid_uuid) is None

    # Create directory and check paths
    paths = run_manager.create_run_dirs(valid_uuid)
    assert paths["trace_dir"].exists()
    assert paths["checkpoint_dir"].exists()

    found = run_manager.get_run_paths(valid_uuid)
    assert found is not None
    assert found["root"] == tmp_path / valid_uuid


def test_set_runs_root_persists_and_switches(tmp_path, monkeypatch):
    current = tmp_path / "current"
    selected = tmp_path / "selected"
    settings_path = tmp_path / "config" / "settings.json"
    current.mkdir()
    monkeypatch.setattr(run_manager, "RUNS_ROOT", current)
    monkeypatch.setattr(run_manager, "RUNS_ROOT_SOURCE", "default")
    monkeypatch.setattr(run_manager, "RUNS_ROOT_LOCKED", False)
    monkeypatch.setattr(run_manager.storage, "settings_file_path", lambda: settings_path)
    monkeypatch.setattr(run_manager, "_ACTIVE_RUNS", {})

    state = run_manager.set_runs_root(str(selected))

    assert run_manager.RUNS_ROOT == selected.resolve()
    assert state["source"] == "saved"
    assert state["writable"] is True
    assert state["recent_roots"] == [str(current.resolve())]
    assert json.loads(settings_path.read_text(encoding="utf-8"))["runs_root"] == str(selected.resolve())


def test_set_runs_root_rejects_locked_or_active_session(tmp_path, monkeypatch):
    current = tmp_path / "current"
    current.mkdir()
    monkeypatch.setattr(run_manager, "RUNS_ROOT", current)
    monkeypatch.setattr(run_manager, "RUNS_ROOT_LOCKED", True)

    with pytest.raises(run_manager.storage.StorageConfigurationError, match="locked"):
        run_manager.set_runs_root(str(tmp_path / "locked-target"))
    assert run_manager.RUNS_ROOT == current

    class _AliveThread:
        @staticmethod
        def is_alive():
            return True

    monkeypatch.setattr(run_manager, "RUNS_ROOT_LOCKED", False)
    monkeypatch.setattr(run_manager, "_ACTIVE_RUNS", {"active": {"thread": _AliveThread()}})
    with pytest.raises(RuntimeError, match="while a run is active"):
        run_manager.set_runs_root(str(tmp_path / "active-target"))
    assert run_manager.RUNS_ROOT == current


def test_write_redacted_config(tmp_path):
    req = CreateRunRequest(
        run_name="secret-run",
        documents_path=str(tmp_path),
        llm_provider="openai",
        llm_model="gpt-4o",
        api_key="super-secret-key-12345",
        dataset_llm_provider="openai",
        dataset_llm_model="gpt-4o-mini",
        dataset_api_key="dataset-super-secret",
        reranker_llm_provider="openai",
        reranker_llm_model="gpt-4o-mini",
        reranker_llm_api_key="reranker-llm-secret",
        reranker_base_url="https://reranker.example/v1/rerank",
        reranker_api_key="remote-reranker-secret",
        reranker_model="rerank-v2",
        search_space=SearchSpaceDict(
            reranking=["cross_encoder", "none"],
            reranking_model=[" BAAI/bge-reranker-base ", "org/custom-reranker"],
        ),
        custom_pricing=[{
            "provider": "OpenAI",
            "model": "gpt-4o",
            "input_usd_per_million_tokens": 1.25,
            "output_usd_per_million_tokens": 5,
        }],
    )
    run_manager.write_redacted_config(tmp_path, req, "test-id", "2026-09-05T12:00:00")
    config_file = tmp_path / "config.json"
    assert config_file.exists()

    content = config_file.read_text(encoding="utf-8")
    assert "super-secret-key-12345" not in content
    assert "dataset-super-secret" not in content
    assert "reranker-llm-secret" not in content
    assert "remote-reranker-secret" not in content
    saved = json.loads(content)
    assert "api_key" not in saved
    assert "dataset_api_key" not in saved
    assert "reranker_llm_api_key" not in saved
    assert "reranker_api_key" not in saved
    assert saved["reranker_base_url"] == "https://reranker.example/v1/rerank"
    assert saved["reranker_model"] == "rerank-v2"
    assert saved["dataset_llm_model"] == "gpt-4o-mini"
    assert saved["custom_pricing"][0]["provider"] == "openai"
    assert saved["custom_pricing"][0]["input_usd_per_million_tokens"] == 1.25
    assert saved["pricing_models"][0]["pricing_source"] == "custom"
    assert saved["search_space"]["reranking_model"] == [
        "BAAI/bge-reranker-base",
        "org/custom-reranker",
    ]


def test_write_redacted_config_snapshots_prompts_once_with_hashes(tmp_path):
    req = CreateRunRequest(
        run_name="prompt-snapshot",
        language="en",
        documents_path=str(tmp_path),
        llm_provider="ollama",
        llm_model="local",
        eval_dataset_mode="existing",
        eval_dataset_path=str(tmp_path / "eval.csv"),
        metrics=["answer_correctness"],
        search_space=SearchSpaceDict(query_expansion=["none"], reranking=["none"]),
        prompt_overrides={"generation": "Use {context} to answer {question}."},
    )
    run_manager.write_redacted_config(
        tmp_path, req, "prompt-run", "2026-09-09T00:00:00"
    )
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["language"] == "en"
    assert saved["prompt_overrides"] == req.prompt_overrides
    assert saved["resolved_prompts"]["generation"] == req.prompt_overrides["generation"]
    assert len(saved["prompt_hashes"]["generation"]) == 64


def test_launch_run_success(tmp_path, monkeypatch):
    monkeypatch.setattr(run_manager, "RUNS_ROOT", tmp_path)

    import Composer.composer as composer_module

    class _FakeSearch:
        def __init__(self, **kwargs):
            self.stopped_early = False
            self.stop_reason = None

        def run(self, **kwargs):
            from Composer.results.trial import TrialResult

            return [
                TrialResult(
                    trial_id=0,
                    config={"k": 3},
                    error="one configuration failed",
                    error_code="TEST_FAILURE",
                ),
                TrialResult(
                    trial_id=1,
                    config={"k": 5},
                    metrics={"recall": 1.0},
                    composite_score=1.0,
                ),
            ]

    monkeypatch.setattr(composer_module, "GridSearch", _FakeSearch)
    monkeypatch.setattr(
        composer_module.MuffakirComposer, "_index_documents", lambda self, combos: None
    )
    monkeypatch.setattr(
        composer_module.DatasetLoader, "load_or_generate", staticmethod(lambda **kwargs: [QAPair(question="What happened?", answer="A valid answer", context="Reference context")])
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    req = CreateRunRequest(
        run_name="fast-test-run",
        documents_path=str(data_dir),
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        api_key="sk-test",
        search_space=SearchSpaceDict(k=[3, 5]),
        max_eval_samples=2,
    )

    run_info = run_manager.launch_run(req)
    assert run_info["run_id"]
    # Wait for thread to finish
    run_info["thread"].join(timeout=10.0)
    assert not run_info["thread"].is_alive()

    # Discover runs should list it
    runs = run_manager.discover_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_info["run_id"]
    assert runs[0]["run_name"] == "fast-test-run"
    assert runs[0]["status"] == "completed"
    paths = run_manager.get_run_paths(run_info["run_id"])
    saved_report = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    assert saved_report["summary"]["successful_trials"] == 1
    assert saved_report["summary"]["failed_trials"] == 1


def test_launch_run_all_trials_failed_keeps_report_and_marks_failed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(run_manager, "RUNS_ROOT", tmp_path)
    info_messages = []
    error_messages = []
    monkeypatch.setattr(
        run_manager.logger,
        "info",
        lambda message, *args, **kwargs: info_messages.append(message),
    )
    monkeypatch.setattr(
        run_manager.logger,
        "error",
        lambda message, *args, **kwargs: error_messages.append(message),
    )

    import Composer.composer as composer_module

    class _AllFailedSearch:
        def __init__(self, **kwargs):
            self.stopped_early = False
            self.stop_reason = None

        def run(self, **kwargs):
            from Composer.results.trial import TrialResult

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
        composer_module.MuffakirComposer, "_index_documents", lambda self, combos: None
    )
    monkeypatch.setattr(
        composer_module.DatasetLoader,
        "load_or_generate",
        staticmethod(lambda **kwargs: [QAPair(question="What happened?", answer="A valid answer", context="Reference context")]),
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    request = CreateRunRequest(
        run_name="all-failed-run",
        documents_path=str(data_dir),
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        api_key="sk-test",
        search_space=SearchSpaceDict(k=[3]),
    )

    run_info = run_manager.launch_run(request)
    run_info["thread"].join(timeout=10.0)
    assert not run_info["thread"].is_alive()

    paths = run_manager.get_run_paths(run_info["run_id"])
    assert paths is not None
    manifest = trace_reader.read_manifest(paths["trace_dir"])
    assert manifest["status"] == "failed"

    saved_report = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    assert saved_report["summary"]["successful_trials"] == 0
    assert saved_report["summary"]["failed_trials"] == 1
    assert saved_report["trials"][0]["error"] == "simulated trial failure"

    assert run_manager.discover_runs()[0]["status"] == "failed"
    assert "run completed" not in info_messages
    assert "run failed: no successful trials or valid result" in error_messages


def test_empty_auto_dataset_fails_run_and_records_dataset_error(tmp_path, monkeypatch):
    monkeypatch.setattr(run_manager, "RUNS_ROOT", tmp_path)
    import Composer.composer as composer_module
    import Muffakir.dependency_validation as dependency_validation
    from Pricing.price_map import PriceMap

    monkeypatch.setattr(
        dependency_validation, "validate_composer_dependencies",
        lambda config, search_space: None,
    )
    monkeypatch.setattr(PriceMap, "load", lambda self, **kwargs: None)

    def _empty_generation(**kwargs):
        from Trace.observability import emit_stage_outcome

        emit_stage_outcome(
            "dataset_generation", "llm", outcome="error",
            recovery="retry", provider="gemini",
            error=TypeError("generate_content() got unexpected max_retries"),
        )
        return []

    monkeypatch.setattr(
        composer_module.DatasetLoader, "load_or_generate",
        staticmethod(_empty_generation),
    )
    monkeypatch.setattr(
        composer_module.MuffakirComposer, "_index_documents",
        lambda self, combos: pytest.fail("zero-sample run must not index"),
    )

    documents = tmp_path / "documents"
    documents.mkdir()
    request = CreateRunRequest(
        run_name="empty-auto-dataset",
        documents_path=str(documents),
        llm_provider="gemini",
        llm_model="test-model",
        api_key="test-secret",
        search_space=SearchSpaceDict(k=[3]),
    )

    run_info = run_manager.launch_run(request)
    run_info["thread"].join(timeout=10.0)
    assert not run_info["thread"].is_alive()
    paths = run_manager.get_run_paths(run_info["run_id"])
    assert trace_reader.read_manifest(paths["trace_dir"])["status"] == "failed"
    assert "no valid Q&A samples" in paths["error_file"].read_text(encoding="utf-8")
    assert run_manager.discover_runs()[0]["status"] == "failed"
    assert not paths["report_json"].exists()
    assert trace_reader.count_trial_lines(paths["trace_dir"]) == 0

    rates = trace_reader.read_error_rates(paths["trace_dir"])
    assert rates["errors"] == 2
    assert rates["unrecovered_errors"] == 1
    assert any(
        stage["stage"] == "dataset_generation" and stage["component"] == "llm"
        for stage in rates["stages"]
    )
    events, _ = trace_reader.read_errors_after(paths["trace_dir"])
    assert {event["stage"] for event in events} == {
        "dataset_generation", "dataset_load"
    }


def test_launch_run_crash_safety(tmp_path, monkeypatch):
    monkeypatch.setattr(run_manager, "RUNS_ROOT", tmp_path)

    import Composer.composer as composer_module

    def _failing_fit(self, **kwargs):
        raise RuntimeError("Simulated crash outside composer trace writer")

    monkeypatch.setattr(composer_module.MuffakirComposer, "fit", _failing_fit)

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    req = CreateRunRequest(
        run_name="crash-test-run",
        documents_path=str(data_dir),
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        api_key="sk-test",
    )

    run_info = run_manager.launch_run(req)
    run_info["thread"].join(timeout=10.0)

    run_paths = run_manager.get_run_paths(run_info["run_id"])
    assert run_paths is not None

    # Error file must be written
    assert run_paths["error_file"].exists()
    assert "Simulated crash" in run_paths["error_file"].read_text(encoding="utf-8")

    # Manifest must exist and report status="failed"
    manifest = trace_reader.read_manifest(run_paths["trace_dir"])
    assert manifest is not None
    assert manifest["status"] == "failed"

    # discover_runs must show status="failed"
    runs = run_manager.discover_runs()
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"


def test_discover_runs_corrects_legacy_completed_manifest_when_all_trials_failed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(run_manager, "RUNS_ROOT", tmp_path)
    run_id = "12345678-1234-5678-1234-567812345678"
    paths = run_manager.create_run_dirs(run_id)
    paths["config_file"].write_text(
        json.dumps(
            {
                "run_name": "legacy-all-failed",
                "created_at": "2026-09-10T00:00:00",
            }
        ),
        encoding="utf-8",
    )
    (paths["trace_dir"] / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "status": "completed", "total_trials": 2}),
        encoding="utf-8",
    )
    paths["report_json"].write_text(
        json.dumps(
            {
                "trials": [
                    {"trial_id": 0, "error": "failure one"},
                    {"trial_id": 1, "error": "failure two"},
                ],
                "best_trial": None,
                "best_result": None,
                "summary": {
                    "total_trials": 2,
                    "successful_trials": 0,
                    "failed_trials": 2,
                },
            }
        ),
        encoding="utf-8",
    )

    runs = run_manager.discover_runs()
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
