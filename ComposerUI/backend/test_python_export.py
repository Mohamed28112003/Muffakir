import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from ComposerUI.backend.app import app
from ComposerUI.backend import run_manager
from Composer.test_python_export import trial_config

RUN = "00000000-0000-0000-0000-000000000007"


@pytest.fixture
def export_client(tmp_path, monkeypatch):
    monkeypatch.setattr(run_manager, "RUNS_ROOT", tmp_path)
    paths = run_manager.create_run_dirs(RUN)
    paths["config_file"].write_text(json.dumps({"llm_model": "run-default-model"}), encoding="utf-8")
    records = [
        {"trial_id": 0, "status": "success", "composite_score": 1,
         "resolved_rag_config": trial_config(llm_model="best-model")},
        {"trial_id": 7, "status": "failed", "composite_score": 0,
         "resolved_rag_config": trial_config(llm_model="selected-model", api_key="never-export-me")},
        {"trial_id": 8, "status": "failed", "resolved_rag_config": {}},
    ]
    (paths["trace_dir"] / "trials.jsonl").write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
    return TestClient(app), paths


def test_preview_and_downloads_select_failed_non_best_trial(export_client):
    client, _ = export_client
    endpoint = f"/api/runs/{RUN}/trials/7/export"
    preview = client.get(endpoint)
    assert preview.status_code == 200
    data = preview.json()
    assert data["metadata"]["trial_id"] == 7
    assert data["metadata"]["status"] == "failed"
    assert "selected-model" in data["code"]
    for excluded in ("best-model", "run-default-model", "never-export-me"):
        assert excluded not in preview.text
    script = client.get(endpoint + "?format=python")
    assert script.text == data["code"]
    assert 'filename="rag_app.py"' in script.headers["content-disposition"]
    archive = client.get(endpoint + "?format=zip")
    assert archive.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(archive.content)) as zipped:
        assert zipped.read("muffakir-trial-7/rag_app.py").decode() == script.text


def test_validation_and_legacy_checkpoint(export_client):
    client, paths = export_client
    assert client.get(f"/api/runs/{RUN}/trials/900/export").status_code == 404
    assert client.get("/api/runs/not-a-run/trials/7/export").status_code == 404
    assert client.get(f"/api/runs/{RUN}/trials/8/export").status_code == 422
    assert client.get(f"/api/runs/{RUN}/trials/7/export?format=exe").status_code == 422
    checkpoint = {"trials": [{"trial_id": 9, "error": "failed", "resolved_config": trial_config(k=9)}]}
    (paths["checkpoint_dir"] / "composer_checkpoint.json").write_text(json.dumps(checkpoint))
    response = client.get(f"/api/runs/{RUN}/trials/9/export")
    assert response.status_code == 200
    assert response.json()["metadata"]["trial_id"] == 9
    assert "'k': 9" in response.json()["code"]


def test_export_controls_in_trial_inspector(export_client):
    client, _ = export_client
    script = client.get("/static/js/run_detail.js").text
    assert 'id: "export", label: "Export Python"' in script
    assert "Download project ZIP" in script
    assert "inspectorData === selected" in script
    assert "navigator.clipboard.writeText(data.code)" in script
