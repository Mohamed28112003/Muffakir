"""Pure, credential-free Python export from persisted Composer trial records."""
from __future__ import annotations

import io
import json
import re
import uuid
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from pprint import pformat
from typing import Any

from .export_runtime import RUNTIME


class ExportError(ValueError):
    """The saved trial does not contain sufficient portable configuration."""


@dataclass
class PythonExport:
    code: str
    requirements: str
    environment: list[dict]
    notices: list[str]
    metadata: dict
    readme: str

    def preview(self) -> dict:
        return {key: getattr(self, key) for key in
                ("code", "requirements", "environment", "notices", "metadata", "readme")}

    def zip_bytes(self) -> bytes:
        output = io.BytesIO()
        prefix = f"muffakir-trial-{self.metadata['trial_id']}/"
        env = "\n".join(
            f"# {item['description']} ({'required' if item['required'] else 'optional'})\n"
            f"{item['name']}={item.get('default', '')}" for item in self.environment
        ) + "\n"
        files = {"rag_app.py": self.code, "requirements.txt": self.requirements,
                 ".env.example": env, "README.md": self.readme,
                 ".gitignore": ".env\n.venv/\n__pycache__/\nindex/\nknowledge-base/\n"}
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, content in files.items():
                archive.writestr(prefix + name, content)
        return output.getvalue()


def export_trial(run_id: str, trial: dict, run_config: dict | None = None,
                 manifest: dict | None = None) -> PythonExport:
    """Render one trial without importing providers, inspecting a corpus, or writing files."""
    from Muffakir import __version__
    from Muffakir.constants import DEFAULT_RAG_CONFIG, DEFAULT_SEARCH_CONFIG
    from Muffakir.dependency_validation import (
        required_features_for_rag, required_features_for_search, required_features_for_retrieval,
    )
    from PromptManager.registry import resolve_prompt_keys

    run_config, manifest = run_config or {}, manifest or {}
    resolved = trial.get("resolved_rag_config") or trial.get("resolved_config")
    if not isinstance(resolved, dict) or not resolved:
        raise ExportError("This trial has no saved resolved configuration. Export a trial with a recorded configuration.")
    trial_id = trial.get("trial_id")
    if not isinstance(trial_id, int) or isinstance(trial_id, bool) or trial_id < 0:
        raise ExportError("Invalid saved trial identifier.")
    source = {**(manifest.get("base_config") or {}), **run_config, **resolved}
    web = source.get("retrieval_source") == "web_search_only"
    retrieval = source.get("pipeline_mode") == "retrieval_only"
    if web and retrieval:
        raise ExportError("Web-search-only and retrieval-only modes cannot be combined.")
    required = [] if retrieval else ["llm_provider", "llm_model"]
    if web or source.get("adaptive_web_search"):
        required += ["search_provider"]
    if not web:
        required += ["embedding_model", "vector_db_provider", "chunking_method", "chunk_size", "chunk_overlap", "k", "retrieval_method"]
    missing = [key for key in required if source.get(key) is None or source.get(key) == ""]
    if missing:
        raise ExportError("Saved configuration is incomplete: " + ", ".join(missing) + ". No model or trial choices were substituted.")
    if any(source.get(key) is not None for key in ("chunking", "embedding_provider_instance", "db_manager")):
        raise ExportError("This trial uses injected Python components that cannot be reconstructed from JSON.")

    defaults = DEFAULT_SEARCH_CONFIG if web else DEFAULT_RAG_CONFIG
    allowed = set(defaults) | {
        "llm_parameters", "query_transform_llm_parameters", "reranker_llm_parameters",
        "pipeline_mode", "retrieval_source", "query_transform_api_key", "query_transform_base_url",
        "reranker_llm_provider", "reranker_llm_model", "reranker_llm_api_key", "reranker_llm_base_url",
        "reranker_base_url", "reranker_model", "reranker_api_key", "reranker_timeout_seconds", "reranker_options",
    }
    config = deepcopy({**defaults, **{k: v for k, v in source.items() if k in allowed}})
    for key in ("data_dir", "db_path", "collection_name", "chunking", "skip_document_ingestion"):
        config.pop(key, None)
    config["pipeline_mode"] = "retrieval_only" if retrieval else "full_rag"
    config["retrieval_source"] = "web_search_only" if web else "vector_db"
    if retrieval:
        config["hallucination_check"] = False
        config["adaptive_web_search"] = False
        # This facade's fallback differs from MuffakirRAG's default (15).
        config["fetch_k"] = source.get("fetch_k", 7)
        for key in ("llm_provider", "llm_model"):
            config.pop(key, None)
    for key in ("vector_db_config", "search_provider_config", "document_parser_config", "reranker_options"):
        if key in config:
            if config[key] is None:
                config[key] = {}
            elif not isinstance(config[key], dict):
                raise ExportError(f"Saved {key} must be an object.")
    if not config.get("query_transformer") and not (retrieval and config.get("reranking_method") == "llm"):
        for key in list(config):
            if key.startswith("query_transform_"):
                config.pop(key)
    if not config.get("reranking"):
        for key in list(config):
            if key.startswith("reranker_"):
                config.pop(key)
    notices = []
    status = trial.get("status") or ("failed" if trial.get("error") else "success")
    if status not in {"success", "failed", "running", "completed", "cancelled", "pruned"}:
        status = "unknown"
    if status != "success":
        notices.append(f"Original trial status: {status}. This configuration may reproduce its failure or has not finished evaluation.")
    if not manifest.get("base_config"):
        notices.append("Unrecorded optional settings use defaults from the pinned Muffakir version.")

    keys = resolve_prompt_keys(
        pipeline_mode=config["pipeline_mode"], retrieval_source=config["retrieval_source"],
        adaptive_web_search=config.get("adaptive_web_search", False), eval_dataset_mode="file", metrics=[],
        search_space={"query_expansion": [config.get("query_transformer_strategy") if config.get("query_transformer") else "none"],
                      "reranking": [config.get("reranking_method") if config.get("reranking") else "none"]},
        hallucination_check=config.get("hallucination_check", True),
        hallucination_method=config.get("hallucination_method", "text_cleaner"),
    )
    if config.get("hallucination_check") and config.get("hallucination_method") == "context_grounding":
        keys.append("context_grounding")
    snapshot = run_config.get("resolved_prompts") or {}
    overrides = {**(run_config.get("prompt_overrides") or {}), **(resolved.get("prompt_overrides") or {})}
    config["prompt_overrides"] = {key: overrides.get(key, snapshot.get(key)) for key in keys if key in overrides or key in snapshot}
    if any(key not in config["prompt_overrides"] for key in keys):
        notices.append("Some runtime prompt snapshots are unavailable; installed prompt defaults will supply missing templates.")

    # Build role-specific bindings even though normal saved records omit credentials.
    environment: dict[tuple, dict] = {}

    def bind(path, *, required=True, default="", name=None):
        path = tuple(path)
        if path in environment:
            environment[path]["required"] |= required
            return
        env_name = name or "MUFFAKIR_" + re.sub(r"[^A-Z0-9_]", "_", "_".join(map(str, path)).upper())
        if any(item["name"] == env_name for item in environment.values()):
            env_name += "_" + str(len(environment))
        environment[path] = {"path": list(path), "name": env_name, "required": required,
                             "default": default, "description": "Set " + ".".join(map(str, path))}

    def llm_role(provider, key, endpoint):
        if not provider:
            return
        local = str(provider).lower() in {"ollama", "vllm"}
        bind([key], required=not local, default="local" if local else "")
        bind([endpoint], required=bool(config.get(endpoint)) or str(provider).lower() in {"custom", "custom_openai", "vllm", "azure", "azure_openai"})

    if not retrieval:
        llm_role(config.get("llm_provider"), "api_key", "base_url")
    if config.get("query_transformer") or (retrieval and config.get("reranking_method") == "llm"):
        llm_role(config.get("query_transform_llm_provider"), "query_transform_api_key", "query_transform_base_url")
    if config.get("reranking"):
        llm_role(config.get("reranker_llm_provider"), "reranker_llm_api_key", "reranker_llm_base_url")
        if config.get("reranking_method") in {"custom", "remote"}:
            bind(["reranker_base_url"])
            bind(["reranker_api_key"], required=False)
    if web or config.get("adaptive_web_search"):
        config.setdefault("search_provider_config", {})
        bind(["search_provider_config", "api_key"])
    else:
        for key in ("search_provider", "search_provider_config", "fire_crawl_api"):
            config.pop(key, None)
    parser = config.get("document_parser")
    if parser in {"azure", "llama_parse", "llamaparse", "llama-parse"}:
        bind(["document_parser_config", "api_key"])
        if parser == "azure":
            bind(["document_parser_config", "endpoint"])

    if not web:
        vector = config.setdefault("vector_db_config", {}) or {}
        config["vector_db_config"] = vector
        provider = config["vector_db_provider"]
        if provider not in {"chroma", "faiss", "qdrant", "milvus", "pinecone"}:
            raise ExportError("Unsupported saved vector database provider. No provider was substituted.")
        for key in ("path", "db_path", "folder_path", "collection_name", "index_name", "location", "url"):
            vector.pop(key, None)
        if provider == "qdrant":
            bind(["vector_db_config", "location"])
            bind(["vector_db_config", "api_key"], required=False)
        elif provider == "milvus":
            bind(["vector_db_config", "connection_args", "uri"])
            bind(["vector_db_config", "connection_args", "token"], required=False)
        elif provider == "pinecone":
            bind(["vector_db_config", "api_key"])
        if provider in {"qdrant", "milvus", "pinecone"}:
            notices.append("Supply your vector database connection and a dedicated new collection (an existing empty index for Pinecone).")

    secret = re.compile(r"api.?key|token(?!s)|secret|password|authorization|credential|cookie", re.I)
    connection = re.compile(r"url|endpoint|uri$|(?:^|_)path$|directory|location|host$", re.I)

    def sanitize(value, path=()):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                current = (*path, key)
                # Prompt text is user-authored code data, never interpolated or executed.
                if path == ("prompt_overrides",):
                    result[key] = item
                elif secret.search(str(key)) and key not in {"llm_max_tokens", "max_tokens"}:
                    if item is not None or current in environment:
                        bind(current, required=bool(item) and current not in environment)
                    result[key] = None
                elif connection.search(str(key)) or "headers" in path:
                    if item is not None or current in environment:
                        if isinstance(item, (dict, list)):
                            result[key] = sanitize(item, current)
                            continue
                        bind(current, required=bool(item) and current not in environment)
                    result[key] = None
                else:
                    result[key] = sanitize(item, current)
            return result
        if isinstance(value, list):
            return [sanitize(item, (*path, i)) for i, item in enumerate(value)]
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ExportError("Saved configuration contains a non-JSON component.")
        return value

    config = sanitize(config)
    try:
        json.dumps(config, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ExportError("Saved configuration contains values that cannot be represented in portable Python.") from exc
    # Environment bindings create keys omitted by redaction when load_config runs.
    env = list(environment.values())
    if not web:
        for name, default, description in (
            ("MUFFAKIR_DOCUMENTS_DIR", "./knowledge-base", "Directory containing your documents"),
            ("MUFFAKIR_INDEX_DIR", "./index", "Fresh project index directory; includes the local chunk cache"),
            ("MUFFAKIR_COLLECTION", "", "Dedicated collection name; required existing empty index name for Pinecone"),
        ):
            env.append({"name": name, "path": [], "required": name == "MUFFAKIR_COLLECTION" and config["vector_db_provider"] == "pinecone",
                        "default": default, "description": description})

    feature_config = {**config, "data_dir": None, "skip_document_ingestion": False}
    features = required_features_for_search(feature_config) if web else required_features_for_rag(feature_config)
    if retrieval:
        features += required_features_for_retrieval(feature_config)
    if not web:
        features.append("pdf")  # A portable corpus may contain PDFs even if the original is unavailable.
    requirements = f"Muffakir[{','.join(sorted(set(features)))}]=={__version__}\npython-dotenv>=1.0,<2\n"
    metadata = {"run_id": run_id, "trial_id": trial_id, "status": status,
                "pipeline_mode": config["pipeline_mode"], "retrieval_source": config["retrieval_source"],
                "export_version": 1, "muffakir_version": __version__,
                "export_id": uuid.uuid5(uuid.NAMESPACE_URL, f"muffakir:{run_id}:{trial_id}").hex[:12]}
    code = ('"""Exported Composer trial. Set environment variables; index once, then query."""\n'
            'import json\nimport os\nfrom pathlib import Path\n\nROOT = Path(__file__).resolve().parent\n\n'
            + "EXPORT_METADATA = " + pformat(metadata, sort_dicts=False) + "\n\n"
            + "CONFIG = " + pformat(config, sort_dicts=False, width=100) + "\n\n"
            + "ENVIRONMENT = " + pformat(env, sort_dicts=False, width=100) + "\n" + RUNTIME)
    readme = f"""# Muffakir trial {trial_id}

Selected trial: {trial_id}. Original status: {status}. Muffakir: {__version__}.
This application runs the selected pipeline without executing Composer search or evaluation.

## Setup

Use Python 3.11+. Create and activate a virtual environment, then:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the indicated variables. Existing environment
variables take precedence. For a script-only download, install the requirements shown
in ComposerUI and set the variables listed in `ENVIRONMENT` at the top of the script.

If using an unreleased local Muffakir checkout, replace the Muffakir line in
requirements.txt with `-e /path/to/Muffakir_Arabic_RAG[{','.join(sorted(set(features)))}]`
before installing. Exported code
requires the matching implementation, not merely a matching package version number.

{'Web Search Only needs no documents or index.' if web else 'Set MUFFAKIR_DOCUMENTS_DIR, then run `python rag_app.py --index` once.'}

```bash
python rag_app.py --question "What does the documentation say?"
```

You can import `build_pipeline` from `rag_app` and reuse one initialized pipeline.
Retrieval-only mode returns documents and metadata; other modes return answers and available sources.

## Index and services

Indexing requires a fresh MUFFAKIR_INDEX_DIR. An interrupted build reserves that location;
use a new location and collection to retry. Querying never ingests documents. To change
the corpus or index settings, build into a fresh directory and dedicated collection.
Do not delete existing databases to reuse their names. Keep index/ready.json, chunks.json,
and the vector index together. They contain your data and should remain private.

Qdrant and Milvus require your configured service URI. Pinecone requires a dedicated
empty index with dimensions matching the selected embedding model. Hosted providers
require their credentials. Ollama/custom servers must be running and have the model available.
Hugging Face models download on first initialization and use the standard cache.
The saved CPU/CUDA setting is preserved; CUDA needs compatible hardware and PyTorch.
Optional OCR/parsers may require system tools such as Tesseract/Poppler or provider accounts;
see Muffakir's installation and parser documentation for the selected backend.

## Export notes

""" + "\n".join("- " + notice for notice in notices) + """

Credentials and connection values are supplied by you. Review custom prompts before sharing.
Configuration reproduction does not guarantee identical answers or scores: documents,
model versions, sampling, hardware, and live web results can change.
"""
    return PythonExport(code, requirements, env, notices, metadata, readme)
