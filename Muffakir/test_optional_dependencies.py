import json
import subprocess
import sys

import pytest

from Muffakir.exceptions import ConfigurationError, MissingOptionalDependencyError
from Muffakir import optional_dependencies as optional
from Muffakir.dependency_validation import required_features_for_composer


def test_missing_optional_dependency_is_typed_and_actionable():
    error = MissingOptionalDependencyError(
        feature="Example provider",
        missing_imports=["example_sdk"],
        extra="example",
    )
    assert isinstance(error, ConfigurationError)
    assert isinstance(error, ImportError)
    assert error.error_code == "OPTIONAL_DEPENDENCY_MISSING"
    assert error.extra == "example"
    assert error.install_command == 'pip install "Muffakir[example]"'
    assert error.context["missing_imports"] == ["example_sdk"]


def test_require_optional_dependencies_reports_every_extra(monkeypatch):
    monkeypatch.setattr(optional, "is_importable", lambda _name: False)
    with pytest.raises(MissingOptionalDependencyError) as caught:
        optional.require_optional_dependencies(["openai", "chroma"])

    context = caught.value.context
    assert [item["extra"] for item in context["missing_features"]] == ["openai", "chroma"]
    assert context["install_commands"] == [
        'pip install "Muffakir[openai]"',
        'pip install "Muffakir[chroma]"',
    ]


def test_selected_factory_uses_typed_dependency_error(monkeypatch):
    from VectorDB.factory import create_vector_db

    monkeypatch.setattr(optional, "is_importable", lambda _name: False)
    with pytest.raises(MissingOptionalDependencyError, match=r'Muffakir\[faiss\]'):
        create_vector_db("faiss", embedding_provider=object())


def test_composer_feature_resolution_uses_search_space_union(tmp_path):
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")
    features = required_features_for_composer(
        {
            "data_dir": str(tmp_path),
            "llm_provider": "openai",
            "adaptive_web_search": True,
            "search_provider": "tavily",
        },
        {
            "retrieval": ["similarity_search", "hybrid"],
            "reranking": ["none", "bm25"],
            "chunking": [{"method": "token", "size": 100, "overlap": 10}],
            "vector_db_provider": ["chroma", "qdrant"],
        },
    )
    assert features == [
        "rag", "datasets", "local", "chroma", "qdrant", "token",
        "bm25", "pdf", "openai", "tavily",
    ]


def test_core_import_is_fast_and_does_not_load_heavy_modules():
    script = """
import json, sys, time
start = time.perf_counter()
import Muffakir
elapsed = time.perf_counter() - start
heavy = [name for name in (
    'chromadb', 'pandas', 'torch', 'transformers',
    'sentence_transformers', 'fastapi', 'langchain_openai', 'langchain_groq'
) if name in sys.modules]
print(json.dumps({'elapsed': elapsed, 'heavy': heavy, 'version': Muffakir.__version__}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout.strip())
    assert result["elapsed"] < 2.
    assert result["heavy"] == []
    assert result["version"] == "0.3.0"


def test_public_classes_resolve_without_importing_optional_runtimes():
    script = """
import json, sys
from Muffakir import (
    MuffakirRAG, MuffakirSearch, MuffakirRetrieval,
    MuffakirSyntheticData, MuffakirEvaluation, MuffakirComposer, MuffakirPrompt,
)
from VectorDB import (
    ChromaDBManager, FAISSDBManager, QdrantDBManager,
    MilvusDBManager, PineconeDBManager,
)
from RAGPipeline import RetrieveMethods, RAGPipelineManager
from QueryTransformer import QueryTransformer
from Evaluation import EvaluationReport
from Reranker import Reranker
heavy = [name for name in (
    'chromadb', 'pandas', 'torch', 'transformers',
    'sentence_transformers', 'fastapi', 'langchain_openai', 'langchain_groq'
) if name in sys.modules]
print(json.dumps({'heavy': heavy}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout.strip())["heavy"] == []
