"""
Eval-dataset helpers for ComposerUI: train/test split and inline synthetic
Q&A dataset generation (wraps MuffakirSyntheticData).
"""

import json
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from SyntheticData.models import QAPair
from ComposerUI.backend import run_manager

if TYPE_CHECKING:
    from ComposerUI.backend.schemas import GenerateDatasetRequest


def split_train_test(
    pairs: List[QAPair],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[QAPair], List[QAPair]]:
    """Deterministically split QAPairs into (train, test). Stdlib only.

    The train split is returned but not consumed anywhere yet — reserved for
    future use, per design. Only `test` is meant to be passed as
    `composer.fit(eval_dataset=...)` so trials are scored on held-out data.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1 (exclusive), got {test_size}")
    if not pairs:
        return [], []

    indices = list(range(len(pairs)))
    random.Random(random_state).shuffle(indices)

    n_test = max(1, round(len(pairs) * test_size))
    n_test = min(n_test, len(pairs))
    test_indices = set(indices[:n_test])

    train = [p for i, p in enumerate(pairs) if i not in test_indices]
    test = [p for i, p in enumerate(pairs) if i in test_indices]
    return train, test


def generate_synthetic_dataset(request: "GenerateDatasetRequest") -> Dict[str, Any]:
    """Generate a synthetic Q&A dataset via MuffakirSyntheticData and save it
    to a staging directory. Returns {dataset_path, preview, row_count}.
    """
    from Muffakir import MuffakirSyntheticData

    staging_dir = run_manager.RUNS_ROOT / "_datasets" / uuid.uuid4().hex
    staging_dir.mkdir(parents=True, exist_ok=True)

    config: Dict[str, Any] = {
        "data_dir": request.documents_path,
        "api_key": request.api_key,
        "llm_provider": request.llm_provider,
        "llm_model": request.llm_model,
        "llm_parameters": request.llm_parameters,
        "language": request.language,
        "output_dir": str(staging_dir),
        "output_format": ["csv"],
    }
    if request.base_url:
        config["base_url"] = request.base_url
    if request.document_parser:
        config["document_parser"] = request.document_parser
        config["document_parser_config"] = request.document_parser_config or {}
    if request.use_ocr:
        config["use_ocr"] = True

    generator = MuffakirSyntheticData(config=config)
    df = generator.generate_dataset(max_chunks=request.max_chunks)

    dataset_path = staging_dir / "synthetic_qa_final.csv"

    from Evaluation.dataset import load_evaluation_dataset

    # Raises ValueError if generation produced 0 valid rows or a bad schema —
    # surfaces as a clean 400 instead of handing back a silently-broken file.
    load_evaluation_dataset(str(dataset_path))

    preview: List[Dict[str, Any]] = json.loads(
        df.head(5).to_json(orient="records", force_ascii=False)
    )

    return {
        "dataset_path": str(dataset_path),
        "preview": preview,
        "row_count": len(df),
    }
