from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from SyntheticData.models import QAPair

DatasetInput = Union[str, Path, pd.DataFrame, List[Dict[str, Any]]]

REQUIRED_COLUMNS = ("question", "answer", "context")


def _dataframe_from_path(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(path)
        if suffix == ".jsonl":
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            if not records:
                raise ValueError(f"JSONL dataset is empty: {path}")
            return pd.DataFrame(records)
        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # allow {"data": [...]} or single-record wrappers
                if "data" in data and isinstance(data["data"], list):
                    data = data["data"]
                else:
                    data = [data]
            if not isinstance(data, list):
                raise ValueError(f"JSON dataset must be a list of records: {path}")
            return pd.DataFrame(data)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to read dataset file '{path}': {e}") from e
    raise ValueError(
        f"Unsupported dataset file type '{suffix}'. Use .csv, .xlsx, .xls, or .json."
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: str(c).strip().lower() for c in df.columns}
    return df.rename(columns=rename)


def load_evaluation_dataset(
    dataset: DatasetInput,
    fail_fast: bool = True,
    max_samples: Optional[int] = None,
) -> List[QAPair]:
    """
    Load and validate an evaluation dataset into QAPair objects.

    Accepts file path, DataFrame, or list of dicts.
    Required columns: question, answer, context.
    Optional: chunk_id, source_file.
    """
    if isinstance(dataset, (str, Path)):
        path = Path(dataset)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        df = _dataframe_from_path(path)
    elif isinstance(dataset, pd.DataFrame):
        df = dataset.copy()
    elif isinstance(dataset, list):
        df = pd.DataFrame(dataset)
    else:
        raise TypeError(
            "dataset must be a file path, pandas DataFrame, or list of dicts "
            f"(got {type(dataset).__name__})."
        )

    if df.empty:
        raise ValueError("Evaluation dataset is empty.")

    df = _normalize_columns(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns {missing}. "
            f"Required: {list(REQUIRED_COLUMNS)}. Found: {list(df.columns)}. "
            "Use the same schema as MuffakirSyntheticData / QAPair."
        )

    if max_samples is not None:
        df = df.head(int(max_samples))

    pairs: List[QAPair] = []
    errors: List[str] = []

    for idx, row in df.iterrows():
        record = {
            "question": row.get("question"),
            "answer": row.get("answer"),
            "context": row.get("context"),
            "chunk_id": row.get("chunk_id", 0) if "chunk_id" in df.columns else 0,
            "source_file": (
                row.get("source_file", "unknown")
                if "source_file" in df.columns
                else "unknown"
            ),
        }
        # pandas NaN -> None handling
        for key, value in list(record.items()):
            if pd.isna(value):
                if key in ("chunk_id",):
                    record[key] = 0
                elif key == "source_file":
                    record[key] = "unknown"
                else:
                    record[key] = None
        try:
            if record.get("chunk_id") is not None:
                record["chunk_id"] = int(record["chunk_id"])
            pairs.append(QAPair.model_validate(record))
        except Exception as e:
            errors.append(f"row {idx}: {e}")

    if errors and fail_fast:
        preview = "\n".join(errors[:20])
        more = f"\n... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(
            f"Dataset validation failed for {len(errors)} row(s).\n{preview}{more}"
        )

    if not pairs:
        raise ValueError("No valid QAPair rows found in dataset.")

    return pairs
