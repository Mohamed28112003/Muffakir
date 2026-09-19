"""
Tests for ComposerUI.backend.dataset_utils.
"""

from pathlib import Path

import pandas as pd
import pytest

from SyntheticData.models import QAPair
from ComposerUI.backend import dataset_utils
from ComposerUI.backend.schemas import GenerateDatasetRequest


def _make_pairs(n: int):
    return [
        QAPair(
            question=f"question number {i}?",
            answer=f"answer number {i}",
            context=f"context text for item {i} " * 3,
            chunk_id=i,
        )
        for i in range(n)
    ]


def test_split_train_test_sizes_and_no_overlap():
    pairs = _make_pairs(10)
    train, test = dataset_utils.split_train_test(pairs, test_size=0.3, random_state=42)

    assert len(test) == 3
    assert len(train) == 7
    assert len(train) + len(test) == len(pairs)

    train_ids = {p.chunk_id for p in train}
    test_ids = {p.chunk_id for p in test}
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == {p.chunk_id for p in pairs}


def test_split_train_test_deterministic():
    pairs = _make_pairs(20)
    train1, test1 = dataset_utils.split_train_test(pairs, test_size=0.25, random_state=7)
    train2, test2 = dataset_utils.split_train_test(pairs, test_size=0.25, random_state=7)
    assert [p.chunk_id for p in test1] == [p.chunk_id for p in test2]
    assert [p.chunk_id for p in train1] == [p.chunk_id for p in train2]


def test_split_train_test_small_dataset_always_yields_at_least_one_test_item():
    pairs = _make_pairs(2)
    train, test = dataset_utils.split_train_test(pairs, test_size=0.1, random_state=1)
    assert len(test) >= 1
    assert len(train) + len(test) == 2


def test_split_train_test_rejects_invalid_test_size():
    pairs = _make_pairs(5)
    with pytest.raises(ValueError):
        dataset_utils.split_train_test(pairs, test_size=0.0)
    with pytest.raises(ValueError):
        dataset_utils.split_train_test(pairs, test_size=1.0)


def test_split_train_test_empty_input():
    train, test = dataset_utils.split_train_test([], test_size=0.2)
    assert train == []
    assert test == []


def test_generate_synthetic_dataset(monkeypatch, tmp_path):
    import Muffakir as muffakir_module

    fake_df = pd.DataFrame([
        {"question": "what is question number one?", "answer": "this is answer number one",
         "context": "context text number one padded to be long enough", "chunk_id": 0, "source_file": "f.txt"},
        {"question": "what is question number two?", "answer": "this is answer number two",
         "context": "context text number two padded to be long enough", "chunk_id": 1, "source_file": "f.txt"},
    ])

    captured_config = {}

    class _FakeSyntheticData:
        def __init__(self, config):
            captured_config.update(config)

        def generate_dataset(self, max_chunks=None):
            # Real MuffakirSyntheticData writes synthetic_qa_final.csv to
            # output_dir as a side effect — mirror that so the caller's
            # post-generation validation has a real file to read.
            out_path = Path(captured_config["output_dir"]) / "synthetic_qa_final.csv"
            fake_df.to_csv(out_path, index=False)
            return fake_df

    monkeypatch.setattr(muffakir_module, "MuffakirSyntheticData", _FakeSyntheticData)
    monkeypatch.setattr(dataset_utils.run_manager, "RUNS_ROOT", tmp_path)

    request = GenerateDatasetRequest(
        documents_path="./data",
        llm_provider="together",
        llm_model="some-model",
        llm_parameters={"temperature": 0.2, "max_retries": 0},
        api_key="sk-test",
        document_parser="docling",
        use_ocr=True,
        max_chunks=5,
    )
    result = dataset_utils.generate_synthetic_dataset(request)

    assert result["row_count"] == 2
    assert result["preview"][0]["question"] == "what is question number one?"
    assert result["dataset_path"].endswith("synthetic_qa_final.csv")
    assert captured_config["data_dir"] == "./data"
    assert captured_config["document_parser"] == "docling"
    assert captured_config["use_ocr"] is True
    assert captured_config["llm_parameters"] == {"temperature": 0.2, "max_retries": 0}
    assert captured_config["output_format"] == ["csv"]
