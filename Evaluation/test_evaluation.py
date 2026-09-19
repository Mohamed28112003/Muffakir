"""
Test suite for the Evaluation module (pytest).

No optional SDKs or real LLM/DB are required: the RAG and metric classes are
mocked. This validates retrieval metrics, matching logic, dataset loading,
EvalRunner (including the double-retrieval elimination fix), and the
MuffakirEvaluation facade.
"""

import csv
import json
import logging
import sys
import types
from pathlib import Path

import pytest

# Make the library importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Evaluation.metrics.retrieval import (
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    dcg_at_k,
    ndcg_at_k,
    score_retrieval,
)
from Evaluation.matching import (
    normalize_text,
    is_relevant_document,
    build_relevance_list,
)
from Evaluation.models import (
    ALL_METRICS,
    DEFAULT_METRICS,
    EvalSampleResult,
    EvaluationReport,
    GenerationScores,
    RetrievalScores,
)
from Evaluation.runner import EvalRunner

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------
def test_recall_at_k():
    assert recall_at_k([0, 0, 1, 0]) == 1.0
    assert recall_at_k([0, 0, 0, 0]) == 0.0
    assert recall_at_k([]) == 0.0


def test_precision_at_k():
    assert precision_at_k([1, 0, 1, 0]) == 0.5
    assert precision_at_k([1, 1]) == 1.0
    assert precision_at_k([]) == 0.0


def test_mrr():
    assert mean_reciprocal_rank([0, 1, 0]) == 0.5
    assert mean_reciprocal_rank([1, 0, 0]) == 1.0
    assert mean_reciprocal_rank([0, 0, 0]) == 0.0


def test_ndcg():
    # All relevant at top → perfect nDCG
    assert ndcg_at_k([1, 1, 0, 0]) == pytest.approx(1.0)
    # Relevant at bottom → nDCG < 1
    assert 0.0 < ndcg_at_k([0, 0, 1, 0]) < 1.0
    # No relevant → 0
    assert ndcg_at_k([0, 0, 0]) == 0.0
    assert ndcg_at_k([]) == 0.0


def test_score_retrieval_dispatch():
    scores = score_retrieval([1, 0, 0], ["recall", "precision", "mrr", "ndcg"])
    assert "recall_at_k" in scores
    assert "precision_at_k" in scores
    assert "mrr" in scores
    assert "ndcg_at_k" in scores
    assert scores["recall_at_k"] == 1.0


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def test_normalize_text_basic():
    assert normalize_text("  Hello   World  ") == "hello world"


def test_normalize_text_arabic():
    # أ/إ/آ → ا, ة → ه, ى → ي
    assert normalize_text("أحمد") == normalize_text("احمد")
    assert normalize_text("إبراهيم") == normalize_text("ابراهيم")
    assert normalize_text("مدرسة") == normalize_text("مدرسه")
    assert normalize_text("على") == normalize_text("علي")


def test_token_jaccard():
    from Evaluation.matching import _token_jaccard_ratio

    assert _token_jaccard_ratio("hello world", "hello world") == 1.0
    assert _token_jaccard_ratio("hello world", "hello") == 1.0  # substring
    assert _token_jaccard_ratio("", "hello") == 0.0
    assert 0.0 < _token_jaccard_ratio("hello world foo", "hello bar baz") < 1.0


def test_is_relevant_by_chunk_id():
    doc = Document(page_content="irrelevant text", metadata={"chunk_id": 42})
    assert is_relevant_document(doc, gold_context="something else", gold_chunk_id=42)


def test_is_relevant_by_text_containment():
    doc = Document(page_content="the gold answer is here", metadata={})
    assert is_relevant_document(doc, gold_context="the gold answer is here")


def test_is_relevant_no_match():
    doc = Document(page_content="completely unrelated content", metadata={})
    assert not is_relevant_document(doc, gold_context="something else entirely")


def test_build_relevance_list_padding():
    """When fewer than k docs retrieved, pad with zeros."""
    docs = [Document(page_content="match", metadata={})]
    relevance = build_relevance_list(docs, gold_context="match", k=3)
    assert len(relevance) == 3
    assert relevance[0] == 1
    assert relevance[1] == 0
    assert relevance[2] == 0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def test_load_dataset_from_list():
    from Evaluation.dataset import load_evaluation_dataset

    data = [
        {"question": "what is the law?", "answer": "the law is X", "context": "context text here"},
        {"question": "what is article 5?", "answer": "it says Y", "context": "more context text"},
    ]
    pairs = load_evaluation_dataset(data)
    assert len(pairs) == 2
    assert pairs[0].question == "what is the law?"


def test_load_dataset_missing_columns():
    from Evaluation.dataset import load_evaluation_dataset

    with pytest.raises(ValueError, match="missing required columns"):
        load_evaluation_dataset([{"question": "only question"}])


def test_load_dataset_empty():
    from Evaluation.dataset import load_evaluation_dataset

    with pytest.raises(ValueError, match="empty"):
        load_evaluation_dataset([])


def test_load_dataset_jsonl(tmp_path):
    from Evaluation.dataset import load_evaluation_dataset

    f = tmp_path / "data.jsonl"
    f.write_text(
        json.dumps({"question": "q one?", "answer": "a one", "context": "c one here"}) + "\n"
        + json.dumps({"question": "q two?", "answer": "a two", "context": "c two here"}) + "\n",
        encoding="utf-8",
    )
    pairs = load_evaluation_dataset(str(f))
    assert len(pairs) == 2
    assert pairs[1].question == "q two?"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def test_eval_sample_to_flat_dict():
    s = EvalSampleResult(
        question="q?",
        gold_answer="ga",
        gold_context="gc",
        predicted_answer="pa",
        retrieved_previews=["doc1", "doc2"],
        retrieval=RetrievalScores(recall_at_k=1.0),
        generation=GenerationScores(faithfulness=0.9, llm_judge_rating=4),
    )
    d = s.to_flat_dict()
    assert d["recall_at_k"] == 1.0
    assert d["faithfulness"] == 0.9
    assert d["llm_judge_rating"] == 4
    assert "doc1" in d["retrieved_previews"]


def test_evaluation_report_to_dataframe_empty_has_expected_schema():
    """An empty report's to_dataframe() must keep the full column schema
    (zero rows), not collapse to zero columns — otherwise
    report.to_dataframe()["question"] raises KeyError on an empty run."""
    report = EvaluationReport(
        n_samples=0, k=3, metrics_enabled=["recall"], retrieval={}, generation={}, samples=[],
    )
    df = report.to_dataframe()
    assert len(df) == 0
    assert "question" in df.columns
    assert "recall_at_k" in df.columns
    assert "llm_judge_rating" in df.columns
    assert str(df["llm_judge_rating"].dtype) == "Int64"


def test_evaluation_dataframe_preserves_nullable_integer_rating(tmp_path):
    report = EvaluationReport(
        n_samples=2,
        k=1,
        metrics_enabled=["llm_judge_rating"],
        generation={"llm_judge_rating": 4.0},
        samples=[
            EvalSampleResult(
                question="q1", gold_answer="a1",
                generation=GenerationScores(llm_judge_rating=4),
            ),
            EvalSampleResult(question="q2", gold_answer="a2"),
        ],
    )

    df = report.to_dataframe()

    assert str(df["llm_judge_rating"].dtype) == "Int64"
    assert df.loc[0, "llm_judge_rating"] == 4
    assert bool(df["llm_judge_rating"].isna().iloc[1])

    output_dir = tmp_path / "rating-report"
    report.save(str(output_dir))
    with open(output_dir / "summary.json", encoding="utf-8") as handle:
        summary = json.load(handle)
    with open(output_dir / "samples.csv", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["generation"]["llm_judge_rating"] == 4.0
    assert rows[0]["llm_judge_rating"] == "4"
    assert rows[1]["llm_judge_rating"] == ""


def test_evaluation_report_save(tmp_path):
    report = EvaluationReport(
        n_samples=1,
        k=3,
        metrics_enabled=["recall"],
        retrieval={"recall@3": 1.0},
        generation={},
        samples=[
            EvalSampleResult(
                question="q?",
                gold_answer="ga",
                gold_context="gc",
                retrieval=RetrievalScores(recall_at_k=1.0),
            )
        ],
    )
    out = str(tmp_path / "eval_out")
    report.save(out)
    assert (tmp_path / "eval_out" / "summary.json").exists()
    assert (tmp_path / "eval_out" / "samples.csv").exists()
    with open(tmp_path / "eval_out" / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["n_samples"] == 1
    assert summary["retrieval"]["recall@3"] == 1.0


# ---------------------------------------------------------------------------
# _parse_score (AnswerCorrectnessMetric)
# ---------------------------------------------------------------------------
def test_parse_score_explicit():
    from Evaluation.metrics.generation import AnswerCorrectnessMetric

    assert AnswerCorrectnessMetric._parse_score("score=0.85") == 0.85
    assert AnswerCorrectnessMetric._parse_score("درجة:0.9") == 0.9


def test_parse_score_float():
    from Evaluation.metrics.generation import AnswerCorrectnessMetric

    assert AnswerCorrectnessMetric._parse_score("the answer is 0.75 correct") == 0.75


def test_parse_score_keywords():
    from Evaluation.metrics.generation import AnswerCorrectnessMetric

    assert AnswerCorrectnessMetric._parse_score("yes this is correct") == 1.0
    assert AnswerCorrectnessMetric._parse_score("لا this is incorrect") == 0.0
    assert AnswerCorrectnessMetric._parse_score("") == 0.0


def test_parse_score_explicit_accepts_leading_dot():
    from Evaluation.metrics.generation import AnswerCorrectnessMetric

    assert AnswerCorrectnessMetric._parse_score("score: .85") == 0.85


def test_parse_score_float_prefers_last_match_over_first():
    """A judge that reasons first and states its final score last must resolve
    to the concluding number, not an incidental one earlier in the text."""
    from Evaluation.metrics.generation import AnswerCorrectnessMetric

    text = "I would rate this 0 out of the given issues, but overall 0.9 is fair"
    assert AnswerCorrectnessMetric._parse_score(text) == 0.9


# ---------------------------------------------------------------------------
# LLMJudgeRatingMetric
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1", 1),
        ("5", 5),
        ("rating: 4", 4),
        ("**LLM Judge Rating:** 3", 3),
        ("التقييم: 2", 2),
        ("التقييم: ٤", 4),
    ],
)
def test_llm_judge_rating_parses_valid_integer_ratings(text, expected):
    from Evaluation.metrics.generation import LLMJudgeRatingMetric

    assert LLMJudgeRatingMetric._parse_rating(text) == expected


@pytest.mark.parametrize(
    "text",
    ["", "0", "6", "4.5", "rating: 0", "rating: 6", "rating: 3.5", "probably four"],
)
def test_llm_judge_rating_rejects_invalid_output(text):
    from Evaluation.metrics.generation import LLMJudgeRatingMetric

    with pytest.raises(ValueError, match="rating|Rating|parse"):
        LLMJudgeRatingMetric._parse_rating(text)


def test_llm_judge_rating_uses_only_reference_and_generated_answer():
    from unittest.mock import MagicMock
    from Evaluation.metrics.generation import LLMJudgeRatingMetric

    metric = LLMJudgeRatingMetric.__new__(LLMJudgeRatingMetric)
    metric.prompt = "Reference: {gold_answer}\nGenerated: {predicted_answer}"
    metric.llm_provider = MagicMock()
    llm = MagicMock()
    llm.bind.return_value = llm
    llm.invoke.return_value.content = "rating: 5"
    metric.llm_provider.get_llm.return_value = llm

    assert metric.score(gold_answer="gold", predicted_answer="prediction") == 5
    llm.invoke.assert_called_once_with("Reference: gold\nGenerated: prediction")


def test_llm_judge_rating_provider_failure_is_typed():
    from unittest.mock import MagicMock
    from Evaluation.metrics.generation import LLMJudgeRatingMetric
    from Muffakir.exceptions import ProviderError

    metric = LLMJudgeRatingMetric.__new__(LLMJudgeRatingMetric)
    metric.prompt = "Reference: {gold_answer}\nGenerated: {predicted_answer}"
    metric.llm_provider = MagicMock()
    llm = MagicMock()
    llm.bind.return_value = llm
    llm.invoke.side_effect = RuntimeError("judge unavailable")
    metric.llm_provider.get_llm.return_value = llm

    with pytest.raises(ProviderError):
        metric.score(gold_answer="gold", predicted_answer="prediction")


# ---------------------------------------------------------------------------
# FaithfulnessMetric / AnswerCorrectnessMetric — failure propagation
# ---------------------------------------------------------------------------
def test_faithfulness_metric_propagates_checker_failure():
    from unittest.mock import MagicMock
    from Evaluation.metrics.generation import FaithfulnessMetric
    from Muffakir.exceptions import HallucinationCheckError

    metric = FaithfulnessMetric.__new__(FaithfulnessMetric)
    metric.checker = MagicMock()
    metric.checker.check.side_effect = HallucinationCheckError("boom")

    with pytest.raises(HallucinationCheckError):
        metric.score(answer="ans", context="ctx")


def test_answer_correctness_metric_prompt_bug_propagates_untyped():
    """A prompt-formatting bug is a per-sample data issue, not a systemic
    provider failure — it must propagate as-is (untyped), not be wrapped."""
    from unittest.mock import MagicMock
    from Evaluation.metrics.generation import AnswerCorrectnessMetric

    metric = AnswerCorrectnessMetric.__new__(AnswerCorrectnessMetric)
    metric.llm_provider = MagicMock()
    metric.prompt_manager = MagicMock()
    metric.prompt = "{this_key_does_not_exist}"

    with pytest.raises(KeyError):
        metric.score(question="q", gold_answer="a", predicted_answer="p")


def test_answer_correctness_metric_judge_failure_raises_provider_error():
    from unittest.mock import MagicMock
    from Evaluation.metrics.generation import AnswerCorrectnessMetric
    from Muffakir.exceptions import ProviderError

    metric = AnswerCorrectnessMetric.__new__(AnswerCorrectnessMetric)
    metric.llm_provider = MagicMock()
    metric.prompt_manager = MagicMock()
    metric.prompt = "Q: {question} Gold: {gold_answer} Pred: {predicted_answer}"

    mock_llm = MagicMock(spec=["invoke"])
    mock_llm.invoke.side_effect = RuntimeError("judge LLM down")
    metric.llm_provider.get_llm.return_value = mock_llm

    with pytest.raises(ProviderError) as excinfo:
        metric.score(question="q", gold_answer="a", predicted_answer="p")
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_eval_runner_aborts_on_judge_provider_error(monkeypatch):
    """A systemic judge-LLM failure must abort the whole eval run, not just
    exclude one sample — matches _evaluate_one's existing MuffakirError path."""
    from Muffakir.exceptions import ProviderError

    class _FlakyMetric:
        def __init__(self, **kwargs):
            pass

        def score(self, **kwargs):
            raise ProviderError("judge down")

    monkeypatch.setattr("Evaluation.runner.AnswerCorrectnessMetric", _FlakyMetric)

    mock_rag = _MockRAG([Document(page_content="gold context text here", metadata={})])
    pairs = [_make_pair(), _make_pair(q="another question?")]

    runner = EvalRunner(
        rag=mock_rag, llm_provider=None, prompt_manager=None,
        metrics=["recall", "answer_correctness"], k=1,
    )
    with pytest.raises(ProviderError):
        runner.run(pairs)


# ---------------------------------------------------------------------------
# EvalRunner — retrieval-only (no LLM needed)
# ---------------------------------------------------------------------------
class _MockRAG:
    """Mock RAG with ask() and get_similar_documents()."""

    def __init__(self, docs, answer="test answer"):
        self._docs = docs
        self._answer = answer
        self.ask_calls = 0
        self.get_similar_calls = 0

    def ask(self, question, k=5):
        self.ask_calls += 1
        return {
            "answer": self._answer,
            "retrieved_documents": [doc.page_content for doc in self._docs[:k]],
        }

    def get_similar_documents(self, query, k=5):
        self.get_similar_calls += 1
        return self._docs[:k]


def _make_pair(q="what is the law?", a="the answer is X", c="gold context text here"):
    from SyntheticData.models import QAPair

    return QAPair(question=q, answer=a, context=c)


def test_eval_runner_retrieval_only():
    """Retrieval-only metrics: get_similar_documents called, ask() NOT called."""
    mock_rag = _MockRAG([Document(page_content="gold context text here", metadata={})])
    pairs = [_make_pair()]

    runner = EvalRunner(
        rag=mock_rag, llm_provider=None, prompt_manager=None, metrics=["recall"], k=1,
    )
    report = runner.run(pairs)

    assert mock_rag.get_similar_calls == 1
    assert mock_rag.ask_calls == 0
    assert report.retrieval["recall@1"] == 1.0
    assert report.generation == {}


def test_eval_runner_trace_keeps_transformed_query_and_refusal_signal():
    class _TransformedRefusalRag(_MockRAG):
        def ask(self, question, k=5):
            response = super().ask(question, k=k)
            response.update({
                "answer": "I cannot answer this question.",
                "transformed_query": [question, "broader retrieval query"],
                "query_transform_strategy": "step_back",
            })
            return response

    runner = EvalRunner(
        rag=_TransformedRefusalRag([]), llm_provider=None, prompt_manager=None,
        metrics=["recall"], k=1, execute_generation=True,
    )
    runner.run([_make_pair()])
    trace = runner.collected_sample_traces[0]

    assert trace["transformed_query"] == ["what is the law?", "broader retrieval query"]
    assert trace["query_transform_strategy"] == "step_back"
    assert trace["generation_attempted"] is True
    assert trace["answer_refusal"] is True
    assert trace["answer_refusal_reason"] == "insufficient_information"


def test_eval_runner_full_rag_generates_with_retrieval_metrics_only():
    """Pipeline execution is independent from which metrics are scored."""

    class _TracedRAG(_MockRAG):
        def ask(self, question, k=5):
            response = super().ask(question, k=k)
            response["pipeline_latency_ms"] = 25.0
            response["stage_timings_ms"] = {
                "query_embedding_ms": 2.0,
                "vector_search_ms": 3.0,
                "generation_ms": 20.0,
            }
            return response

    mock_rag = _TracedRAG(
        [Document(page_content="gold context text here", metadata={})],
        answer="generated answer",
    )
    runner = EvalRunner(
        rag=mock_rag,
        llm_provider=None,
        prompt_manager=None,
        metrics=["recall", "mrr"],
        k=1,
        execute_generation=True,
    )

    report = runner.run([_make_pair()])

    assert mock_rag.ask_calls == 1
    assert mock_rag.get_similar_calls == 0
    assert report.samples[0].predicted_answer == "generated answer"
    assert report.samples[0].stage_timings_ms["generation_ms"] == 20.0
    assert runner.collected_sample_traces[0]["generation_ms"] == 20.0
    assert report.retrieval["recall@1"] == 1.0
    assert report.generation == {}


def test_eval_runner_rejects_generation_metrics_when_generation_is_disabled():
    with pytest.raises(ValueError, match="Generation metrics require answer generation"):
        EvalRunner(
            rag=_MockRAG([]),
            llm_provider=None,
            prompt_manager=None,
            metrics=["answer_correctness"],
            execute_generation=False,
        )


def test_eval_runner_metric_validation():
    """Unknown metric raises ValueError in EvalRunner.__init__."""
    with pytest.raises(ValueError, match="Unknown evaluation metrics"):
        EvalRunner(
            rag=_MockRAG([]), llm_provider=None, prompt_manager=None, metrics=["bogus_metric"],
        )


def test_llm_judge_rating_is_supported_but_not_implicit_default():
    assert "llm_judge_rating" in ALL_METRICS
    assert "llm_judge_rating" not in DEFAULT_METRICS

    from Muffakir.MuffakirEvaluation import MuffakirEvaluation

    evaluator = MuffakirEvaluation.__new__(MuffakirEvaluation)
    evaluator.config = {"metrics": None}
    assert evaluator._resolve_metrics() == list(DEFAULT_METRICS)


def test_eval_runner_records_integer_ratings_and_decimal_mean(monkeypatch):
    ratings = iter([4, 5])

    class _MockRatingMetric:
        def __init__(self, **kwargs):
            pass

        def score(self, **kwargs):
            return next(ratings)

    monkeypatch.setattr("Evaluation.runner.LLMJudgeRatingMetric", _MockRatingMetric)
    runner = EvalRunner(
        rag=_MockRAG([], answer="generated"),
        llm_provider=None,
        prompt_manager=None,
        metrics=["llm_judge_rating"],
        k=1,
    )

    report = runner.run([_make_pair(), _make_pair(q="another question?")])

    assert [sample.generation.llm_judge_rating for sample in report.samples] == [4, 5]
    assert report.generation["llm_judge_rating"] == pytest.approx(4.5)
    assert [trace["metrics"]["llm_judge_rating"] for trace in runner.collected_sample_traces] == [4, 5]


def test_eval_runner_records_malformed_rating_as_sample_error(monkeypatch):
    class _BrokenRatingMetric:
        def __init__(self, **kwargs):
            pass

        def score(self, **kwargs):
            raise ValueError("invalid rating")

    monkeypatch.setattr("Evaluation.runner.LLMJudgeRatingMetric", _BrokenRatingMetric)
    runner = EvalRunner(
        rag=_MockRAG([], answer="generated"),
        llm_provider=None,
        prompt_manager=None,
        metrics=["llm_judge_rating"],
    )

    report = runner.run([_make_pair()])

    assert report.samples[0].error == "invalid rating"
    assert report.generation == {}


def test_eval_runner_double_retrieval_elimination(monkeypatch):
    """Both retrieval+generation: ask() called once, get_similar_documents NOT called."""
    class _MockMetric:
        def __init__(self, **kwargs):
            pass
        def score(self, **kwargs):
            return 1.0

    monkeypatch.setattr("Evaluation.runner.FaithfulnessMetric", _MockMetric)
    monkeypatch.setattr("Evaluation.runner.AnswerCorrectnessMetric", _MockMetric)

    mock_rag = _MockRAG([Document(page_content="gold context text here", metadata={})])
    pairs = [_make_pair()]

    runner = EvalRunner(
        rag=mock_rag, llm_provider=None, prompt_manager=None,
        metrics=["recall", "faithfulness"], k=1,
    )
    report = runner.run(pairs)

    assert mock_rag.ask_calls == 1
    assert mock_rag.get_similar_calls == 0
    assert "recall@1" in report.retrieval
    assert report.generation["faithfulness"] == 1.0


def test_eval_runner_error_handling():
    """Per-sample errors don't abort the run."""
    mock_rag = _MockRAG([])

    def flaky(query, k=5):
        raise RuntimeError("boom")

    mock_rag.get_similar_documents = flaky
    pairs = [_make_pair(), _make_pair(q="another question?")]

    runner = EvalRunner(
        rag=mock_rag, llm_provider=None, prompt_manager=None, metrics=["recall"], k=1,
    )
    report = runner.run(pairs)

    assert report.n_samples == 2
    assert all(s.error is not None for s in report.samples)


# ---------------------------------------------------------------------------
# EvalRunner — concurrency (max_workers)
# ---------------------------------------------------------------------------
class _SlowMockRAG(_MockRAG):
    """Like _MockRAG, but get_similar_documents() sleeps a per-query amount
    (indexed by call order) so later-submitted work can finish first — used to
    prove output ordering survives concurrent execution."""

    def __init__(self, docs, delays):
        super().__init__(docs)
        self._delays = delays
        self._call_index = 0
        self._lock = __import__("threading").Lock()

    def get_similar_documents(self, query, k=5):
        with self._lock:
            idx = self._call_index
            self._call_index += 1
        import time
        time.sleep(self._delays[idx])
        self.get_similar_calls += 1
        return self._docs[:k]


def test_eval_runner_max_workers_zero_raises():
    with pytest.raises(ValueError, match="max_workers"):
        EvalRunner(
            rag=_MockRAG([]), llm_provider=None, prompt_manager=None,
            metrics=["recall"], max_workers=0,
        )


def test_eval_runner_concurrent_preserves_output_order():
    """Later-submitted pairs finishing first must not reorder report.samples —
    it must always match the input pairs order."""
    docs = [Document(page_content="gold context text here", metadata={})]
    # First pair sleeps longest, so if execution were completion-ordered the
    # last pair would land first in `samples`.
    mock_rag = _SlowMockRAG(docs, delays=[0.1, 0.02, 0.0])
    pairs = [
        _make_pair(q="question 0"),
        _make_pair(q="question 1"),
        _make_pair(q="question 2"),
    ]

    runner = EvalRunner(
        rag=mock_rag, llm_provider=None, prompt_manager=None,
        metrics=["recall"], k=1, max_workers=4,
    )
    report = runner.run(pairs)

    assert [s.question for s in report.samples] == ["question 0", "question 1", "question 2"]


def test_eval_runner_concurrent_matches_sequential_metrics():
    """Same pairs through max_workers=1 vs max_workers=4 must yield identical
    aggregated metrics — concurrency must not change results, only ordering
    of execution."""
    docs = [Document(page_content="gold context text here", metadata={})]
    pairs = [_make_pair(q=f"question number {i}") for i in range(5)]

    seq_runner = EvalRunner(
        rag=_MockRAG(docs), llm_provider=None, prompt_manager=None,
        metrics=["recall"], k=1, max_workers=1,
    )
    par_runner = EvalRunner(
        rag=_MockRAG(docs), llm_provider=None, prompt_manager=None,
        metrics=["recall"], k=1, max_workers=4,
    )

    seq_report = seq_runner.run(pairs)
    par_report = par_runner.run(pairs)

    assert seq_report.retrieval == par_report.retrieval
    assert seq_report.n_samples == par_report.n_samples == 5


def test_eval_runner_concurrent_aborts_on_provider_error(monkeypatch):
    """A MuffakirError from any sample must still abort run() under concurrency."""
    from Muffakir.exceptions import ProviderError

    class _MockMetric:
        def __init__(self, **kwargs):
            pass

        def score(self, **kwargs):
            raise ProviderError("judge down", error_code="PROVIDER_ERROR")

    monkeypatch.setattr("Evaluation.runner.FaithfulnessMetric", _MockMetric)

    mock_rag = _MockRAG([Document(page_content="gold context text here", metadata={})])
    pairs = [_make_pair(q=f"question number {i}") for i in range(4)]

    runner = EvalRunner(
        rag=mock_rag, llm_provider=None, prompt_manager=None,
        metrics=["faithfulness"], k=1, max_workers=4,
    )
    with pytest.raises(ProviderError):
        runner.run(pairs)


def test_eval_runner_sequential_path_does_not_use_thread_pool(monkeypatch):
    """max_workers=1 (the default) must take the exact untouched sequential
    path — never construct a ThreadPoolExecutor."""
    def _boom(*a, **kw):
        raise AssertionError("ThreadPoolExecutor must not be used when max_workers=1")

    monkeypatch.setattr("Evaluation.runner.ThreadPoolExecutor", _boom)

    mock_rag = _MockRAG([Document(page_content="gold context text here", metadata={})])
    pairs = [_make_pair()]

    runner = EvalRunner(
        rag=mock_rag, llm_provider=None, prompt_manager=None, metrics=["recall"], k=1,
    )
    report = runner.run(pairs)
    assert report.n_samples == 1


# ---------------------------------------------------------------------------
# MuffakirEvaluation facade
# ---------------------------------------------------------------------------
def test_muffakir_evaluation_no_basicconfig():
    """P0 regression: instantiation must not mutate root logger handlers."""
    root_before = list(logging.getLogger().handlers)
    from Muffakir.MuffakirEvaluation import MuffakirEvaluation

    MuffakirEvaluation()
    assert list(logging.getLogger().handlers) == root_before


def test_muffakir_evaluation_duck_type_check():
    """P2: evaluate() accepts any object with ask(), not just MuffakirRAG."""
    from Muffakir.MuffakirEvaluation import MuffakirEvaluation

    ev = MuffakirEvaluation()
    with pytest.raises(TypeError, match="must expose an ask"):
        ev.evaluate(rag=object(), dataset=[])


# ---------------------------------------------------------------------------
# EvalRunner — trace_queue wiring (Trace package)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("rag_config", "expected_execute_generation"),
    [
        ({}, True),
        ({"pipeline_mode": "full_rag"}, True),
        ({"pipeline_mode": "retrieval_only"}, False),
    ],
)
def test_muffakir_evaluation_infers_generation_execution_from_pipeline_mode(
    monkeypatch, rag_config, expected_execute_generation
):
    from Muffakir.MuffakirEvaluation import MuffakirEvaluation
    from Evaluation.models import EvaluationReport

    captured = {}

    class _Rag:
        config = rag_config

        @staticmethod
        def ask(question, k=5):
            return {"answer": "answer"}

    class _FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, pairs):
            return EvaluationReport(n_samples=0, k=1, metrics_enabled=["recall"])

    monkeypatch.setattr(
        "Muffakir.dependency_validation.validate_evaluation_dependencies",
        lambda config, metrics: None,
    )
    monkeypatch.setattr("Evaluation.dataset.load_evaluation_dataset", lambda **kwargs: [])
    monkeypatch.setattr("Evaluation.runner.EvalRunner", _FakeRunner)

    evaluator = MuffakirEvaluation(config={"metrics": ["recall"]})
    evaluator.evaluate(rag=_Rag(), dataset=[])

    assert captured["execute_generation"] is expected_execute_generation
    assert captured["llm_provider"] is None


def test_muffakir_evaluation_retrieval_only_rejects_generation_metrics(monkeypatch):
    from Muffakir.MuffakirEvaluation import MuffakirEvaluation

    class _RetrievalRag:
        config = {"pipeline_mode": "retrieval_only"}

        @staticmethod
        def ask(question, k=5):
            raise AssertionError("retrieval-only must not generate")

    monkeypatch.setattr(
        "Muffakir.dependency_validation.validate_evaluation_dependencies",
        lambda config, metrics: None,
    )

    evaluator = MuffakirEvaluation(config={"metrics": ["answer_correctness"]})
    with pytest.raises(ValueError, match="retrieval_only.*generation metrics"):
        evaluator.evaluate(rag=_RetrievalRag(), dataset=[])


def test_trace_disabled_still_collects_in_memory_telemetry_without_a_queue():
    """ComposerReport timing must not depend on on-disk tracing being enabled."""
    mock_rag = _MockRAG([Document(page_content="gold context text here", metadata={})])
    runner = EvalRunner(rag=mock_rag, llm_provider=None, prompt_manager=None, metrics=["recall"], k=1)
    report = runner.run([_make_pair()])
    assert report.n_samples == 1
    assert len(runner.collected_sample_traces) == 1
    assert runner.collected_sample_traces[0]["pipeline_latency_ms"] is not None
    assert runner.trace_queue is None


def test_trace_queue_receives_one_record_per_sample():
    import queue as queue_module

    mock_rag = _MockRAG([Document(page_content="gold context text here", metadata={})])
    trace_queue = queue_module.Queue()
    runner = EvalRunner(
        rag=mock_rag, llm_provider=None, prompt_manager=None, metrics=["recall"], k=1,
        trace_queue=trace_queue, trial_id=3,
    )
    pairs = [_make_pair(q="question one"), _make_pair(q="question two")]
    runner.run(pairs)

    pushed = []
    while not trace_queue.empty():
        pushed.append(trace_queue.get_nowait())

    assert len(pushed) == 2
    assert all(kind == "sample" for kind, _ in pushed)
    assert {rec["question"] for _, rec in pushed} == {"question one", "question two"}
    assert all(rec["trial_id"] == 3 for _, rec in pushed)
    assert len(runner.collected_sample_traces) == 2


def test_retrieval_only_uses_telemetry_aware_path_and_separates_eval_overhead():
    from Muffakir.telemetry import RetrievalTelemetryResult, empty_stage_timings

    class _TelemetryRag:
        def get_similar_documents_with_trace(self, query, k=5):
            timings = empty_stage_timings()
            timings.update({
                "query_embedding_ms": 2.0,
                "vector_search_ms": 7.0,
                "rerank_ms": 3.0,
            })
            return RetrievalTelemetryResult(
                documents=[Document(page_content="gold context text here")],
                pipeline_latency_ms=12.5,
                stage_timings_ms=timings,
            )

    runner = EvalRunner(
        rag=_TelemetryRag(), llm_provider=None, prompt_manager=None,
        metrics=["recall"], k=1,
    )
    report = runner.run([_make_pair()])
    record = runner.collected_sample_traces[0]

    assert report.samples[0].pipeline_latency_ms == 12.5
    assert report.samples[0].evaluation_overhead_ms is not None
    assert record["pipeline_latency_ms"] == 12.5
    assert record["query_embedding_ms"] == 2.0
    assert record["vector_search_ms"] == 7.0
    assert record["rerank_ms"] == 3.0


def test_trace_record_includes_stage_timings_from_ask_response(monkeypatch):
    import queue as queue_module

    class _MockMetric:
        def __init__(self, **kwargs):
            pass

        def score(self, **kwargs):
            return 1.0

    monkeypatch.setattr("Evaluation.runner.FaithfulnessMetric", _MockMetric)

    class _RagWithTimings(_MockRAG):
        def ask(self, question, k=5):
            resp = super().ask(question, k=k)
            resp["stage_timings_ms"] = {
                "query_transform_ms": None,
                "retrieval_ms": 12.0,
                "rerank_ms": None,
                "generation_ms": 30.0,
                "hallucination_check_ms": None,
            }
            return resp

    mock_rag = _RagWithTimings([Document(page_content="gold context text here", metadata={})])
    trace_queue = queue_module.Queue()
    runner = EvalRunner(
        rag=mock_rag, llm_provider=None, prompt_manager=None, metrics=["recall", "faithfulness"], k=1,
        trace_queue=trace_queue, trial_id=0,
    )
    runner.run([_make_pair()])

    _, record = trace_queue.get_nowait()
    assert record["generation_ms"] == 30.0
    assert record["query_transform_ms"] is None


def test_trace_record_reflects_sample_error():
    trace_queue = __import__("queue").Queue()
    mock_rag = _MockRAG([])

    def flaky(query, k=5):
        raise RuntimeError("boom")

    mock_rag.get_similar_documents = flaky
    runner = EvalRunner(
        rag=mock_rag, llm_provider=None, prompt_manager=None, metrics=["recall"], k=1,
        trace_queue=trace_queue, trial_id=0,
    )
    runner.run([_make_pair()])

    _, record = trace_queue.get_nowait()
    assert record["error"] is not None


def test_sample_trace_records_web_search_used_when_context_source_is_web_search(monkeypatch):
    import queue as queue_module

    class _MockMetric:
        def __init__(self, **kwargs):
            pass

        def score(self, **kwargs):
            return 1.0

    monkeypatch.setattr("Evaluation.runner.FaithfulnessMetric", _MockMetric)

    class _RagWithWebSearchFallback(_MockRAG):
        def ask(self, question, k=5):
            resp = super().ask(question, k=k)
            resp["context_source"] = "web_search"
            resp["used_context"] = "Web result context"
            resp["web_sources"] = [{"title": "Source", "url": "https://example.test"}]
            resp["retrieved_documents"] = []
            return resp

    mock_rag = _RagWithWebSearchFallback([Document(page_content="gold context text here", metadata={})])
    trace_queue = queue_module.Queue()
    runner = EvalRunner(
        rag=mock_rag, llm_provider=None, prompt_manager=None, metrics=["recall", "faithfulness"], k=1,
        trace_queue=trace_queue, trial_id=0,
    )
    runner.run([_make_pair()])

    _, record = trace_queue.get_nowait()
    assert record["web_search_used"] is True
    assert record["context_source"] == "web_search"
    assert record["retrieved_candidates"][0]["page_content"] == "Web result context"
    assert record["retrieved_candidates"][0]["metadata"]["sources"][0]["title"] == "Source"
    assert runner.collected_sample_traces[0]["web_search_used"] is True


def test_sample_trace_web_search_used_false_by_default(monkeypatch):
    """A plain rag.ask() response with no context_source (vector-DB-only,
    no adaptive fallback) must record web_search_used=False, not crash or
    default to True."""
    import queue as queue_module

    class _MockMetric:
        def __init__(self, **kwargs):
            pass

        def score(self, **kwargs):
            return 1.0

    monkeypatch.setattr("Evaluation.runner.FaithfulnessMetric", _MockMetric)

    mock_rag = _MockRAG([Document(page_content="gold context text here", metadata={})])
    trace_queue = queue_module.Queue()
    runner = EvalRunner(
        rag=mock_rag, llm_provider=None, prompt_manager=None, metrics=["recall", "faithfulness"], k=1,
        trace_queue=trace_queue, trial_id=0,
    )
    runner.run([_make_pair()])

    _, record = trace_queue.get_nowait()
    assert record["web_search_used"] is False


def test_concurrent_per_sample_attribution_does_not_cross_contaminate(monkeypatch):
    """The core correctness case the thread-local design exists for: samples
    running concurrently through the SAME shared LLM/embedding provider must
    each get their OWN token usage / embedding time in their trace record,
    never another in-flight sample's."""
    import queue as queue_module
    import time as time_module

    from LLMProvider.usage import UsageCallbackHandler
    from Embedding.timing import EmbeddingTimingTracker

    class _MockMetric:
        def __init__(self, **kwargs):
            pass

        def score(self, **kwargs):
            return 1.0

    monkeypatch.setattr("Evaluation.runner.FaithfulnessMetric", _MockMetric)

    class _Provider:
        value = "openai"

    class _FakeLLMProvider:
        def __init__(self):
            self._usage_callback = UsageCallbackHandler()
            self.provider = _Provider()
            self.model = "fake-model"

        def get_sample_usage_totals(self, trial_id, sample_index):
            return self._usage_callback.get_sample_totals(trial_id, sample_index)

    class _FakeEmbeddingProvider:
        def __init__(self):
            self.timing = EmbeddingTimingTracker()

    class _ConcurrentRag:
        """Each ask() call sleeps briefly (forcing thread interleaving) then
        records usage/timing sized to the question -- proving attribution
        stays correct even when multiple samples are mid-flight at once."""

        def __init__(self):
            self.llm_provider = _FakeLLMProvider()
            self.embedding_provider = _FakeEmbeddingProvider()
            self.query_transformer = None

        def ask(self, question, k=5):
            n = int(question.split()[-1])  # distinguishable size per sample
            start = time_module.perf_counter()
            time_module.sleep(0.02)
            self.embedding_provider.timing.record((time_module.perf_counter() - start) * 1000.0)

            class _FakeResp:
                llm_output = {"token_usage": {"prompt_tokens": n, "completion_tokens": n, "total_tokens": 2 * n}}
                generations = [[]]

            self.llm_provider._usage_callback.on_llm_end(_FakeResp())
            return {
                "answer": "a",
                "retrieved_documents": [],
                "stage_timings_ms": {"retrieval_ms": 100.0},
            }

    rag = _ConcurrentRag()
    trace_queue = queue_module.Queue()
    pairs = [_make_pair(q=f"question {i}") for i in range(1, 6)]  # sizes 1..5

    runner = EvalRunner(
        rag=rag, llm_provider=None, prompt_manager=None, metrics=["recall", "faithfulness"], k=1,
        max_workers=5, trace_queue=trace_queue, trial_id=0,
    )
    runner.run(pairs)

    pushed = {}
    while not trace_queue.empty():
        _, rec = trace_queue.get_nowait()
        pushed[rec["question"]] = rec

    assert len(pushed) == 5
    for i in range(1, 6):
        rec = pushed[f"question {i}"]
        assert rec["token_usage"] == {"prompt_tokens": i, "completion_tokens": i, "total_tokens": 2 * i}


if __name__ == "__main__":
    import pytest as _pytest

    sys.exit(_pytest.main([__file__, "-v"]))
