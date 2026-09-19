from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from SyntheticData.models import QAPair
from Evaluation.models import (
    ALL_METRICS,
    DEFAULT_METRICS,
    GENERATION_METRICS,
    RETRIEVAL_METRICS,
    EvalSampleResult,
    EvaluationReport,
    GenerationScores,
    RetrievalScores,
)
from Evaluation.matching import build_relevance_list
from Evaluation.metrics.retrieval import score_retrieval
from Evaluation.metrics.generation import (
    AnswerCorrectnessMetric,
    FaithfulnessMetric,
    LLMJudgeRatingMetric,
)
from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import MuffakirPrompt
from Trace.context import set_current_sample, clear_current_sample
from Trace.cost import dedupe_llm_providers, aggregate_usage_and_cost
from .refusals import detect_answer_refusal

logger = logging.getLogger(__name__)


def _mean(values: List[float]) -> Optional[float]:
    """Mean of *values*, or None when there are no valid values to average.

    Returning None (instead of 0.0) lets ``_aggregate`` distinguish "no sample
    produced a value for this metric" from a genuine all-zero result.
    """
    if not values:
        return None
    return float(sum(values) / len(values))


class EvalRunner:
    """
    Evaluation loop over QAPair samples against an injected MuffakirRAG.

    Sequential by default (``max_workers=1``). Pass ``max_workers>1`` to
    evaluate samples concurrently via a thread pool — safe because each
    sample's LLM/retrieval calls are I/O-bound and every shared collaborator
    (``rag``, ``llm_provider``, the metric objects) is either stateless per
    call or has had its shared-state races fixed (see ``MuffakirRAG.ask()``).
    """

    def __init__(
        self,
        rag: Any,
        llm_provider: LLMProvider,
        prompt_manager: MuffakirPrompt,
        metrics: Sequence[str],
        k: int = 5,
        max_workers: int = 1,
        trace_queue: Optional[Any] = None,
        trial_id: Optional[int] = None,
        price_map: Optional[Any] = None,
        execute_generation: Optional[bool] = None,
        trace_attempt: Optional[int] = None,
        observation_queue: Optional[Any] = None,
    ):
        """
        trace_queue/trial_id/price_map: optional trace-capture wiring (see
        Trace package). When trace_queue is None (default), tracing is fully
        off and behavior is byte-for-byte identical to before it existed.
        When set, trial_id must also be set -- every pushed SampleTraceRecord
        needs it to know which trial it belongs to.

        execute_generation controls pipeline execution independently from
        generation-metric scoring. ``True`` always executes ``rag.ask()``;
        ``False`` executes retrieval only; ``None`` preserves the legacy
        metric-driven behavior for direct low-level callers.
        """
        self.rag = rag
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager
        self.metrics = [m.lower().strip() for m in metrics] or list(DEFAULT_METRICS)
        self.k = int(k)
        self.max_workers = int(max_workers)
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self.trace_queue = trace_queue
        self.trial_id = trial_id
        self.price_map = price_map
        self.trace_attempt = trace_attempt
        self.observation_queue = observation_queue
        self.collected_sample_traces: List[Dict[str, Any]] = []
        self._collected_traces_lock = threading.Lock()

        # Validate metric names early (fail-fast).
        unknown = [m for m in self.metrics if m not in ALL_METRICS]
        if unknown:
            raise ValueError(
                f"Unknown evaluation metrics: {unknown}. "
                f"Available: {list(ALL_METRICS)}"
            )

        self.need_retrieval = bool(set(self.metrics) & RETRIEVAL_METRICS)
        self.need_generation = bool(set(self.metrics) & GENERATION_METRICS)
        self.execute_generation = (
            self.need_generation
            if execute_generation is None
            else bool(execute_generation)
        )
        if not self.execute_generation and self.need_generation:
            raise ValueError(
                "Generation metrics require answer generation; "
                "execute_generation cannot be False."
            )

        self.faithfulness_metric: Optional[FaithfulnessMetric] = None
        self.correctness_metric: Optional[AnswerCorrectnessMetric] = None
        self.llm_judge_rating_metric: Optional[LLMJudgeRatingMetric] = None

        if "faithfulness" in self.metrics:
            self.faithfulness_metric = FaithfulnessMetric(
                llm_provider=self.llm_provider,
                prompt_manager=self.prompt_manager,
            )
        if "answer_correctness" in self.metrics:
            self.correctness_metric = AnswerCorrectnessMetric(
                llm_provider=self.llm_provider,
                prompt_manager=self.prompt_manager,
            )
        if "llm_judge_rating" in self.metrics:
            self.llm_judge_rating_metric = LLMJudgeRatingMetric(
                llm_provider=self.llm_provider,
                prompt_manager=self.prompt_manager,
            )

    def run(self, pairs: List[QAPair]) -> EvaluationReport:
        if self.max_workers <= 1:
            samples: List[EvalSampleResult] = []
            for idx, pair in enumerate(tqdm(pairs, desc="Evaluating", unit="sample")):
                samples.append(self._evaluate_one(pair, idx))
            return self._aggregate(samples)

        return self._run_concurrent(pairs)

    def _run_concurrent(self, pairs: List[QAPair]) -> EvaluationReport:
        results: List[Optional[EvalSampleResult]] = [None] * len(pairs)
        abort_error: Optional[Exception] = None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._evaluate_one, pair, i): i
                for i, pair in enumerate(pairs)
            }
            # Only the main thread ever touches pbar — no cross-thread contention.
            with tqdm(total=len(pairs), desc="Evaluating", unit="sample") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    pbar.update(1)
                    try:
                        results[idx] = future.result()
                    except Exception as exc:
                        if abort_error is None:
                            abort_error = exc
                            # Best-effort: skips not-yet-started futures. Already
                            # running futures can't be force-cancelled and are
                            # simply awaited to completion (their results are
                            # discarded below since we raise instead of returning).
                            for f in future_to_idx:
                                f.cancel()

        if abort_error is not None:
            raise abort_error
        return self._aggregate(results)

    def _evaluate_one(self, pair: QAPair, sample_index: int) -> EvalSampleResult:
        trace_trial_id = self.trial_id if self.trial_id is not None else -1
        # Always tag the sample. Telemetry is part of EvaluationReport/ComposerReport
        # even when no on-disk trace queue was requested.
        set_current_sample(trace_trial_id, sample_index)
        from Trace.observability import observation_context, observe_stage, provider_identity

        judge_provider, judge_model = provider_identity(self.llm_provider)

        observation_scope = observation_context(
            self.observation_queue,
            trial_id=trace_trial_id,
            sample_index=sample_index,
            attempt=self.trace_attempt,
        )
        observation_scope.__enter__()

        sample = EvalSampleResult(
            question=pair.question,
            gold_answer=pair.answer,
            gold_context=pair.context,
        )
        start = time.perf_counter()
        stage_timings_ms: Dict[str, Optional[float]] = {}
        retrieved_candidates: List[Dict[str, Any]] = []
        web_search_used = False
        pipeline_latency_ms: Optional[float] = None
        evaluation_overhead_ms: Optional[float] = None
        evaluation_start: Optional[float] = None
        transformed_query: Optional[Any] = None
        query_transform_strategy: Optional[str] = None

        try:
            retrieved_docs = []
            predicted_answer = ""
            retrieved_context = ""

            if self.execute_generation:
                # Full-RAG execution: call ask() once (it does its own retrieval
                # + generation internally), regardless of which metrics are
                # selected. Reuse the documents that ask() actually used so
                # retrieval scores reflect the real pipeline instead of a
                # separate retrieval call that may diverge.
                pipeline_start = time.perf_counter()
                response = self.rag.ask(pair.question, k=self.k) or {}
                measured_pipeline_ms = (time.perf_counter() - pipeline_start) * 1000.0
                pipeline_latency_ms = response.get("pipeline_latency_ms")
                if pipeline_latency_ms is None:
                    pipeline_latency_ms = measured_pipeline_ms
                predicted_answer = response.get("answer", "") or ""
                sample.predicted_answer = predicted_answer
                stage_timings_ms = response.get("stage_timings_ms") or {}
                transformed_query = response.get("transformed_query")
                query_transform_strategy = response.get("query_transform_strategy")
                web_search_used = response.get("context_source") == "web_search"

                docs_from_ask = response.get("retrieved_documents") or []
                if docs_from_ask:
                    # ask() returns page_content strings plus an index-aligned
                    # source_metadata list; wrap both for the Document interface
                    # expected by build_relevance_list, so chunk_id-based
                    # relevance matching (Evaluation/matching.py) still works
                    # in this combined retrieval+generation path instead of
                    # silently falling back to text-overlap-only matching.
                    metas_from_ask = response.get("source_metadata") or []
                    retrieved_docs = [
                        Document(
                            page_content=str(d),
                            metadata=metas_from_ask[i] if i < len(metas_from_ask) else {},
                        )
                        for i, d in enumerate(docs_from_ask)
                    ]
                    retrieved_context = "\n\n".join(str(d) for d in docs_from_ask)
                elif web_search_used and response.get("used_context"):
                    # Adaptive web search uses a synthesized text context rather
                    # than Document objects. Preserve that exact generation
                    # context plus its source list for the UI inspector.
                    retrieved_context = str(response.get("used_context") or "")
                    retrieved_candidates = [{
                        "rank": 1,
                        "page_content": retrieved_context,
                        "page_content_preview": retrieved_context[:200],
                        "metadata": {
                            "context_source": "web_search",
                            "sources": response.get("web_sources") or [],
                        },
                    }]
            elif self.need_retrieval:
                # Prefer the telemetry-aware path while retaining compatibility
                # with third-party RAG objects that only expose the legacy method.
                pipeline_start = time.perf_counter()
                traced_retrieval = getattr(self.rag, "get_similar_documents_with_trace", None)
                if traced_retrieval is not None:
                    retrieval_result = traced_retrieval(query=pair.question, k=self.k)
                    if isinstance(retrieval_result, dict):
                        retrieved_docs = retrieval_result.get("documents") or []
                        stage_timings_ms = retrieval_result.get("stage_timings_ms") or {}
                        pipeline_latency_ms = retrieval_result.get("pipeline_latency_ms")
                        transformed_query = retrieval_result.get("transformed_query")
                        query_transform_strategy = retrieval_result.get("query_transform_strategy")
                    else:
                        retrieved_docs = retrieval_result.documents or []
                        stage_timings_ms = retrieval_result.stage_timings_ms or {}
                        pipeline_latency_ms = retrieval_result.pipeline_latency_ms
                        transformed_query = getattr(retrieval_result, "transformed_query", None)
                        query_transform_strategy = getattr(retrieval_result, "query_transform_strategy", None)
                else:
                    retrieved_docs = self.rag.get_similar_documents(
                        query=pair.question,
                        k=self.k,
                    ) or []
                if pipeline_latency_ms is None:
                    pipeline_latency_ms = (time.perf_counter() - pipeline_start) * 1000.0

            sample.retrieved_previews = [
                (getattr(doc, "page_content", "") or "")[:200]
                for doc in retrieved_docs[: self.k]
            ]
            if retrieved_docs:
                retrieved_candidates = [
                    {
                        "rank": rank,
                        "page_content": getattr(doc, "page_content", "") or "",
                        "page_content_preview": (getattr(doc, "page_content", "") or "")[:200],
                        "metadata": dict(getattr(doc, "metadata", {}) or {}),
                    }
                    for rank, doc in enumerate(retrieved_docs[: self.k], start=1)
                ]

            evaluation_start = time.perf_counter()
            if self.need_retrieval:
                with observe_stage("retrieval_metrics", "evaluation"):
                    relevance = build_relevance_list(
                        retrieved_docs=retrieved_docs,
                        gold_context=pair.context,
                        gold_chunk_id=pair.chunk_id,
                        k=self.k,
                    )
                    scores = score_retrieval(relevance, self.metrics)
                sample.retrieval = RetrievalScores(
                    recall_at_k=scores.get("recall_at_k"),
                    precision_at_k=scores.get("precision_at_k"),
                    mrr=scores.get("mrr"),
                    ndcg_at_k=scores.get("ndcg_at_k"),
                )

            gen = GenerationScores()
            if self.faithfulness_metric is not None:
                with observe_stage(
                    "faithfulness_judge", "llm",
                    provider=judge_provider, model=judge_model,
                ):
                    gen.faithfulness = self.faithfulness_metric.score(
                        answer=predicted_answer,
                        context=retrieved_context,
                        query=pair.question,
                    )
            if self.correctness_metric is not None:
                with observe_stage(
                    "answer_correctness_judge", "llm",
                    provider=judge_provider, model=judge_model,
                ):
                    gen.answer_correctness = self.correctness_metric.score(
                        question=pair.question,
                        gold_answer=pair.answer,
                        predicted_answer=predicted_answer,
                    )
            if self.llm_judge_rating_metric is not None:
                with observe_stage(
                    "llm_judge_rating", "llm",
                    provider=judge_provider, model=judge_model,
                ):
                    gen.llm_judge_rating = self.llm_judge_rating_metric.score(
                        gold_answer=pair.answer,
                        predicted_answer=predicted_answer,
                    )
            sample.generation = gen
            evaluation_overhead_ms = (time.perf_counter() - evaluation_start) * 1000.0

        except Exception as e:
            from Muffakir.exceptions import MuffakirError
            from Trace.observability import record_exception_outcome

            if isinstance(e, MuffakirError):
                # Typed, systemic failure (provider auth, retrieval, generation, ...).
                # Every remaining sample would likely fail identically - surface it
                # instead of masking it as N per-sample failures.
                clear_current_sample()
                observation_scope.__exit__(None, None, None)
                raise
            logger.error(f"Sample evaluation failed: {e}", exc_info=True)
            sample.error = str(e)
            record_exception_outcome(
                e,
                recovery="continued",
                fallback_stage="sample_evaluation",
                fallback_component="evaluation",
            )

        sample.latency_ms = (time.perf_counter() - start) * 1000.0
        sample.pipeline_latency_ms = pipeline_latency_ms
        sample.evaluation_overhead_ms = evaluation_overhead_ms
        sample.stage_timings_ms = dict(stage_timings_ms)
        sample.transformed_query = transformed_query
        sample.query_transform_strategy = query_transform_strategy
        if self.execute_generation:
            sample.answer_refusal_reason = detect_answer_refusal(sample.predicted_answer)
            sample.answer_refusal = bool(sample.answer_refusal_reason)

        try:
            self._record_sample_trace(
                sample, sample_index, stage_timings_ms, retrieved_candidates, web_search_used
            )
        finally:
            clear_current_sample()
            observation_scope.__exit__(None, None, None)

        return sample

    def _record_sample_trace(
        self,
        sample: EvalSampleResult,
        sample_index: int,
        stage_timings_ms: Dict[str, Optional[float]],
        retrieved_candidates: List[Dict[str, Any]],
        web_search_used: bool = False,
    ) -> None:
        """Collect one sample record and optionally persist it to the trace queue."""
        from Trace.models import SampleTraceRecord

        query_embedding_ms = stage_timings_ms.get("query_embedding_ms")
        vector_search_ms = stage_timings_ms.get("vector_search_ms")
        retrieval_ms = stage_timings_ms.get("retrieval_ms")
        embedding_provider = getattr(self.rag, "embedding_provider", None)
        # Compatibility fallback for older ask() implementations that expose
        # only an aggregate retrieval_ms value.
        if query_embedding_ms is None and retrieval_ms is not None and embedding_provider is not None:
            query_embedding_ms = embedding_provider.timing.get_sample_total_ms(
                self.trial_id if self.trial_id is not None else -1, sample_index
            )
            vector_search_ms = max(retrieval_ms - query_embedding_ms, 0.0)

        providers = dedupe_llm_providers(self.rag, self.llm_provider)

        def _sample_usage_getter(provider: Any) -> Dict[str, int]:
            get_sample_usage_totals = getattr(provider, "get_sample_usage_totals", None)
            if get_sample_usage_totals is None:
                return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            return get_sample_usage_totals(
                self.trial_id if self.trial_id is not None else -1, sample_index
            )

        token_usage, cost_usd = aggregate_usage_and_cost(providers, _sample_usage_getter, self.price_map)

        metrics: Dict[str, float] = {}
        for name, value in (
            ("recall_at_k", sample.retrieval.recall_at_k),
            ("precision_at_k", sample.retrieval.precision_at_k),
            ("mrr", sample.retrieval.mrr),
            ("ndcg_at_k", sample.retrieval.ndcg_at_k),
            ("faithfulness", sample.generation.faithfulness),
            ("answer_correctness", sample.generation.answer_correctness),
            ("llm_judge_rating", sample.generation.llm_judge_rating),
        ):
            if value is not None:
                metrics[name] = value

        record = SampleTraceRecord(
            trial_id=self.trial_id if self.trial_id is not None else -1,
            sample_index=sample_index,
            question=sample.question,
            transformed_query=sample.transformed_query,
            query_transform_strategy=sample.query_transform_strategy,
            gold_answer=sample.gold_answer,
            gold_context=sample.gold_context,
            predicted_answer=sample.predicted_answer,
            context_source="web_search" if web_search_used else "vector_db",
            latency_ms=sample.latency_ms,
            pipeline_latency_ms=sample.pipeline_latency_ms,
            evaluation_overhead_ms=sample.evaluation_overhead_ms,
            query_transform_ms=stage_timings_ms.get("query_transform_ms"),
            query_embedding_ms=query_embedding_ms,
            vector_search_ms=vector_search_ms,
            rerank_ms=stage_timings_ms.get("rerank_ms"),
            relevance_check_ms=stage_timings_ms.get("relevance_check_ms"),
            web_search_ms=stage_timings_ms.get("web_search_ms"),
            generation_ms=stage_timings_ms.get("generation_ms"),
            hallucination_check_ms=stage_timings_ms.get("hallucination_check_ms"),
            retrieved_candidates=retrieved_candidates,
            metrics=metrics,
            token_usage=token_usage,
            cost_usd=cost_usd,
            error=sample.error,
            web_search_used=web_search_used,
            generation_attempted=self.execute_generation,
            answer_refusal=sample.answer_refusal,
            answer_refusal_reason=sample.answer_refusal_reason,
        )
        record_dict = record.to_dict()
        if self.trace_queue is not None:
            self.trace_queue.put(("sample", record_dict))
        with self._collected_traces_lock:
            self.collected_sample_traces.append(record_dict)

    def _aggregate(self, samples: List[EvalSampleResult]) -> EvaluationReport:
        retrieval: Dict[str, float] = {}
        generation: Dict[str, float] = {}

        def _put(target: Dict[str, float], key: str, values: List[Optional[float]]) -> None:
            # Omit the key entirely when there are no valid values, rather than
            # reporting a misleading 0.0 indistinguishable from a genuine
            # all-zero result (e.g. every sample errored before scoring).
            mean = _mean([v for v in values if v is not None])
            if mean is not None:
                target[key] = mean

        if "recall" in self.metrics:
            _put(retrieval, f"recall@{self.k}", [s.retrieval.recall_at_k for s in samples])
        if "precision" in self.metrics:
            _put(retrieval, f"precision@{self.k}", [s.retrieval.precision_at_k for s in samples])
        if "mrr" in self.metrics:
            _put(retrieval, "mrr", [s.retrieval.mrr for s in samples])
        if "ndcg" in self.metrics:
            _put(retrieval, f"ndcg@{self.k}", [s.retrieval.ndcg_at_k for s in samples])
        if "faithfulness" in self.metrics:
            _put(generation, "faithfulness", [s.generation.faithfulness for s in samples])
        if "answer_correctness" in self.metrics:
            _put(generation, "answer_correctness", [s.generation.answer_correctness for s in samples])
        if "llm_judge_rating" in self.metrics:
            _put(generation, "llm_judge_rating", [s.generation.llm_judge_rating for s in samples])

        return EvaluationReport(
            n_samples=len(samples),
            k=self.k,
            metrics_enabled=list(self.metrics),
            retrieval=retrieval,
            generation=generation,
            samples=samples,
        )
