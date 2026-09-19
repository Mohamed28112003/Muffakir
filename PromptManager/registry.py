"""Prompt contracts and workflow relevance rules.

This module deliberately has no dependency on Composer or ComposerUI so the
same rules can be used by the library, API validation, and frontend endpoint.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PromptSpec:
    key: str
    label: str
    category: str
    description: str
    stage: str
    required_variables: Tuple[str, ...]
    allowed_variables: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["required_variables"] = list(self.required_variables)
        value["allowed_variables"] = list(self.allowed_variables)
        return value


def _spec(
    key: str,
    label: str,
    category: str,
    description: str,
    stage: str,
    variables: Sequence[str],
) -> PromptSpec:
    names = tuple(variables)
    return PromptSpec(key, label, category, description, stage, names, names)


# Registry order is also the stable display order returned to clients.
PROMPT_SPECS: Dict[str, PromptSpec] = {
    item.key: item
    for item in (
        _spec("QA", "QA Pair Generation", "Dataset", "Creates grounded evaluation questions and answers from document chunks.", "Evaluation dataset generation", ["context"]),
        _spec("query_rewrite", "Query Rewrite", "Query transformation", "Rewrites the user's query before retrieval.", "Query transformation", ["original_query"]),
        _spec("multi_query_expansion", "Multi-query Expansion", "Query transformation", "Produces multiple retrieval queries from the original question.", "Query transformation", ["original_query"]),
        _spec("query_decomposition", "Query Decomposition", "Query transformation", "Splits a complex question into retrieval sub-questions.", "Query transformation", ["original_query"]),
        _spec("hyde", "HyDE", "Query transformation", "Generates a hypothetical answer used to improve retrieval.", "Query transformation", ["original_query"]),
        _spec("step_back", "Step-back Query", "Query transformation", "Creates a broader question before retrieval.", "Query transformation", ["original_query"]),
        _spec("reranker_scoring", "LLM Reranker", "Retrieval", "Scores the relevance of each retrieved document.", "Reranking", ["query", "document"]),
        _spec("generation", "Answer Generation", "Generation", "Generates the final answer from the selected context.", "Answer generation", ["context", "question"]),
        _spec("context_relevance", "Adaptive Context Relevance", "Generation", "Decides whether local context is sufficient or web fallback is needed.", "Adaptive relevance check", ["question", "context"]),
        _spec("hallucination_check_prompt", "Answer Text Cleaner", "Generation", "Cleans the generated answer in the default hallucination-processing stage.", "Hallucination processing", ["answer"]),
        _spec("answer_correctness", "Answer Correctness Judge", "Evaluation", "Scores a predicted answer against the expected answer.", "Evaluation judge", ["question", "gold_answer", "predicted_answer"]),
        _spec("llm_judge_rating", "LLM Judge Rating", "Evaluation", "Rates semantic correctness against the reference answer on a 1–5 scale.", "Evaluation judge", ["gold_answer", "predicted_answer"]),
        _spec("context_grounding", "Context Grounding Judge", "Evaluation", "Checks whether the generated answer is supported by its context.", "Faithfulness evaluation", ["context", "answer"]),
        # Public library prompts that are not exposed by the Composer workflow.
        _spec("MCQ", "Multiple-choice Generation", "Library", "Creates multiple-choice questions from context.", "Synthetic data", ["context"]),
        _spec("history_classification_prompt", "History Classification", "Library", "Classifies whether a query depends on conversation history.", "Conversation", ["conversation_history", "new_query"]),
        _spec("history_query_prompt", "History Query", "Library", "Answers a query using recent conversation history.", "Conversation", ["last_query", "last_response", "new_query"]),
        _spec("summary_generation", "Summary Generation", "Library", "Summarizes input text.", "Summary", ["text"]),
        _spec("question_generation", "Question Generation", "Library", "Creates questions from input text.", "Question generation", ["text"]),
        _spec("search_query", "Search Query", "Library", "Rewrites a query for web search.", "Web search", ["original_query"]),
        _spec("faithfulness_extraction", "Faithfulness Extraction", "Library", "Extracts claims from an answer.", "Legacy evaluation", ["answer"]),
        _spec("faithfulness_verification", "Faithfulness Verification", "Library", "Checks a claim against context.", "Legacy evaluation", ["claim", "context"]),
        _spec("youtube", "YouTube Query", "Library", "Rewrites a query for YouTube search.", "YouTube", ["original_query"]),
        _spec("mindmap", "Mind Map", "Library", "Creates a mind-map representation from context.", "Mind map", ["context"]),
        _spec("summary_map", "Summary Map", "Library", "Summarizes one document section.", "Summary", ["text"]),
        _spec("summary_combine", "Summary Combine", "Library", "Combines section summaries.", "Summary", ["text"]),
        _spec("summary_direct", "Direct Summary", "Library", "Directly summarizes input text.", "Summary", ["text"]),
        _spec("summary_check", "Summary Check", "Library", "Evaluates a generated summary.", "Summary", ["original_text", "summary"]),
    )
}


QUERY_TRANSFORM_PROMPTS = {
    "rewrite": "query_rewrite",
    "multi_query": "multi_query_expansion",
    "multi_query_expansion": "multi_query_expansion",
    "decomposition": "query_decomposition",
    "query_decomposition": "query_decomposition",
    "hyde": "hyde",
    "step_back": "step_back",
}

LLM_RERANKERS = {"llm", "llm_reranker", "llm-based", "llm_based"}


def _values(search_space: Mapping[str, Any], key: str) -> Iterable[Any]:
    value = search_space.get(key) or []
    return value if isinstance(value, (list, tuple, set)) else [value]


def resolve_prompt_keys(
    *,
    pipeline_mode: str = "full_rag",
    retrieval_source: str = "vector_db",
    adaptive_web_search: bool = False,
    eval_dataset_mode: str = "auto",
    search_space: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Sequence[str]] = None,
    hallucination_check: bool = True,
    hallucination_method: str = "text_cleaner",
) -> List[str]:
    """Return the union of prompts executable by any selected combination."""
    search_space = search_space or {}
    requested = set()

    if eval_dataset_mode == "auto":
        requested.add("QA")

    is_web_only = retrieval_source == "web_search_only"
    is_full_rag = pipeline_mode == "full_rag"

    if not is_web_only:
        for strategy in _values(search_space, "query_expansion"):
            prompt_key = QUERY_TRANSFORM_PROMPTS.get(str(strategy).strip().lower())
            if prompt_key:
                requested.add(prompt_key)
        if any(str(value).strip().lower() in LLM_RERANKERS for value in _values(search_space, "reranking")):
            requested.add("reranker_scoring")

    if is_full_rag:
        requested.add("generation")
        if not is_web_only:
            if adaptive_web_search:
                requested.add("context_relevance")
            if hallucination_check and str(hallucination_method).lower() in {"text_cleaner", "default", "llm"}:
                requested.add("hallucination_check_prompt")

        normalized_metrics = {str(metric).strip().lower() for metric in (metrics or [])}
        if "answer_correctness" in normalized_metrics:
            requested.add("answer_correctness")
        if "llm_judge_rating" in normalized_metrics:
            requested.add("llm_judge_rating")
        if "faithfulness" in normalized_metrics:
            requested.add("context_grounding")

    return [key for key in PROMPT_SPECS if key in requested]
