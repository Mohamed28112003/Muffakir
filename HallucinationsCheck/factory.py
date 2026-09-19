from typing import Optional, Any
from .base import BaseHallucinationChecker
from Muffakir.optional_dependencies import require_optional_dependency

AVAILABLE_METHODS = ("context_grounding", "nli", "semantic_similarity", "text_cleaner")


def create_hallucination_checker(
    method: str = "text_cleaner",
    llm_provider: Optional[Any] = None,
    prompt_manager: Optional[Any] = None,
    embedding_provider: Optional[Any] = None,
    **kwargs: Any
) -> BaseHallucinationChecker:
    """
    Factory function to instantiate a Hallucination Checker strategy.

    Args:
        method (str): Checking strategy name. Options:
            - 'context_grounding': LLM-as-Judge via Pydantic structured output.
              Checks if answer is fully supported by context.
            - 'nli': NLI cross-encoder model (ENTAIL/CONTRADICT/NEUTRAL).
              No LLM API calls required.
            - 'semantic_similarity': Cosine similarity between answer and context.
              Uses EmbeddingProvider; no LLM required.
            - 'text_cleaner': LLM post-processes and cleans the answer.
              Preserves the original HallucinationsCheck behavior.
        llm_provider: LLMProvider instance (for context_grounding and text_cleaner).
        prompt_manager: MuffakirPrompt instance (for text_cleaner).
        embedding_provider: EmbeddingProvider instance (for semantic_similarity).
        **kwargs: Additional strategy-specific options.

    Returns:
        BaseHallucinationChecker: Instantiated checker strategy.

    Raises:
        ValueError: If ``method`` is empty or not a recognized strategy name.
    """
    if not isinstance(method, str) or not method.strip():
        raise ValueError(
            f"method must be a non-empty string. Available: {list(AVAILABLE_METHODS)}"
        )

    method_clean = method.lower().strip()

    if method_clean in ("context_grounding", "grounding", "llm_judge"):
        from .context_grounding import ContextGroundingChecker
        return ContextGroundingChecker(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager
        )

    elif method_clean in ("nli", "nli_faithfulness", "natural_language_inference"):
        require_optional_dependency("local")
        from .nli import NLIChecker
        return NLIChecker(
            model_name=kwargs.get("model_name", "cross-encoder/nli-deberta-v3-small"),
            entailment_threshold=kwargs.get("entailment_threshold", 0.7),
            contradiction_threshold=kwargs.get("contradiction_threshold", 0.7)
        )

    elif method_clean in ("semantic_similarity", "cosine", "embedding"):
        require_optional_dependency("local")
        from .semantic_similarity import SemanticSimilarityChecker
        return SemanticSimilarityChecker(
            embedding_provider=embedding_provider,
            model_name=kwargs.get("model_name", "mohamed2811/Muffakir_Embedding"),
            similarity_threshold=kwargs.get("similarity_threshold", 0.3)
        )

    elif method_clean in ("text_cleaner", "cleaner", "post_processor"):
        from .text_cleaner import TextCleanerChecker
        return TextCleanerChecker(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            prompt_key=kwargs.get("prompt_key", "hallucination_check_prompt")
        )

    raise ValueError(
        f"Unknown hallucination checker method: '{method}'. "
        "Available: 'context_grounding', 'nli', 'semantic_similarity', 'text_cleaner'."
    )
