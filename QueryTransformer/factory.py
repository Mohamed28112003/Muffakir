from typing import Optional, Any
from .base import BaseQueryTransformer
from .rewriter import QueryRewriter
from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import MuffakirPrompt

from .multi_query import MultiQueryExpansion
from .query_decomposition import QueryDecomposition
from .hyde import HyDEQueryTransformer
from .step_back import StepBackQueryTransformer

AVAILABLE_STRATEGIES = ("rewrite", "multi_query", "decomposition", "hyde", "step_back")

def create_query_transformer(
    strategy: str = "rewrite", 
    llm_provider: Optional[LLMProvider] = None,
    prompt_manager: Optional[MuffakirPrompt] = None,
    **kwargs: Any
) -> BaseQueryTransformer:
    """
    Factory function to initialize a Query Transformer strategy.

    Args:
        strategy (str): Strategy name ('rewrite', 'multi_query', 'decomposition', 'hyde', 'step_back', etc.). Default is 'rewrite'.
        llm_provider (LLMProvider): Instantiated LLM provider.
        prompt_manager (MuffakirPrompt, optional): Instantiated prompt manager.
        **kwargs: Strategy-specific parameters.

    Returns:
        BaseQueryTransformer: Strategy instance.

    Raises:
        ValueError: If ``llm_provider`` is missing or ``strategy`` is unknown.
    """
    if llm_provider is None:
        raise ValueError("llm_provider is required for query transformers.")

    strategy = str(strategy).lower().strip()

    if strategy in ("rewrite", "query_rewrite"):
        return QueryRewriter(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            **kwargs
        )

    elif strategy in ("multi_query", "multi_query_expansion", "query_expansion"):
        return MultiQueryExpansion(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            **kwargs
        )

    elif strategy in ("decomposition", "query_decomposition", "sub_query"):
        return QueryDecomposition(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            **kwargs
        )

    elif strategy in ("hyde", "hypothetical_document"):
        return HyDEQueryTransformer(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            **kwargs
        )

    elif strategy in ("step_back", "stepback"):
        return StepBackQueryTransformer(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            **kwargs
        )

    raise ValueError(
        f"Unknown query transformer strategy: '{strategy}'. "
        f"Available: {list(AVAILABLE_STRATEGIES)}"
    )
