from typing import List, Optional, Union, Dict, Any
from .base import BaseQueryTransformer
from .factory import create_query_transformer
from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import MuffakirPrompt

class QueryTransformer:
    """
    Main Orchestrator class for query transformation in Muffakir RAG.

    Supports pluggable transformation strategies (Query Rewriting, Multi-Query Expansion, etc.)
    and integrates directly with LLMProvider and MuffakirPrompt.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: Optional[MuffakirPrompt] = None,
        prompt: str = "query_rewrite",  # legacy/ignored — kept for backward compatibility
        strategy: Union[str, BaseQueryTransformer] = "rewrite",
        strategy_config: Optional[Dict[str, Any]] = None
    ):
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager or MuffakirPrompt(language="ar")
        
        if strategy_config is None:
            strategy_config = {}

        if isinstance(strategy, str):
            self.transformer = create_query_transformer(
                strategy=strategy,
                llm_provider=self.llm_provider,
                prompt_manager=self.prompt_manager,
                **strategy_config
            )
        else:
            self.transformer = strategy

    def transform_query(
        self, 
        original_query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Union[str, List[str]]:
        """
        Transform the input query into an optimized representation for vector search.
        """
        return self.transformer.transform(
            query=original_query, 
            conversation_history=conversation_history
        )

    @property
    def name(self) -> str:
        return self.transformer.name
