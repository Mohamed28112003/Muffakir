import logging
from typing import List, Optional, Dict, Any
from .base import BaseQueryTransformer
from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import MuffakirPrompt


class QueryRewriter(BaseQueryTransformer):
    """
    Query Rewriting Strategy.

    Transforms a noisy, contextual, or poorly phrased user message into 
    a clean, direct, keyword-rich standalone query optimized for vector database retrieval.

    Features:
    - Temperature = 0.0 for predictable, deterministic rewrites.
    - Prompts loaded directly from MuffakirPrompt (PromptManager).
    - Fully bilingual (Arabic & English).
    - Robust fallback to original query on exception.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: Optional[MuffakirPrompt] = None,
        prompt_key: str = "query_rewrite"
    ):
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager or MuffakirPrompt(language="ar")
        self.prompt_key = prompt_key
        
        self.logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "rewrite"

    def transform(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        if not query or not str(query).strip():
            return ""

        try:
            llm = self.llm_provider.get_llm()
            
            # Deterministic rewriting WITHOUT mutating shared LLM state:
            # prefer ``bind`` (LangChain standard); fall back to unbound LLM.
            try:
                llm = llm.bind(temperature=0.0)
            except (AttributeError, TypeError):
                pass

            # Get template directly from PromptManager
            prompt_template = self.prompt_manager.get_prompt(self.prompt_key)

            # Build query string including conversation history context if provided
            query_with_context = self._build_query_with_context(
                query, conversation_history, latest_label="Latest Query"
            )

            # Format template with original query
            formatted_prompt = prompt_template.format(original_query=query_with_context)

            response = llm.invoke(formatted_prompt)
            rewritten = response.content if hasattr(response, "content") else str(response)

            rewritten_clean = str(rewritten).strip().strip('"').strip("'")
            return rewritten_clean or query

        except Exception as e:
            self.logger.warning(f"QueryRewriter error: {e}. Returning original query.")
            return query
