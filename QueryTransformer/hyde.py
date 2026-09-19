import logging
from typing import List, Optional, Dict, Any
from .base import BaseQueryTransformer
from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import MuffakirPrompt


class HyDEQueryTransformer(BaseQueryTransformer):
    """
    Hypothetical Document Embeddings (HyDE) Strategy.

    Generates a plausible, hypothetical answer or document snippet for the user query.
    The hypothetical document is then used as the query text for vector database 
    embedding search, bridging the structural gap between questions and answer passages.

    Features:
    - Temperature = 0.3 for natural document formatting and phrasing variance.
    - Prompts loaded directly from MuffakirPrompt (PromptManager).
    - Fully bilingual (Arabic & English).
    - Robust fallback to original query string on exception.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: Optional[MuffakirPrompt] = None,
        prompt_key: str = "hyde",
        temperature: float = 0.3
    ):
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager or MuffakirPrompt(language="ar")
        self.prompt_key = prompt_key
        self.temperature = temperature

        self.logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "hyde"

    def transform(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        if not query or not str(query).strip():
            return ""

        try:
            llm = self.llm_provider.get_llm()

            # Natural document phrasing WITHOUT mutating shared LLM state:
            # prefer ``bind`` (LangChain standard); fall back to unbound LLM.
            try:
                llm = llm.bind(temperature=self.temperature)
            except (AttributeError, TypeError):
                pass

            prompt_template = self.prompt_manager.get_prompt(self.prompt_key)

            query_with_context = self._build_query_with_context(
                query, conversation_history, latest_label="Query"
            )

            formatted_prompt = prompt_template.format(original_query=query_with_context)

            response = llm.invoke(formatted_prompt)
            hypothetical_doc = response.content if hasattr(response, "content") else str(response)

            cleaned_doc = str(hypothetical_doc).strip()
            return cleaned_doc or query

        except Exception as e:
            self.logger.warning(f"HyDEQueryTransformer error: {e}. Returning original query.")
            return query
