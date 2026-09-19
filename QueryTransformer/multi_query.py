import logging
import re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .base import BaseQueryTransformer
from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import MuffakirPrompt


class QueryExpansionOutput(BaseModel):
    expanded_queries: List[str] = Field(
        description="A list of 3 to 5 distinct variations of the user query optimized for vector search."
    )


class MultiQueryExpansion(BaseQueryTransformer):
    """
    Multi-Query Expansion Strategy.

    Expands a single user query into 3 to 5 distinct variations using synonyms,
    different phrasing, and related terminology to optimize vector database retrieval.

    Features:
    - Temperature = 0.2 for slight creative variance in synonyms.
    - Pydantic structured output with robust fallback parser.
    - Ensures the original query is always preserved as the first element.
    - Prompts loaded directly from MuffakirPrompt (PromptManager).
    - Fully bilingual (Arabic & English).
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: Optional[MuffakirPrompt] = None,
        prompt_key: str = "multi_query_expansion",
        temperature: float = 0.2
    ):
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager or MuffakirPrompt(language="ar")
        self.prompt_key = prompt_key
        self.temperature = temperature

        self.logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "multi_query"

    def transform(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[str]:
        if not query or not str(query).strip():
            return [query] if query else []

        try:
            llm = self.llm_provider.get_llm()

            # Creative variance WITHOUT mutating shared LLM state: prefer
            # ``bind`` (LangChain standard); fall back to unbound LLM.
            try:
                llm = llm.bind(temperature=self.temperature)
            except (AttributeError, TypeError):
                pass

            prompt_template = self.prompt_manager.get_prompt(self.prompt_key)

            query_with_context = self._build_query_with_context(
                query, conversation_history, latest_label="Latest Query"
            )

            formatted_prompt = prompt_template.format(original_query=query_with_context)

            # Try structured output first (OpenAI / Pydantic supported models)
            expanded_list = []
            if hasattr(llm, "with_structured_output"):
                try:
                    structured_llm = llm.with_structured_output(QueryExpansionOutput)
                    res = structured_llm.invoke(formatted_prompt)
                    if isinstance(res, QueryExpansionOutput):
                        expanded_list = res.expanded_queries
                    elif isinstance(res, dict) and "expanded_queries" in res:
                        expanded_list = res["expanded_queries"]
                except Exception as ex:
                    self.logger.debug(f"Structured output fallback: {ex}")

            # If structured output was not supported or returned empty, call standard LLM & parse
            if not expanded_list:
                response = llm.invoke(formatted_prompt)
                raw_text = response.content if hasattr(response, "content") else str(response)
                expanded_list = self._parse_lines(raw_text)

            # Clean and filter empty strings
            cleaned_queries = [q.strip() for q in expanded_list if q and q.strip()]

            # Rule: Ensure original query is the first item in the list
            if query not in cleaned_queries:
                cleaned_queries.insert(0, query)
            else:
                # Move original query to index 0 if present elsewhere
                cleaned_queries.remove(query)
                cleaned_queries.insert(0, query)

            return cleaned_queries

        except Exception as e:
            self.logger.warning(f"MultiQueryExpansion error: {e}. Returning original query list.")
            return [query]

    @staticmethod
    def _parse_lines(text: str) -> List[str]:
        """Parse numbered/bulleted lines or line breaks into a list of queries."""
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove leading numbers or bullet points (e.g. "1. ", "- ", "* ")
            cleaned = re.sub(r'^(?:\d+[\.\)]|[-*•])\s*', '', line).strip()
            if cleaned:
                lines.append(cleaned)
        return lines
