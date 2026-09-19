import logging
import re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .base import BaseQueryTransformer
from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import MuffakirPrompt


class SubQueryDecomposition(BaseModel):
    sub_queries: List[str] = Field(
        description="A list of 2 to 4 distinct, standalone sub-queries extracted from the main complex query."
    )
    rationale: str = Field(
        description="A brief explanation of why this query needed to be decomposed."
    )


class QueryDecomposition(BaseQueryTransformer):
    """
    Query Decomposition Strategy.

    Splits a complex, compound, or multi-step user query into distinct, 
    independent sub-queries optimized for isolated vector database retrieval.

    Features:
    - Temperature = 0.0 for deterministic, precise splitting logic.
    - Pydantic structured output with robust fallback line parser.
    - Preserves simple queries as single-item lists without over-decomposing.
    - Prompts loaded directly from MuffakirPrompt (PromptManager).
    - Fully bilingual (Arabic & English).
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: Optional[MuffakirPrompt] = None,
        prompt_key: str = "query_decomposition"
    ):
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager or MuffakirPrompt(language="ar")
        self.prompt_key = prompt_key

        self.logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "decomposition"

    def transform(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[str]:
        if not query or not str(query).strip():
            return [query] if query else []

        try:
            llm = self.llm_provider.get_llm()

            # Deterministic decomposition WITHOUT mutating shared LLM state:
            # prefer ``bind`` (LangChain standard); fall back to unbound LLM.
            try:
                llm = llm.bind(temperature=0.0)
            except (AttributeError, TypeError):
                pass

            prompt_template = self.prompt_manager.get_prompt(self.prompt_key)

            query_with_context = self._build_query_with_context(
                query, conversation_history, latest_label="Complex Query"
            )

            formatted_prompt = prompt_template.format(original_query=query_with_context)

            # Try structured output first
            sub_queries = []
            if hasattr(llm, "with_structured_output"):
                try:
                    structured_llm = llm.with_structured_output(SubQueryDecomposition)
                    res = structured_llm.invoke(formatted_prompt)
                    if isinstance(res, SubQueryDecomposition):
                        sub_queries = res.sub_queries
                    elif isinstance(res, dict) and "sub_queries" in res:
                        sub_queries = res["sub_queries"]
                except Exception as ex:
                    self.logger.debug(f"Structured output fallback: {ex}")

            # Standard invocation fallback
            if not sub_queries:
                response = llm.invoke(formatted_prompt)
                raw_text = response.content if hasattr(response, "content") else str(response)
                sub_queries = self._parse_lines(raw_text)

            cleaned_sub_queries = [q.strip() for q in sub_queries if q and q.strip()]

            return cleaned_sub_queries or [query]

        except Exception as e:
            self.logger.warning(f"QueryDecomposition error: {e}. Returning original query.")
            return [query]

    @staticmethod
    def _parse_lines(text: str) -> List[str]:
        """Parse numbered/bulleted lines or line breaks into a list of sub-queries."""
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("Rationale:") or line.startswith("التبرير:"):
                continue
            cleaned = re.sub(r'^(?:\d+[\.\)]|[-*•])\s*', '', line).strip()
            if cleaned and not cleaned.startswith("{") and not cleaned.startswith("}"):
                lines.append(cleaned)
        return lines
