import logging
import re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .base import BaseQueryTransformer
from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import MuffakirPrompt


class StepBackQueryOutput(BaseModel):
    step_back_query: str = Field(
        description="A broader, generic, or high-level foundational question derived from the user's specific query."
    )


class StepBackQueryTransformer(BaseQueryTransformer):
    """
    Step-Back Prompting Strategy.

    Takes a highly technical or specific user query and generates a broader, high-level
    'step-back' question to retrieve foundational domain concepts along with specific facts.

    Features:
    - Temperature = 0.0 for deterministic abstraction logic.
    - Pydantic structured output with fallback parser.
    - Returns a list containing both the original specific query and the step-back query:
      [original_query, step_back_query] so vector retrieval searches for both concepts.
    - Prompts loaded directly from MuffakirPrompt (PromptManager).
    - Fully bilingual (Arabic & English).
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: Optional[MuffakirPrompt] = None,
        prompt_key: str = "step_back"
    ):
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager or MuffakirPrompt(language="ar")
        self.prompt_key = prompt_key

        self.logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "step_back"

    def transform(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[str]:
        if not query or not str(query).strip():
            return [query] if query else []

        try:
            llm = self.llm_provider.get_llm()

            # Deterministic abstraction WITHOUT mutating shared LLM state:
            # prefer ``bind`` (LangChain standard); fall back to unbound LLM.
            try:
                llm = llm.bind(temperature=0.0)
            except (AttributeError, TypeError):
                pass

            prompt_template = self.prompt_manager.get_prompt(self.prompt_key)

            query_with_context = self._build_query_with_context(
                query, conversation_history, latest_label="Specific Query"
            )

            formatted_prompt = prompt_template.format(original_query=query_with_context)

            step_back_str = ""

            # Try structured output first
            if hasattr(llm, "with_structured_output"):
                try:
                    structured_llm = llm.with_structured_output(StepBackQueryOutput)
                    res = structured_llm.invoke(formatted_prompt)
                    if isinstance(res, StepBackQueryOutput):
                        step_back_str = res.step_back_query
                    elif isinstance(res, dict) and "step_back_query" in res:
                        step_back_str = res["step_back_query"]
                except Exception as ex:
                    self.logger.debug(f"Structured output fallback: {ex}")

            # Fallback standard invocation
            if not step_back_str:
                response = llm.invoke(formatted_prompt)
                raw_text = response.content if hasattr(response, "content") else str(response)
                step_back_str = str(raw_text).strip()

            step_back_clean = step_back_str.strip().strip('"').strip("'")

            # Return both original specific query and step-back query for dual-vector retrieval
            queries = [query]
            if step_back_clean and step_back_clean.lower() != query.lower():
                queries.append(step_back_clean)

            return queries

        except Exception as e:
            self.logger.warning(f"StepBackQueryTransformer error: {e}. Returning original query.")
            return [query]
