from typing import List, Union
import logging

from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import MuffakirPrompt
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generates an LLM answer from a query and retrieved context.

    Formats the ``generation`` prompt template with the context and question,
    invokes the LLM, and returns the response text.  Handles both string
    and ``List[Document]`` context inputs, with a fallback prompt if the
    template fails to format.
    """

    def __init__(self, llm_provider: LLMProvider, prompt_manager: MuffakirPrompt):
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager
        self.logger = logging.getLogger(__name__)
        self.generation_prompt = self.prompt_manager.get_prompt("generation")

    def generate_answer(
        self,
        query: str,
        documents: Union[List[Document], str]
    ) -> str:
        """Generate an answer from the given query and context documents.

        Args:
            query: The user's question.
            documents: Either a pre-joined context string or a list of
                LangChain ``Document`` objects (``page_content`` will be
                joined with ``"\\n\\n"``).

        Returns:
            The generated answer text.

        Raises:
            GenerationError: If the LLM call itself fails (auth, timeout,
                rate limit, or any other provider failure).
        """
        if isinstance(documents, str):
            context = documents
        else:
            try:
                context = "\n\n".join(doc.page_content for doc in documents)
            except Exception as e:
                from Trace.observability import mark_current_stage_error

                mark_current_stage_error(e, "fallback")
                self.logger.error(f"Failed to join page_content: {e}", exc_info=True)
                context = str(documents)

        try:
            prompt = self.generation_prompt.format(
                context=context,
                question=query
            )
        except Exception as e:
            from Trace.observability import mark_current_stage_error

            mark_current_stage_error(e, "fallback")
            self.logger.error(f"Failed to format generation prompt: {e}", exc_info=True)
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        try:
            llm = self.llm_provider.get_llm()
            if hasattr(llm, "invoke"):
                response = llm.invoke(prompt)
            else:
                response = llm(prompt)

            return self._extract_content(response)

        except Exception as e:
            from Muffakir.exceptions import GenerationError

            self.logger.error(f"Error generating answer from LLM: {e}", exc_info=True)
            raise GenerationError(f"LLM failed to generate an answer: {e}") from e

    @staticmethod
    def _extract_content(response) -> str:
        """Extract text from an LLM response, handling thinking/reasoning models.

        Some models (Qwen3, DeepSeek-R1, o1-mini …) separate *reasoning* from
        *answer*.  LangChain surfaces this in different ways depending on the
        provider adapter:

        1. ``response.content`` contains the final answer (normal case).
        2. ``response.content`` is empty and the answer is in
           ``response.additional_kwargs["reasoning_content"]`` (SovereignEG /
           some OpenAI-compatible wrappers).
        3. ``response.content`` contains the full output including
           ``<think>…</think>`` tags — strip them to get the clean answer.

        Returns the best non-empty string found, or an empty string if nothing
        usable is present.
        """
        import re as _re

        # --- 1. Primary: response.content ---
        content = ""
        if hasattr(response, "content"):
            content = response.content or ""

        # --- 2. Strip <think>…</think> blocks (keeps text after the block) ---
        if content:
            stripped = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()
            if stripped:
                return stripped
            # content was only thinking tokens → fall through

        # --- 3. Fallback: additional_kwargs["reasoning_content"] ---
        if hasattr(response, "additional_kwargs") and response.additional_kwargs:
            reasoning = response.additional_kwargs.get("reasoning_content", "")
            if reasoning and reasoning.strip():
                return reasoning.strip()

        # If it's a message object with empty content, do not return str(response)
        if hasattr(response, "content"):
            return ""

        return str(response) if response is not None else ""
