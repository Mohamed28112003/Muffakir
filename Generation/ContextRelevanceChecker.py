import logging
import re
from typing import List, Union

from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import MuffakirPrompt

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


class ContextRelevanceChecker:
    """
    Grades whether retrieved context is relevant/sufficient to answer a query.
    Used by Adaptive RAG (Mode B) before deciding to fall back to web search.
    """

    def __init__(self, llm_provider: LLMProvider, prompt_manager: MuffakirPrompt):
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager
        self.logger = logging.getLogger(__name__)
        self.relevance_prompt = self.prompt_manager.get_prompt("context_relevance")

    def _documents_to_context(self, documents: Union[List[Document], str]) -> str:
        if isinstance(documents, str):
            return documents
        try:
            return "\n\n".join(doc.page_content for doc in documents)
        except Exception as e:
            from Trace.observability import mark_current_stage_error

            mark_current_stage_error(e, "fallback")
            self.logger.error(f"Failed to join document content: {e}", exc_info=True)
            return str(documents)

    def is_relevant(self, query: str, documents: Union[List[Document], str]) -> bool:
        """
        Returns True if context is graded relevant, False if not_relevant.
        On grader failure, defaults to True (keep vector context / avoid unnecessary web calls).
        """
        context = self._documents_to_context(documents)
        if not context or not context.strip():
            return False

        try:
            prompt = self.relevance_prompt.format(question=query, context=context)
        except Exception as e:
            from Trace.observability import mark_current_stage_error

            mark_current_stage_error(e, "fallback")
            self.logger.error(f"Failed to format context_relevance prompt: {e}", exc_info=True)
            return True

        try:
            llm = self.llm_provider.get_llm()
            if hasattr(llm, "invoke"):
                response = llm.invoke(prompt)
            else:
                response = llm(prompt)

            if hasattr(response, "content"):
                text = response.content
            else:
                text = str(response)

            cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).strip().lower()
            # Prefer explicit not_relevant match
            if "not_relevant" in cleaned.replace(" ", "_") or "not-relevant" in cleaned:
                return False
            if cleaned.startswith("relevant") or cleaned == "relevant":
                return True
            # Fallback token scan
            tokens = re.findall(r"[a-z_]+", cleaned.replace("-", "_"))
            if "not_relevant" in tokens:
                return False
            if "relevant" in tokens:
                return True

            from Trace.observability import mark_current_stage_error

            mark_current_stage_error(
                ValueError("Ambiguous relevance grader response"), "fallback"
            )
            self.logger.warning(f"Ambiguous relevance grade '{cleaned}', defaulting to relevant")
            return True
        except Exception as e:
            from Trace.observability import mark_current_stage_error

            mark_current_stage_error(e, "fallback")
            self.logger.error(f"Context relevance check failed: {e}", exc_info=True)
            return True
