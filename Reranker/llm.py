import re
import logging
from typing import List, Tuple, Optional
from pydantic import BaseModel, Field

from .base import BaseReranker
from LLMProvider.LLMProvider import LLMProvider
from PromptManager.PromptManager import MuffakirPrompt

try:
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    from langchain.schema import Document
    from langchain.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


# ── Pydantic structured output schema ──────────────────────────────────────────

class RerankerScore(BaseModel):
    """Structured relevance score for LLM-based document reranking."""

    score: float = Field(
        description=(
            "Relevance score from 0.0 to 1.0. "
            "1.0 = document directly and fully answers the query. "
            "0.0 = document is completely irrelevant to the query."
        )
    )
    reasoning: str = Field(
        description="One sentence explanation of why this score was assigned."
    )


# ── LLM Reranker ───────────────────────────────────────────────────────────────

class LLMReranker(BaseReranker):
    """
    LLM-Based Reranker using Pydantic Structured Output.

    Uses the configured LLM to independently score each (query, document) pair
    on a continuous relevance scale of 0.0 to 1.0.

    Architecture:
    - Primary: Uses `llm.with_structured_output(RerankerScore)` for JSON-enforced
      structured output. This works with OpenAI, Anthropic, Groq, and other
      function-calling capable providers.
    - Fallback: If the provider does not support structured output, falls back to
      plain text generation + regex score extraction.

    The prompt is loaded from PromptManager (`reranker_scoring` key), ensuring
    full bilingual support (Arabic & English). Temperature is forced to 0.0
    for deterministic, consistent scoring.

    Why continuous score vs. binary (yes/no)?
    - Binary grading (yes/no) is appropriate for filtering decisions
      (e.g., hallucination checking, context relevance).
    - Continuous scoring (0.0–1.0) is required for reranking because we need
      to sort documents by relative relevance, not just keep/discard them.
    """

    # Structured output prompt template (system + human)
    _SCORING_TEMPLATE = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are an expert relevance evaluator for RAG (Retrieval-Augmented Generation) systems.\n"
                "Your task is to score how relevant a retrieved document is to the user's query.\n"
                "Score on a scale from 0.0 to 1.0:\n"
                "  - 1.0: Document directly and fully answers the query.\n"
                "  - 0.7–0.9: Document is highly relevant and contains important related information.\n"
                "  - 0.4–0.6: Document is moderately relevant; related to topic but does not directly answer.\n"
                "  - 0.1–0.3: Document is weakly relevant; loosely connected to the query.\n"
                "  - 0.0: Document is completely irrelevant.\n"
                "Be precise and consistent."
            )
        ),
        (
            "human",
            "Query: {query}\n\nDocument: {document}"
        )
    ])

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: Optional[MuffakirPrompt] = None,
        prompt_key: str = "reranker_scoring",
        use_structured_output: bool = True
    ):
        if llm_provider is None:
            raise ValueError("llm_provider is required for LLMReranker.")

        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager or MuffakirPrompt(language="ar")
        self.prompt_key = prompt_key
        self.prompt_template = self.prompt_manager.get_prompt(self.prompt_key)
        self.use_structured_output = use_structured_output

        # Try to build a structured output chain at init time
        self._structured_chain = None
        if self.use_structured_output:
            try:
                llm = self.llm_provider.get_llm()
                if hasattr(llm, "temperature"):
                    llm.temperature = 0.0
                structured_llm = llm.with_structured_output(RerankerScore)
                managed_prompt = ChatPromptTemplate.from_messages([
                    ("human", self.prompt_template)
                ])
                self._structured_chain = managed_prompt | structured_llm
                logger.info("LLMReranker: Using Pydantic structured output (function calling).")
            except Exception as e:
                logger.warning(
                    f"LLMReranker: Provider does not support structured output ({e}). "
                    "Falling back to regex score extraction."
                )
                self._structured_chain = None

    @property
    def name(self) -> str:
        return "llm"

    def _score_with_structured_output(self, query: str, document: str) -> Optional[float]:
        """Score using Pydantic structured output chain."""
        try:
            result: RerankerScore = self._structured_chain.invoke({
                "query": query,
                "document": document
            })
            score = max(0.0, min(1.0, float(result.score)))
            logger.debug(
                f"Structured score={score:.3f} | reason='{result.reasoning[:60]}...' "
                f"| doc='{document[:40]}...'"
            )
            return score
        except Exception as e:
            logger.debug(f"Structured output scoring failed: {e}")
            return None

    def _score_with_fallback(self, query: str, document: str, llm) -> float:
        """Score via plain text generation + regex parsing (fallback)."""
        try:
            formatted_prompt = self.prompt_template.format(
                query=query,
                document=document
            )
            response = llm.invoke(formatted_prompt)
            raw = response.content if hasattr(response, "content") else str(response)

            # Extract first float in [0.0, 1.0] range
            matches = re.findall(r"\b([01]\.\d+|0|1)\b", raw.strip())
            if matches:
                return max(0.0, min(1.0, float(matches[0])))
        except Exception as e:
            from Muffakir.exceptions import classify_provider_exception

            logger.error(f"Fallback scoring failed: {e}")
            raise classify_provider_exception(e, provider="reranker_llm") from e
        return 0.0

    def score(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Score each document independently against the query.
        Prefers structured output; falls back to regex parsing.
        """
        scored = []

        # Lazily get fallback LLM only if needed
        fallback_llm = None

        for doc in documents:
            doc_score = None

            # Primary: structured output
            if self._structured_chain is not None:
                doc_score = self._score_with_structured_output(query, doc.page_content)

            # Fallback: plain text + regex
            if doc_score is None:
                if fallback_llm is None:
                    fallback_llm = self.llm_provider.get_llm()
                    if hasattr(fallback_llm, "temperature"):
                        fallback_llm.temperature = 0.0
                doc_score = self._score_with_fallback(query, doc.page_content, fallback_llm)

            scored.append((doc, doc_score))

        return sorted(scored, key=lambda x: x[1], reverse=True)
