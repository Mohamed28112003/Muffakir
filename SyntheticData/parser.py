import re
import logging
from typing import Optional, Dict, Any
from .models import LLMQAOutput
from PromptManager.PromptManager import MuffakirPrompt

try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    from langchain.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class QAParser:
    """
    Parses LLM responses into question and answer pairs.
    Prefers Pydantic structured output (`LLMQAOutput`), falling back to regex.
    """

    _QA_TEMPLATE = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are an expert Q&A generator for building high-quality RAG datasets.\n"
                "Based on the provided document snippet, generate one clear, direct question "
                "and its detailed, accurate answer.\n"
                "Both question and answer MUST be based exclusively on the given text."
            )
        ),
        (
            "human",
            "Text snippet:\n{context}"
        )
    ])

    def __init__(self, prompt_manager: Optional[MuffakirPrompt] = None):
        self.prompt_manager = prompt_manager or MuffakirPrompt(language="ar")

    def generate_and_parse(
        self,
        context: str,
        llm_provider: Any,
        custom_prompt_key: str = "QA"
    ) -> Dict[str, str]:
        """
        Generate Q&A from context and parse the result.
        Returns a dict with 'question' and 'answer' keys.
        """
        llm = llm_provider.get_llm()
        prompt_template = self.prompt_manager.get_prompt(custom_prompt_key)

        # Strategy 1: Pydantic Structured Output
        try:
            if hasattr(llm, "with_structured_output"):
                structured_llm = llm.with_structured_output(LLMQAOutput)
                managed_prompt = ChatPromptTemplate.from_messages([
                    ("human", prompt_template)
                ])
                chain = managed_prompt | structured_llm
                res: LLMQAOutput = chain.invoke({"context": context})
                if res and res.question and res.answer:
                    logger.debug("QAParser: Successfully parsed via Pydantic structured output.")
                    return {"question": res.question.strip(), "answer": res.answer.strip()}
        except Exception as e:
            logger.debug(f"QAParser structured output attempt failed/unavailable: {e}")

        # Strategy 2: PromptManager + LLM invoke + Regex fallback
        from Trace.observability import mark_current_stage_error, observe_stage

        provider_value = getattr(llm_provider, "provider", None)
        provider_name = getattr(provider_value, "value", provider_value)
        with observe_stage(
            "dataset_generation", "llm",
            provider=str(provider_name) if provider_name is not None else None,
        ):
            try:
                prompt_str = prompt_template.format(context=context)
                response = llm.invoke(prompt_str)
                raw_text = response.content if hasattr(response, "content") else str(response)
                return self.parse_raw_text(raw_text)
            except Exception as e:
                # Generation may retry the chunk. Preserve that behavior, but
                # never let a swallowed provider exception disappear from the
                # Composer error-rate panel.
                mark_current_stage_error(e, recovery="retry")
                logger.error(f"QAParser raw text generation/parsing error: {e}")
                return {"question": "", "answer": ""}

    def parse_raw_text(self, response: str) -> Dict[str, str]:
        """
        Parse raw LLM response text using regex and line scanning fallback.
        Supports Arabic (السؤال / الإجابة) and English (Question / Answer).
        """
        if not response:
            return {"question": "", "answer": ""}

        response = response.strip()

        # Primary regex match
        question_pattern = r'(?:السؤال|Question)\s*:\s*([^\n]+)'
        answer_pattern = r'(?:الإجابة|Answer)\s*:\s*([^\n]+(?:\n(?!(?:السؤال|Question|الإجابة|Answer)\s*:)[^\n]+)*)'

        question_match = re.search(question_pattern, response, re.IGNORECASE)
        answer_match = re.search(answer_pattern, response, re.IGNORECASE)

        question = question_match.group(1).strip() if question_match else ""
        answer = answer_match.group(1).strip() if answer_match else ""

        # Line-by-line scan fallback
        if not question or not answer:
            arabic_q = {'السؤال'}
            arabic_a = {'الإجابة'}
            english_q = {'question'}
            english_a = {'answer'}

            lines = response.split('\n')
            for i, line in enumerate(lines):
                if ':' not in line:
                    continue
                label, _, rest = line.partition(':')
                label_norm = label.strip().lower()

                if label_norm in english_q or label.strip() in arabic_q:
                    if not question:
                        question = rest.strip()
                elif label_norm in english_a or label.strip() in arabic_a:
                    if not answer:
                        answer = rest.strip()
                        stop_labels = arabic_q | arabic_a | english_q | english_a
                        for j in range(i + 1, len(lines)):
                            next_line = lines[j].strip()
                            if not next_line:
                                continue
                            next_label = next_line.split(':')[0].strip().lower()
                            if next_label in stop_labels or next_line.split(':')[0].strip() in stop_labels:
                                break
                            answer += ' ' + next_line

        return {"question": question, "answer": answer}
