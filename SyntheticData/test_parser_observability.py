"""Dataset-generation provider errors must be visible to ComposerUI."""

from queue import Queue

from SyntheticData.parser import QAParser
from Trace.observability import observation_context


def test_qa_parser_records_failed_llm_call_as_retryable_observation():
    class _PromptManager:
        @staticmethod
        def get_prompt(key):
            assert key == "QA"
            return "Generate a question and answer from: {context}"

    class _FailingLLM:
        @staticmethod
        def invoke(prompt):
            raise TypeError(
                "GenerativeServiceClient.generate_content() got an "
                "unexpected keyword argument 'max_retries'"
            )

    class _Provider:
        provider = "gemini"
        model = "must-not-be-in-error-event"

        @staticmethod
        def get_llm():
            return _FailingLLM()

    queue = Queue()
    with observation_context(queue):
        result = QAParser(prompt_manager=_PromptManager()).generate_and_parse(
            context="Document text", llm_provider=_Provider()
        )

    assert result == {"question": "", "answer": ""}
    kind, event = queue.get_nowait()
    assert kind == "operation"
    assert event["stage"] == "dataset_generation"
    assert event["component"] == "llm"
    assert event["provider"] == "gemini"
    assert event["model"] is None
    assert event["outcome"] == "error"
    assert event["recovery"] == "retry"
    assert "max_retries" in event["message"]
