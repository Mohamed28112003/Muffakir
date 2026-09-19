"""Tests for explicit no-answer observability detection."""

from Evaluation.refusals import detect_answer_refusal


def test_detects_the_configured_english_and_arabic_refusal_answers():
    assert detect_answer_refusal("I cannot answer this question.") == "insufficient_information"
    assert detect_answer_refusal("لا يمكنني الإجابة على هذا السؤال") == "insufficient_information"


def test_detects_common_short_refusal_variants_without_flagging_real_answers():
    assert detect_answer_refusal("I don't have enough context") == "insufficient_information"
    assert detect_answer_refusal("لا توجد معلومات كافية") == "insufficient_information"
    assert detect_answer_refusal("The report says: I cannot answer this question.") is None
    assert detect_answer_refusal("Here is the answer based on the available context.") is None
