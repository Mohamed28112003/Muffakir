"""
Test suite for the PromptManager module (pytest).

Runs against BOTH the real shipped prompts/ directory (default loads,
bilingual coverage of consumer-critical keys, placeholder spot checks) and
tmp_path custom prompt directories (fallbacks and error paths). No network
or LLM required.
"""

import logging
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PromptManager.PromptManager import MuffakirPrompt, PromptManager
from Muffakir.exceptions import PromptValidationError
from PromptManager.registry import resolve_prompt_keys


# Keys consumed by real components across the library (Generation, Evaluation,
# HallucinationsCheck, QueryTransformer, Reranker, SyntheticData).
CONSUMER_CRITICAL_KEYS = [
    "generation", "query_rewrite", "context_relevance", "answer_correctness",
    "llm_judge_rating",
    "MCQ", "QA", "history_classification_prompt", "history_query_prompt",
    "summary_generation", "question_generation", "search_query",
    "faithfulness_extraction", "faithfulness_verification", "youtube",
    "mindmap", "summary_map", "summary_combine", "summary_direct",
    "summary_check", "hallucination_check_prompt", "reranker_scoring",
    "multi_query_expansion", "hyde", "step_back", "query_decomposition",
    "context_grounding",
]


def _write_yaml(dir_path, name, data):
    p = Path(dir_path) / f"{name}.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return str(dir_path)


# ---------------------------------------------------------------------------
# Real shipped prompts/ directory
# ---------------------------------------------------------------------------
def test_default_ar_load():
    pm = MuffakirPrompt()  # default language="ar"
    assert len(pm.prompts) > 0
    for key in CONSUMER_CRITICAL_KEYS:
        assert key in pm.prompts, f"missing consumer-critical key: {key}"
        assert isinstance(pm.get_prompt(key), str)


def test_default_en_load_with_critical_keys():
    pm = MuffakirPrompt(language="en")
    for key in CONSUMER_CRITICAL_KEYS:
        assert key in pm.prompts, f"en.yaml missing consumer-critical key: {key}"


def test_shipped_hallucination_prompt_has_answer_placeholder():
    for lang in ("ar", "en"):
        tpl = MuffakirPrompt(language=lang).get_prompt("hallucination_check_prompt")
        assert "{answer}" in tpl, f"{lang} hallucination_check_prompt lacks {{answer}}"


def test_generation_template_placeholders():
    tpl = MuffakirPrompt(language="ar").get_prompt("generation")
    assert "{context}" in tpl and "{question}" in tpl


def test_language_normalized_lower():
    pm = MuffakirPrompt(language="EN")  # shipped dir has en.yaml
    assert pm.language == "en"


# ---------------------------------------------------------------------------
# Loading: fallbacks & error paths (custom dirs)
# ---------------------------------------------------------------------------
def test_custom_dir_fallback_merge(tmp_path):
    _write_yaml(tmp_path, "ar", {"ar_only": "AR-only", "shared": "AR-shared"})
    _write_yaml(tmp_path, "en", {"shared": "EN-shared"})
    pm = MuffakirPrompt(language="en", custom_prompts_dir=str(tmp_path))
    assert pm.get_prompt("shared") == "EN-shared"     # en wins
    assert pm.get_prompt("ar_only") == "AR-only"      # ar fills the gap


def test_unknown_language_falls_back_to_ar(tmp_path, caplog):
    _write_yaml(tmp_path, "ar", {"a": "AR-a"})
    with caplog.at_level(logging.WARNING):
        pm = MuffakirPrompt(language="fr", custom_prompts_dir=str(tmp_path))
    assert pm.get_prompt("a") == "AR-a"
    assert any("falling back" in r.message for r in caplog.records)


def test_missing_both_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Could not load prompts"):
        MuffakirPrompt(language="en", custom_prompts_dir=str(tmp_path))


def test_malformed_yaml_raises_runtimeerror(tmp_path):
    (tmp_path / "en.yaml").write_text("key: [unclosed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Failed to load prompts"):
        MuffakirPrompt(language="en", custom_prompts_dir=str(tmp_path))


def test_malformed_yaml_chains_original_exception(tmp_path):
    (tmp_path / "en.yaml").write_text("key: [unclosed", encoding="utf-8")
    with pytest.raises(RuntimeError) as excinfo:
        MuffakirPrompt(language="en", custom_prompts_dir=str(tmp_path))
    assert excinfo.value.__cause__ is not None


def test_empty_yaml_loads_as_empty(tmp_path):
    _write_yaml(tmp_path, "ar", {})
    pm = MuffakirPrompt(custom_prompts_dir=str(tmp_path))
    assert pm.prompts == {}


def test_non_string_value_warns_but_still_loads(tmp_path, caplog):
    (tmp_path / "ar.yaml").write_text(
        "good: fine template\nnested:\n  key: oops\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        pm = MuffakirPrompt(custom_prompts_dir=str(tmp_path))
    assert pm.get_prompt("good") == "fine template"
    assert any("not a plain string" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# get_prompt / update_prompt / add_prompt
# ---------------------------------------------------------------------------
def test_get_prompt_unknown_key_raises(tmp_path):
    _write_yaml(tmp_path, "ar", {"a": "x"})
    pm = MuffakirPrompt(custom_prompts_dir=str(tmp_path))
    with pytest.raises(ValueError, match="not found"):
        pm.get_prompt("nope")


def test_update_prompt_sets_value(tmp_path):
    _write_yaml(tmp_path, "ar", {"generation": "orig"})
    pm = MuffakirPrompt(custom_prompts_dir=str(tmp_path))
    pm.update_prompt("generation", "new {context} / {question}")
    assert pm.get_prompt("generation") == "new {context} / {question}"


def test_update_prompt_warns_on_missing_placeholders(tmp_path, caplog):
    _write_yaml(tmp_path, "ar", {"generation": "orig"})
    pm = MuffakirPrompt(custom_prompts_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING):
        pm.update_prompt("generation", "no placeholders here")
    assert any(
        "missing required placeholders" in r.message for r in caplog.records
    )
    # Update is still applied (warning, not error) — documented behavior.
    assert pm.get_prompt("generation") == "no placeholders here"


def test_update_prompt_no_warning_when_complete(tmp_path, caplog):
    _write_yaml(tmp_path, "ar", {"generation": "orig"})
    pm = MuffakirPrompt(custom_prompts_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING):
        pm.update_prompt("generation", "{context} + {question}")
    assert not any(
        "missing required placeholders" in r.message for r in caplog.records
    )


def test_update_hallucination_check_requires_answer(tmp_path, caplog):
    """P2 regression: new placeholder rule for hallucination_check_prompt."""
    _write_yaml(tmp_path, "ar", {"hallucination_check_prompt": "x"})
    pm = MuffakirPrompt(custom_prompts_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING):
        pm.update_prompt("hallucination_check_prompt", "missing the slot")
    assert any("{answer}" in r.message for r in caplog.records)


def test_update_reranker_scoring_requires_both(tmp_path, caplog):
    _write_yaml(tmp_path, "ar", {"reranker_scoring": "x"})
    pm = MuffakirPrompt(custom_prompts_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING):
        pm.update_prompt("reranker_scoring", "only {query} here")
    assert any("{document}" in r.message for r in caplog.records)


def test_update_new_key_without_rule_no_warning(tmp_path, caplog):
    """Keys not in the rules table are stored freely (e.g. custom keys)."""
    pm = MuffakirPrompt(language="en")
    with caplog.at_level(logging.WARNING):
        pm.update_prompt("totally_custom", "whatever {free} content")
    assert not any(
        "missing required placeholders" in r.message for r in caplog.records
    )
    assert pm.get_prompt("totally_custom") == "whatever {free} content"


def test_add_prompt_is_alias_for_update(tmp_path):
    _write_yaml(tmp_path, "ar", {"a": "x"})
    pm = MuffakirPrompt(custom_prompts_dir=str(tmp_path))
    pm.add_prompt("brand_new", "{text}")
    assert pm.get_prompt("brand_new") == "{text}"


# ---------------------------------------------------------------------------
# get_all_prompts / aliases / package surface / regressions
# ---------------------------------------------------------------------------
def test_get_all_prompts_returns_copy(tmp_path):
    _write_yaml(tmp_path, "ar", {"a": "x"})
    pm = MuffakirPrompt(custom_prompts_dir=str(tmp_path))
    snapshot = pm.get_all_prompts()
    snapshot["injected"] = "bad"
    assert "injected" not in pm.prompts


def test_get_all_prompts_dynamic_language_with_fallback(tmp_path):
    _write_yaml(tmp_path, "ar", {"ar_only": "AR", "shared": "AR-s"})
    _write_yaml(tmp_path, "en", {"shared": "EN-s"})
    pm = MuffakirPrompt(language="ar", custom_prompts_dir=str(tmp_path))
    en = pm.get_all_prompts("en")
    assert en["shared"] == "EN-s"
    assert en["ar_only"] == "AR"            # ar fallback applied on-demand too
    assert pm.get_prompt("shared") == "AR-s"  # internal state untouched


def test_get_all_prompts_unknown_language_falls_back_to_ar(tmp_path, caplog):
    """Unknown language resolves via the ar.yaml fallback (with a warning)."""
    _write_yaml(tmp_path, "ar", {"a": "x"})
    pm = MuffakirPrompt(custom_prompts_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING):
        result = pm.get_all_prompts("zz")
    assert result == {"a": "x"}
    assert any("falling back" in r.message for r in caplog.records)


def test_get_all_prompts_load_failure_returns_empty(tmp_path):
    """Fail-safe: if neither the requested nor the ar fallback file loads,
    an empty dict is returned instead of raising."""
    _write_yaml(tmp_path, "ar", {"a": "x"})
    pm = MuffakirPrompt(custom_prompts_dir=str(tmp_path))
    empty_dir = tmp_path / "nowhere"
    empty_dir.mkdir()
    pm.prompts_dir = str(empty_dir)  # simulate a broken/unavailable source
    assert pm.get_all_prompts("zz") == {}


def test_backward_compatible_class_alias():
    assert PromptManager is MuffakirPrompt


def test_package_reexports():
    import PromptManager as pkg
    assert pkg.MuffakirPrompt is MuffakirPrompt
    assert pkg.PromptManager is PromptManager
    assert set(pkg.__all__) == {"MuffakirPrompt", "PromptManager"}


def test_no_basicconfig_regression():
    """Regression: constructing managers must not mutate root logger config."""
    root_before = list(logging.getLogger().handlers)
    level_before = logging.getLogger().level
    MuffakirPrompt()                      # shipped dir, ar
    MuffakirPrompt(language="en")         # shipped dir, en (+fallback merge)
    assert list(logging.getLogger().handlers) == root_before
    assert logging.getLogger().level == level_before


# ---------------------------------------------------------------------------
# Strict per-run overrides and Composer relevance
# ---------------------------------------------------------------------------
def test_strict_override_is_isolated_from_defaults():
    custom = "Use {context} to answer {question}."
    overridden = MuffakirPrompt(language="en", overrides={"generation": custom})
    assert overridden.get_prompt("generation") == custom
    assert MuffakirPrompt(language="en").get_prompt("generation") != custom


@pytest.mark.parametrize("template", [
    "Only {context}",
    "Use {context and {question}",
    "Use {context!r} and {question}",
    "Use {context:>10} and {question}",
    "Use {context}, {question}, and {unknown}",
    "   ",
])
def test_strict_validation_rejects_pipeline_unsafe_templates(template):
    with pytest.raises(PromptValidationError):
        MuffakirPrompt(language="en", overrides={"generation": template})


def test_strict_validation_allows_escaped_literal_braces():
    template = "Return JSON like {{\"answer\": \"...\"}} using {context} for {question}."
    manager = MuffakirPrompt(language="en", overrides={"generation": template})
    assert manager.get_prompt("generation") == template


def test_validate_prompt_is_non_mutating_and_rejects_unknown_key():
    manager = MuffakirPrompt(language="en")
    original = manager.get_prompt("generation")
    manager.validate_prompt("generation", "Answer {question} from {context}.")
    assert manager.get_prompt("generation") == original
    with pytest.raises(PromptValidationError, match="Unsupported prompt key"):
        manager.validate_prompt("not_a_prompt", "anything")


def test_retrieval_only_existing_without_transform_has_no_prompts():
    assert resolve_prompt_keys(
        pipeline_mode="retrieval_only",
        eval_dataset_mode="existing",
        search_space={"query_expansion": ["none"], "reranking": ["none"]},
        metrics=["recall"],
    ) == []


def test_relevance_resolver_uses_union_and_mode_boundaries():
    keys = resolve_prompt_keys(
        pipeline_mode="full_rag",
        retrieval_source="vector_db",
        adaptive_web_search=True,
        eval_dataset_mode="auto",
        search_space={
            "query_expansion": ["none", "rewrite", "hyde"],
            "reranking": ["semantic_similarity", "llm"],
        },
        metrics=["recall", "faithfulness", "answer_correctness", "llm_judge_rating"],
    )
    assert set(keys) == {
        "QA", "query_rewrite", "hyde", "reranker_scoring", "generation",
        "context_relevance", "hallucination_check_prompt",
        "answer_correctness", "llm_judge_rating", "context_grounding",
    }

    web_keys = resolve_prompt_keys(
        pipeline_mode="full_rag",
        retrieval_source="web_search_only",
        eval_dataset_mode="existing",
        search_space={"query_expansion": ["rewrite"], "reranking": ["llm"]},
        metrics=["faithfulness"],
    )
    assert web_keys == ["generation", "context_grounding"]


if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))
