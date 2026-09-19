"""Shared prompt resolution, override validation, and run snapshots."""

import hashlib
from typing import Any, Dict, List

from Composer.config_space import ConfigSpace
from PromptManager import MuffakirPrompt, PROMPT_SPECS, resolve_prompt_keys


DEFAULT_METRICS = ["recall", "faithfulness", "answer_correctness"]


def applicable_prompt_keys(context: Any) -> List[str]:
    requested_space = context.search_space.to_composer_dict()
    resolved_space = ConfigSpace(search_space=requested_space or None).search_space
    return resolve_prompt_keys(
        pipeline_mode=context.pipeline_mode,
        retrieval_source=context.retrieval_source,
        adaptive_web_search=context.adaptive_web_search,
        eval_dataset_mode=context.eval_dataset_mode,
        search_space=resolved_space,
        metrics=context.metrics if context.metrics is not None else DEFAULT_METRICS,
        hallucination_check=getattr(context, "hallucination_check", True),
        hallucination_method=getattr(context, "hallucination_method", "text_cleaner"),
    )


def resolve_definitions(context: Any) -> List[Dict[str, Any]]:
    manager = MuffakirPrompt(language=context.language)
    definitions = []
    for key in applicable_prompt_keys(context):
        definition = PROMPT_SPECS[key].to_dict()
        definition["default_template"] = manager.get_prompt(key)
        definitions.append(definition)
    return definitions


def validate_overrides(context: Any) -> MuffakirPrompt:
    applicable = set(applicable_prompt_keys(context))
    overrides = dict(getattr(context, "prompt_overrides", {}) or {})
    irrelevant = sorted(key for key in overrides if key in PROMPT_SPECS and key not in applicable)
    if irrelevant:
        raise ValueError(
            "Prompt overrides are not applicable to this run: " + ", ".join(irrelevant)
        )
    # Constructor performs strict validation, including unsupported keys.
    return MuffakirPrompt(language=context.language, overrides=overrides)


def build_prompt_snapshot(context: Any) -> Dict[str, Any]:
    manager = validate_overrides(context)
    keys = applicable_prompt_keys(context)
    resolved = {key: manager.get_prompt(key) for key in keys}
    return {
        "language": context.language,
        "overrides": dict(getattr(context, "prompt_overrides", {}) or {}),
        "resolved_prompts": resolved,
        "prompt_hashes": {
            key: hashlib.sha256(template.encode("utf-8")).hexdigest()
            for key, template in resolved.items()
        },
    }
