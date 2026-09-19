import os
import yaml
import logging
from string import Formatter
from typing import Dict, Any, Mapping, Optional

from .registry import PROMPT_SPECS

class MuffakirPrompt:
    """
    A centralized manager for all prompt templates used in the pipeline,
    supporting localized prompt loading, dynamic listing, and dynamic edits.
    """
    def __init__(
        self,
        language: str = "ar",
        custom_prompts_dir: Optional[str] = None,
        overrides: Optional[Mapping[str, str]] = None,
    ):
        """
        Initialize the MuffakirPrompt manager.
        
        Args:
            language (str): Target language for prompt templates (e.g. "ar" or "en"). Defaults to "ar".
            custom_prompts_dir (str, optional): Custom path to folder containing localized prompt files.
        """
        self.language = language.lower()
        self.prompts: Dict[str, str] = {}
        self.logger = logging.getLogger(__name__)
        
        # Determine the prompts directory path
        if custom_prompts_dir:
            self.prompts_dir = custom_prompts_dir
        else:
            self.prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
            
        self._load_prompts()
        for key, template in dict(overrides or {}).items():
            self.validate_prompt(key, template)
            self.prompts[key] = template
        
    def _load_prompts(self):
        # Load base prompts
        self.prompts = self._load_from_file(self.language)
        
        # If language is not 'ar' (e.g. 'en'), load 'ar' as fallback for any missing keys
        if self.language != "ar":
            try:
                ar_prompts = self._load_from_file("ar")
                fallback_keys = [k for k in ar_prompts if k not in self.prompts]
                for key, value in ar_prompts.items():
                    if key not in self.prompts:
                        self.prompts[key] = value
                if fallback_keys:
                    self.logger.debug(
                        "%d prompt key(s) missing from '%s.yaml'; filled from "
                        "Arabic fallback: %s",
                        len(fallback_keys), self.language,
                        ", ".join(sorted(fallback_keys)),
                    )
            except Exception as e:
                self.logger.warning(f"Failed to load fallback prompts from 'ar': {e}")

    def _load_from_file(self, lang: str) -> Dict[str, str]:
        """Helper to load prompts from a specific language YAML file."""
        lang_file = os.path.join(self.prompts_dir, f"{lang.lower()}.yaml")
        
        # Fallback to 'ar' if file does not exist (logged so the silent
        # language substitution is visible to users).
        if not os.path.exists(lang_file):
            self.logger.warning(
                "Prompt file '%s.yaml' not found in %s; falling back to "
                "'ar.yaml'. Requested language will behave as Arabic.",
                lang.lower(), self.prompts_dir,
            )
            lang_file = os.path.join(self.prompts_dir, "ar.yaml")
            if not os.path.exists(lang_file):
                raise FileNotFoundError(
                    f"Could not load prompts. Neither {lang}.yaml nor ar.yaml found in {self.prompts_dir}"
                )
        
        try:
            with open(lang_file, "r", encoding="utf-8") as f:
                prompts = yaml.safe_load(f) or {}
        except Exception as e:
            from Muffakir.exceptions import PromptLoadError

            raise PromptLoadError(f"Failed to load prompts from {lang_file}: {e}") from e

        # Warn on non-string template values (usually YAML indentation bugs).
        # Values are still returned as-is to preserve backward compatibility.
        for key, value in prompts.items():
            if not isinstance(value, str):
                self.logger.warning(
                    "Prompt '%s' in %s is not a plain string (got %s); it may "
                    "fail when formatted downstream.",
                    key, lang_file, type(value).__name__,
                )
        return prompts

    def get_prompt(self, key: str) -> str:
        """Get a specific prompt template by key."""
        if key not in self.prompts:
            raise ValueError(f"Prompt '{key}' not found in MuffakirPrompt.")
        return self.prompts[key]

    def validate_prompt(self, key: str, template: str) -> None:
        """Strictly validate *template* without changing this manager."""
        # Lazy import avoids PromptManager -> Muffakir.__init__ -> PromptManager
        # during package initialization.
        from Muffakir.exceptions import PromptValidationError

        if key not in PROMPT_SPECS:
            raise PromptValidationError(f"Unsupported prompt key: '{key}'.")
        if not isinstance(template, str) or not template.strip():
            raise PromptValidationError(f"Prompt '{key}' cannot be empty.")

        spec = PROMPT_SPECS[key]
        variables = set()
        try:
            for _literal, field_name, format_spec, conversion in Formatter().parse(template):
                if field_name is None:
                    continue
                if not field_name:
                    raise PromptValidationError(
                        f"Prompt '{key}' contains an empty placeholder."
                    )
                if conversion is not None:
                    raise PromptValidationError(
                        f"Prompt '{key}' cannot use conversion '!{conversion}'."
                    )
                if format_spec:
                    raise PromptValidationError(
                        f"Prompt '{key}' cannot use format specifications."
                    )
                if field_name not in spec.allowed_variables:
                    raise PromptValidationError(
                        f"Prompt '{key}' contains unknown variable '{{{field_name}}}'. "
                        f"Allowed: {', '.join(spec.allowed_variables)}."
                    )
                variables.add(field_name)
        except PromptValidationError:
            raise
        except ValueError as exc:
            raise PromptValidationError(
                f"Prompt '{key}' contains malformed braces: {exc}."
            ) from exc

        missing = [name for name in spec.required_variables if name not in variables]
        if missing:
            formatted = ", ".join(f"{{{name}}}" for name in missing)
            raise PromptValidationError(
                f"Prompt '{key}' is missing required variables: {formatted}."
            )

    def update_prompt(self, key: str, template: str) -> None:
        """
        Update or add a prompt template in-memory.
        Validates placeholders to warn the developer of potential omissions.
        """
        if key in PROMPT_SPECS:
            missing = [
                f"{{{name}}}"
                for name in PROMPT_SPECS[key].required_variables
                if f"{{{name}}}" not in template
            ]
            if missing:
                self.logger.warning(
                    f"⚠️ Warning: The updated prompt '{key}' is missing required placeholders: {', '.join(missing)}. "
                    "This may cause runtime failures in the pipeline."
                )

        self.prompts[key] = template

    def add_prompt(self, key: str, prompt_template: str):
        """Backwards compatible alias for update_prompt."""
        self.update_prompt(key, prompt_template)

    def get_all_prompts(self, language: Optional[str] = None) -> Dict[str, str]:
        """
        Return all prompts for the selected language.
        If language is specified and is different from self.language, load it dynamically on-demand.
        """
        target_lang = (language or self.language).lower()
        if target_lang == self.language:
            return self.prompts.copy()
        
        # Load from file dynamically
        try:
            prompts = self._load_from_file(target_lang)
            # Apply Arabic fallbacks if necessary
            if target_lang != "ar":
                ar_prompts = self._load_from_file("ar")
                for key, value in ar_prompts.items():
                    if key not in prompts:
                        prompts[key] = value
            return prompts
        except Exception as e:
            self.logger.error(f"Failed to load prompts for language '{target_lang}': {e}")
            return {}

    def print_all_prompts(self, language: Optional[str] = None) -> None:
        """
        Print all loaded prompts for the selected language in a clear, formatted terminal layout.
        """
        target_lang = (language or self.language).lower()
        prompts = self.get_all_prompts(target_lang)
        
        print("\n" + "="*80)
        print(f" MUFAKKIR PROMPT TEMPLATES (Language: {target_lang.upper()}) ".center(80, "="))
        print("="*80)
        
        if not prompts:
            print(f"No prompts found or failed to load prompts for language: {target_lang}")
            return
            
        for key, template in sorted(prompts.items()):
            print(f"\n🔑 Key: {key}")
            print("-" * (len(key) + 8))
            print(template.strip())
            print("-" * 80)
            
        print("\n" + "="*80 + "\n")

# Backwards compatible alias
PromptManager = MuffakirPrompt
