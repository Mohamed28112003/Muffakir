"""
ConfigSpace - Search space definitions for architecture search.

Defines the available options for each pipeline stage and provides
utilities for generating combinations.
"""

from typing import Dict, List, Any, Optional
from itertools import product
from copy import deepcopy
from LLMProvider.parameters import validate_parameters, resolve_parameters
import logging

from .index_key import compute_index_key

logger = logging.getLogger(__name__)


# Default search space for MVP (4 critical stages)
DEFAULT_SEARCH_SPACE: Dict[str, List[Any]] = {
    "query_expansion": ["none", "multi_query", "hyde", "step_back"],
    "retrieval": ["similarity_search", "max_marginal_relevance", "hybrid"],
    "reranking": ["none", "semantic_similarity", "cross_encoder"],
    "k": [3, 5, 10],
}
# Total combinations: 4 × 3 × 3 × 3 = 108


# Mapping from search space keys to MuffakirRAG config keys.
# Each entry maps a Composer search-space stage to one or more MuffakirRAG config keys:
#   - config_key: the primary MuffakirRAG config key produced from this stage
#   - transform: how to convert the search-space value into the primary config value
#   - method_key (optional): a secondary config key carrying the concrete method/strategy name
#   - method_transform (optional): how to convert the search-space value into the method_key value
CONFIG_MAPPING: Dict[str, Dict[str, Any]] = {
    "query_expansion": {
        "config_key": "query_transformer",          # bool: enable/disable query transformation
        "transform": lambda v: v != "none",
        "method_key": "query_transformer_strategy",  # str: rewrite/multi_query/hyde/step_back/...
        "method_transform": lambda v: v if v != "none" else "rewrite",
    },
    "retrieval": {
        "config_key": "retrieval_method",
        "transform": lambda v: v,
    },
    "reranking": {
        "config_key": "reranking",                  # bool: enable/disable reranking
        "transform": lambda v: v != "none",
        "method_key": "reranking_method",            # str: semantic_similarity/cross_encoder/...
        "method_transform": lambda v: v if v != "none" else "semantic_similarity",
    },
    "reranking_model": {
        "config_key": "reranking_model",
        "transform": lambda v: str(v).strip(),
    },
    "k": {
        "config_key": "k",
        "transform": lambda v: v,
    },
    "llm": {
        "config_key": "llm_provider",
        "transform": lambda v: v["provider"],
        "method_key": "llm_model",
        "method_transform": lambda v: v["model"],
    },
    "chunking": {
        "config_key": "chunking_method",
        "transform": lambda v: v["method"],
        "extra_keys": {
            "chunk_size": lambda v: v["size"],
            "chunk_overlap": lambda v: v["overlap"],
        },
    },
    "embedding_model": {
        "config_key": "embedding_model",
        "transform": lambda v: v,
    },
    "vector_db_provider": {
        "config_key": "vector_db_provider",
        "transform": lambda v: v,
    },
}

# Stages recognized by the Composer. Unknown stages are rejected to avoid silently
# injecting arbitrary keys into the downstream MuffakirRAG config.
ALLOWED_STAGES = set(CONFIG_MAPPING.keys())

# Stages whose search-space values are compound (dict) rather than a flat
# string/int, plus the keys each entry must contain. Validated explicitly
# since a malformed entry here would otherwise surface as a confusing
# KeyError deep inside trial_config_to_rag_config() instead of a clear
# validation error at ConfigSpace construction time.
COMPOUND_STAGE_REQUIRED_KEYS: Dict[str, List[str]] = {
    "llm": ["provider", "model"],
    "chunking": ["method", "size", "overlap"],
}


class ConfigSpace:
    """
    Manages the search space for architecture optimization.
    
    Provides utilities for:
    - Validating search space configurations
    - Generating all combinations (grid search)
    - Computing total number of trials
    """
    
    def __init__(self, search_space: Optional[Dict[str, List[Any]]] = None):
        """
        Initialize ConfigSpace.
        
        Args:
            search_space: Custom search space dictionary.
                         If None, uses DEFAULT_SEARCH_SPACE.
        """
        if search_space is None:
            self.search_space = deepcopy(DEFAULT_SEARCH_SPACE)
        else:
            # Deep copy to avoid sharing mutable lists with caller / module globals
            self.search_space = deepcopy(search_space)
        self._validate()
    
    def _validate(self):
        """Validate the search space configuration."""
        if not self.search_space:
            raise ValueError("Search space cannot be empty")
        
        unknown = set(self.search_space.keys()) - ALLOWED_STAGES
        if unknown:
            raise ValueError(
                f"Unknown search space stage(s): {sorted(unknown)}. "
                f"Allowed stages: {sorted(ALLOWED_STAGES)}"
            )
        
        for key, values in self.search_space.items():
            if not isinstance(values, list):
                raise ValueError(
                    f"Search space values for '{key}' must be a list, got {type(values)}"
                )
            if not values:
                raise ValueError(f"Search space for '{key}' cannot be empty")

        if "reranking_model" in self.search_space:
            normalized_models = []
            seen_models = set()
            for model in self.search_space["reranking_model"]:
                if not isinstance(model, str) or not model.strip():
                    raise ValueError(
                        "Search space entries for 'reranking_model' must be "
                        "non-empty Hugging Face model IDs."
                    )
                normalized = model.strip()
                if normalized in seen_models:
                    raise ValueError(
                        f"Duplicate reranking model in search space: '{normalized}'."
                    )
                seen_models.add(normalized)
                normalized_models.append(normalized)
            self.search_space["reranking_model"] = normalized_models

            reranking_methods = self.search_space.get("reranking")
            if not reranking_methods or not any(
                str(method).strip().lower() in {"cross_encoder", "pointwise"}
                for method in reranking_methods
            ):
                raise ValueError(
                    "The 'reranking_model' search dimension requires at least "
                    "one compatible reranking method: cross_encoder or pointwise."
                )

        for key, values in self.search_space.items():
            required_keys = COMPOUND_STAGE_REQUIRED_KEYS.get(key)
            if not required_keys:
                continue
            for entry in values:
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"Search space entries for '{key}' must be dicts with keys "
                        f"{required_keys}, got {type(entry).__name__}"
                    )
                missing = [k for k in required_keys if k not in entry]
                if missing:
                    raise ValueError(
                        f"Search space entry for '{key}' missing required key(s): {missing}"
                    )
                if key == "llm":
                    validate_parameters(entry.get("parameters"), entry["provider"])
        if "llm" in self.search_space:
            import json
            from LLMProvider.parameters import canonical_provider
            identities = [json.dumps([canonical_provider(e["provider"]), e["model"],
                                     validate_parameters(e.get("parameters"))], sort_keys=True)
                          for e in self.search_space["llm"]]
            if len(identities) != len(set(identities)):
                raise ValueError("Duplicate LLM variant: provider, model and parameters must differ")
    
    @property
    def stages(self) -> List[str]:
        """Get list of stage names in the search space."""
        return list(self.search_space.keys())
    
    @property
    def total_combinations(self) -> int:
        """Calculate total number of pipeline combinations."""
        return len(self.generate_combinations())
    
    def get_options(self, stage: str) -> List[Any]:
        """
        Get available options for a specific stage.
        
        Args:
            stage: Stage name
            
        Returns:
            List of available options for that stage
        """
        return self.search_space.get(stage, [])
    
    def generate_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all possible pipeline configurations.
        
        Uses itertools.product to create the Cartesian product
        of all stage options.
        
        Returns:
            List of configuration dictionaries, one per combination
        """
        stages = list(self.search_space.keys())
        options = [self.search_space[stage] for stage in stages]
        
        combinations = []
        seen = set()
        for combo in product(*options):
            config = dict(zip(stages, combo))
            # A local Hugging Face reranker model only affects CrossEncoder-
            # based methods. Remove the irrelevant dimension before deduping so
            # non-local methods produce exactly one trial regardless of how
            # many local model IDs were selected.
            reranking_method = str(config.get("reranking", "")).strip().lower()
            if (
                "reranking_model" in config
                and "reranking" in config
                and reranking_method not in {"cross_encoder", "pointwise"}
            ):
                config.pop("reranking_model", None)
            signature = repr(sorted(config.items(), key=lambda item: item[0]))
            if signature in seen:
                continue
            seen.add(signature)
            combinations.append(config)
        
        logger.info(f"Generated {len(combinations)} pipeline combinations")
        return combinations
    
    def generate_combinations_with_ids(self) -> List[tuple]:
        """
        Generate combinations with trial IDs.
        
        Returns:
            List of (trial_id, config_dict) tuples
        """
        combinations = self.generate_combinations()
        return list(enumerate(combinations))
    
    def filter_combinations(
        self,
        completed_ids: set,
        combinations: Optional[List[tuple]] = None
    ) -> List[tuple]:
        """
        Filter out already completed combinations.
        
        Args:
            completed_ids: Set of trial IDs that have been completed
            combinations: Optional pre-generated combinations with IDs
            
        Returns:
            List of (trial_id, config_dict) tuples for remaining trials
        """
        if combinations is None:
            combinations = self.generate_combinations_with_ids()
        
        remaining = [(tid, cfg) for tid, cfg in combinations if tid not in completed_ids]
        logger.info(
            f"Filtered combinations: {len(remaining)} remaining "
            f"(skipped {len(completed_ids)} completed)"
        )
        return remaining
    
    def summary(self) -> str:
        """Get a text summary of the search space."""
        lines = [
            "=" * 50,
            "SEARCH SPACE SUMMARY",
            "=" * 50,
        ]
        
        for stage, options in self.search_space.items():
            lines.append(f"\n{stage}:")
            for opt in options:
                lines.append(f"  - {opt}")
        
        lines.extend([
            "",
            "-" * 50,
            f"Total combinations: {self.total_combinations}",
            "=" * 50,
        ])
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"ConfigSpace(stages={len(self.stages)}, combinations={self.total_combinations})"


def trial_config_to_rag_config(trial_config: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a trial configuration to a MuffakirRAG configuration.
    
    Args:
        trial_config: Trial-specific configuration from search space
        base_config: Base configuration shared across all trials
        
    Returns:
        Complete MuffakirRAG configuration dictionary
    """
    # Deep copy so trial configs never mutate the shared base config
    rag_config = deepcopy(base_config)
    
    for key, value in trial_config.items():
        mapping = CONFIG_MAPPING.get(key)
        if mapping:
            # Primary config key
            rag_config[mapping["config_key"]] = mapping["transform"](value)
            # Secondary method/strategy key (e.g. query_transformer_strategy, reranking_method)
            if "method_key" in mapping:
                method_transform = mapping.get("method_transform", lambda v: v)
                rag_config[mapping["method_key"]] = method_transform(value)
            # Any further derived keys (e.g. chunking's size/overlap alongside its method)
            for extra_key, extra_transform in mapping.get("extra_keys", {}).items():
                rag_config[extra_key] = extra_transform(value)
        else:
            # Should not happen (unknown stages are rejected by ConfigSpace), but keep safe.
            logger.warning(f"Unmapped search-space key '{key}' passed through to RAG config")
            rag_config[key] = value

    # Index-time dimensions (chunking, embedding_model, vector_db_provider) change
    if "llm" in trial_config and trial_config["llm"].get("parameters") is not None:
        rag_config["llm_parameters"] = {**resolve_parameters(base_config), **trial_config["llm"]["parameters"]}

    # Index-time dimensions (chunking, embedding_model, vector_db_provider) change
    # the indexed corpus itself, so each distinct combination needs its own on-disk
    # VectorDB. Namespace db_path/collection_name by a deterministic key computed
    # from the *resolved* config (not the raw base config) so _index_documents()
    # and evaluate_trial() always agree on which index a given trial should use,
    # regardless of whether these values came from a trial override or the base
    # default. Trials that don't vary any index-time dimension all resolve to the
    # same key, so they keep sharing one index exactly as before this change.
    index_key = compute_index_key(rag_config)
    base_db_path = base_config.get("db_path", "./muffakir_db")
    base_collection = base_config.get("collection_name", "MuffakirComposer")
    rag_config["db_path"] = f"{base_db_path}/{index_key}"
    rag_config["collection_name"] = f"{base_collection}_{index_key}"

    return rag_config
