"""
Centralized Configuration Defaults and Constants for Muffakir Arabic RAG.

Single source of truth for library default values across all modules and facades.
"""

from typing import Dict, Any

# ---------------------------------------------------------------------------
# Core Default Values
# ---------------------------------------------------------------------------

DEFAULT_LANGUAGE: str = "ar"

# ComposerUI Server defaults
DEFAULT_UI_PORT: int = 2811
DEFAULT_UI_HOST: str = "127.0.0.1"

# Embedding defaults
DEFAULT_EMBEDDING_MODEL: str = "mohamed2811/Muffakir_Embedding"
DEFAULT_EMBEDDING_PROVIDER: str = "sentence_transformers"
DEFAULT_EMBEDDING_BATCH_SIZE: int = 16

# Chunking & Document processing defaults
DEFAULT_CHUNK_SIZE: int = 600
DEFAULT_CHUNK_OVERLAP: int = 200
DEFAULT_CHUNKING_METHOD: str = "recursive"

# Vector Database defaults
DEFAULT_VECTOR_DB_PROVIDER: str = "chroma"
DEFAULT_DB_PATH: str = "./muffakir_db"
DEFAULT_COLLECTION_NAME: str = "ArabicBooks"

# Retrieval defaults
DEFAULT_RETRIEVAL_METHOD: str = "max_marginal_relevance"
DEFAULT_K: int = 5
DEFAULT_FETCH_K: int = 15

# LLM defaults
DEFAULT_LLM_TEMPERATURE: float = 0.0
DEFAULT_LLM_MAX_TOKENS: int = 4096

# Generation & Post-processing defaults
DEFAULT_HALLUCINATION_CHECK: bool = True
DEFAULT_HALLUCINATION_METHOD: str = "text_cleaner"

# Query Transformer defaults
DEFAULT_QUERY_TRANSFORMER_STRATEGY: str = "rewrite"

# Reranker defaults
DEFAULT_RERANKING_METHOD: str = "semantic_similarity"
DEFAULT_RERANKING_MODEL: str = "BAAI/bge-reranker-base"

# Web Search defaults
DEFAULT_SEARCH_PROVIDER: str = "tavily"

# Device selection (embedding + reranker local backends)
DEFAULT_DEVICE: str = "auto"


# ---------------------------------------------------------------------------
# Standard Facade Default Configurations
# ---------------------------------------------------------------------------

DEFAULT_RAG_CONFIG: Dict[str, Any] = {
    # Required parameters
    "data_dir": None,
    "api_key": None,

    # LLM Configuration
    "llm_provider": None,
    "llm_model": None,
    "llm_temperature": DEFAULT_LLM_TEMPERATURE,
    "llm_max_tokens": DEFAULT_LLM_MAX_TOKENS,
    "llm_base_url": None,
    "base_url": None,

    # Embedding Configuration
    "embedding_model": DEFAULT_EMBEDDING_MODEL,
    "embedding_provider": DEFAULT_EMBEDDING_PROVIDER,
    "embedding_batch_size": DEFAULT_EMBEDDING_BATCH_SIZE,
    "device": DEFAULT_DEVICE,

    # Document Processing
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    "chunking_method": DEFAULT_CHUNKING_METHOD,

    # Vector Database
    "vector_db_provider": DEFAULT_VECTOR_DB_PROVIDER,
    "vector_db_config": {},
    "db_path": DEFAULT_DB_PATH,
    "collection_name": DEFAULT_COLLECTION_NAME,

    # Retrieval Configuration
    "retrieval_method": DEFAULT_RETRIEVAL_METHOD,
    "k": DEFAULT_K,
    "fetch_k": DEFAULT_FETCH_K,

    # Language & Injected Chunking
    "chunking": None,
    "language": DEFAULT_LANGUAGE,
    "prompt_overrides": {},

    # Query Processing
    "query_transformer": False,
    "query_transformer_strategy": DEFAULT_QUERY_TRANSFORMER_STRATEGY,
    "query_transform_llm_provider": None,
    "query_transform_llm_model": None,
    "skip_document_ingestion": False,
    "hallucination_check": DEFAULT_HALLUCINATION_CHECK,
    "hallucination_method": DEFAULT_HALLUCINATION_METHOD,

    # OCR Configuration
    "use_ocr": False,
    "azure_endpoint": None,
    "azure_api_key": None,

    # Document Parsing
    "document_parser": None,
    "document_parser_config": {},

    # Reranking
    "reranking": False,
    "reranking_method": DEFAULT_RERANKING_METHOD,
    "reranking_model": DEFAULT_RERANKING_MODEL,

    # Adaptive RAG Web Search Fallback
    "adaptive_web_search": False,
    "search_provider": DEFAULT_SEARCH_PROVIDER,
    "search_provider_config": {},
    "fire_crawl_api": None,
}


DEFAULT_COMPOSER_CONFIG: Dict[str, Any] = {
    "data_dir": None,
    "api_key": None,
    "llm_provider": None,
    "llm_model": None,
    "base_url": None,
    "llm_base_url": None,

    "embedding_model": DEFAULT_EMBEDDING_MODEL,
    "embedding_provider": DEFAULT_EMBEDDING_PROVIDER,
    "embedding_batch_size": DEFAULT_EMBEDDING_BATCH_SIZE,
    "device": DEFAULT_DEVICE,

    "chunk_size": DEFAULT_CHUNK_SIZE,
    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    "chunking_method": DEFAULT_CHUNKING_METHOD,

    "vector_db_provider": DEFAULT_VECTOR_DB_PROVIDER,
    "vector_db_config": {},
    "db_path": DEFAULT_DB_PATH,
    "collection_name": "MuffakirComposer",

    "language": DEFAULT_LANGUAGE,
    "prompt_overrides": {},
    "llm_temperature": DEFAULT_LLM_TEMPERATURE,
    "llm_max_tokens": DEFAULT_LLM_MAX_TOKENS,
}


DEFAULT_EVALUATION_CONFIG: Dict[str, Any] = {
    "k": DEFAULT_K,
    "metrics": None,
    "language": DEFAULT_LANGUAGE,
    "prompt_overrides": {},
    "api_key": None,
    "llm_provider": None,
    "llm_model": None,
    "llm_temperature": DEFAULT_LLM_TEMPERATURE,
    "llm_max_tokens": 1000,
    "max_samples": None,
    "output_dir": "./muffakir_eval_results",
    "fail_fast_dataset": True,
    "base_url": None,
    "llm_base_url": None,
}


DEFAULT_SEARCH_CONFIG: Dict[str, Any] = {
    "api_key": None,
    "fire_crawl_api": None,
    "llm_provider": None,
    "llm_model": None,
    "llm_temperature": DEFAULT_LLM_TEMPERATURE,
    "llm_max_tokens": DEFAULT_LLM_MAX_TOKENS,
    "language": DEFAULT_LANGUAGE,
    "prompt_overrides": {},
    "base_url": None,
    "llm_base_url": None,

    "search_provider": "firecrawl",
    "search_provider_config": {},

    "max_depth": 1,
    "time_limit": 30,
    "max_urls": 5,
}


DEFAULT_SYNTHETIC_DATA_CONFIG: Dict[str, Any] = {
    "data_dir": None,
    "api_key": None,
    "llm_provider": None,
    "llm_model": None,
    "base_url": None,
    "llm_base_url": None,

    "llm_temperature": 0.3,
    "llm_max_tokens": 2000,
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    "chunking_method": DEFAULT_CHUNKING_METHOD,
    "language": DEFAULT_LANGUAGE,
    "prompt_overrides": {},
    "output_dir": "./muffakir_synthetic_data",
    "max_retries": 3,
}
