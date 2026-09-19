"""
Search strategies for Muffakir Composer architecture optimization.

Provides different search algorithms for finding optimal RAG pipeline configurations:
- GridSearch: Brute-force search with parallel execution
- BayesianSearch: (Future) Intelligent sampling using Optuna
"""

from .base_search import BaseSearch
from .grid_search import GridSearch

__all__ = ["BaseSearch", "GridSearch"]