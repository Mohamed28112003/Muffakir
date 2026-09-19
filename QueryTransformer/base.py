from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict

class BaseQueryTransformer(ABC):
    """
    Abstract base class for all Query Transformation strategies.
    Supported strategies include Query Rewriting, Multi-Query Expansion, HyDE, etc.
    """

    @abstractmethod
    def transform(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Union[str, List[str]]:
        """
        Transform the input query into an optimized representation (or list of representations) 
        for RAG vector database retrieval.

        Args:
            query (str): The raw input user query.
            conversation_history (list of dict, optional): Past conversation messages 
                e.g. [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]

        Returns:
            Union[str, List[str]]: Transformed query string or list of query strings.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name (e.g. 'rewrite', 'multi_query')."""
        pass

    def _build_query_with_context(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        latest_label: str = "Latest Query",
    ) -> str:
        """
        Combine conversation history with the latest query into one string.

        Shared by all strategies so the formatting stays consistent; the
        output layout is preserved exactly from the original per-strategy
        implementations.
        """
        if not conversation_history:
            return query
        history_str = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in conversation_history
        )
        return f"[Context History]:\n{history_str}\n\n[{latest_label}]: {query}"
