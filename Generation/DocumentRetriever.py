from typing import TYPE_CHECKING, List, Dict, Any, Optional, Union

if TYPE_CHECKING:
    # Deferred: Generation/__init__.py eagerly imports this module, and
    # Muffakir/__init__.py's eager chain imports back into Generation — a
    # module-level import here would be circular. Only needed for the type hint.
    from Muffakir.Enums import RetrievalMethod

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


class DocumentRetriever:
    """Retrieves documents from the RAG pipeline and formats them.

    Supports single-query retrieval and multi-query expansion (deduplicates
    results across queries by page_content).
    """

    def __init__(self, pipeline_manager: "RAGPipelineManager"):
        """
        Args:
            pipeline_manager: the ``RAGPipelineManager`` instance (must expose
                ``query_similar_documents(query, k) -> List[Document]``).
        """
        self.pipeline_manager = pipeline_manager

    def retrieve_documents(
        self,
        query: Union[str, List[str]],
        k: int = 2,
        method: Optional["RetrievalMethod"] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k similar documents, preserving full metadata.

        Supports single query string or list of queries (e.g. from Multi-Query
        Expansion).  Deduplicates results across multiple queries by page_content.

        :param method: override the retrieval method for this call only (passed
            straight through to ``pipeline_manager.query_similar_documents`` as an
            explicit argument, never via shared mutable state).

        Returns a list of dicts with keys ``page_content``, ``source``, and
        ``metadata`` (the full original metadata dict).
        """
        if isinstance(query, list):
            seen_content = set()
            all_docs = []
            for q in query:
                if not q or not str(q).strip():
                    continue
                results = self.pipeline_manager.query_similar_documents(q, k, method=method)
                for doc in results:
                    content = doc.page_content.strip()
                    if content not in seen_content:
                        seen_content.add(content)
                        all_docs.append({
                            "source": doc.metadata.get("source", "Unknown"),
                            "page_content": doc.page_content,
                            "metadata": dict(doc.metadata),
                        })
            return all_docs
        else:
            results = self.pipeline_manager.query_similar_documents(query, k, method=method)
            return [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "page_content": doc.page_content,
                    "metadata": dict(doc.metadata),
                }
                for doc in results
            ]

    def format_documents(self, retrieval_result: List[Dict[str, Any]]) -> List[Document]:
        """Format retrieved data into LangChain ``Document`` objects.

        Preserves the full ``metadata`` dict from each retrieved document so
        downstream consumers (e.g. Evaluation matching by ``chunk_id``) can
        access all original metadata.
        """
        return [
            Document(page_content=item["page_content"], metadata=item.get("metadata", {"source": item.get("source", "Unknown")}))
            for item in retrieval_result
        ]
