"""Vector database public API.

Backends import their optional SDKs only when instantiated, so exporting the
manager classes here is lightweight and keeps same-named submodules from
shadowing the historical class imports.
"""

from .base import BaseVectorDBManager
from .factory import create_vector_db
from .ChromaDBManager import ChromaDBManager
from .QdrantDBManager import QdrantDBManager
from .PineconeDBManager import PineconeDBManager
from .FAISSDBManager import FAISSDBManager
from .MilvusDBManager import MilvusDBManager

__all__ = [
    "BaseVectorDBManager",
    "ChromaDBManager",
    "QdrantDBManager",
    "PineconeDBManager",
    "FAISSDBManager",
    "MilvusDBManager",
    "create_vector_db",
]
