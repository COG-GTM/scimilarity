"""kNN backend registry and public exports."""

from typing import Dict, Type

from .base import KNNBackend
from .hnswlib_backend import HNSWLibBackend
from .tiledb_backend import TileDBVectorBackend

KNN_BACKENDS: Dict[str, Type[KNNBackend]] = {
    "hnswlib": HNSWLibBackend,
    "tiledb_vector_search": TileDBVectorBackend,
}

__all__ = [
    "KNNBackend",
    "KNN_BACKENDS",
    "HNSWLibBackend",
    "TileDBVectorBackend",
]
