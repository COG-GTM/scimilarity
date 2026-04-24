"""TileDB Vector Search kNN backend."""

import math
from typing import Any, Tuple

import numpy


class TileDBVectorBackend:
    """KNNBackend backed by ``tiledb-vector-search``."""

    def __init__(self, memory_budget: int = 50000000):
        self.memory_budget = memory_budget
        self._index = None

    @property
    def supports_gpu(self) -> bool:
        return False

    @property
    def partitions(self) -> int:
        if self._index is None:
            raise RuntimeError("Index not built or loaded.")
        return self._index.partitions

    def build(self, embeddings: numpy.ndarray, **kwargs: Any) -> None:
        import tiledb
        import tiledb.vector_search as vs
        from tiledb.vector_search import _tiledbvspy as vspy

        index_uri = kwargs.get("index_uri")
        if index_uri is None:
            raise ValueError("index_uri is required for TileDB build.")

        self._index = vs.ingest(
            index_type="IVF_FLAT",
            index_uri=index_uri,
            input_vectors=embeddings,
            distance_metric=vspy.DistanceMetric.COSINE,
            normalized=True,
            filters=tiledb.FilterList([tiledb.LZ4Filter()]),
        )
        self._index.vacuum()

    def query(
        self, vectors: numpy.ndarray, k: int, ef: int | None = None
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        if self._index is None:
            raise RuntimeError("Index not built or loaded.")
        nprobe = int(math.sqrt(self._index.partitions))
        distances, indices = self._index.query(vectors, k=k, nprobe=nprobe)
        return indices, distances

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        import tiledb.vector_search as vs

        self._index = vs.IVFFlatIndex(path, memory_budget=self.memory_budget)
