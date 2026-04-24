"""HNSWLib kNN backend."""

from typing import Any, Optional, Tuple

import numpy


class HNSWLibBackend:
    """KNNBackend backed by ``hnswlib``."""

    def __init__(
        self,
        space: str = "cosine",
        dim: Optional[int] = None,
        ef_construction: int = 1000,
        M: int = 80,
        ef: int = 100,
    ):
        self.space = space
        self.dim = dim
        self.ef_construction = ef_construction
        self.M = M
        self.ef = ef
        self._index = None

    @property
    def supports_gpu(self) -> bool:
        return False

    def build(self, embeddings: numpy.ndarray, **kwargs: Any) -> None:
        import hnswlib

        n_elements, dim = embeddings.shape
        if self.dim is None:
            self.dim = dim

        ef_construction = kwargs.get("ef_construction", self.ef_construction)
        M = kwargs.get("M", self.M)

        self._index = hnswlib.Index(space=self.space, dim=self.dim)
        self._index.init_index(
            max_elements=n_elements,
            ef_construction=ef_construction,
            M=M,
        )
        self._index.set_ef(ef_construction)
        self._index.add_items(embeddings, list(range(n_elements)))

    def query(
        self, vectors: numpy.ndarray, k: int, ef: int | None = None
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        if self._index is None:
            raise RuntimeError("Index not built or loaded.")
        self._index.set_ef(ef if ef is not None else self.ef)
        indices, distances = self._index.knn_query(vectors, k=k)
        return indices, distances

    def save(self, path: str) -> None:
        if self._index is None:
            raise RuntimeError("Index not built or loaded.")
        self._index.save_index(path)

    def load(self, path: str) -> None:
        import hnswlib

        if self.dim is None:
            raise RuntimeError("dim must be set before loading an index.")
        self._index = hnswlib.Index(space=self.space, dim=self.dim)
        self._index.load_index(path)

    # --- hnswlib-specific helpers used by CellAnnotation ---

    def set_ef(self, ef: int) -> None:
        if self._index is not None:
            self._index.set_ef(ef)

    def mark_deleted(self, label: int) -> None:
        if self._index is not None:
            self._index.mark_deleted(label)

    def unmark_deleted(self, label: int) -> None:
        if self._index is not None:
            self._index.unmark_deleted(label)
