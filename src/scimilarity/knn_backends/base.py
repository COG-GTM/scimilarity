"""KNNBackend protocol defining the interface all kNN backends must implement."""

from typing import Any, Protocol, Tuple, runtime_checkable

import numpy


@runtime_checkable
class KNNBackend(Protocol):
    """Protocol for kNN backend implementations.

    Every backend must expose ``build``, ``query``, ``save``, ``load``, and a
    ``supports_gpu`` property so that callers can swap implementations without
    changing their own code.
    """

    @property
    def supports_gpu(self) -> bool:
        """Whether this backend can leverage GPU acceleration."""
        ...

    def build(self, embeddings: numpy.ndarray, **kwargs: Any) -> None:
        """Build an index from *embeddings*.

        Parameters
        ----------
        embeddings : numpy.ndarray
            2-D array of shape ``(n_vectors, dim)``.
        **kwargs
            Backend-specific construction parameters.
        """
        ...

    def query(
        self, vectors: numpy.ndarray, k: int, ef: int | None = None
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Return the *k* nearest neighbours for each row in *vectors*.

        Parameters
        ----------
        vectors : numpy.ndarray
            2-D query matrix of shape ``(n_queries, dim)``.
        k : int
            Number of neighbours to retrieve.
        ef : int, optional
            Search-time parameter (interpretation is backend-specific).

        Returns
        -------
        indices : numpy.ndarray
            Shape ``(n_queries, k)`` — neighbour indices.
        distances : numpy.ndarray
            Shape ``(n_queries, k)`` — neighbour distances.
        """
        ...

    def save(self, path: str) -> None:
        """Persist the index to *path*."""
        ...

    def load(self, path: str) -> None:
        """Load a previously saved index from *path*."""
        ...
