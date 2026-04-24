from typing import Tuple, TYPE_CHECKING

from .cell_embedding import CellEmbedding
from .knn_backends import KNN_BACKENDS

if TYPE_CHECKING:
    import numpy


class CellSearchKNN(CellEmbedding):
    """A class for searching similar cells using cell embeddings kNN.

    Parameters
    ----------
    model_path: str
        Path to the directory containing model files.
    knn_type: str, default: "hnswlib"
        What type of knn to use, options are ["hnswlib", "tiledb_vector_search"]
    use_gpu: bool, default: False
        Use GPU instead of CPU.

    Examples
    --------
    >>> cs = CellSearchKNN(model_path="/opt/data/model")
    """

    def __init__(
        self,
        model_path: str,
        knn_type: str,
        use_gpu: bool = False,
    ):
        super().__init__(
            model_path=model_path,
            use_gpu=use_gpu,
        )

        if knn_type not in KNN_BACKENDS:
            raise ValueError(
                f"Unknown knn_type {knn_type!r}. "
                f"Available backends: {sorted(KNN_BACKENDS)}"
            )

        self.knn_type = knn_type
        self._backend = None
        self.knn = None
        self.safelist = None
        self.blocklist = None

    def _get_backend(self):
        """Lazily instantiate and return the kNN backend."""
        if self._backend is None:
            backend_cls = KNN_BACKENDS[self.knn_type]
            if self.knn_type == "hnswlib":
                self._backend = backend_cls(
                    space="cosine", dim=self.model.latent_dim
                )
            else:
                self._backend = backend_cls()
        return self._backend

    def load_knn_index(self, knn_file: str, memory_budget: int = 50000000):
        """Load the kNN index file

        Parameters
        ----------
        knn_file: str
            Filename of the kNN index.
        memory_budget: int, default: 50000000
            Memory budget for tiledb vector search.
        """

        import os

        backend = self._get_backend()

        if self.knn_type == "hnswlib" and os.path.isfile(knn_file):
            backend.load(knn_file)
            self.knn = backend
        elif self.knn_type == "tiledb_vector_search" and os.path.isdir(knn_file):
            if hasattr(backend, "memory_budget"):
                backend.memory_budget = memory_budget
            backend.load(knn_file)
            self.knn = backend
        else:
            print(f"Warning: No KNN index found at {knn_file}")
            self.knn = None

    def get_nearest_neighbors(
        self, embeddings: "numpy.ndarray", k: int = 50, ef: int = 100
    ) -> Tuple["numpy.ndarray", "numpy.ndarray"]:
        """Get nearest neighbors.
        Used by classes that inherit from CellEmbedding and have an instantiated kNN.

        Parameters
        ----------
        embeddings: numpy.ndarray
            Embeddings as a 2D numpy array.
        k: int, default: 50
            The number of nearest neighbors.
        ef: int, default: 100
            The size of the dynamic list for the nearest neighbors for hnswlib.
            See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md

        Returns
        -------
        nn_idxs: numpy.ndarray
            A 2D numpy array of nearest neighbor indices [num_embeddings x k].
        nn_dists: numpy.ndarray
            A 2D numpy array of nearest neighbor distances [num_embeddings x k].

        Examples
        --------
        >>> nn_idxs, nn_dists = get_nearest_neighbors(embeddings)
        """

        if self.knn is None:
            raise RuntimeError("kNN is not initialized.")
        return self.knn.query(embeddings, k=k, ef=ef)
