from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    import scipy.sparse
    import numpy


class CellEmbedding:
    """A class that embeds cell gene expression data using an ML model.

    Parameters
    ----------
    model_path: str
        Path to the directory containing model files.
    use_gpu: bool, default: False
        Use GPU instead of CPU.

    Examples
    --------
    >>> ce = CellEmbedding(model_path="/opt/data/model")
    """

    def __init__(
        self,
        model_path: str,
        use_gpu: bool = False,
    ):
        import json
        import os
        import pandas as pd
        from .nn_models import Encoder, TransformerMoEEncoder

        self.model_path = model_path
        self.use_gpu = use_gpu

        self.filenames = {
            "model": os.path.join(self.model_path, "encoder.ckpt"),
            "gene_order": os.path.join(self.model_path, "gene_order.tsv"),
        }

        # get gene order
        with open(self.filenames["gene_order"], "r") as fh:
            self.gene_order = [line.strip() for line in fh]

        # detect architecture
        model_config_path = os.path.join(self.model_path, "model_config.json")
        if os.path.exists(model_config_path):
            with open(model_config_path, "r") as fh:
                model_config = json.load(fh)
            self.architecture = model_config.get("architecture", "mlp")
        else:
            self.architecture = "mlp"

        self.n_genes = len(self.gene_order)

        if self.architecture == "transformer_moe":
            self.latent_dim = model_config.get("latent_dim", 128)
            self.model = TransformerMoEEncoder(
                n_genes=self.n_genes,
                latent_dim=self.latent_dim,
                d_model=model_config.get("d_model", 256),
                n_heads=model_config.get("n_heads", 8),
                n_layers=model_config.get("n_layers", 4),
                d_ff=model_config.get("d_ff", 512),
                num_experts=model_config.get("num_experts", 8),
                top_k=model_config.get("top_k", 2),
                patch_size=model_config.get("patch_size", 160),
                input_dropout=0.0,
                dropout=0.0,
            )
        else:
            # get neural network model and infer network size
            with open(os.path.join(self.model_path, "layer_sizes.json"), "r") as fh:
                layer_sizes = json.load(fh)
            # keys: network.1.weight, network.2.weight, ..., network.n.weight
            layers = [
                (key, layer_sizes[key])
                for key in sorted(list(layer_sizes.keys()))
                if "weight" in key and len(layer_sizes[key]) > 1
            ]
            parameters = {
                "latent_dim": layers[-1][1][0],  # last
                "hidden_dim": [layer[1][0] for layer in layers][0:-1],  # all but last
            }
            self.latent_dim = parameters["latent_dim"]
            self.model = Encoder(
                n_genes=self.n_genes,
                latent_dim=parameters["latent_dim"],
                hidden_dim=parameters["hidden_dim"],
            )

        if self.use_gpu is True:
            self.model.cuda()
        self.model.load_state(self.filenames["model"])
        self.model.eval()

        self.int2label = pd.read_csv(
            os.path.join(self.model_path, "label_ints.csv"), index_col=0
        )["0"].to_dict()
        self.label2int = {value: key for key, value in self.int2label.items()}

    def get_embeddings(
        self,
        X: Union["scipy.sparse.csr_matrix", "scipy.sparse.csc_matrix", "numpy.ndarray"],
        num_cells: int = -1,
        buffer_size: int = 10000,
    ) -> "numpy.ndarray":
        """Calculate embeddings for lognormed gene expression matrix.

        Parameters
        ----------
        X: scipy.sparse.csr_matrix, scipy.sparse.csc_matrix, numpy.ndarray
            Gene space aligned and log normalized (tp10k) gene expression matrix.
        num_cells: int, default: -1
            The number of cells to embed, starting from index 0.
            A value of -1 will embed all cells.
        buffer_size: int, default: 10000
            The number of cells to embed in one batch.

        Returns
        -------
        numpy.ndarray
            A 2D numpy array of embeddings [num_cells x latent_space_dimensions].

        Examples
        --------
        >>> from scimilarity.utils import align_dataset, lognorm_counts
        >>> ce = CellEmbedding(model_path="/opt/data/model")
        >>> data = align_dataset(data, ce.gene_order)
        >>> data = lognorm_counts(data)
        >>> embeddings = ce.get_embeddings(data.X)
        """

        import numpy as np
        from scipy.sparse import csr_matrix, csc_matrix
        import torch
        import zarr

        if num_cells == -1:
            num_cells = X.shape[0]

        if (
            (isinstance(X, csr_matrix) or isinstance(X, csc_matrix))
            and (
                isinstance(X.data, zarr.Array)
                or isinstance(X.indices, zarr.Array)
                or isinstance(X.indptr, zarr.Array)
            )
            and num_cells <= buffer_size
        ):
            X.data = X.data[...]
            X.indices = X.indices[...]
            X.indptr = X.indptr[...]

        embedding_parts = []
        with torch.inference_mode():  # disable gradients, not needed for inference
            for i in range(0, num_cells, buffer_size):
                profiles = None
                if isinstance(X, np.ndarray):
                    profiles = torch.Tensor(X[i : i + buffer_size])
                elif isinstance(X, torch.Tensor):
                    profiles = X[i : i + buffer_size]
                elif isinstance(X, csr_matrix) or isinstance(X, csc_matrix):
                    profiles = torch.Tensor(X[i : i + buffer_size].toarray())

                if profiles is None:
                    raise RuntimeError(f"Unknown data type {type(X)}.")

                if self.use_gpu is True:
                    profiles = profiles.cuda()
                output = self.model(profiles)
                if isinstance(output, tuple):
                    output = output[0]  # discard aux_loss during inference
                embedding_parts.append(output)

        if not embedding_parts:
            raise RuntimeError("No valid cells detected.")

        if self.use_gpu:
            # detach, move from gpu into cpu, return as numpy array
            embedding = torch.vstack(embedding_parts).detach().cpu().numpy()
        else:
            # detach, return as numpy array
            embedding = torch.vstack(embedding_parts).detach().numpy()

        if np.isnan(embedding).any():
            raise RuntimeError("NaN detected in embeddings.")

        return embedding
