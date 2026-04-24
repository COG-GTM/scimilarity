"""
This file contains the neural network architectures.
These are all you need for inference.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple
import math


class Encoder(nn.Module):
    """A class that encapsulates the encoder.

    Parameters
    ----------
    n_genes: int
        The number of genes in the gene space, representing the input dimensions.
    latent_dim: int, default: 128
        The latent space dimensions
    hidden_dim: List[int], default: [1024, 1024]
        A list of hidden layer dimensions, describing the number of layers and their dimensions.
        Hidden layers are constructed in the order of the list for the encoder and in reverse
        for the decoder.
    dropout: float, default: 0.5
        The dropout rate for hidden layers
    input_dropout: float, default: 0.4
        The dropout rate for the input layer
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [1024, 1024],
        dropout: float = 0.5,
        input_dropout: float = 0.4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        for i in range(len(hidden_dim)):
            if i == 0:  # input layer
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=input_dropout),
                        nn.Linear(n_genes, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # hidden layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # output layer
        self.network.append(nn.Linear(hidden_dim[-1], latent_dim))

    def forward(self, x) -> torch.Tensor:
        """Forward.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor corresponding to input layer.

        Returns
        -------
        torch.Tensor
            Output tensor corresponding to output layer.
        """

        for i, layer in enumerate(self.network):
            x = layer(x)
        return F.normalize(x, p=2, dim=1)

    def save_state(self, filename: str):
        """Save model state.

        Parameters
        ----------
        filename: str
            Filename to save the model state.
        """

        torch.save({"state_dict": self.state_dict()}, filename)

    def load_state(self, filename: str, use_gpu: bool = False):
        """Load model state.

        Parameters
        ----------
        filename: str
            Filename containing the model state.
        use_gpu: bool, default: False
            Boolean indicating whether or not to use GPUs.
        """

        if not use_gpu:
            ckpt = torch.load(
                filename, map_location=torch.device("cpu"), weights_only=False
            )
        else:
            ckpt = torch.load(filename, weights_only=False)
        self.load_state_dict(ckpt["state_dict"])


class Decoder(nn.Module):
    """A class that encapsulates the decoder.

    Parameters
    ----------
    n_genes: int
        The number of genes in the gene space, representing the input dimensions.
    latent_dim: int, default: 128
        The latent space dimensions
    hidden_dim: List[int], default: [1024, 1024]
        A list of hidden layer dimensions, describing the number of layers and their dimensions.
        Hidden layers are constructed in the order of the list for the encoder and in reverse
        for the decoder.
    dropout: float, default: 0.5
        The dropout rate for hidden layers
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [1024, 1024],
        dropout: float = 0.5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        for i in range(len(hidden_dim)):
            if i == 0:  # first hidden layer
                self.network.append(
                    nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # other hidden layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # reconstruction layer
        self.network.append(nn.Linear(hidden_dim[-1], n_genes))

    def forward(self, x) -> torch.Tensor:
        """Forward.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor corresponding to input layer.

        Returns
        -------
        torch.Tensor
            Output tensor corresponding to output layer.
        """
        for i, layer in enumerate(self.network):
            x = layer(x)
        return x

    def save_state(self, filename: str):
        """Save model state.

        Parameters
        ----------
        filename: str
            Filename to save the model state.
        """

        torch.save({"state_dict": self.state_dict()}, filename)

    def load_state(self, filename: str, use_gpu: bool = False):
        """Load model state.

        Parameters
        ----------
        filename: str
            Filename containing the model state.
        use_gpu: bool, default: False
            Boolean indicating whether or not to use GPUs.
        """

        if not use_gpu:
            ckpt = torch.load(
                filename, map_location=torch.device("cpu"), weights_only=False
            )
        else:
            ckpt = torch.load(filename, weights_only=False)
        self.load_state_dict(ckpt["state_dict"])


class ExpertFFN(nn.Module):
    """A small 2-layer FFN used as a single expert.

    Parameters
    ----------
    d_model: int
        Model dimension.
    d_ff: int
        Feed-forward hidden dimension.
    dropout: float, default: 0.1
        Dropout rate.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELayer(nn.Module):
    """Mixture of Experts layer.

    Parameters
    ----------
    d_model: int
        Model dimension.
    d_ff: int
        Feed-forward hidden dimension per expert.
    num_experts: int, default: 8
        Number of experts.
    top_k: int, default: 2
        Number of experts selected per token.
    dropout: float, default: 0.1
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList(
            [ExpertFFN(d_model, d_ff, dropout) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]
            weight = top_k_probs[:, k].unsqueeze(-1)
            for i in range(self.num_experts):
                mask = expert_idx == i
                if mask.any():
                    output[mask] += weight[mask] * self.experts[i](x_flat[mask])

        # Auxiliary load-balancing loss (Switch Transformer formulation)
        tokens_per_expert = torch.zeros(
            self.num_experts, device=x.device, dtype=x.dtype
        )
        for i in range(self.num_experts):
            tokens_per_expert[i] = (top_k_indices == i).float().sum()
        f_i = tokens_per_expert / (x_flat.shape[0] * self.top_k)
        P_i = router_probs.mean(dim=0)
        aux_loss = self.num_experts * (f_i * P_i).sum()

        return output.view(batch, seq_len, d_model), aux_loss


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with MoE FFN.

    Parameters
    ----------
    d_model: int
        Model dimension.
    n_heads: int
        Number of attention heads.
    d_ff: int
        Feed-forward hidden dimension.
    num_experts: int, default: 8
        Number of experts.
    top_k: int, default: 2
        Number of experts selected per token.
    dropout: float, default: 0.1
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.dropout(attn_out)

        normed = self.norm2(x)
        moe_out, aux_loss = self.moe(normed)
        x = x + self.dropout(moe_out)

        return x, aux_loss


class TransformerMoEEncoder(nn.Module):
    """Transformer+MoE encoder for single-cell gene expression.

    Parameters
    ----------
    n_genes: int
        Number of genes (input dimension).
    latent_dim: int, default: 128
        Latent space dimension.
    d_model: int, default: 256
        Transformer model dimension.
    n_heads: int, default: 8
        Number of attention heads.
    n_layers: int, default: 4
        Number of transformer blocks.
    d_ff: int, default: 512
        Feed-forward hidden dimension.
    num_experts: int, default: 8
        Number of experts in MoE layers.
    top_k: int, default: 2
        Number of experts selected per token.
    patch_size: int, default: 160
        Size of each gene patch.
    input_dropout: float, default: 0.4
        Dropout rate for input.
    dropout: float, default: 0.1
        Dropout rate for transformer layers.
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        num_experts: int = 8,
        top_k: int = 2,
        patch_size: int = 160,
        input_dropout: float = 0.4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.patch_size = patch_size
        self.num_patches = math.ceil(n_genes / patch_size)
        self.latent_dim = latent_dim
        self.pad_size = self.num_patches * patch_size - n_genes

        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02
        )
        self.input_dropout = nn.Dropout(input_dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, num_experts, top_k, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pad_size > 0:
            x = F.pad(x, (0, self.pad_size))
        x = x.view(x.shape[0], self.num_patches, self.patch_size)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.input_dropout(x)

        total_aux_loss = torch.tensor(0.0, device=x.device)
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.final_norm(x)
        x = x.mean(dim=1)
        x = self.proj(x)
        x = F.normalize(x, p=2, dim=1)
        return x, total_aux_loss

    def save_state(self, filename: str):
        """Save model state."""
        torch.save({"state_dict": self.state_dict()}, filename)

    def load_state(self, filename: str, use_gpu: bool = False):
        """Load model state."""
        if not use_gpu:
            ckpt = torch.load(
                filename, map_location=torch.device("cpu"), weights_only=False
            )
        else:
            ckpt = torch.load(filename, weights_only=False)
        self.load_state_dict(ckpt["state_dict"])


class TransformerMoEDecoder(nn.Module):
    """Transformer+MoE decoder for single-cell gene expression reconstruction.

    Parameters
    ----------
    n_genes: int
        Number of genes (output dimension).
    latent_dim: int, default: 128
        Latent space dimension.
    d_model: int, default: 256
        Transformer model dimension.
    n_heads: int, default: 8
        Number of attention heads.
    n_layers: int, default: 4
        Number of transformer blocks.
    d_ff: int, default: 512
        Feed-forward hidden dimension.
    num_experts: int, default: 8
        Number of experts in MoE layers.
    top_k: int, default: 2
        Number of experts selected per token.
    patch_size: int, default: 160
        Size of each gene patch.
    dropout: float, default: 0.1
        Dropout rate for transformer layers.
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        num_experts: int = 8,
        top_k: int = 2,
        patch_size: int = 160,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.patch_size = patch_size
        self.num_patches = math.ceil(n_genes / patch_size)
        self.pad_size = self.num_patches * patch_size - n_genes

        self.expand = nn.Linear(latent_dim, self.num_patches * d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, num_experts, top_k, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, patch_size)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = z.shape[0]
        x = self.expand(z).view(batch, self.num_patches, -1)
        x = x + self.pos_embed

        total_aux_loss = torch.tensor(0.0, device=z.device)
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.final_norm(x)
        x = self.output_proj(x)
        x = x.reshape(batch, self.num_patches * self.patch_size)
        x = x[:, : self.n_genes]
        return x, total_aux_loss

    def save_state(self, filename: str):
        """Save model state."""
        torch.save({"state_dict": self.state_dict()}, filename)

    def load_state(self, filename: str, use_gpu: bool = False):
        """Load model state."""
        if not use_gpu:
            ckpt = torch.load(
                filename, map_location=torch.device("cpu"), weights_only=False
            )
        else:
            ckpt = torch.load(filename, weights_only=False)
        self.load_state_dict(ckpt["state_dict"])
