import torch
import pytest
from scimilarity.nn_models import (
    Encoder,
    Decoder,
    ExpertFFN,
    MoELayer,
    TransformerBlock,
    TransformerMoEEncoder,
    TransformerMoEDecoder,
)

N_GENES = 1000  # small for testing
BATCH_SIZE = 4


class TestExpertFFN:
    def test_output_shape(self):
        expert = ExpertFFN(d_model=64, d_ff=128)
        x = torch.randn(BATCH_SIZE, 10, 64)
        out = expert(x)
        assert out.shape == (BATCH_SIZE, 10, 64)


class TestMoELayer:
    def test_output_shape_and_aux_loss(self):
        moe = MoELayer(d_model=64, d_ff=128, num_experts=4, top_k=2)
        x = torch.randn(BATCH_SIZE, 10, 64)
        out, aux_loss = moe(x)
        assert out.shape == (BATCH_SIZE, 10, 64)
        assert aux_loss.ndim == 0  # scalar
        assert aux_loss >= 0


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(d_model=64, n_heads=4, d_ff=128, num_experts=4, top_k=2)
        x = torch.randn(BATCH_SIZE, 10, 64)
        out, aux_loss = block(x)
        assert out.shape == (BATCH_SIZE, 10, 64)
        assert aux_loss.ndim == 0


class TestTransformerMoEEncoder:
    def test_output_shape(self):
        enc = TransformerMoEEncoder(
            n_genes=N_GENES, latent_dim=64,
            d_model=64, n_heads=4, n_layers=2,
            d_ff=128, num_experts=4, top_k=2,
            patch_size=100,
        )
        x = torch.randn(BATCH_SIZE, N_GENES)
        emb, aux_loss = enc(x)
        assert emb.shape == (BATCH_SIZE, 64)
        assert aux_loss.ndim == 0

    def test_l2_normalized(self):
        enc = TransformerMoEEncoder(
            n_genes=N_GENES, latent_dim=64,
            d_model=64, n_heads=4, n_layers=2,
            d_ff=128, num_experts=4, top_k=2,
            patch_size=100,
        )
        x = torch.randn(BATCH_SIZE, N_GENES)
        emb, _ = enc(x)
        norms = torch.norm(emb, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(BATCH_SIZE), atol=1e-5)

    def test_save_load_state(self, tmp_path):
        enc = TransformerMoEEncoder(
            n_genes=N_GENES, latent_dim=64,
            d_model=64, n_heads=4, n_layers=2,
            d_ff=128, num_experts=4, top_k=2,
            patch_size=100,
        )
        filepath = str(tmp_path / "encoder.ckpt")
        enc.save_state(filepath)
        enc2 = TransformerMoEEncoder(
            n_genes=N_GENES, latent_dim=64,
            d_model=64, n_heads=4, n_layers=2,
            d_ff=128, num_experts=4, top_k=2,
            patch_size=100,
        )
        enc2.load_state(filepath)
        enc.eval()
        enc2.eval()
        x = torch.randn(1, N_GENES)
        with torch.no_grad():
            out1, _ = enc(x)
            out2, _ = enc2(x)
        assert torch.allclose(out1, out2, atol=1e-5)


class TestTransformerMoEDecoder:
    def test_output_shape(self):
        dec = TransformerMoEDecoder(
            n_genes=N_GENES, latent_dim=64,
            d_model=64, n_heads=4, n_layers=2,
            d_ff=128, num_experts=4, top_k=2,
            patch_size=100,
        )
        z = torch.randn(BATCH_SIZE, 64)
        recon, aux_loss = dec(z)
        assert recon.shape == (BATCH_SIZE, N_GENES)
        assert aux_loss.ndim == 0

    def test_save_load_state(self, tmp_path):
        dec = TransformerMoEDecoder(
            n_genes=N_GENES, latent_dim=64,
            d_model=64, n_heads=4, n_layers=2,
            d_ff=128, num_experts=4, top_k=2,
            patch_size=100,
        )
        filepath = str(tmp_path / "decoder.ckpt")
        dec.save_state(filepath)
        dec2 = TransformerMoEDecoder(
            n_genes=N_GENES, latent_dim=64,
            d_model=64, n_heads=4, n_layers=2,
            d_ff=128, num_experts=4, top_k=2,
            patch_size=100,
        )
        dec2.load_state(filepath)
        dec.eval()
        dec2.eval()
        z = torch.randn(1, 64)
        with torch.no_grad():
            out1, _ = dec(z)
            out2, _ = dec2(z)
        assert torch.allclose(out1, out2, atol=1e-5)


class TestExistingModels:
    """Verify existing Encoder and Decoder still work (backward compatibility)."""

    def test_encoder_output_shape(self):
        enc = Encoder(n_genes=N_GENES, latent_dim=64, hidden_dim=[128, 128])
        x = torch.randn(BATCH_SIZE, N_GENES)
        out = enc(x)
        assert out.shape == (BATCH_SIZE, 64)

    def test_decoder_output_shape(self):
        dec = Decoder(n_genes=N_GENES, latent_dim=64, hidden_dim=[128, 128])
        z = torch.randn(BATCH_SIZE, 64)
        out = dec(z)
        assert out.shape == (BATCH_SIZE, N_GENES)
