"""Spec tests for the RQ-VAE and semantic-ID assignment.

Spec:
- Residual quantization round trip: the quantized latent equals the sum of the selected
  codebook vectors, so codes -> codebook lookup reproduces exactly what the decoder sees.
- Each level quantizes the residual left by the previous levels, so the residual norm does
  not grow across levels for a trained quantizer.
- Training on clustered data drives reconstruction error down and uses many codes.
- Semantic IDs: a 4th token disambiguates items that share all 3 codes; the final IDs are
  unique, and the 4th token is the item's index within its collision group.
"""
import numpy as np
import pytest
import torch

from visid.ids.rqvae import RQVAE, train_rqvae
from visid.ids.semantic_ids import add_collision_token, codebook_stats


def _clustered(n=2000, d=32, k=40, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * 3
    lab = rng.integers(0, k, size=n)
    return (centers[lab] + rng.normal(size=(n, d)) * 0.3).astype(np.float32)


def test_quantizer_round_trip_is_exact():
    torch.manual_seed(0)
    m = RQVAE(in_dim=32, hidden=[64], latent_dim=8, n_levels=3, codebook_size=16)
    x = torch.from_numpy(_clustered(256))
    m.quantizer.kmeans_init(m.encoder(x).detach())
    m.eval()
    with torch.no_grad():
        z = m.encoder(x)
        codes, zq, _ = m.quantizer(z)
        assert codes.shape == (256, 3)
        assert torch.allclose(m.quantizer.decode_codes(codes), zq, atol=1e-5)
        assert torch.equal(m.encode(x), codes)  # deterministic in eval mode


def test_residual_norm_non_increasing_after_kmeans_init():
    torch.manual_seed(0)
    m = RQVAE(in_dim=32, hidden=[64], latent_dim=8, n_levels=3, codebook_size=32)
    x = torch.from_numpy(_clustered(1024))
    with torch.no_grad():
        z = m.encoder(x)
        m.quantizer.kmeans_init(z)
        norms = m.quantizer.residual_norms(z)
    assert len(norms) == 4
    for a, b in zip(norms[:-1], norms[1:]):
        assert b <= a + 1e-6


def test_gradients_reach_encoder_and_codebooks():
    torch.manual_seed(0)
    m = RQVAE(in_dim=32, hidden=[64], latent_dim=8, n_levels=3, codebook_size=16)
    x = torch.from_numpy(_clustered(128))
    m.quantizer.kmeans_init(m.encoder(x).detach())
    out = m(x)
    out["loss"].backward()
    assert m.encoder.net[0].weight.grad is not None and m.encoder.net[0].weight.grad.abs().sum() > 0
    for cb in m.quantizer.codebooks:
        assert cb.weight.grad is not None and cb.weight.grad.abs().sum() > 0


def test_training_reduces_reconstruction_and_uses_codes():
    X = _clustered(2000)
    cfg = dict(hidden=[64], latent_dim=16, n_levels=3, codebook_size=32, epochs=60,
               batch_size=256, lr=1e-3, beta=0.25, reset_dead_every=10, seed=0)
    model, history = train_rqvae(X, cfg, device="cpu", verbose=False)
    assert history[-1]["recon"] < 0.5 * history[0]["recon"]
    assert history[-1]["util_l0"] > 0.5  # at least half of level-0 codes in use


def test_collision_token_makes_ids_unique():
    codes = np.array([[1, 2, 3], [1, 2, 3], [0, 0, 0], [1, 2, 3], [0, 0, 1]])
    ids, info = add_collision_token(codes)
    assert ids.shape == (5, 4)
    assert len({tuple(r) for r in ids}) == 5
    np.testing.assert_array_equal(ids[:, 3], [0, 1, 0, 2, 0])  # index within collision group
    assert info["max_group"] == 3
    assert info["collision_rate"] == pytest.approx(3 / 5)


def test_codebook_stats():
    codes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 1]])
    st = codebook_stats(codes, codebook_size=4)
    assert st["util_l0"] == pytest.approx(1.0)
    assert st["util_l1"] == pytest.approx(0.25)
    assert st["perplexity_l0"] == pytest.approx(4.0)
    assert st["perplexity_l1"] == pytest.approx(1.0)
