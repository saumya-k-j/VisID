import numpy as np
import pytest

from visid.ids.encoders import fuse, project_common, item_text, attribute_embeddings


def test_item_text_template():
    r = {"title": "Rose oil", "brand": "X", "categories": ["Beauty", "Oils"], "price": "$9.5"}
    assert item_text(r) == "Title: Rose oil. Brand: X. Categories: Beauty, Oils. Price: $9.5"
    assert item_text({}) == "unknown item"


def test_fuse_weights_modalities_equally():
    rng = np.random.default_rng(0)
    t, i = rng.normal(size=(5, 8)) * 100, rng.normal(size=(5, 3))
    f = fuse(t, i, 0.5)
    assert f.shape == (5, 11)
    assert np.allclose(np.linalg.norm(f[:, :8], axis=1) ** 2, 0.5)
    assert np.allclose(np.linalg.norm(f, axis=1), 1.0)


def test_project_common_keeps_padding_row():
    rng = np.random.default_rng(0)
    X = np.concatenate([np.zeros((1, 20)), rng.normal(size=(50, 20))]).astype(np.float32)
    Z = project_common(X, 8)
    assert Z.shape == (51, 8)
    assert np.all(Z[0] == 0)
    assert np.allclose(np.linalg.norm(Z[1:], axis=1), 1.0, atol=1e-5)


def test_attribute_embeddings_separate_different_bags():
    items = [{"attributes": [1, 2]}, {"attributes": [1, 2]}, {"attributes": [3]}, {"attributes": [4, 5]}]
    E = attribute_embeddings(items, dim=3)[1:]
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    assert E[0] @ E[1] > 0.99          # identical bags -> (almost) identical embeddings
    assert E[0] @ E[2] < 0.9


def test_image_features_tensor_handles_old_and_new_transformers():
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from visid.ids.encoders import image_features_tensor
    t = torch.randn(2, 8)
    assert image_features_tensor(t) is t                                  # transformers < 5
    out = BaseModelOutputWithPooling(last_hidden_state=torch.randn(2, 5, 8), pooler_output=t)
    assert image_features_tensor(out) is t                                # transformers >= 5
