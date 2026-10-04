"""Item content embeddings, one per ID variant.

text  : Sentence-T5 (sentence-transformers/sentence-t5-base, TIGER's encoder) over
        "Title: ... Brand: ... Categories: ... Price: ..."
image : CLIP (openai/clip-vit-base-patch32) or SigLIP (google/siglip-base-patch16-224)
        image features; items without a downloaded image get the mean image embedding and
        are counted in the coverage report
fused : [l2norm(text) * w, l2norm(image) * (1-w)] concatenated
attr  : bag of S3Rec attribute ids (brand/category) -> TF-IDF -> SVD. A sandbox stand-in used
        only when titles/images are unavailable (data.source=s3rec_txt)

Every variant is then projected to a common dimension with PCA so the RQ-VAE sees
inputs of identical size and scale across variants.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def item_text(r: dict) -> str:
    parts = []
    if r.get("title"):
        parts.append(f"Title: {r['title']}")
    if r.get("brand"):
        parts.append(f"Brand: {r['brand']}")
    if r.get("categories"):
        parts.append("Categories: " + ", ".join(r["categories"]))
    if r.get("price"):
        parts.append(f"Price: {r['price']}")
    return ". ".join(parts) or "unknown item"


def text_embeddings(items: list[dict], model_name="sentence-transformers/sentence-t5-base",
                    batch_size=128, device=None) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode([item_text(r) for r in items], batch_size=batch_size, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    return _with_pad_row(emb)


def image_embeddings(items: list[dict], image_dir: Path, model_name="google/siglip-base-patch16-224",
                     batch_size=64, device=None) -> tuple[np.ndarray, dict]:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    proc = AutoProcessor.from_pretrained(model_name)
    feats: dict[int, np.ndarray] = {}
    paths = [(r["item_id"], Path(image_dir) / f"{r['item_id']}.jpg") for r in items]
    paths = [(i, p) for i, p in paths if p.exists()]
    for s in range(0, len(paths), batch_size):
        chunk = paths[s:s + batch_size]
        imgs, ids = [], []
        for i, p in chunk:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                ids.append(i)
            except Exception:
                pass
        if not imgs:
            continue
        with torch.no_grad():
            x = proc(images=imgs, return_tensors="pt").to(device)
            f = image_features_tensor(model.get_image_features(**x))
            f = torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()
        feats.update(zip(ids, f))
    dim = next(iter(feats.values())).shape[0]
    mean = np.mean(np.stack(list(feats.values())), 0)
    emb = np.stack([feats.get(r["item_id"], mean) for r in items]).astype(np.float32)
    cov = {"items": len(items), "with_image": len(feats), "coverage": len(feats) / len(items), "dim": dim}
    return _with_pad_row(emb), cov


def image_features_tensor(out):
    """transformers < 5 returns the image embedding as a tensor; transformers >= 5 returns a
    BaseModelOutputWithPooling whose `pooler_output` is that embedding (projected for CLIP,
    the attention-pool head for SigLIP)."""
    import torch
    if torch.is_tensor(out):
        return out
    for name in ("image_embeds", "pooler_output"):
        v = getattr(out, name, None)
        if v is not None:
            return v
    raise TypeError(f"cannot find image features in {type(out).__name__}")


def attribute_embeddings(items: list[dict], dim: int = 128, seed: int = 0) -> np.ndarray:
    n_attr = 1 + max((a for r in items for a in r.get("attributes", [])), default=0)
    X = np.zeros((len(items), n_attr), dtype=np.float32)
    for row, r in enumerate(items):
        for a in r.get("attributes", []):
            X[row, a] = 1.0
    df = (X > 0).sum(0) + 1
    X = X * np.log(len(items) / df)[None]
    X /= np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)
    rng = np.random.default_rng(seed)
    X = X + 1e-3 * rng.normal(size=X.shape).astype(np.float32)  # break exact ties between identical bags
    return _with_pad_row(pca(X, min(dim, n_attr - 1)))


def fuse(text: np.ndarray, image: np.ndarray, text_weight: float = 0.5) -> np.ndarray:
    return np.concatenate([l2(text) * np.sqrt(text_weight), l2(image) * np.sqrt(1 - text_weight)], 1)


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)


def pca(X: np.ndarray, dim: int) -> np.ndarray:
    """Project rows of X onto the top `dim` principal components (deterministic)."""
    Xc = X - X.mean(0, keepdims=True)
    dim = min(dim, Xc.shape[1], Xc.shape[0])
    _, _, Vt = np.linalg.svd(Xc.astype(np.float64), full_matrices=False)
    return (Xc @ Vt[:dim].T).astype(np.float32)


def project_common(emb_with_pad: np.ndarray, dim: int) -> np.ndarray:
    """PCA-project item rows (skipping padding row 0) to `dim`, then l2-normalize."""
    Z = l2(pca(emb_with_pad[1:], dim))
    return _with_pad_row(Z)


def _with_pad_row(emb: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros((1, emb.shape[1]), dtype=np.float32), emb.astype(np.float32)], 0)
