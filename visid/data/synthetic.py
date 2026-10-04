"""Synthetic catalog for smoke tests (`make smoke`). NOT a benchmark.

Items belong to latent clusters; users drift between related clusters. The "text" and
"image" embeddings are noisy views of the cluster (and of a finer sub-style), so the whole
pipeline - IDs, generator, baselines, benchmark - can be exercised end to end in minutes on
a CPU. Numbers produced on this data say nothing about Amazon Beauty.
"""
from __future__ import annotations

import numpy as np


def make_synthetic(n_users=2000, n_items=1500, n_clusters=30, dim=64, seed=0):
    rng = np.random.default_rng(seed)
    cluster = rng.integers(0, n_clusters, size=n_items + 1)
    centers = rng.normal(size=(n_clusters, dim))
    style = rng.normal(size=(n_items + 1, dim)) * 0.5
    text = centers[cluster] + style + rng.normal(size=(n_items + 1, dim)) * 0.8
    image = centers[cluster] + style + rng.normal(size=(n_items + 1, dim)) * 1.2
    text[0] = image[0] = 0
    # cluster transition matrix with locality
    trans = np.exp(-np.abs(np.subtract.outer(np.arange(n_clusters), np.arange(n_clusters))) / 2.0)
    trans /= trans.sum(1, keepdims=True)
    by_cluster = [np.where(cluster[1:] == c)[0] + 1 for c in range(n_clusters)]
    pop = rng.zipf(1.6, size=n_items + 1).astype(float)
    seqs = {}
    for u in range(1, n_users + 1):
        c = rng.integers(n_clusters)
        L = rng.integers(5, 15)
        s = []
        for _ in range(L):
            c = rng.choice(n_clusters, p=trans[c])
            cand = by_cluster[c]
            if len(cand) == 0:
                continue
            p = pop[cand] / pop[cand].sum()
            s.append(int(rng.choice(cand, p=p)))
        if len(s) >= 5:
            seqs[u] = s
    items = [{"item_id": i, "asin": f"SYN{i:05d}", "title": f"synthetic item {i}", "brand": f"b{cluster[i] % 7}",
              "categories": [f"c{cluster[i]}"], "price": "", "image_url": ""} for i in range(1, n_items + 1)]
    return seqs, items, text.astype(np.float32), image.astype(np.float32)
