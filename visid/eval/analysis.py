"""Diagnostics that explain *why* one ID variant beats another.

1. Prefix agreement of consecutive training pairs (i_t -> i_{t+1}) at depth 1..3, against the
   chance rate sum_c p(c)^2. If consecutive purchases share prefixes more often under variant
   A than B, A's IDs encode behaviourally useful similarity and the generator's first tokens
   are easier to predict.
2. The same agreement for cold test cases (last history item -> cold target): this is the
   signal a generator can use to reach an item it never saw.
3. Content kNN co-occurrence precision: for each item, the fraction of its 10 nearest content
   neighbours that are adjacent to it in some training sequence. Measures how predictive each
   modality's notion of similarity is of behaviour, independent of the RQ-VAE.
4. Collision rate / unique prefixes, from the ID build.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from visid.data.splits import Split


def prefix_agreement(ids: np.ndarray, pairs: np.ndarray, depth: int) -> tuple[float, float]:
    a, b = ids[pairs[:, 0], :depth], ids[pairs[:, 1], :depth]
    agree = float((a == b).all(1).mean()) if len(pairs) else 0.0
    _, cnt = np.unique(ids[1:, :depth], axis=0, return_counts=True)
    p = cnt / cnt.sum()
    return agree, float((p ** 2).sum())


def consecutive_pairs(split: Split) -> np.ndarray:
    out = [(s[i], s[i + 1]) for s in split.train.values() for i in range(len(s) - 1) if s[i] != s[i + 1]]
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def cold_pairs(split: Split) -> np.ndarray:
    out = [(h[-1], t) for h, t in split.test.values() if t in split.cold_items and h]
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def knn_cooccurrence_precision(emb: np.ndarray, split: Split, k: int = 10, n_items: int = 3000,
                               seed: int = 0) -> float:
    adj = defaultdict(set)
    for s in split.train.values():
        for a, b in zip(s[:-1], s[1:]):
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
    cand = np.array(sorted(i for i in adj if len(adj[i]) > 0))
    rng = np.random.default_rng(seed)
    pick = rng.choice(cand, size=min(n_items, len(cand)), replace=False)
    E = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)
    E[0] = 0
    warm = np.zeros(len(E), dtype=bool)
    warm[cand] = True
    sims = E[pick] @ E.T
    sims[:, ~warm] = -np.inf
    sims[np.arange(len(pick)), pick] = -np.inf
    nn = np.argpartition(-sims, k, axis=1)[:, :k]
    hits = [len(adj[i] & set(row.tolist())) / k for i, row in zip(pick, nn)]
    return float(np.mean(hits))


def analyze_variant(ids: np.ndarray, split: Split) -> dict:
    out = {}
    cp, cold = consecutive_pairs(split), cold_pairs(split)
    for d in (1, 2, 3):
        agree, chance = prefix_agreement(ids, cp, d)
        out[f"consec_agree@{d}"] = agree
        out[f"chance@{d}"] = chance
        out[f"lift@{d}"] = agree / chance if chance else float("nan")
        out[f"cold_agree@{d}"] = prefix_agreement(ids, cold, d)[0] if len(cold) else float("nan")
    out["n_consecutive_pairs"] = int(len(cp))
    out["n_cold_pairs"] = int(len(cold))
    return out
