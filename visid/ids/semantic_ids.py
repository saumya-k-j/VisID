"""Semantic-ID assembly: collision token and codebook diagnostics."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def add_collision_token(codes: np.ndarray) -> tuple[np.ndarray, dict]:
    """Append a token that is the item's index within the group of items sharing all codes
    (TIGER's disambiguation token). Items are numbered in row order within a group."""
    codes = np.asarray(codes)
    groups: dict[tuple, int] = defaultdict(int)
    extra = np.zeros(len(codes), dtype=np.int64)
    for i, row in enumerate(map(tuple, codes)):
        extra[i] = groups[row]
        groups[row] += 1
    sizes = np.array(list(groups.values()))
    ids = np.concatenate([codes.astype(np.int64), extra[:, None]], axis=1)
    info = {
        "n_items": int(len(codes)),
        "n_unique_prefixes": int(len(groups)),
        "max_group": int(sizes.max()) if len(sizes) else 0,
        "collision_rate": float(sizes[sizes > 1].sum() / max(len(codes), 1)),
    }
    return ids, info


def codebook_stats(codes: np.ndarray, codebook_size: int) -> dict:
    """Per-level utilization (fraction of codes used) and perplexity (exp entropy of the code
    distribution; equals codebook_size for perfectly uniform use, 1 for total collapse)."""
    codes = np.asarray(codes)
    out = {}
    for l in range(codes.shape[1]):
        cnt = np.bincount(codes[:, l], minlength=codebook_size).astype(np.float64)
        p = cnt / cnt.sum()
        nz = p[p > 0]
        out[f"util_l{l}"] = float((cnt > 0).mean())
        out[f"perplexity_l{l}"] = float(np.exp(-(nz * np.log(nz)).sum()))
    prefixes = {tuple(r) for r in codes}
    out["unique_prefix_frac"] = len(prefixes) / max(len(codes), 1)
    return out
