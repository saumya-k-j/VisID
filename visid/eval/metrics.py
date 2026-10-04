"""Ranking metrics for leave-one-out evaluation (exactly one relevant item per user)."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch


def target_ranks(topk: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """0-based position of each target in its top-k list, or -1 if absent."""
    topk = np.asarray(topk)
    targets = np.asarray(targets)
    if topk.size == 0:
        return np.full(len(targets), -1, dtype=np.int64)
    hit = topk == targets[:, None]
    ranks = np.where(hit.any(1), hit.argmax(1), -1)
    return ranks.astype(np.int64)


def compute_metrics(ranks: np.ndarray, ks: Iterable[int] = (5, 10)) -> dict:
    ranks = np.asarray(ranks)
    out = {"n": int(len(ranks))}
    for k in ks:
        if len(ranks) == 0:
            out[f"recall@{k}"] = 0.0
            out[f"ndcg@{k}"] = 0.0
            continue
        hit = (ranks >= 0) & (ranks < k)
        gain = np.where(hit, 1.0 / np.log2(np.clip(ranks, 0, None) + 2.0), 0.0)
        out[f"recall@{k}"] = float(hit.mean())
        out[f"ndcg@{k}"] = float(gain.mean())
    return out


def topk_from_scores(scores: torch.Tensor, k: int,
                     exclude: Sequence[Sequence[int]] | None = None) -> np.ndarray:
    """Top-k item ids from a [N, num_items + 1] score matrix. Column 0 (padding) is never
    returned; `exclude` optionally masks per-row item ids (e.g. history items)."""
    scores = scores.clone().float()
    scores[:, 0] = -float("inf")
    if exclude is not None:
        for r, items in enumerate(exclude):
            if len(items):
                scores[r, list(items)] = -float("inf")
    return scores.topk(k, dim=1).indices.cpu().numpy()


def split_metrics(topk: np.ndarray, targets: np.ndarray, is_cold: np.ndarray,
                  ks: Iterable[int] = (5, 10)) -> dict:
    """Metrics on all / warm / cold test cases."""
    ranks = target_ranks(topk, targets)
    is_cold = np.asarray(is_cold, dtype=bool)
    return {
        "all": compute_metrics(ranks, ks),
        "warm": compute_metrics(ranks[~is_cold], ks),
        "cold": compute_metrics(ranks[is_cold], ks),
    }


def list_stats(topk: np.ndarray, num_items: int, cold_items: set[int] | None = None) -> dict:
    """Catalog coverage (fraction of items recommended to anyone) and the share of
    recommendation slots taken by cold items."""
    flat = topk[topk > 0]
    out = {"coverage": float(len(np.unique(flat)) / max(num_items, 1))}
    if cold_items:
        out["cold_slot_share"] = float(np.isin(flat, list(cold_items)).mean()) if flat.size else 0.0
    return out
