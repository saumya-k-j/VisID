"""Spec tests for ranking metrics, checked against hand-computed values.

Leave-one-out has exactly one relevant item per user, so
  Recall@K = 1 if the target is in the top K else 0
  NDCG@K   = 1 / log2(rank + 2) if rank < K else 0   (rank is 0-based)
averaged over users.
"""
import math

import numpy as np
import pytest
import torch

from visid.eval.metrics import target_ranks, compute_metrics, topk_from_scores


def test_target_ranks():
    topk = np.array([[5, 3, 9], [1, 2, 3], [7, 7, 7], [4, 8, 6]])
    targets = np.array([5, 3, 2, 6])
    np.testing.assert_array_equal(target_ranks(topk, targets), [0, 2, -1, 2])


def test_hand_computed_example():
    # ranks: hit at position 0, hit at 2, hit at 7, miss
    ranks = np.array([0, 2, 7, -1])
    m = compute_metrics(ranks, ks=(5, 10))
    # Recall@5: users 0 and 1 hit -> 2/4
    assert m["recall@5"] == pytest.approx(0.5)
    # Recall@10: users 0, 1, 2 hit -> 3/4
    assert m["recall@10"] == pytest.approx(0.75)
    # NDCG@5: (1/log2(2) + 1/log2(4)) / 4 = (1 + 0.5) / 4
    assert m["ndcg@5"] == pytest.approx(0.375)
    # NDCG@10: (1 + 0.5 + 1/log2(9)) / 4
    assert m["ndcg@10"] == pytest.approx((1 + 0.5 + 1 / math.log2(9)) / 4)
    assert m["n"] == 4


def test_empty_input():
    m = compute_metrics(np.array([], dtype=int), ks=(5,))
    assert m["n"] == 0 and m["recall@5"] == 0.0


def test_topk_from_scores_excludes_padding_and_masks():
    scores = torch.tensor([[100.0, 1.0, 5.0, 3.0, 4.0], [100.0, 9.0, 8.0, 7.0, 6.0]])
    top = topk_from_scores(scores, k=2)
    # column 0 is the padding item and must never be returned
    np.testing.assert_array_equal(top, [[2, 4], [1, 2]])
    exclude = [[2], []]
    top = topk_from_scores(scores, k=2, exclude=exclude)
    np.testing.assert_array_equal(top, [[4, 3], [1, 2]])
