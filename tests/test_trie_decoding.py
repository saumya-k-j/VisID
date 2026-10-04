"""Spec tests for the semantic-ID prefix trie and trie-constrained beam search.

Spec:
- The trie answers "which next tokens are valid after this prefix" for every level, and maps
  a full semantic ID back to its item (or -1 if the ID is not in the catalog).
- Constrained beam search only ever returns valid catalog IDs, returns distinct items,
  sorts them by score, and degrades gracefully when the catalog is smaller than the beam.
- With a beam wide enough to be exhaustive, beam search equals brute-force ranking of all
  valid IDs under the model's score.
"""
import itertools

import numpy as np
import pytest
import torch

from visid.ids.trie import SemanticIDTrie
from visid.models.decoding import constrained_beam_search

SIZES = [4, 4, 4, 3]


def _catalog(n=25, seed=0):
    rng = np.random.default_rng(seed)
    all_ids = list(itertools.product(*[range(s) for s in SIZES]))
    pick = rng.choice(len(all_ids), size=n, replace=False)
    ids = np.array([all_ids[i] for i in pick], dtype=np.int64)
    table = np.full((n + 1, 4), -1, dtype=np.int64)  # row 0 = padding item
    table[1:] = ids
    return table


def test_allowed_mask_matches_brute_force():
    table = _catalog()
    trie = SemanticIDTrie(table, SIZES)
    valid = {tuple(r) for r in table[1:]}
    for t in range(4):
        prefixes = sorted({r[:t] for r in valid}) + [tuple([3] * t)] if t else [()]
        pref = torch.tensor(prefixes, dtype=torch.long).reshape(len(prefixes), t)
        mask = trie.allowed_mask(pref)
        assert mask.shape == (len(prefixes), SIZES[t])
        for row, p in enumerate(prefixes):
            expected = {v[t] for v in valid if v[:t] == tuple(p)}
            assert set(torch.nonzero(mask[row]).flatten().tolist()) == expected


def test_lookup_items():
    table = _catalog()
    trie = SemanticIDTrie(table, SIZES)
    full = torch.tensor(table[1:], dtype=torch.long)
    np.testing.assert_array_equal(trie.lookup_items(full).numpy(), np.arange(1, len(table)))
    valid = {tuple(r) for r in table[1:]}
    bogus = next(p for p in itertools.product(*[range(s) for s in SIZES]) if p not in valid)
    assert trie.lookup_items(torch.tensor([bogus])).item() == -1


def _random_step_fn(batch, seed=0):
    g = torch.Generator().manual_seed(seed)
    tables = [torch.randn(batch, *([s for s in SIZES[:t]]), SIZES[t], generator=g) for t in range(4)]

    def step_fn(tokens):  # tokens: [B, W, t]
        B, W, t = tokens.shape
        out = torch.empty(B, W, SIZES[t])
        for b in range(B):
            for w in range(W):
                idx = (b,) + tuple(tokens[b, w].tolist())
                out[b, w] = tables[t][idx].log_softmax(-1)
        return out

    return step_fn


def test_beam_search_returns_only_valid_distinct_sorted():
    table = _catalog()
    trie = SemanticIDTrie(table, SIZES)
    valid = {tuple(r) for r in table[1:]}
    B, W = 3, 8
    tokens, scores, items = constrained_beam_search(_random_step_fn(B), B, trie, W)
    assert tokens.shape == (B, W, 4)
    for b in range(B):
        for w in range(W):
            assert tuple(tokens[b, w].tolist()) in valid
        assert (items[b] > 0).all()
        assert len(set(items[b].tolist())) == W
        assert (scores[b][:-1] >= scores[b][1:]).all()


def test_beam_wider_than_catalog():
    table = _catalog(n=5)
    trie = SemanticIDTrie(table, SIZES)
    tokens, scores, items = constrained_beam_search(_random_step_fn(2), 2, trie, 12)
    for b in range(2):
        found = items[b][items[b] > 0].tolist()
        assert sorted(found) == [1, 2, 3, 4, 5]
        assert torch.isinf(scores[b][items[b] <= 0]).all()


def test_exhaustive_beam_equals_brute_force():
    table = _catalog(n=25)
    trie = SemanticIDTrie(table, SIZES)
    step_fn = _random_step_fn(1, seed=3)
    tokens, scores, items = constrained_beam_search(step_fn, 1, trie, beam_size=64)
    # brute force: score every valid ID with the same conditional distributions
    brute = []
    for i in range(1, len(table)):
        s = 0.0
        for t in range(4):
            prefix = torch.tensor(table[i][:t], dtype=torch.long).view(1, 1, t)
            s += step_fn(prefix)[0, 0, table[i][t]].item()
        brute.append((s, i))
    brute.sort(reverse=True)
    got = [i for i in items[0].tolist() if i > 0]
    assert got[:10] == [i for _, i in brute[:10]]


def test_unconstrained_can_emit_invalid_ids():
    table = _catalog(n=5)
    trie = SemanticIDTrie(table, SIZES)
    _, _, items = constrained_beam_search(_random_step_fn(2), 2, trie, 20, constrained=False)
    assert (items == -1).any(), "with 5 valid IDs out of 192, unconstrained beams must hit invalid IDs"
