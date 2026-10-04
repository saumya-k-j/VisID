"""Spec tests for leave-one-out + cold-start splits (written before visid/data/splits.py).

Spec:
- Leave-one-out as in TIGER: last item = test target, second-to-last = validation target,
  everything before = training sequence.
- Cold start: a fraction of *test-target* items is held out of training entirely. Their
  interactions are removed from every non-test position, so they never appear in a
  training sequence, a validation history/target, or a test history.
- Test targets are never changed by the cold-start filtering.
- Splitting is deterministic given the seed.
"""
import random
from pathlib import Path

import pytest

from visid.data.splits import make_splits, split_stats, Split


def _random_seqs(n_users=300, n_items=80, seed=0):
    rng = random.Random(seed)
    return {u: [rng.randint(1, n_items) for _ in range(rng.randint(5, 12))] for u in range(1, n_users + 1)}


def test_leave_one_out_positions():
    seqs = {1: [1, 2, 3, 4, 5], 2: [3, 4, 5, 6, 7, 8]}
    sp = make_splits(seqs, cold_frac=0.0, seed=0)
    assert sp.train[1] == [1, 2, 3]
    assert sp.val[1] == ([1, 2, 3], 4)
    assert sp.test[1] == ([1, 2, 3, 4], 5)
    assert sp.train[2] == [3, 4, 5, 6]
    assert sp.test[2] == ([3, 4, 5, 6, 7], 8)
    assert sp.cold_items == set()


def test_cold_items_never_seen_in_training():
    seqs = _random_seqs()
    sp = make_splits(seqs, cold_frac=0.2, seed=1)
    assert len(sp.cold_items) > 0
    original_test_targets = {s[-1] for s in seqs.values()}
    assert sp.cold_items <= original_test_targets
    for u, seq in sp.train.items():
        assert not (set(seq) & sp.cold_items), "cold item leaked into a training sequence"
    for u, (hist, tgt) in sp.val.items():
        assert tgt not in sp.cold_items and not (set(hist) & sp.cold_items)
    for u, (hist, tgt) in sp.test.items():
        assert not (set(hist) & sp.cold_items), "cold item leaked into a test history"


def test_test_targets_preserved_and_flags_correct():
    seqs = _random_seqs()
    sp = make_splits(seqs, cold_frac=0.2, seed=1)
    for u, (hist, tgt) in sp.test.items():
        assert tgt == seqs[u][-1]
    cold_users = set(sp.cold_test_users())
    warm_users = set(sp.warm_test_users())
    assert cold_users and warm_users
    assert cold_users.isdisjoint(warm_users)
    assert cold_users | warm_users == set(sp.test)
    for u in cold_users:
        assert sp.test[u][1] in sp.cold_items


def test_histories_are_consistent():
    seqs = _random_seqs()
    sp = make_splits(seqs, cold_frac=0.1, seed=3)
    for u in sp.test:
        train = sp.train[u]
        vh, vt = sp.val[u]
        th, _ = sp.test[u]
        assert vh == train
        assert th == train + [vt]
        # history order is the original chronological order minus cold items
        expected = [i for i in seqs[u][:-1] if i not in sp.cold_items]
        assert th == expected


def test_deterministic_and_seed_sensitive():
    seqs = _random_seqs()
    a = make_splits(seqs, cold_frac=0.2, seed=7)
    b = make_splits(seqs, cold_frac=0.2, seed=7)
    c = make_splits(seqs, cold_frac=0.2, seed=8)
    assert a.cold_items == b.cold_items
    assert a.cold_items != c.cold_items


def test_json_round_trip(tmp_path):
    sp = make_splits(_random_seqs(), cold_frac=0.2, seed=1)
    p = tmp_path / "split.json"
    sp.save(p)
    sp2 = Split.load(p)
    assert sp2.train == sp.train and sp2.val == sp.val and sp2.test == sp.test
    assert sp2.cold_items == sp.cold_items and sp2.num_items == sp.num_items


def test_stats_hand_computed():
    seqs = {1: [1, 2, 3, 4, 5], 2: [3, 4, 5, 6, 7, 8]}
    st = split_stats(seqs)
    assert st["users"] == 2
    assert st["items"] == 8
    assert st["interactions"] == 11
    assert st["avg_seq_len"] == pytest.approx(5.5)


S3REC = Path(__file__).resolve().parents[1] / "data" / "raw" / "Beauty.txt"


@pytest.mark.skipif(not S3REC.exists(), reason="real Beauty 5-core file not downloaded")
def test_real_beauty_matches_published_statistics():
    from visid.data.preprocess import load_s3rec_txt

    seqs = load_s3rec_txt(S3REC)
    st = split_stats(seqs)
    # Published Amazon Beauty 5-core statistics (TIGER / S3Rec / P5)
    assert (st["users"], st["items"], st["interactions"]) == (22363, 12101, 198502)
