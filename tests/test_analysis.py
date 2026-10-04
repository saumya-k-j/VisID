import numpy as np
import pytest

from visid.eval.analysis import prefix_agreement


def test_prefix_agreement_hand_computed():
    ids = np.array([[-1, -1], [0, 0], [0, 1], [1, 0], [1, 1]])  # row 0 = padding
    pairs = np.array([[1, 2], [1, 3], [3, 4]])
    agree, chance = prefix_agreement(ids, pairs, depth=1)
    assert agree == pytest.approx(2 / 3)       # (1,2) and (3,4) share the first code
    assert chance == pytest.approx(0.5)        # two codes, each used by half the items
    agree2, chance2 = prefix_agreement(ids, pairs, depth=2)
    assert agree2 == 0.0 and chance2 == pytest.approx(0.25)
