"""The table aggregator must average every test-result block across seeds and ignore
non-result fields, even ones whose names start with 'test'."""
import json

import pytest

from visid.cli import cmd_table


def _metrics(r5):
    part = {"n": 10, "recall@5": r5, "recall@10": r5, "ndcg@5": r5, "ndcg@10": r5}
    return {"all": part, "warm": part, "cold": part, "lists": {"coverage": 0.5}}


def test_table_mean_std_and_ignores_scalars(tmp_path):
    root = tmp_path / "toy"
    for seed, r5 in ((0, 0.1), (1, 0.3)):
        d = root / "tiger_text" / f"seed{seed}"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({"test": _metrics(r5), "test_eps0.1": _metrics(r5 / 2),
                                                    "test_beam": 20, "seed": seed}))
    cmd_table({"results_dir": str(tmp_path), "data": {"name": "toy"}}, None)
    agg = json.loads((root / "table.json").read_text())
    assert agg["tiger_text"]["warm"]["recall@5"] == pytest.approx([0.2, 0.1])  # mean, std
    assert agg["tiger_text (eps0.1)"]["cold"]["recall@5"][0] == pytest.approx(0.1)
    assert "tiger_text (beam)" not in agg
