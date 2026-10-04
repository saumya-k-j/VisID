"""Raw-format loaders on tiny fixtures in each source's real format (the raw Amazon files are
not downloaded in CI)."""
import gzip
import json

from visid.data.preprocess import load_amazon2014, load_amazon2023, k_core, build_sequences


def _gz(path, lines):
    with gzip.open(path, "wt") as f:
        for l in lines:
            f.write(l + "\n")


def test_k_core_iterates_to_fixed_point():
    inter = [("u1", "a", 1), ("u1", "b", 2), ("u2", "a", 1), ("u2", "b", 2), ("u3", "a", 1)]
    # 2-core: u3 has one interaction -> dropped; then every remaining user/item has >= 2
    assert sorted(k_core(inter, 2)) == sorted(inter[:4])


def test_sequences_are_chronological_with_stable_ties():
    inter = [("u1", "b", 5), ("u1", "a", 1), ("u1", "c", 5)]
    seqs, _, item_map = build_sequences(inter)
    inv = {v: k for k, v in item_map.items()}
    assert [inv[i] for i in seqs[1]] == ["a", "b", "c"]


def test_amazon2014_format(tmp_path):
    reviews = [json.dumps({"reviewerID": u, "asin": a, "unixReviewTime": t})
               for u, a, t in [("U1", "A", 1), ("U1", "B", 2), ("U2", "B", 1), ("U2", "A", 3)]]
    # 2014 metadata lines are Python literals (single quotes), not JSON
    meta = [str({"asin": "A", "title": "Rose oil", "brand": "X", "categories": [["Beauty", "Oils"]],
                 "price": 9.5, "imUrl": "http://img/a.jpg"}),
            str({"asin": "B", "title": "Shampoo", "categories": [["Beauty", "Hair"]]})]
    _gz(tmp_path / "r.json.gz", reviews)
    _gz(tmp_path / "m.json.gz", meta)
    seqs, items = load_amazon2014(tmp_path / "r.json.gz", tmp_path / "m.json.gz", k=2)
    by_asin = {r["asin"]: r for r in items}
    assert by_asin["A"]["title"] == "Rose oil" and by_asin["A"]["price"] == "$9.5"
    assert by_asin["A"]["categories"] == ["Beauty", "Oils"]
    assert by_asin["A"]["image_url"] == "http://img/a.jpg"
    assert [r["item_id"] for r in items] == [1, 2]
    a, b = by_asin["A"]["item_id"], by_asin["B"]["item_id"]
    assert seqs == {1: [a, b], 2: [b, a]}


def test_amazon2023_format(tmp_path):
    reviews = [json.dumps({"user_id": u, "asin": a + "x", "parent_asin": a, "timestamp": t})
               for u, a, t in [("U1", "P", 10), ("U1", "Q", 20), ("U2", "Q", 5), ("U2", "P", 7)]]
    meta = [json.dumps({"parent_asin": "P", "title": "Serum", "store": "S", "categories": ["Skin"],
                        "price": None, "images": [{"variant": "PT01", "large": "l2"},
                                                  {"variant": "MAIN", "large": "l1"}]}),
            json.dumps({"parent_asin": "Q", "title": "Mask", "images": []})]
    _gz(tmp_path / "r.jsonl.gz", reviews)
    _gz(tmp_path / "m.jsonl.gz", meta)
    seqs, items = load_amazon2023(tmp_path / "r.jsonl.gz", tmp_path / "m.jsonl.gz", k=2)
    by = {r["asin"]: r for r in items}
    assert by["P"]["image_url"] == "l1" and by["P"]["brand"] == "S" and by["P"]["price"] == ""
    assert by["Q"]["image_url"] == ""
    assert len(seqs) == 2
