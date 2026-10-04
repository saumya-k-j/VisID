"""Raw data -> chronological user sequences + item metadata table.

Sources (config `data.source`):
  amazon2014   reviews_<Cat>_5.json.gz + meta_<Cat>.json.gz (McAuley 2014; TIGER's data)
  amazon2023   <Cat>.jsonl.gz + meta_<Cat>.jsonl.gz (Amazon Reviews 2023), k-core filtered
  s3rec_txt    S3Rec's preprocessed Beauty.txt (+ item2attributes.json); interactions only,
               same 5-core sequences as TIGER, no titles or images
  synthetic    generated clustered catalog for smoke tests (see visid/data/synthetic.py)

Outputs in data/processed/<name>/:
  sequences.json   {user_id: [item_id, ...]} chronological, ids are 1-based ints
  items.jsonl      one row per item_id: asin, title, brand, categories, price, image_url
"""
from __future__ import annotations

import ast
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

from visid.data.splits import split_stats


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")


def _lines(path: Path) -> Iterator[dict]:
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:  # 2014 metadata is Python-literal, not JSON
                yield ast.literal_eval(line)


def k_core(inter: list[tuple[str, str, int]], k: int) -> list[tuple[str, str, int]]:
    while True:
        uc = Counter(u for u, _, _ in inter)
        ic = Counter(i for _, i, _ in inter)
        keep = [x for x in inter if uc[x[0]] >= k and ic[x[1]] >= k]
        if len(keep) == len(inter):
            return keep
        inter = keep


def build_sequences(inter: list[tuple[str, str, int]]):
    """Chronological sequences; ties broken by file order (stable sort). Items are numbered
    by first appearance in user order, users by sorted raw id (S3Rec / P5 convention)."""
    by_user: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for n, (u, i, t) in enumerate(inter):
        by_user[u].append((t, n, i))
    user_map, item_map, seqs = {}, {}, {}
    for u in sorted(by_user):
        uid = user_map.setdefault(u, len(user_map) + 1)
        s = []
        for _, _, i in sorted(by_user[u]):
            s.append(item_map.setdefault(i, len(item_map) + 1))
        seqs[uid] = s
    return seqs, user_map, item_map


def _price(p) -> str:
    if p in (None, "", "None"):
        return ""
    return f"${p}" if isinstance(p, (int, float)) else str(p)


def load_amazon2014(reviews: Path, meta: Path, k: int = 5):
    inter = [(r["reviewerID"], r["asin"], int(r.get("unixReviewTime", 0))) for r in _lines(reviews)]
    inter = k_core(inter, k)
    seqs, _, item_map = build_sequences(inter)
    want = set(item_map)
    meta_rows = {}
    for m in _lines(meta):
        if m.get("asin") in want:
            cats = m.get("categories") or []
            flat = []
            for path in cats:
                for c in path:
                    if c not in flat:
                        flat.append(c)
            meta_rows[m["asin"]] = {
                "title": m.get("title", ""), "brand": m.get("brand", ""), "categories": flat,
                "price": _price(m.get("price")), "image_url": m.get("imUrl", ""),
                "description": (m.get("description") or "")[:500],
            }
    items = _item_table(item_map, meta_rows)
    return seqs, items


def load_amazon2023(reviews: Path, meta: Path, k: int = 5):
    inter = [(r["user_id"], r.get("parent_asin") or r["asin"], int(r["timestamp"])) for r in _lines(reviews)]
    inter = k_core(inter, k)
    seqs, _, item_map = build_sequences(inter)
    want = set(item_map)
    meta_rows = {}
    for m in _lines(meta):
        a = m.get("parent_asin")
        if a in want:
            imgs = m.get("images") or []
            url = ""
            for im in imgs:  # prefer the MAIN large image
                if im.get("variant", "MAIN") == "MAIN":
                    url = im.get("large") or im.get("hi_res") or im.get("thumb") or ""
                    if url:
                        break
            if not url and imgs:
                url = imgs[0].get("large") or imgs[0].get("thumb") or ""
            meta_rows[a] = {
                "title": m.get("title", ""), "brand": m.get("store", "") or (m.get("details") or {}).get("Brand", ""),
                "categories": m.get("categories") or [m.get("main_category", "")],
                "price": _price(m.get("price")), "image_url": url,
                "description": " ".join(m.get("description") or [])[:500],
            }
    return seqs, _item_table(item_map, meta_rows)


def load_s3rec_txt(path: Path) -> dict[int, list[int]]:
    seqs = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if parts:
                seqs[int(parts[0])] = [int(x) for x in parts[1:]]
    return seqs


def load_s3rec(path: Path, attr_path: Path | None = None):
    seqs = load_s3rec_txt(path)
    n_items = max(i for s in seqs.values() for i in s)
    attrs = json.loads(Path(attr_path).read_text()) if attr_path and Path(attr_path).exists() else {}
    items = [{"item_id": i, "asin": "", "title": "", "brand": "", "categories": [], "price": "",
              "image_url": "", "attributes": attrs.get(str(i), [])} for i in range(1, n_items + 1)]
    return seqs, items


def _item_table(item_map: dict[str, int], meta_rows: dict[str, dict]) -> list[dict]:
    rows = []
    for asin, iid in sorted(item_map.items(), key=lambda x: x[1]):
        m = meta_rows.get(asin, {})
        rows.append({"item_id": iid, "asin": asin, "title": m.get("title", ""), "brand": m.get("brand", ""),
                     "categories": m.get("categories", []), "price": m.get("price", ""),
                     "image_url": m.get("image_url", ""), "description": m.get("description", "")})
    return rows


def write_processed(out_dir: Path, seqs: dict, items: list[dict]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sequences.json").write_text(json.dumps({str(u): s for u, s in seqs.items()}))
    with open(out_dir / "items.jsonl", "w") as f:
        for r in items:
            f.write(json.dumps(r) + "\n")
    st = split_stats(seqs)
    st["items_with_title"] = sum(1 for r in items if r.get("title"))
    st["items_with_image_url"] = sum(1 for r in items if r.get("image_url"))
    (out_dir / "stats.json").write_text(json.dumps(st, indent=2))
    return st


def load_processed(out_dir: Path):
    seqs = {int(u): s for u, s in json.loads((out_dir / "sequences.json").read_text()).items()}
    items = [json.loads(l) for l in open(out_dir / "items.jsonl")]
    return seqs, items
