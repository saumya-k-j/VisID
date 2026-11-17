"""MongoDB Atlas Vector Search as a third ANN backend, next to FAISS HNSW and IVF.

The same two-tower item embeddings FAISS indexes are loaded into a collection
(`_id` = item_id, `embedding`, `category`, `is_cold`), indexed with a `vectorSearch` index that
uses the same similarity as the FAISS indexes (inner product -> dotProduct), and queried with the
`$vectorSearch` aggregation stage. `numCandidates` is swept like FAISS HNSW `efSearch`.

Timing matches `latency.bench_ann`: a batch is timed end to end, from user histories through the
two-tower user encoder to top-k item ids. FAISS answers a batch in one call; Atlas has no batch
query, so a batch is `len(batch)` `$vectorSearch` calls (concurrency set by `query_concurrency`).

This module never reads paths or environment variables; `cli.py` passes the connection in.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from visid.bench.latency import _summary, _time_batches, build_faiss
from visid.eval.metrics import overlap_at_k
from visid.models.sasrec import pad_left


# ----------------------------------------------------------------------------- documents & index
def atlas_similarity(faiss_metric: int) -> str:
    """The Atlas similarity that ranks exactly like a FAISS metric."""
    import faiss
    if faiss_metric == faiss.METRIC_INNER_PRODUCT:
        return "dotProduct"
    if faiss_metric == faiss.METRIC_L2:
        return "euclidean"
    raise ValueError(f"no Atlas similarity for FAISS metric {faiss_metric}")


def category_of(item: dict, level: int) -> str:
    """Category at `level` of the item's category path (deepest available if the path is shorter)."""
    cats = [c for c in (item.get("categories") or []) if c]
    if not cats:
        return "unknown"
    return str(cats[min(level, len(cats) - 1)])


def item_documents(item_vecs: np.ndarray, categories: list[str], cold_items: set[int]) -> list[dict]:
    """One document per item. `item_vecs` row r is item r + 1 (the padding row is already dropped)."""
    assert len(item_vecs) == len(categories)
    return [{"_id": i + 1, "embedding": [float(x) for x in v], "category": categories[i],
             "is_cold": (i + 1) in cold_items} for i, v in enumerate(np.asarray(item_vecs, dtype=np.float32))]


def index_definition(dim: int, similarity: str, path: str = "embedding", filter_paths=(),
                     hnsw_options: dict | None = None, quantization: str | None = None) -> dict:
    vec = {"type": "vector", "path": path, "numDimensions": int(dim), "similarity": similarity}
    if quantization and quantization != "none":
        vec["quantization"] = quantization
    if hnsw_options:
        vec["hnswOptions"] = dict(hnsw_options)
    return {"fields": [vec] + [{"type": "filter", "path": p} for p in filter_paths]}


def load_collection(coll, docs: list[dict], batch: int = 5000) -> None:
    """Replace the collection's contents with `docs` (search indexes are dropped with it)."""
    coll.drop()
    for s in range(0, len(docs), batch):
        coll.insert_many(docs[s:s + batch], ordered=False)


def create_vector_index(coll, definition: dict, name: str, timeout_s: float = 600, poll_s: float = 1.0) -> dict:
    """Create the vectorSearch index and block until it is queryable on the loaded documents."""
    from pymongo.operations import SearchIndexModel
    for ix in coll.list_search_indexes(name):
        coll.drop_search_index(name)
        _wait(lambda: not list(coll.list_search_indexes(name)), timeout_s, poll_s, f"drop of {name}")
    coll.create_search_index(model=SearchIndexModel(definition=definition, name=name, type="vectorSearch"))
    t0 = time.perf_counter()

    def ready():
        ix = next(iter(coll.list_search_indexes(name)), None)
        if ix and ix.get("status") == "FAILED":
            raise RuntimeError(f"vector index {name} failed: {ix}")
        return bool(ix and ix.get("queryable") and ix.get("status") == "READY")

    _wait(ready, timeout_s, poll_s, f"index {name}")
    return {"build_s": time.perf_counter() - t0}


def _wait(cond, timeout_s, poll_s, what):
    t_end = time.time() + timeout_s
    while not cond():
        if time.time() > t_end:
            raise TimeoutError(f"timed out waiting for {what}")
        time.sleep(poll_s)


# ----------------------------------------------------------------------------- queries
def vector_search_pipeline(query, k: int, num_candidates: int | None, index: str, path: str = "embedding",
                           filter: dict | None = None, exact: bool = False) -> list[dict]:
    vs = {"index": index, "path": path, "queryVector": [float(x) for x in query]}
    if exact:
        vs["exact"] = True
    else:
        if num_candidates is None or num_candidates < k:
            raise ValueError(f"ANN $vectorSearch needs numCandidates >= limit ({num_candidates} < {k})")
        vs["numCandidates"] = int(num_candidates)
    vs["limit"] = int(k)
    if filter:
        vs["filter"] = filter
    return [{"$vectorSearch": vs}, {"$project": {"_id": 1}}]


def search_topk(coll, query, k: int, num_candidates: int | None, index: str, path: str = "embedding",
                filter: dict | None = None, exact: bool = False) -> np.ndarray:
    """Item ids of one query's top-k, in rank order."""
    pipe = vector_search_pipeline(query, k, num_candidates, index, path, filter, exact)
    return np.array([d["_id"] for d in coll.aggregate(pipe)], dtype=np.int64)


def _pad(ids: np.ndarray, k: int) -> np.ndarray:
    out = np.full(k, -1, dtype=np.int64)
    out[:min(k, len(ids))] = ids[:k]
    return out


def _user_vectors(model, hists, device):
    with torch.no_grad():
        return model.user_vectors(pad_left(hists, model.max_len).to(device)).float().cpu().numpy()


def _flat_truth(items, u, k, mask=None):
    """Exact inner-product top-k (FAISS Flat), optionally restricted to the items in `mask`."""
    import faiss
    ids = np.arange(len(items)) if mask is None else np.where(mask)[0]
    faiss.omp_set_num_threads(1)  # same as bench.faiss_threads; avoids the macOS dual-OpenMP crash
    index = faiss.IndexFlatIP(items.shape[1])
    index.add(np.ascontiguousarray(items[ids]))
    return ids[index.search(np.ascontiguousarray(u), k)[1]] + 1


# ----------------------------------------------------------------------------- benchmark
def bench_atlas(model, split, users, coll, acfg: dict, cfg: dict, device, batch_size: int) -> list[dict]:
    """Two-tower + Atlas $vectorSearch sweep over numCandidates, plus exact (ENN) search."""
    model.eval()
    with torch.no_grad():
        items = model.items.all_items()[1:].float().cpu().numpy()
    hists = [split.test[u][0] for u in users]
    targets = np.array([split.test[u][1] for u in users])
    is_cold = np.array([t in split.cold_items for t in targets])
    batches = [hists[i:i + batch_size] for i in range(0, len(hists), batch_size)]
    k, name = cfg.get("k", 10), acfg.get("index_name", "item_embedding")
    truth = np.concatenate([_flat_truth(items, _user_vectors(model, b, device), k) for b in batches])
    pool = ThreadPoolExecutor(max_workers=acfg.get("query_concurrency", 1))
    rows = []
    settings = [("hnsw", nc, False) for nc in acfg.get("num_candidates", [16, 32, 64, 128, 256])]
    if acfg.get("include_exact", True):
        settings.append(("enn", None, True))
    for kind, nc, exact in settings:
        def run(b):
            u = _user_vectors(model, b, device)
            res = pool.map(lambda v: _pad(search_topk(coll, v, k, nc, name, exact=exact), k), u)
            return np.stack(list(res))

        lat, topk = _time_batches(run, batches, device)
        rows.append({"system": "two-tower+atlas", "index": kind, "knob": nc, "batch_size": batch_size,
                     **_summary(lat, topk, targets, is_cold), "ann_recall@10": overlap_at_k(topk, truth, k)})
    pool.shutdown()
    return rows


def bench_filtered(model, split, users, coll, categories: list[str], acfg: dict, cfg: dict, device) -> list[dict]:
    """Filtered vector search, batch 1: each user's query is restricted to the category of their most
    recent history item. Ground truth is the exact inner-product top-k inside that category.
    Atlas pre-filters with `filter`; FAISS HNSW is run with an IDSelector at the same efSearch for
    reference (FAISS filters during graph traversal and keeps no metadata of its own)."""
    import faiss
    fcfg = acfg.get("filtered", {})
    model.eval()
    with torch.no_grad():
        items = model.items.all_items()[1:].float().cpu().numpy()
    cats = np.array(categories)
    users = users[:fcfg.get("n_users", 500)]
    hists = [split.test[u][0] for u in users]
    qcat = [cats[h[-1] - 1] for h in hists]
    k, name = cfg.get("k", 10), acfg.get("index_name", "item_embedding")
    u_all = np.concatenate([_user_vectors(model, [h], device) for h in hists])
    truth = np.stack([_flat_truth(items, u_all[i:i + 1], k, mask=cats == c)[0] for i, c in enumerate(qcat)])
    selectivity = np.array([(cats == c).mean() for c in qcat])
    sel_ids = {c: np.where(cats == c)[0].astype(np.int64) for c in set(qcat)}
    hnsw = build_faiss(items, "hnsw", M=cfg.get("hnsw_M", 32))
    batches = list(range(len(users)))
    rows = []
    for nc in fcfg.get("num_candidates", [64, 256]):
        def run_atlas(i):
            u = _user_vectors(model, [hists[i]], device)[0]
            return _pad(search_topk(coll, u, k, nc, name, filter={"category": qcat[i]}), k)[None]

        def run_faiss(i):
            u = _user_vectors(model, [hists[i]], device)
            params = faiss.SearchParametersHNSW(sel=faiss.IDSelectorBatch(sel_ids[qcat[i]]), efSearch=nc)
            _, I = hnsw.search(np.ascontiguousarray(u), k, params=params)
            return np.where(I >= 0, I + 1, -1)

        for system, fn in (("two-tower+atlas", run_atlas), ("two-tower+faiss", run_faiss)):
            lat, topk = _time_batches(fn, batches, device)
            in_cat = topk[topk > 0]
            rows.append({"system": system, "index": "hnsw+filter", "knob": nc, "batch_size": 1,
                         "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
                         "mean_ms": float(lat.mean()), "n_queries": len(lat),
                         "ann_recall@10": overlap_at_k(topk, truth, k),
                         "match_rate": float(np.mean(cats[in_cat - 1] == np.repeat(
                             qcat, (topk > 0).sum(1)))) if in_cat.size else 0.0,
                         "median_filter_selectivity": float(np.median(selectivity))})
    return rows
