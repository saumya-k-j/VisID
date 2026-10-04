"""Systems benchmark: trie-constrained generative retrieval vs FAISS ANN retrieval.

Both pipelines are timed end to end from a batch of user histories to top-k item ids:
  generative : TIGER encoder + constrained beam search        (sweep: beam size)
  ANN        : two-tower user encoder + FAISS search           (sweep: index type / efSearch / nprobe)
Latency is wall-clock per batch (p50 / p95 over batches, after warm-up, CUDA synchronized),
recall@10 is measured on the same test users. Batch sizes 1 and 64 by default.
"""
from __future__ import annotations

import time

import faiss
import numpy as np
import torch

from visid.eval.metrics import split_metrics
from visid.models.sasrec import pad_left


def _sync(device):
    """GPU work is asynchronous: wait for it before reading the clock."""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elif str(device).startswith("mps"):
        torch.mps.synchronize()


def _time_batches(fn, batches, device, warmup: int = 3):
    for b in batches[:warmup]:
        fn(b)
    _sync(device)
    lat, outs = [], []
    for b in batches:
        t0 = time.perf_counter()
        out = fn(b)
        _sync(device)
        lat.append((time.perf_counter() - t0) * 1000)
        outs.append(out)
    return np.array(lat), np.concatenate(outs)


def _summary(lat, topk, targets, is_cold):
    m = split_metrics(topk, targets, is_cold)
    return {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
            "mean_ms": float(lat.mean()), "n_batches": len(lat),
            "recall@10": m["all"]["recall@10"], "ndcg@10": m["all"]["ndcg@10"],
            "cold_recall@10": m["cold"]["recall@10"], "warm_recall@10": m["warm"]["recall@10"]}


def build_faiss(item_vecs: np.ndarray, kind: str, **kw):
    d = item_vecs.shape[1]
    x = np.ascontiguousarray(item_vecs.astype(np.float32))
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, kw.get("M", 32), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = kw.get("efConstruction", 200)
    elif kind == "ivf":
        nlist = kw.get("nlist", max(1, int(np.sqrt(len(x)))))
        q = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(q, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(x)
    else:
        raise ValueError(kind)
    index.add(x)
    return index


def bench_ann(model, split, users, cfg, device, batch_size: int) -> list[dict]:
    """Two-tower + FAISS sweep."""
    model.eval()
    with torch.no_grad():
        items = model.items.all_items()[1:].float().cpu().numpy()  # row r -> item r+1
    hists = [split.test[u][0] for u in users]
    targets = np.array([split.test[u][1] for u in users])
    is_cold = np.array([t in split.cold_items for t in targets])
    batches = [hists[i:i + batch_size] for i in range(0, len(hists), batch_size)]
    k = cfg.get("k", 10)
    rows = []
    settings = [("flat", {}, None)]
    settings += [("hnsw", {"M": cfg.get("hnsw_M", 32)}, ef) for ef in cfg.get("hnsw_ef", [16, 32, 64, 128, 256])]
    settings += [("ivf", {}, npb) for npb in cfg.get("ivf_nprobe", [1, 4, 16, 64])]
    faiss.omp_set_num_threads(cfg.get("faiss_threads", 1))
    for kind, kw, knob in settings:
        index = build_faiss(items, kind, **kw)
        if kind == "hnsw":
            index.hnsw.efSearch = knob
        if kind == "ivf":
            index.nprobe = knob

        def run(b):
            with torch.no_grad():
                u = model.user_vectors(pad_left(b, model.max_len).to(device)).float().cpu().numpy()
            _, I = index.search(np.ascontiguousarray(u), k)
            return I + 1

        lat, topk = _time_batches(run, batches, device)
        rows.append({"system": "two-tower+faiss", "index": kind, "knob": knob, "batch_size": batch_size,
                     **_summary(lat, topk, targets, is_cold)})
    return rows


def bench_generative(model, tok, trie, split, users, cfg, device, batch_size: int, name="tiger") -> list[dict]:
    hists = [split.test[u][0] for u in users]
    targets = np.array([split.test[u][1] for u in users])
    is_cold = np.array([t in split.cold_items for t in targets])
    idx = [list(range(i, min(i + batch_size, len(users)))) for i in range(0, len(users), batch_size)]
    k = cfg.get("k", 10)
    rows = []
    model.eval()
    for beam in cfg.get("beams", [10, 20, 40]):
        def run(ix):
            src = tok.encode_histories([users[i] for i in ix], [hists[i] for i in ix]).to(device)
            items, _ = model.generate(src, trie, beam_size=beam)
            out = np.full((len(ix), k), -1, dtype=np.int64)
            it = items[:, :k].cpu().numpy()
            out[:, :it.shape[1]] = it
            return out

        lat, topk = _time_batches(run, idx, device)
        rows.append({"system": name, "index": "trie-beam", "knob": beam, "batch_size": batch_size,
                     **_summary(lat, topk, targets, is_cold)})
    return rows
