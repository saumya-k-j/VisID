"""Single entry point: `python -m visid.cli <command> --config configs/<x>.yaml [opts]`.

Commands (in pipeline order):
  prepare          raw data -> sequences, item table, leave-one-out + cold-start split
  check-images     Day-1 check: fraction of dead image URLs (rule: >15% -> switch to 2023 data)
  download-images  fetch item images
  embed            item embeddings for --variant {text,image,fused,attr}
  build-ids        RQ-VAE semantic IDs for --variant, --seed
  train-sasrec     SASRec baseline
  train-twotower   two-tower (content + ID item tower), evaluated exactly and through FAISS
  train-tiger      generative retriever on --variant semantic IDs
  eval-tiger       re-evaluate a trained generator (e.g. --set tiger.test_beam=40) without retraining
  bench            recall vs p50/p95 latency: TIGER beam search vs FAISS (two-tower)
  table            aggregate every metrics.json into results/<data>/table.md (mean ± std over seeds)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

if sys.platform == "darwin":
    # Apple Silicon: run any op MPS lacks on the CPU instead of crashing, and tolerate the
    # second OpenMP runtime that faiss-cpu and torch each bring on macOS.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import urllib.request
from pathlib import Path

import numpy as np
import torch

from visid.utils import (JsonlLogger, get_device, load_config, make_run_dir, resolve, save_json, set_seed)

# ----------------------------------------------------------------------------- paths
def proc_dir(cfg) -> Path:
    return resolve("data/processed") / cfg["data"]["name"]


def emb_path(cfg, variant: str) -> Path:
    return resolve("data/embeddings") / cfg["data"]["name"] / f"{variant}.npy"


def ids_dir(cfg, variant: str, seed: int) -> Path:
    return resolve("data/ids") / cfg["data"]["name"] / variant / f"seed{seed}"


def load_split(cfg):
    from visid.data.splits import Split
    return Split.load(proc_dir(cfg) / "split.json")


def load_items(cfg):
    return [json.loads(l) for l in open(proc_dir(cfg) / "items.jsonl")]


# ----------------------------------------------------------------------------- prepare
RAW_URLS = {
    "amazon2014": [
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/{f}",
        "https://jmcauley.ucsd.edu/data/amazon/productGraph/categoryFiles/{f}",
    ],
    "amazon2023": [
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{f}",
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/{f}",
    ],
    "s3rec_txt": ["https://raw.githubusercontent.com/RUCAIBox/CIKM2020-S3Rec/master/data/{f}"],
}


def _fetch(source: str, fname: str, raw_dir: Path) -> Path:
    path = raw_dir / fname
    if path.exists():
        return path
    raw_dir.mkdir(parents=True, exist_ok=True)
    for tmpl in RAW_URLS.get(source, []):
        url = tmpl.format(f=fname)
        try:
            print(f"downloading {url}", flush=True)
            urllib.request.urlretrieve(url, path)
            return path
        except Exception as e:  # try the next mirror
            print(f"  failed: {e}", flush=True)
    sys.exit(f"Could not download {fname}. Download it manually into {raw_dir}/ (see README, 'Data').")


def cmd_prepare(cfg, args):
    from visid.data import preprocess as pp
    from visid.data.splits import make_splits

    d = cfg["data"]
    raw = resolve(d.get("raw_dir", "data/raw"))
    src = d["source"]
    if src == "amazon2014":
        seqs, items = pp.load_amazon2014(_fetch(src, d["reviews_file"], raw), _fetch(src, d["meta_file"], raw), d["k_core"])
    elif src == "amazon2023":
        seqs, items = pp.load_amazon2023(_fetch(src, d["reviews_file"], raw), _fetch(src, d["meta_file"], raw), d["k_core"])
    elif src == "s3rec_txt":
        seqs, items = pp.load_s3rec(_fetch(src, d["reviews_file"], raw), _fetch(src, d["attr_file"], raw))
    elif src == "synthetic":
        from visid.data.synthetic import make_synthetic
        seqs, items, text, image = make_synthetic(**d.get("synthetic", {}))
        for v, e in (("text", text), ("image", image)):
            emb_path(cfg, v).parent.mkdir(parents=True, exist_ok=True)
            np.save(emb_path(cfg, v), e)
    else:
        raise ValueError(src)
    st = pp.write_processed(proc_dir(cfg), seqs, items)
    split = make_splits(seqs, d["cold_frac"], d["split_seed"], num_items=len(items))
    split.save(proc_dir(cfg) / "split.json")
    st["split"] = split.info
    save_json(st, proc_dir(cfg) / "stats.json")
    print(json.dumps(st, indent=2))
    exp = d.get("expected_stats")
    if exp:
        got = {k: st[k] for k in exp}
        if got != exp:
            sys.exit(f"STATS MISMATCH: expected {exp}, got {got}. Do not start modeling until this passes.")
        print(f"stats match published numbers: {exp}")


def cmd_check_images(cfg, args):
    from visid.data.images import check_urls
    res = check_urls(load_items(cfg), sample=cfg["data"].get("image_check_sample", 500))
    save_json(res, proc_dir(cfg) / "image_check.json")
    print(json.dumps(res, indent=2))
    if res["dead_frac"] > cfg["data"].get("dead_image_threshold", 0.15):
        print("More than the threshold of image URLs are dead: switch to configs/beauty2023.yaml "
              "and note the dataset change in the results (see README).")


def cmd_download_images(cfg, args):
    from visid.data.images import download_images
    res = download_images(load_items(cfg), resolve("data/images") / cfg["data"]["name"])
    print(json.dumps(res, indent=2))


# ----------------------------------------------------------------------------- embeddings & IDs
def cmd_embed(cfg, args):
    from visid.ids import encoders as enc
    e, v = cfg["embed"], args.variant
    items = load_items(cfg)
    out = emb_path(cfg, v)
    out.parent.mkdir(parents=True, exist_ok=True)
    if v == "text":
        X = enc.text_embeddings(items, e["text_model"], device=None)
    elif v == "image":
        X, cov = enc.image_embeddings(items, resolve("data/images") / cfg["data"]["name"], e["image_model"])
        save_json(cov, out.with_suffix(".coverage.json"))
        print(cov)
    elif v == "fused":
        t, i = np.load(emb_path(cfg, "text")), np.load(emb_path(cfg, "image"))
        X = enc._with_pad_row(enc.fuse(t[1:], i[1:], e.get("fused_text_weight", 0.5)))
    elif v == "attr":
        X = enc.attribute_embeddings(items, dim=e.get("attr_dim", 128))
    else:
        raise ValueError(v)
    np.save(out, X.astype(np.float32))
    print(f"saved {out} {X.shape}")


def common_embedding(cfg, variant: str) -> np.ndarray:
    from visid.ids.encoders import project_common
    if variant == "fused" and not emb_path(cfg, "fused").exists():
        from visid.ids.encoders import fuse, _with_pad_row
        t, i = np.load(emb_path(cfg, "text")), np.load(emb_path(cfg, "image"))
        X = _with_pad_row(fuse(t[1:], i[1:], cfg["embed"].get("fused_text_weight", 0.5)))
    else:
        X = np.load(emb_path(cfg, variant))
    return project_common(X, cfg["embed"]["common_dim"])


def cmd_build_ids(cfg, args):
    from visid.ids.rqvae import train_rqvae, encode_items
    from visid.ids.semantic_ids import add_collision_token, codebook_stats

    seed, v = args.seed, args.variant
    set_seed(seed)
    out = ids_dir(cfg, v, seed)
    out.mkdir(parents=True, exist_ok=True)
    X = common_embedding(cfg, v)[1:]
    device = get_device(cfg.get("device", "auto"))
    icfg = {**cfg["ids"], "seed": seed}
    logger = JsonlLogger(out / "rqvae_log.jsonl")
    t0 = time.time()
    model, hist = train_rqvae(X, icfg, device=device, log_fn=logger.log)
    codes = encode_items(model, X, device)
    ids, info = add_collision_token(codes)
    table = np.full((len(ids) + 1, ids.shape[1]), -1, dtype=np.int64)
    table[1:] = ids
    K = icfg["codebook_size"]
    sizes = [K] * icfg["n_levels"] + [max(info["max_group"], 1)]
    info.update(codebook_stats(codes, K))
    info.update({"level_sizes": sizes, "variant": v, "seed": seed, "train_sec": round(time.time() - t0, 1),
                 "final_recon": hist[-1]["recon"]})
    np.save(out / "ids.npy", table)
    save_json(info, out / "info.json")
    torch.save(model.state_dict(), out / "rqvae.pt")
    print(json.dumps(info, indent=2))


# ----------------------------------------------------------------------------- models
def build_seqrec(cfg, split, content: str | None):
    from visid.models.sasrec import SeqRecModel, IDItemTower, ContentItemTower
    m = cfg["twotower"] if content else cfg["sasrec"]
    if content and content != "none":
        tower = ContentItemTower(common_embedding(cfg, content), m["d_model"], use_id=m.get("use_id", True),
                                 id_dropout=m.get("id_dropout", 0.5), cold_items=split.cold_items)
    else:
        tower = IDItemTower(split.num_items, m["d_model"])
    return SeqRecModel(tower, m["d_model"], m["n_layers"], m["n_heads"], m["max_len"], m["dropout"])


def build_tiger(cfg, variant: str, seed: int):
    from visid.ids.trie import SemanticIDTrie
    from visid.models.tiger import TIGER, TigerTokenizer
    d = ids_dir(cfg, variant, seed)
    if not (d / "ids.npy").exists():
        sys.exit(f"missing semantic IDs {d}; run build-ids --variant {variant} --seed {seed} first")
    ids = np.load(d / "ids.npy")
    sizes = json.loads((d / "info.json").read_text())["level_sizes"]
    t = cfg["tiger"]
    tok = TigerTokenizer(ids, sizes, t["n_user_buckets"], t["max_hist"], t.get("history_align", "left"))
    model = TIGER(tok.vocab_size, sizes, tok.level_offsets, t["d_model"], t["n_heads"], t["n_enc_layers"],
                  t["n_dec_layers"], t["d_ff"], t["dropout"], tok.max_src_len)
    return model, tok, SemanticIDTrie(ids, sizes), ids


def cmd_train_sasrec(cfg, args):
    from visid.models.train import train_seqrec, evaluate_seqrec
    seed = args.seed
    set_seed(seed)
    split = load_split(cfg)
    device = get_device(cfg.get("device", "auto"))
    name = "sasrec" if cfg["sasrec"].get("loss", "ce") == "ce" else f"sasrec_{cfg['sasrec']['loss']}"
    run = make_run_dir(cfg, name, seed)
    model = build_seqrec(cfg, split, None)
    model = train_seqrec(model, split, {**cfg["sasrec"], "seed": seed}, device, JsonlLogger(run / "train_log.jsonl"))
    res = {"test": evaluate_seqrec(model, split, "test", device), "model": name, "seed": seed}
    save_json(res, run / "metrics.json")
    torch.save(model.state_dict(), run / "model.pt")
    _print_res(res["test"])


def cmd_train_twotower(cfg, args):
    from visid.bench.latency import build_faiss
    from visid.eval.metrics import split_metrics, list_stats
    from visid.models.sasrec import pad_left
    from visid.models.train import train_seqrec, evaluate_seqrec
    seed, content = args.seed, args.content or cfg["twotower"]["content"]
    set_seed(seed)
    split = load_split(cfg)
    device = get_device(cfg.get("device", "auto"))
    run = make_run_dir(cfg, f"twotower_{content}", seed)
    model = build_seqrec(cfg, split, content)
    model = train_seqrec(model, split, {**cfg["twotower"], "seed": seed}, device, JsonlLogger(run / "train_log.jsonl"))
    exact = evaluate_seqrec(model, split, "test", device)
    # ANN retrieval through FAISS HNSW at the configured operating point
    model.eval()
    with torch.no_grad():
        item_vecs = model.items.all_items()[1:].float().cpu().numpy()
    index = build_faiss(item_vecs, "hnsw", M=cfg["bench"].get("hnsw_M", 32))
    index.hnsw.efSearch = cfg["twotower"].get("ef_search", 64)
    users = sorted(split.test)
    tops = []
    for s in range(0, len(users), 512):
        with torch.no_grad():
            u = model.user_vectors(pad_left([split.test[x][0] for x in users[s:s + 512]], model.max_len).to(device))
        tops.append(index.search(np.ascontiguousarray(u.float().cpu().numpy()), 10)[1] + 1)
    topk = np.concatenate(tops)
    targets = np.array([split.test[u][1] for u in users])
    ann = split_metrics(topk, targets, np.array([t in split.cold_items for t in targets]))
    ann["lists"] = list_stats(topk, split.num_items, split.cold_items)
    res = {"test": ann, "test_exact": exact, "model": f"twotower_{content}", "seed": seed,
           "faiss": {"index": "hnsw", "M": cfg["bench"].get("hnsw_M", 32), "efSearch": index.hnsw.efSearch}}
    save_json(res, run / "metrics.json")
    torch.save(model.state_dict(), run / "model.pt")
    _print_res(ann)


def cmd_train_tiger(cfg, args):
    from visid.models.train import train_tiger
    seed, v = args.seed, args.variant
    set_seed(seed)
    split = load_split(cfg)
    device = get_device(cfg.get("device", "auto"))
    run = make_run_dir(cfg, f"tiger_{v}" + (f"_{args.tag}" if args.tag else ""), seed)
    model, tok, trie, ids = build_tiger(cfg, v, seed)
    trie.to(device)
    model = train_tiger(model, tok, trie, split, {**cfg["tiger"], "seed": seed}, device,
                        JsonlLogger(run / "train_log.jsonl"))
    torch.save(model.state_dict(), run / "model.pt")
    _eval_tiger_run(cfg, v, seed, split, model, tok, trie, ids, device, run)


def cmd_eval_tiger(cfg, args):
    """Re-evaluate a trained TIGER model (e.g. new beam size or epsilon) without retraining."""
    seed, v = args.seed, args.variant
    split = load_split(cfg)
    device = get_device(cfg.get("device", "auto"))
    run = (resolve(cfg.get("results_dir", "results")) / cfg["data"]["name"]
           / (f"tiger_{v}" + (f"_{args.tag}" if args.tag else "")) / f"seed{seed}")
    model, tok, trie, ids = build_tiger(cfg, v, seed)
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    _eval_tiger_run(cfg, v, seed, split, model.to(device), tok, trie.to(device), ids, device, run)


def _eval_tiger_run(cfg, v, seed, split, model, tok, trie, ids, device, run):
    from visid.models.train import evaluate_tiger, cold_prefix_map
    t = cfg["tiger"]
    cmap = cold_prefix_map(ids, split.cold_items)
    res = {"model": run.parent.name, "seed": seed, "beam": t["test_beam"],
           "history_align": t.get("history_align", "left"),
           "ids": json.loads((ids_dir(cfg, v, seed) / "info.json").read_text())}
    for eps in t.get("cold_eps", [0.0]):
        key = "test" if eps == 0 else f"test_eps{eps}"
        res[key] = evaluate_tiger(model, tok, trie, split, "test", device, beam=t["test_beam"], cold_map=cmap, eps=eps)
        print(f"eps={eps}")
        _print_res(res[key])
    if t.get("measure_unconstrained", True):
        # stretch goal: does free decoding produce IDs that are not in the catalog?
        # Both decodings are measured on the SAME users so their recalls are comparable.
        sub = sorted(split.test)[: t.get("unconstrained_users", 2000)]
        con = evaluate_tiger(model, tok, trie, split, "test", device, beam=t["test_beam"], users=sub)
        unc = evaluate_tiger(model, tok, trie, split, "test", device, beam=t["test_beam"], users=sub, constrained=False)
        res["unconstrained"] = {"n_users": len(sub), "invalid_id_rate@10": unc["invalid_id_rate"],
                                "recall@10_unconstrained": unc["all"]["recall@10"],
                                "recall@10_constrained_same_users": con["all"]["recall@10"]}
        print("unconstrained vs constrained (same users):", res["unconstrained"])
    save_json(res, run / "metrics.json")


def _print_res(r):
    for part in ("all", "warm", "cold"):
        m = r[part]
        print(f"  {part:5s} n={m['n']:6d}  R@5={m['recall@5']:.4f}  R@10={m['recall@10']:.4f}  "
              f"N@5={m['ndcg@5']:.4f}  N@10={m['ndcg@10']:.4f}", flush=True)


# ----------------------------------------------------------------------------- bench
def cmd_bench(cfg, args):
    from visid.bench.latency import bench_ann, bench_generative
    from visid.bench.plot import plot_recall_latency
    seed = args.seed
    set_seed(seed)
    split = load_split(cfg)
    device = get_device(cfg.get("device", "auto"))
    b = cfg["bench"]
    root = resolve(cfg.get("results_dir", "results")) / cfg["data"]["name"]
    users = sorted(split.test)
    if b.get("n_users") and b["n_users"] < len(users):
        users = sorted(np.random.default_rng(seed).choice(users, b["n_users"], replace=False).tolist())
    rows = []
    content = args.content or cfg["twotower"]["content"]
    tt_path = root / f"twotower_{content}" / f"seed{seed}" / "model.pt"
    tt = build_seqrec(cfg, split, content)
    tt.load_state_dict(torch.load(tt_path, map_location="cpu"))
    tt.to(device).eval()
    variants = args.variants.split(",") if args.variants else cfg["bench"].get("variants", ["text"])
    gens = []
    for v in variants:
        p = root / f"tiger_{v}" / f"seed{seed}" / "model.pt"
        if not p.exists():
            print(f"skip tiger_{v}: no trained model at {p}")
            continue
        model, tok, trie, _ = build_tiger(cfg, v, seed)
        model.load_state_dict(torch.load(p, map_location="cpu"))
        gens.append((f"tiger_{v}", model.to(device).eval(), tok, trie.to(device)))
    torch.set_num_threads(b.get("torch_threads", torch.get_num_threads()))
    for bs in b["batch_sizes"]:
        n = len(users) if bs > 1 else min(len(users), b.get("n_users_bs1", 300))
        sub = users[:n]
        rows += bench_ann(tt, split, sub, b, device, bs)
        for name, model, tok, trie in gens:
            rows += bench_generative(model, tok, trie, split, sub, b, device, bs, name=name)
        for r in rows[-(len(gens) * len(b["beams"]) + 1 + len(b["hnsw_ef"]) + len(b["ivf_nprobe"])):]:
            print(f"bs={r['batch_size']:3d} {r['system']:16s} {r['index']:9s} knob={str(r['knob']):5s} "
                  f"R@10={r['recall@10']:.4f} p50={r['p50_ms']:.2f}ms p95={r['p95_ms']:.2f}ms", flush=True)
    meta = {"device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_threads": torch.get_num_threads(), "seed": seed, "n_users": len(users)}
    out = root / "bench" / f"seed{seed}"
    save_json({"meta": meta, "rows": rows}, out / "bench.json")
    plot_recall_latency(rows, out / "recall_vs_latency.png",
                        title=f"{cfg['data']['name']} | {meta['gpu'] or 'CPU'} | seed {seed}")
    print(f"wrote {out}/bench.json and recall_vs_latency.png")


# ----------------------------------------------------------------------------- table
ORDER = ["sasrec", "twotower", "tiger_text", "tiger_image", "tiger_fused", "tiger_attr"]


def cmd_table(cfg, args):
    root = resolve(cfg.get("results_dir", "results")) / cfg["data"]["name"]
    runs: dict[str, list[dict]] = {}
    for p in sorted(root.glob("*/seed*/metrics.json")):
        runs.setdefault(p.parent.parent.name, []).append(json.loads(p.read_text()))
    if not runs:
        sys.exit(f"no metrics under {root}")

    def key(name):
        for i, o in enumerate(ORDER):
            if name.startswith(o):
                return (i, name)
        return (99, name)

    metrics = ["recall@5", "recall@10", "ndcg@5", "ndcg@10"]
    lines = [f"# Results: {cfg['data']['name']}", "",
             "Mean ± std over seeds (n = number of seeds). Leave-one-out test set, full-catalog ranking.", ""]
    agg = {}
    for part in ("warm", "cold", "all"):
        lines += [f"## {part.capitalize()} test cases", "",
                  "| Model | seeds | " + " | ".join(m.replace("recall", "R").replace("ndcg", "NDCG") for m in metrics) + " |",
                  "|---|---|" + "---|" * len(metrics)]
        for name in sorted(runs, key=key):
            for tkey in sorted({k for r in runs[name] for k in r if k.startswith("test") and k != "test_exact" and isinstance(r[k], dict)}):
                rs = [r[tkey] for r in runs[name] if tkey in r]
                label = name + ("" if tkey == "test" else f" ({tkey[5:]})")
                cells = []
                for m in metrics:
                    vals = np.array([r[part][m] for r in rs])
                    agg.setdefault(label, {}).setdefault(part, {})[m] = [float(vals.mean()), float(vals.std())]
                    cells.append(f"{vals.mean():.4f} ± {vals.std():.4f}")
                n_cases = rs[0][part]["n"]
                lines.append(f"| {label} (n_cases={n_cases}) | {len(rs)} | " + " | ".join(cells) + " |")
        lines.append("")
    extra = ["## Diagnostics", "", "| Model | coverage@10 | cold slot share | invalid-ID rate (unconstrained) |",
             "|---|---|---|---|"]
    for name in sorted(runs, key=key):
        rs = runs[name]
        cov = np.mean([r["test"]["lists"]["coverage"] for r in rs if "lists" in r["test"]])
        cs = np.mean([r["test"]["lists"].get("cold_slot_share", 0) for r in rs if "lists" in r["test"]])
        inv = [r["unconstrained"]["invalid_id_rate@10"] for r in rs if "unconstrained" in r]
        extra.append(f"| {name} | {cov:.3f} | {cs:.4f} | {np.mean(inv):.3f} |" if inv else
                     f"| {name} | {cov:.3f} | {cs:.4f} | n/a |")
    lines += extra
    (root / "table.md").write_text("\n".join(lines) + "\n")
    save_json(agg, root / "table.json")
    print("\n".join(lines))


# ----------------------------------------------------------------------------- analysis
def cmd_analyze(cfg, args):
    from visid.eval.analysis import analyze_variant, knn_cooccurrence_precision
    split = load_split(cfg)
    seed = args.seed
    variants = args.variants.split(",") if args.variants else ["text", "image", "fused", "attr"]
    rows = {}
    for v in variants:
        d = ids_dir(cfg, v, seed)
        if not (d / "ids.npy").exists():
            continue
        r = analyze_variant(np.load(d / "ids.npy"), split)
        r.update({k: json.loads((d / "info.json").read_text())[k] for k in ("collision_rate", "n_unique_prefixes", "max_group")})
        if emb_path(cfg, v).exists() or v == "fused":
            r["knn_cooc_precision@10"] = knn_cooccurrence_precision(common_embedding(cfg, v), split)
        rows[v] = r
    if not rows:
        sys.exit("no semantic IDs found; run build-ids first")
    root = resolve(cfg.get("results_dir", "results")) / cfg["data"]["name"] / "analysis"
    save_json(rows, root / f"seed{seed}.json")
    cols = ["consec_agree@1", "lift@1", "consec_agree@2", "lift@2", "cold_agree@1", "cold_agree@2",
            "knn_cooc_precision@10", "collision_rate", "n_unique_prefixes"]
    lines = ["| variant | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for v, r in rows.items():
        lines.append(f"| {v} | " + " | ".join(f"{r.get(c, float('nan')):.4g}" for c in cols) + " |")
    (root / f"seed{seed}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ----------------------------------------------------------------------------- main
COMMANDS = {
    "analyze": cmd_analyze, "eval-tiger": cmd_eval_tiger,
    "prepare": cmd_prepare, "check-images": cmd_check_images, "download-images": cmd_download_images,
    "embed": cmd_embed, "build-ids": cmd_build_ids, "train-sasrec": cmd_train_sasrec,
    "train-twotower": cmd_train_twotower, "train-tiger": cmd_train_tiger, "bench": cmd_bench, "table": cmd_table,
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="visid", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", default="text", help="text | image | fused | attr")
    ap.add_argument("--content", default=None, help="two-tower content embedding (default from config)")
    ap.add_argument("--variants", default=None, help="bench: comma-separated TIGER variants")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="train/eval-tiger: suffix for the results folder, e.g. --tag right")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. tiger.epochs=10")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere+ GPUs (A100): faster, same results to ~1e-3
    COMMANDS[args.command](cfg, args)


if __name__ == "__main__":
    main()
