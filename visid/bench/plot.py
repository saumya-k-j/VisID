"""Recall@10 vs p95 latency plot for the systems benchmark."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {"two-tower+faiss:hnsw": ("#2a7ab0", "o"), "two-tower+faiss:ivf": ("#7fb8da", "s"),
         "two-tower+faiss:flat": ("#123b5a", "*"), "two-tower+atlas:hnsw": ("#13aa52", "^"),
         "two-tower+atlas:enn": ("#0b5d2e", "P")}
GEN_COLORS = ["#d1495b", "#edae49", "#66a182", "#8d6a9f"]


def plot_recall_latency(rows: list[dict], out: Path, title: str = "", metric: str = "recall@10") -> None:
    """y = `metric`: "recall@10" (held-out test item) or "ann_recall@10" (overlap with FAISS Flat top-10)."""
    rows = [r for r in rows if metric in r]
    batch_sizes = sorted({r["batch_size"] for r in rows})
    fig, axes = plt.subplots(1, len(batch_sizes), figsize=(5.2 * len(batch_sizes), 4), squeeze=False)
    gen_systems = sorted({r["system"] for r in rows if r["index"] == "trie-beam"})
    for ax, bs in zip(axes[0], batch_sizes):
        sub = [r for r in rows if r["batch_size"] == bs]
        groups: dict[str, list[dict]] = {}
        for r in sub:
            key = f"{r['system']}:{r['index']}" if r["index"] != "trie-beam" else r["system"]
            groups.setdefault(key, []).append(r)
        for key, rs in groups.items():
            rs = sorted(rs, key=lambda r: r["p95_ms"])
            if key in STYLE:
                color, marker = STYLE[key]
            else:
                color, marker = GEN_COLORS[gen_systems.index(key) % len(GEN_COLORS)], "D"
            ax.plot([r["p95_ms"] for r in rs], [r[metric] for r in rs], marker=marker, color=color,
                    label=key, lw=1.5)
            for r in rs:
                if r["knob"] is not None:
                    ax.annotate(str(r["knob"]), (r["p95_ms"], r[metric]), fontsize=7,
                                xytext=(3, 3), textcoords="offset points", color=color)
        ax.set_xscale("log")
        ax.set_xlabel("p95 latency per batch (ms, log)")
        ax.set_ylabel("ANN Recall@10 (vs FAISS Flat)" if metric == "ann_recall@10" else "Recall@10")
        ax.set_title(f"batch size {bs}")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def bench_table(rows: list[dict], filtered: list[dict] | None = None, meta: dict | None = None) -> str:
    """Markdown tables for bench.json. Recall@10 is against the held-out test item (warm / cold
    split by the target item); ANN recall@10 is overlap with the exact FAISS Flat top-10."""
    def f(x, fmt):
        return format(x, fmt) if isinstance(x, (int, float)) else "–"

    lines = []
    if meta:
        lines += [f"Catalog: {meta.get('catalog_size')} items, embedding dim {meta.get('embedding_dim')}, "
                  f"device {meta.get('gpu') or meta.get('device')}, {meta.get('n_users')} test users, seed {meta.get('seed')}.", ""]
    for bs in sorted({r["batch_size"] for r in rows}):
        lines += [f"## Batch size {bs}", "",
                  "| system | index | knob | ANN R@10 (vs Flat) | R@10 | warm R@10 | cold R@10 | p50 ms | p95 ms |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in [r for r in rows if r["batch_size"] == bs]:
            lines.append(f"| {r['system']} | {r['index']} | {r['knob'] if r['knob'] is not None else '–'} | "
                         f"{f(r.get('ann_recall@10'), '.4f')} | {r['recall@10']:.4f} | {r['warm_recall@10']:.4f} | "
                         f"{r['cold_recall@10']:.4f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} |")
        lines.append("")
    if filtered:
        lines += ["## Filtered search (category of the user's last item, batch 1)", "",
                  "Ground truth: exact inner-product top-10 within the category. Atlas pre-filters with "
                  "`$vectorSearch.filter`; FAISS uses HNSW with an IDSelector at the same efSearch.", "",
                  "| system | knob | ANN R@10 (filtered) | in-category rate | median selectivity | p50 ms | p95 ms |",
                  "|---|---|---|---|---|---|---|"]
        for r in filtered:
            lines.append(f"| {r['system']} | {r['knob']} | {r['ann_recall@10']:.4f} | {r['match_rate']:.4f} | "
                         f"{r['median_filter_selectivity']:.3f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} |")
        lines.append("")
    return "\n".join(lines)


def resume_facts(meta: dict, rows: list[dict], filtered: list[dict]) -> str:
    """Only numbers measured in this run: FAISS HNSW efSearch=X next to Atlas numCandidates=X."""
    def pick(system, index, knob, bs):
        return next((r for r in rows if r["system"] == system and r["index"] == index
                     and r["knob"] == knob and r["batch_size"] == bs), None)

    a = meta.get("atlas", {})
    out = ["RESUME FACTS (measured in this run; nothing estimated)",
           f"catalog size: {meta['catalog_size']} items | embedding dim: {meta['embedding_dim']} | "
           f"device: {meta.get('gpu') or meta.get('device')} | test users: {meta.get('n_users_bs1', '?')} at batch 1, {meta['n_users']} at batch 64 | "
           f"Atlas server {a.get('server_version')}, index build {a.get('index_build_s', float('nan')):.1f}s"]
    for bs in sorted({r["batch_size"] for r in rows}):
        for r in rows:
            if r["system"] != "two-tower+atlas" or r["index"] != "hnsw" or r["batch_size"] != bs:
                continue
            f = pick("two-tower+faiss", "hnsw", r["knob"], bs)
            if f is None:
                continue
            out.append(f"batch {bs:>2}, efSearch = numCandidates = {r['knob']:>3}: "
                       f"FAISS HNSW Recall@10 vs Flat {f['ann_recall@10']:.3f}, p95 {f['p95_ms']:.2f} ms | "
                       f"Atlas Recall@10 vs Flat {r['ann_recall@10']:.3f}, p95 {r['p95_ms']:.2f} ms")
    for r in filtered:
        out.append(f"filtered (category pre-filter, median selectivity {r['median_filter_selectivity']:.3f}), "
                   f"{r['system']} at {r['knob']}: Recall@10 vs exact filtered {r['ann_recall@10']:.3f}, "
                   f"in-category {r['match_rate']:.3f}, p95 {r['p95_ms']:.2f} ms")
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    p = Path(sys.argv[1])
    plot_recall_latency(json.loads(p.read_text()), p.with_suffix(".png"))
