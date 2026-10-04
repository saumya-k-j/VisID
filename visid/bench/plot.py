"""Recall@10 vs p95 latency plot for the systems benchmark."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {"two-tower+faiss:hnsw": ("#2a7ab0", "o"), "two-tower+faiss:ivf": ("#7fb8da", "s"),
         "two-tower+faiss:flat": ("#123b5a", "*")}
GEN_COLORS = ["#d1495b", "#edae49", "#66a182", "#8d6a9f"]


def plot_recall_latency(rows: list[dict], out: Path, title: str = "") -> None:
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
            ax.plot([r["p95_ms"] for r in rs], [r["recall@10"] for r in rs], marker=marker, color=color,
                    label=key, lw=1.5)
            for r in rs:
                if r["knob"] is not None:
                    ax.annotate(str(r["knob"]), (r["p95_ms"], r["recall@10"]), fontsize=7,
                                xytext=(3, 3), textcoords="offset points", color=color)
        ax.set_xscale("log")
        ax.set_xlabel("p95 latency per batch (ms, log)")
        ax.set_ylabel("Recall@10")
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


if __name__ == "__main__":
    import sys

    p = Path(sys.argv[1])
    plot_recall_latency(json.loads(p.read_text()), p.with_suffix(".png"))
