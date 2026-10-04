"""Day-1 image check and image download.

check_urls: probe a random sample of item image URLs concurrently and report the dead
fraction. The plan's rule: if more than ~15% are dead, switch to Amazon Reviews 2023 (which
ships fresh image URLs) and note the dataset change in the results.

download_images: fetch every item image to data/images/<name>/<item_id>.jpg (resumable).
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import random
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (VisID research crawler)"}


def _probe(url: str, timeout: float = 10.0) -> bool:
    if not url:
        return False
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, headers=UA, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status == 200 and r.headers.get("Content-Type", "image").startswith("image"):
                    if method == "HEAD" and int(r.headers.get("Content-Length") or 1) < 200:
                        return False  # Amazon serves a tiny placeholder gif for dead images
                    return True
        except Exception:
            continue
    return False


def check_urls(items: list[dict], sample: int = 500, seed: int = 0, workers: int = 32) -> dict:
    rng = random.Random(seed)
    pool = [r for r in items]
    pick = rng.sample(pool, min(sample, len(pool)))
    with cf.ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(lambda r: _probe(r.get("image_url", "")), pick))
    n_missing = sum(1 for r in pick if not r.get("image_url"))
    dead = 1 - sum(ok) / max(len(pick), 1)
    return {"sampled": len(pick), "alive": int(sum(ok)), "no_url": n_missing, "dead_frac": dead,
            "switch_to_2023": dead > 0.15}


def _download(url: str, path: Path, timeout: float = 20.0) -> bool:
    if path.exists() and path.stat().st_size > 200:
        return True
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        if len(data) < 200:
            return False
        path.write_bytes(data)
        return True
    except Exception:
        return False


def download_images(items: list[dict], out_dir: Path, workers: int = 32) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(r["item_id"], r.get("image_url", "")) for r in items if r.get("image_url")]
    with cf.ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(lambda x: _download(x[1], out_dir / f"{x[0]}.jpg"), todo))
    res = {"items": len(items), "with_url": len(todo), "downloaded": int(sum(ok)),
           "coverage": sum(ok) / max(len(items), 1)}
    (out_dir / "_download_report.json").write_text(json.dumps(res, indent=2))
    return res
