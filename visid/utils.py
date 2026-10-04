"""Config loading, seeding, run directories, and logging helpers."""
from __future__ import annotations

import copy
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- config
def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config. A config may name a parent with `base: other.yaml` (relative
    path); the child is deep-merged over the parent. Overrides are `a.b.c=value` strings
    whose value is parsed as YAML."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    if "base" in cfg:
        parent = load_config(path.parent / cfg.pop("base"))
        cfg = deep_merge(parent, cfg)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# ----------------------------------------------------------------------------- seeds & devices
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------------------- run dirs
def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def make_run_dir(cfg: dict, exp_name: str, seed: int) -> Path:
    """results/<dataset>/<exp_name>/seed<seed>/ with the fully resolved config logged."""
    d = resolve(cfg.get("results_dir", "results")) / cfg["data"]["name"] / exp_name / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    meta = {"git": git_hash(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": seed,
            "torch": str(torch.__version__), "python": platform.python_version(),
            "device": str(get_device(cfg.get("device", "auto")))}
    (d / "config.yaml").write_text(yaml.safe_dump({"meta": meta, **cfg}, sort_keys=False))
    return d


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(type(o))


class JsonlLogger:
    def __init__(self, path: str | Path | None, verbose: bool = True):
        self.path = Path(path) if path else None
        self.verbose = verbose
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def log(self, **row):
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(row, default=_json_default) + "\n")
        if self.verbose:
            print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  flush=True)
