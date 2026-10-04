"""Leave-one-out splits with a TIGER-style cold-start hold-out.

Given chronological user sequences:
  test target  = last item
  val target   = second-to-last item (after cold filtering)
  train seq    = everything before the val target

Cold start (TIGER, Sec. 4.2): a fraction of the items that appear as test targets is held
out of training entirely. Every interaction with a cold item that is *not* a test target is
dropped, so the models never see cold items as inputs or targets. Test cases whose target is
a cold item form the cold test set; the rest form the warm test set.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Split:
    train: dict[int, list[int]]
    val: dict[int, tuple[list[int], int]]
    test: dict[int, tuple[list[int], int]]
    cold_items: set[int]
    num_items: int
    info: dict = field(default_factory=dict)

    def cold_test_users(self) -> list[int]:
        return [u for u, (_, t) in self.test.items() if t in self.cold_items]

    def warm_test_users(self) -> list[int]:
        return [u for u, (_, t) in self.test.items() if t not in self.cold_items]

    def warm_items(self) -> set[int]:
        return set(range(1, self.num_items + 1)) - self.cold_items

    # ---- persistence
    def save(self, path: str | Path) -> None:
        obj = {
            "num_items": self.num_items,
            "cold_items": sorted(self.cold_items),
            "train": {str(u): s for u, s in self.train.items()},
            "val": {str(u): [h, t] for u, (h, t) in self.val.items()},
            "test": {str(u): [h, t] for u, (h, t) in self.test.items()},
            "info": self.info,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj))

    @classmethod
    def load(cls, path: str | Path) -> "Split":
        o = json.loads(Path(path).read_text())
        return cls(
            train={int(u): s for u, s in o["train"].items()},
            val={int(u): (h, t) for u, (h, t) in o["val"].items()},
            test={int(u): (h, t) for u, (h, t) in o["test"].items()},
            cold_items=set(o["cold_items"]),
            num_items=o["num_items"],
            info=o.get("info", {}),
        )


def make_splits(seqs: dict[int, list[int]], cold_frac: float = 0.0, seed: int = 0,
                num_items: int | None = None) -> Split:
    seqs = {u: list(s) for u, s in seqs.items() if len(s) >= 3}
    if num_items is None:
        num_items = max(i for s in seqs.values() for i in s)
    test_targets = sorted({s[-1] for s in seqs.values()})
    rng = random.Random(seed)
    n_cold = int(round(cold_frac * len(test_targets)))
    cold = set(rng.sample(test_targets, n_cold)) if n_cold else set()

    train, val, test = {}, {}, {}
    dropped_users, dropped_interactions = 0, 0
    for u in sorted(seqs):
        s = seqs[u]
        target = s[-1]
        clean = [i for i in s[:-1] if i not in cold]
        dropped_interactions += (len(s) - 1) - len(clean)
        if len(clean) < 2:  # need >= 1 training item and a validation target
            dropped_users += 1
            continue
        test[u] = (clean, target)
        val[u] = (clean[:-1], clean[-1])
        train[u] = clean[:-1]

    info = {"cold_frac": cold_frac, "split_seed": seed, "n_cold_items": len(cold),
            "n_test_target_items": len(test_targets), "dropped_users": dropped_users,
            "dropped_cold_interactions": dropped_interactions,
            "n_cold_test": sum(1 for _, t in test.values() if t in cold),
            "n_warm_test": sum(1 for _, t in test.values() if t not in cold)}
    return Split(train, val, test, cold, num_items, info)


def split_stats(seqs: dict[int, list[int]]) -> dict:
    n_users = len(seqs)
    items = {i for s in seqs.values() for i in s}
    n_inter = sum(len(s) for s in seqs.values())
    return {
        "users": n_users,
        "items": len(items),
        "interactions": n_inter,
        "avg_seq_len": n_inter / max(n_users, 1),
        "density": n_inter / max(n_users * len(items), 1),
    }
