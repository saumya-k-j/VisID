"""Training and inference loops for SASRec / two-tower and TIGER."""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch

from visid.data.splits import Split
from visid.eval.metrics import split_metrics, list_stats, topk_from_scores
from visid.ids.trie import SemanticIDTrie
from visid.models.sasrec import SeqRecModel, pad_left, training_pairs
from visid.models.tiger import TIGER, TigerTokenizer, tiger_examples
from visid.utils import JsonlLogger


def _lr_lambda(warmup: int, total: int):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0))) * 0.9 + 0.1
    return f


def _eval_subset(users: list[int], n: int | None, seed: int = 0) -> list[int]:
    if not n or n >= len(users):
        return users
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(users, size=n, replace=False).tolist())


# ============================================================================ SASRec / two-tower
@torch.no_grad()
def seqrec_topk(model: SeqRecModel, hists: list[list[int]], k: int, device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    items = model.items.all_items()
    for s in range(0, len(hists), batch_size):
        seq = pad_left(hists[s:s + batch_size], model.max_len).to(device)
        scores = model.user_vectors(seq) @ items.t()
        out.append(topk_from_scores(scores, k))
    return np.concatenate(out) if out else np.zeros((0, k), dtype=np.int64)


def evaluate_seqrec(model, split: Split, which: str, device, k: int = 10, users=None) -> dict:
    data = split.val if which == "val" else split.test
    users = users if users is not None else sorted(data)
    hists = [data[u][0] for u in users]
    targets = np.array([data[u][1] for u in users])
    topk = seqrec_topk(model, hists, k, device)
    is_cold = np.array([t in split.cold_items for t in targets])
    res = split_metrics(topk, targets, is_cold)
    res["lists"] = list_stats(topk, split.num_items, split.cold_items)
    return res


def train_seqrec(model: SeqRecModel, split: Split, cfg: dict, device, logger: JsonlLogger) -> SeqRecModel:
    tc = cfg
    inp, tgt = training_pairs(split.train, model.max_len)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.0))
    g = torch.Generator().manual_seed(tc.get("seed", 0))
    val_users = _eval_subset(sorted(split.val), tc.get("val_users"), tc.get("seed", 0))
    best, best_state, bad = -1.0, None, 0
    model.to(device)
    for ep in range(tc["epochs"]):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(inp), generator=g)
        tot, nb = 0.0, 0
        for s in range(0, len(perm), tc["batch_size"]):
            b = perm[s:s + tc["batch_size"]]
            loss = model.loss(inp[b].to(device), tgt[b].to(device), kind=tc.get("loss", "ce"),
                              exclude_items=split.cold_items)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        row = {"epoch": ep, "loss": tot / nb, "sec": round(time.time() - t0, 1)}
        if (ep + 1) % tc.get("eval_every", 5) == 0 or ep == tc["epochs"] - 1:
            m = evaluate_seqrec(model, split, "val", device, users=val_users)["all"]
            row.update({f"val_{k}": v for k, v in m.items() if k != "n"})
            if m["recall@10"] > best:
                best, best_state, bad = m["recall@10"], copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            if bad >= tc.get("patience", 5):
                logger.log(**row, early_stop=True)
                break
        logger.log(**row)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ============================================================================ TIGER
def cold_prefix_map(ids: np.ndarray, cold_items: set[int], n_prefix: int = 3) -> dict[tuple, list[int]]:
    m: dict[tuple, list[int]] = {}
    for i in sorted(cold_items):
        m.setdefault(tuple(int(x) for x in ids[i][:n_prefix]), []).append(i)
    return m


def mix_cold(items: np.ndarray, scores: np.ndarray, tokens: np.ndarray, cold: set[int],
             cold_map: dict[tuple, list[int]], eps: float, k: int) -> np.ndarray:
    """TIGER's cold-start retrieval: reserve up to ceil(eps*k) of the top-k for unseen items
    whose first 3 codes match a generated beam (each inherits that beam's score); the rest are
    the best warm beams. The merged list is re-sorted by score."""
    q = math.ceil(eps * k)
    out = np.full((len(items), k), -1, dtype=np.int64)
    for r in range(len(items)):
        warm, cold_c, seen = [], [], set()
        for w in range(items.shape[1]):
            s = scores[r, w]
            if not np.isfinite(s):
                continue
            it = int(items[r, w])
            if it > 0 and it not in cold and it not in seen:
                warm.append((s, it)); seen.add(it)
            for c in cold_map.get(tuple(int(x) for x in tokens[r, w, :3]), []):
                if c not in seen:
                    cold_c.append((s, c)); seen.add(c)
        n_cold = min(q, len(cold_c))
        chosen = warm[:k - n_cold] + cold_c[:n_cold]
        if len(chosen) < k:  # not enough warm beams: back-fill with remaining cold
            chosen += cold_c[n_cold:n_cold + k - len(chosen)]
        chosen.sort(key=lambda x: -x[0])
        ids_ = [c for _, c in chosen[:k]]
        out[r, :len(ids_)] = ids_
    return out


@torch.no_grad()
def tiger_topk(model: TIGER, tok: TigerTokenizer, trie: SemanticIDTrie, users, hists, k: int, beam: int,
               device, batch_size: int = 256, constrained: bool = True, cold=None, cold_map=None,
               eps: float = 0.0):
    model.eval()
    tops, n_invalid, n_total = [], 0, 0
    for s in range(0, len(hists), batch_size):
        src = tok.encode_histories(users[s:s + batch_size], hists[s:s + batch_size]).to(device)
        items, scores, tokens = model.generate(src, trie, beam_size=beam, constrained=constrained,
                                               return_tokens=True)
        items, scores, tokens = items.cpu().numpy(), scores.cpu().numpy(), tokens.cpu().numpy()
        n_invalid += int((items[:, :k] == -1).sum())
        n_total += items[:, :k].size
        if eps > 0 and cold:
            tops.append(mix_cold(items, scores, tokens, cold, cold_map, eps, k))
        else:
            top = np.full((len(items), k), -1, dtype=np.int64)
            top[:, :min(k, items.shape[1])] = items[:, :k]
            tops.append(top)
    return np.concatenate(tops), n_invalid / max(n_total, 1)


def evaluate_tiger(model, tok, trie, split: Split, which: str, device, k=10, beam=20, users=None,
                   constrained=True, cold_map=None, eps=0.0) -> dict:
    data = split.val if which == "val" else split.test
    users = users if users is not None else sorted(data)
    hists = [data[u][0] for u in users]
    targets = np.array([data[u][1] for u in users])
    topk, invalid = tiger_topk(model, tok, trie, users, hists, k, beam, device, constrained=constrained,
                               cold=split.cold_items, cold_map=cold_map, eps=eps)
    res = split_metrics(topk, targets, np.array([t in split.cold_items for t in targets]))
    res["invalid_id_rate"] = invalid
    res["lists"] = list_stats(topk, split.num_items, split.cold_items)
    return res


def train_tiger(model: TIGER, tok: TigerTokenizer, trie: SemanticIDTrie, split: Split, cfg: dict, device,
                logger: JsonlLogger) -> TIGER:
    ex = tiger_examples(split.train, tok.max_hist)
    users = [u for u, _, _ in ex]
    src_all = tok.encode_histories(users, [h for _, h, _ in ex])
    tgt_all = tok.target_tokens([t for _, _, t in ex])
    bs = cfg["batch_size"]
    steps_per_epoch = math.ceil(len(ex) / bs)
    total = steps_per_epoch * cfg["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(cfg.get("warmup_steps", 1000), total))
    g = torch.Generator().manual_seed(cfg.get("seed", 0))
    val_users = _eval_subset(sorted(split.val), cfg.get("val_users"), cfg.get("seed", 0))
    best, best_state, bad = -1.0, None, 0
    model.to(device)
    logger.log(event="start", n_examples=len(ex), steps_per_epoch=steps_per_epoch,
               n_params=sum(p.numel() for p in model.parameters()))
    for ep in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(ex), generator=g)
        tot, nb = 0.0, 0
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            loss = model.loss(src_all[b].to(device), tgt_all[b].to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        row = {"epoch": ep, "loss": tot / nb, "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
        if (ep + 1) % cfg.get("eval_every", 5) == 0 or ep == cfg["epochs"] - 1:
            m = evaluate_tiger(model, tok, trie, split, "val", device, beam=cfg.get("val_beam", 10),
                               users=val_users)["all"]
            row.update({f"val_{k}": v for k, v in m.items() if k != "n"})
            if m["recall@10"] > best:
                best, best_state, bad = m["recall@10"], copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            if bad >= cfg.get("patience", 5):
                logger.log(**row, early_stop=True)
                break
        logger.log(**row)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
