"""Trie-constrained beam search over semantic IDs.

The model is abstracted as `step_fn(tokens) -> log_probs`, where tokens is [B, W, t]
(level-local codes generated so far for W beams) and the output is [B, W, sizes[t]],
log-probabilities over level t's codebook. At each step invalid continuations are masked to
-inf using the trie, so every finished beam is a real catalog item.
"""
from __future__ import annotations

from typing import Callable

import torch

from visid.ids.trie import SemanticIDTrie


@torch.no_grad()
def constrained_beam_search(step_fn: Callable[[torch.Tensor], torch.Tensor], batch_size: int,
                            trie: SemanticIDTrie, beam_size: int, constrained: bool = True,
                            device: torch.device | str = "cpu"):
    """Returns (tokens [B, W, L], scores [B, W], items [B, W]). Beams that could not be
    filled with a valid ID (catalog smaller than the beam) get score -inf and item -1.
    With constrained=False the trie is ignored and items may be -1 (invalid IDs)."""
    B = batch_size
    tokens = torch.zeros(B, 1, 0, dtype=torch.long, device=device)
    scores = torch.zeros(B, 1, device=device)
    for t, V in enumerate(trie.sizes):
        W = tokens.shape[1]
        logp = step_fn(tokens).float()  # [B, W, V]
        if constrained:
            mask = trie.allowed_mask(tokens.reshape(B * W, t)).view(B, W, V).to(logp.device)
            logp = logp.masked_fill(~mask, float("-inf"))
        cand = (scores[:, :, None] + logp).view(B, W * V)
        k = min(beam_size, W * V)
        scores, idx = cand.topk(k, dim=1)
        beam = idx // V
        nxt = idx % V
        prev = torch.gather(tokens, 1, beam[:, :, None].expand(B, k, t))
        tokens = torch.cat([prev, nxt[:, :, None]], dim=2)
    L = tokens.shape[2]
    items = trie.lookup_items(tokens.reshape(-1, L)).view(B, -1).to(scores.device)
    items = torch.where(torch.isinf(scores), torch.full_like(items, -1), items)
    return tokens, scores, items
