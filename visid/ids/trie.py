"""Prefix trie over semantic IDs, vectorized for batched beam search.

Each prefix of length t is encoded as a mixed-radix integer key. For every level we keep the
sorted keys of all valid prefixes and a boolean mask [n_prefixes, level_size] of the valid
next tokens, so a batch of prefixes is resolved with one `searchsorted`.
"""
from __future__ import annotations

import numpy as np
import torch


class SemanticIDTrie:
    def __init__(self, id_table: np.ndarray, level_sizes: list[int], items: np.ndarray | None = None):
        """id_table: [num_items + 1, L] semantic IDs, row 0 is padding (ignored).
        items: optional subset of item ids to include (default: all rows >= 1)."""
        self.sizes = list(level_sizes)
        self.L = len(self.sizes)
        if items is None:
            items = np.arange(1, len(id_table))
        items = np.asarray(items, dtype=np.int64)
        codes = torch.as_tensor(np.asarray(id_table)[items], dtype=torch.long)
        if codes.shape[1] != self.L:
            raise ValueError("id_table width does not match level_sizes")
        for t in range(self.L):
            if codes[:, t].min() < 0 or codes[:, t].max() >= self.sizes[t]:
                raise ValueError(f"level {t} codes out of range")
        self.keys, self.masks = [], []
        for t in range(self.L):
            pk = self._key(codes[:, :t])
            uniq, inv = torch.unique(pk, return_inverse=True)
            mask = torch.zeros(len(uniq), self.sizes[t], dtype=torch.bool)
            mask[inv, codes[:, t]] = True
            self.keys.append(uniq)
            self.masks.append(mask)
        full = self._key(codes)
        order = torch.argsort(full)
        self.full_keys = full[order]
        if len(torch.unique(self.full_keys)) != len(self.full_keys):
            raise ValueError("semantic IDs are not unique")
        self.full_items = torch.as_tensor(items)[order]
        self.n_items = len(items)

    def _key(self, prefix: torch.Tensor) -> torch.Tensor:
        key = torch.zeros(prefix.shape[0], dtype=torch.long, device=prefix.device)
        for t in range(prefix.shape[1]):
            key = key * self.sizes[t] + prefix[:, t]
        return key

    def to(self, device) -> "SemanticIDTrie":
        self.keys = [k.to(device) for k in self.keys]
        self.masks = [m.to(device) for m in self.masks]
        self.full_keys = self.full_keys.to(device)
        self.full_items = self.full_items.to(device)
        return self

    @staticmethod
    def _find(sorted_keys: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.searchsorted(sorted_keys, q).clamp(max=len(sorted_keys) - 1)
        return pos, sorted_keys[pos] == q

    def allowed_mask(self, prefix: torch.Tensor) -> torch.Tensor:
        """prefix: [N, t] codes -> bool [N, sizes[t]] of valid next tokens."""
        t = prefix.shape[1]
        dev = self.masks[t].device
        prefix = prefix.to(dev)
        q = self._key(prefix.clamp(min=0))
        pos, ok = self._find(self.keys[t], q)
        valid_codes = ((prefix >= 0) & (prefix < torch.tensor(self.sizes[:t], device=dev))).all(1) if t else ok
        return self.masks[t][pos] & (ok & valid_codes)[:, None]

    def lookup_items(self, full: torch.Tensor) -> torch.Tensor:
        """full: [N, L] codes -> item id, or -1 if not a catalog ID."""
        full = full.to(self.full_keys.device)
        q = self._key(full.clamp(min=0))
        pos, ok = self._find(self.full_keys, q)
        in_range = ((full >= 0) & (full < torch.tensor(self.sizes, device=full.device))).all(1)
        return torch.where(ok & in_range, self.full_items[pos], torch.full_like(q, -1))
