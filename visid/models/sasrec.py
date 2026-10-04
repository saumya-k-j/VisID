"""SASRec (Kang & McAuley 2018) with a pluggable item tower.

SeqRecModel is a causal transformer over the item vectors of a user's history; the last
hidden state is the user vector and scores are dot products with all item vectors.
- SASRec          = SeqRecModel(IDItemTower)            scored exhaustively in torch
- Two-tower model = SeqRecModel(ContentItemTower)       item vectors indexed in FAISS
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IDItemTower(nn.Module):
    def __init__(self, num_items: int, d: int):
        super().__init__()
        self.emb = nn.Embedding(num_items + 1, d, padding_idx=0)
        nn.init.normal_(self.emb.weight, std=0.02)
        with torch.no_grad():
            self.emb.weight[0].zero_()

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        return self.emb(items)

    def all_items(self) -> torch.Tensor:
        return self.emb.weight


class ContentItemTower(nn.Module):
    """item vector = MLP(content embedding) + ID embedding.

    The ID embedding is dropped with probability `id_dropout` during training so the tower
    also works from content alone, and it is always zeroed for `cold_items` (which never
    appear in training, so their ID rows are untrained)."""

    def __init__(self, content: np.ndarray, d: int, use_id: bool = True, id_dropout: float = 0.5,
                 cold_items: set[int] | None = None, hidden: int = 256):
        super().__init__()
        c = torch.as_tensor(content, dtype=torch.float32)
        c = c / c.norm(dim=1, keepdim=True).clamp(min=1e-8)
        c[0] = 0
        self.register_buffer("content", c)
        self.mlp = nn.Sequential(nn.Linear(c.shape[1], hidden), nn.GELU(), nn.Linear(hidden, d))
        self.use_id = use_id
        self.id_dropout = id_dropout
        n = c.shape[0]
        self.id = nn.Embedding(n, d, padding_idx=0) if use_id else None
        if self.id is not None:
            nn.init.normal_(self.id.weight, std=0.02)
        warm = torch.ones(n, dtype=torch.bool)
        warm[0] = False
        for i in cold_items or ():
            warm[i] = False
        self.register_buffer("warm", warm)

    def _vec(self, items: torch.Tensor) -> torch.Tensor:
        v = self.mlp(self.content[items])
        if self.id is not None:
            keep = self.warm[items]
            if self.training and self.id_dropout > 0:
                keep = keep & (torch.rand(items.shape, device=items.device) >= self.id_dropout)
            v = v + self.id(items) * keep[..., None].float()
        return v * (items != 0)[..., None].float()

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        return self._vec(items)

    def all_items(self) -> torch.Tensor:
        return self._vec(torch.arange(self.content.shape[0], device=self.content.device))


class SeqRecModel(nn.Module):
    def __init__(self, item_tower: nn.Module, d_model: int = 64, n_layers: int = 2, n_heads: int = 2,
                 max_len: int = 50, dropout: float = 0.2):
        super().__init__()
        self.items = item_tower
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.max_len = max_len
        self.register_buffer("causal", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), 1))

    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: [B, T] left-padded item ids -> hidden states [B, T, d]."""
        T = seq.shape[1]
        x = self.items(seq) * (self.pos.embedding_dim ** 0.5) + self.pos(torch.arange(T, device=seq.device))
        keep = (seq != 0)[..., None].float()
        x = self.drop(x) * keep
        # causal mask + "never attend to padding"; a padded query attends only to itself so
        # no attention row is all -inf (which would produce NaNs that leak into later layers)
        pad = seq == 0
        eye = torch.eye(T, dtype=torch.bool, device=seq.device)
        blocked = self.causal[:T, :T][None] | (pad[:, None, :] & ~eye[None])
        n_heads = self.blocks.layers[0].self_attn.num_heads
        mask = blocked.repeat_interleave(n_heads, dim=0)
        h = self.blocks(x, mask=mask)
        return self.norm(h) * keep

    def user_vectors(self, seq: torch.Tensor) -> torch.Tensor:
        return self.encode_sequence(seq)[:, -1]

    def score(self, seq: torch.Tensor) -> torch.Tensor:
        return self.user_vectors(seq) @ self.items.all_items().t()

    def loss(self, inp: torch.Tensor, tgt: torch.Tensor, kind: str = "ce", n_neg: int = 1,
             exclude_items: set[int] | None = None) -> torch.Tensor:
        """Next-item loss at every non-padded position.
        ce : full softmax over the catalog (works well in practice; gSASRec, Klenitskiy 2023)
        bce: original SASRec, one sampled negative per position
        exclude_items (cold items) are removed from the softmax / negative pool, so they get
        no training signal at all, matching the cold-start protocol."""
        h = self.encode_sequence(inp)  # [B, T, d]
        valid = tgt != 0
        n_all = self.items.all_items().shape[0]
        blocked = torch.zeros(n_all, dtype=torch.bool, device=inp.device)
        blocked[0] = True
        if exclude_items:
            blocked[torch.tensor(sorted(exclude_items), device=inp.device)] = True
        if kind == "ce":
            logits = h[valid] @ self.items.all_items().t()
            logits = logits.masked_fill(blocked[None], float("-inf"))
            return F.cross_entropy(logits, tgt[valid])
        pool = torch.nonzero(~blocked).flatten()
        neg = pool[torch.randint(0, len(pool), (*tgt.shape, n_neg), device=tgt.device)]
        pos_s = (h * self.items(tgt)).sum(-1)
        neg_s = (h[..., None, :] * self.items(neg)).sum(-1)
        loss = F.binary_cross_entropy_with_logits(pos_s, torch.ones_like(pos_s), reduction="none")
        loss = loss + F.binary_cross_entropy_with_logits(neg_s, torch.zeros_like(neg_s), reduction="none").sum(-1)
        return (loss * valid).sum() / valid.sum()


def pad_left(seqs: list[list[int]], max_len: int) -> torch.Tensor:
    out = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for r, s in enumerate(seqs):
        s = s[-max_len:]
        if s:
            out[r, -len(s):] = torch.tensor(s)
    return out


def training_pairs(train: dict[int, list[int]], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One example per user: input seq[:-1], target seq[1:] (SASRec), left padded."""
    seqs = [s for s in train.values() if len(s) >= 2]
    return pad_left([s[:-1] for s in seqs], max_len), pad_left([s[1:] for s in seqs], max_len)
