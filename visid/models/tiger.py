"""TIGER-style generative retriever (Rajput et al., NeurIPS 2023).

Encoder input : [user-hash token, sid(i_1)_0..L-1, ..., sid(i_n)_0..L-1]   (last `max_hist` items)
Decoder       : BOS -> c_0 -> c_1 -> ... -> c_{L-1}  (the next item's semantic ID)

Vocabulary layout (one shared embedding table):
  0 PAD | 1 BOS | level-0 codes | level-1 codes | ... | level-(L-1) codes | user buckets
Each decoding step predicts only over its own level's slice of the vocabulary, both in the
loss and at inference, and inference is constrained by the prefix trie.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from visid.ids.trie import SemanticIDTrie
from visid.models.decoding import constrained_beam_search

PAD, BOS = 0, 1


class TigerTokenizer:
    def __init__(self, id_table: np.ndarray, level_sizes: list[int], n_user_buckets: int = 2000,
                 max_hist: int = 20, align: str = "left"):
        """align='left' : [user, item_1 .. item_n, PAD ...]  (the most recent item's position
                          varies with history length)
           align='right': [user, PAD ..., item_1 .. item_n] (the most recent item always sits
                          in the last positions, so absolute position embeddings encode recency)"""
        if align not in ("left", "right"):
            raise ValueError(align)
        self.align = align
        self.ids = torch.as_tensor(np.asarray(id_table), dtype=torch.long)
        self.sizes = list(level_sizes)
        self.L = len(self.sizes)
        self.level_offsets = [2 + sum(self.sizes[:l]) for l in range(self.L)]
        self.user_offset = 2 + sum(self.sizes)
        self.n_user_buckets = n_user_buckets
        self.vocab_size = self.user_offset + n_user_buckets
        self.max_hist = max_hist
        self.max_src_len = 1 + max_hist * self.L
        self._off = torch.tensor(self.level_offsets, dtype=torch.long)

    def user_token(self, u: int) -> int:
        return self.user_offset + (u * 2654435761 % 2 ** 32) % self.n_user_buckets

    def encode_histories(self, users: list[int], hists: list[list[int]]) -> torch.Tensor:
        out = torch.full((len(hists), self.max_src_len), PAD, dtype=torch.long)
        for r, (u, h) in enumerate(zip(users, hists)):
            h = h[-self.max_hist:]
            toks = (self.ids[h] + self._off).flatten() if h else torch.empty(0, dtype=torch.long)
            out[r, 0] = self.user_token(u)
            if len(toks):
                if self.align == "left":
                    out[r, 1:1 + len(toks)] = toks
                else:
                    out[r, self.max_src_len - len(toks):] = toks
        return out

    def target_tokens(self, items: list[int]) -> torch.Tensor:
        return self.ids[torch.as_tensor(items, dtype=torch.long)]


class TIGER(nn.Module):
    def __init__(self, vocab_size: int, level_sizes: list[int], level_offsets: list[int],
                 d_model: int = 128, n_heads: int = 4, n_enc_layers: int = 4, n_dec_layers: int = 4,
                 d_ff: int = 1024, dropout: float = 0.1, max_src_len: int = 81):
        super().__init__()
        self.sizes, self.offsets, self.L = list(level_sizes), list(level_offsets), len(level_sizes)
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.src_pos = nn.Embedding(max_src_len, d_model)
        self.tgt_pos = nn.Embedding(self.L + 1, d_model)
        self.transformer = nn.Transformer(d_model, n_heads, n_enc_layers, n_dec_layers, d_ff, dropout,
                                          batch_first=True, norm_first=True, activation="gelu")
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("off", torch.tensor(level_offsets, dtype=torch.long))
        self.d = d_model

    # ---------------------------------------------------------------- pieces
    def encode(self, src: torch.Tensor):
        pad = src == PAD
        x = self.emb(src) + self.src_pos(torch.arange(src.shape[1], device=src.device))
        mem = self.transformer.encoder(self.drop(x), src_key_padding_mask=pad)
        return mem, pad

    def decode(self, mem, mem_pad, dec_in: torch.Tensor) -> torch.Tensor:
        T = dec_in.shape[1]
        y = self.emb(dec_in) + self.tgt_pos(torch.arange(T, device=dec_in.device))
        causal = torch.triu(torch.full((T, T), float("-inf"), device=dec_in.device), 1)
        return self.transformer.decoder(self.drop(y), mem, tgt_mask=causal, memory_key_padding_mask=mem_pad)

    def _dec_input(self, local_codes: torch.Tensor) -> torch.Tensor:
        """local codes [N, t] -> decoder input [N, t+1] = BOS + global token ids."""
        bos = torch.full((local_codes.shape[0], 1), BOS, dtype=torch.long, device=local_codes.device)
        t = local_codes.shape[1]
        return torch.cat([bos, local_codes + self.off[:t]], 1)

    def level_logits(self, h: torch.Tensor, level: int) -> torch.Tensor:
        o = self.offsets[level]
        return self.head(h)[..., o:o + self.sizes[level]]

    # ---------------------------------------------------------------- training
    def loss(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        mem, pad = self.encode(src)
        h = self.decode(mem, pad, self._dec_input(tgt[:, :-1]))  # [N, L, d]
        losses = [F.cross_entropy(self.level_logits(h[:, l], l), tgt[:, l]) for l in range(self.L)]
        return torch.stack(losses).mean()

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def generate(self, src: torch.Tensor, trie: SemanticIDTrie, beam_size: int = 20,
                 constrained: bool = True, return_tokens: bool = False):
        mem, pad = self.encode(src)
        B = src.shape[0]

        def step_fn(tokens):  # [B, W, t]
            W, t = tokens.shape[1], tokens.shape[2]
            m = mem[:, None].expand(B, W, *mem.shape[1:]).reshape(B * W, *mem.shape[1:])
            p = pad[:, None].expand(B, W, pad.shape[1]).reshape(B * W, -1)
            h = self.decode(m, p, self._dec_input(tokens.reshape(B * W, t)))[:, -1]
            return self.level_logits(h, t).log_softmax(-1).view(B, W, -1)

        tokens, scores, items = constrained_beam_search(step_fn, B, trie, beam_size, constrained,
                                                        device=src.device)
        if return_tokens:
            return items, scores, tokens
        return items, scores


def tiger_examples(train: dict[int, list[int]], max_hist: int):
    """Every next-item prediction inside each training sequence: (user, history, target)."""
    ex = []
    for u, s in train.items():
        for i in range(1, len(s)):
            ex.append((u, s[max(0, i - max_hist):i], s[i]))
    return ex
