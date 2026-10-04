"""RQ-VAE (Zeghidour et al. 2021; Lee et al. 2022) as used by TIGER to build semantic IDs.

encoder: MLP  x -> z
quantizer: L levels; level l picks the nearest code to the current residual, residual -= code
decoder: MLP  z_q -> x_hat
loss = ||x - x_hat||^2 + sum_l ( ||sg[r_l] - e_l||^2 + beta ||r_l - sg[e_l]||^2 )

Collapse guards: k-means initialization of every level on the first pass, and periodic
re-seeding of dead codes (codes not used during the last `reset_dead_every` epochs) with
random current residuals. Per-level utilization and perplexity are logged every epoch.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from visid.ids.semantic_ids import codebook_stats


class MLP(nn.Module):
    def __init__(self, dims: list[int], dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
                if dropout:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def kmeans(x: torch.Tensor, k: int, iters: int = 25, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = x.shape[0]
    idx = torch.randperm(n, generator=g)[:k].to(x.device)
    if n < k:  # fewer points than codes: repeat points with jitter
        idx = torch.randint(0, n, (k,), generator=g).to(x.device)
    c = x[idx].clone() + 1e-4 * torch.randn(k, x.shape[1], generator=g).to(x.device)
    for _ in range(iters):
        assign = torch.cdist(x, c).argmin(1)
        sums = torch.zeros_like(c).index_add_(0, assign, x)
        cnt = torch.bincount(assign, minlength=k).float()[:, None]
        empty = cnt.squeeze(1) == 0
        c = torch.where(empty[:, None], c, sums / cnt.clamp(min=1))
        if empty.any():  # re-seed empty clusters at random points
            c[empty] = x[torch.randint(0, n, (int(empty.sum()),), generator=g).to(x.device)]
    return c


class ResidualQuantizer(nn.Module):
    def __init__(self, n_levels: int, codebook_size: int, dim: int, beta: float = 0.25):
        super().__init__()
        self.n_levels, self.K, self.dim, self.beta = n_levels, codebook_size, dim, beta
        self.codebooks = nn.ModuleList(nn.Embedding(codebook_size, dim) for _ in range(n_levels))
        for cb in self.codebooks:
            nn.init.normal_(cb.weight, std=0.02)
        self.register_buffer("initialized", torch.tensor(False))

    @staticmethod
    def _nearest(r: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
        d = (r * r).sum(1, keepdim=True) - 2 * r @ cb.t() + (cb * cb).sum(1)[None]
        return d.argmin(1)

    @torch.no_grad()
    def kmeans_init(self, z: torch.Tensor, seed: int = 0) -> None:
        r = z.detach()
        for l, cb in enumerate(self.codebooks):
            c = kmeans(r, self.K, seed=seed + l)
            cb.weight.data.copy_(c)
            r = r - c[self._nearest(r, c)]
        self.initialized.fill_(True)

    def forward(self, z: torch.Tensor):
        residual = z
        zq = torch.zeros_like(z)
        loss = z.new_zeros(())
        codes = []
        for cb in self.codebooks:
            idx = self._nearest(residual.detach(), cb.weight)
            q = cb(idx)
            loss = loss + F.mse_loss(q, residual.detach()) + self.beta * F.mse_loss(residual, q.detach())
            zq = zq + q
            residual = residual - q.detach()
            codes.append(idx)
        return torch.stack(codes, 1), zq, loss

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return sum(cb(codes[:, l]) for l, cb in enumerate(self.codebooks))

    @torch.no_grad()
    def residual_norms(self, z: torch.Tensor) -> list[float]:
        r = z
        norms = [float(r.norm(dim=1).mean())]
        for cb in self.codebooks:
            r = r - cb.weight[self._nearest(r, cb.weight)]
            norms.append(float(r.norm(dim=1).mean()))
        return norms

    @torch.no_grad()
    def reset_dead_codes(self, z: torch.Tensor, usage: list[torch.Tensor], seed: int = 0) -> list[int]:
        """Re-seed codes with zero usage at random residuals of the current data."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        r = z
        n_reset = []
        for l, cb in enumerate(self.codebooks):
            dead = usage[l] == 0
            nd = int(dead.sum())
            if nd:
                pick = torch.randint(0, r.shape[0], (nd,), generator=g).to(r.device)
                cb.weight.data[dead.to(r.device)] = r[pick] + 1e-3 * torch.randn(nd, self.dim, generator=g).to(r.device)
            n_reset.append(nd)
            r = r - cb.weight[self._nearest(r, cb.weight)]
        return n_reset


class RQVAE(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], latent_dim: int, n_levels: int = 3,
                 codebook_size: int = 256, beta: float = 0.25, dropout: float = 0.0):
        super().__init__()
        self.encoder = MLP([in_dim, *hidden, latent_dim], dropout)
        self.decoder = MLP([latent_dim, *reversed(hidden), in_dim], dropout)
        self.quantizer = ResidualQuantizer(n_levels, codebook_size, latent_dim, beta)

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encoder(x)
        codes, zq, qloss = self.quantizer(z)
        z_st = z + (zq - z).detach()  # straight-through estimator
        x_hat = self.decoder(z_st)
        recon = F.mse_loss(x_hat, x)
        return {"loss": recon + qloss, "recon": recon, "qloss": qloss, "codes": codes}

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantizer(self.encoder(x))[0]


def standardize(X: np.ndarray) -> tuple[np.ndarray, dict]:
    mu = X.mean(0, keepdims=True)
    sd = float(X.std()) or 1.0
    return (X - mu) / sd, {"mean": mu, "std": sd}


def train_rqvae(X: np.ndarray, cfg: dict, device: str | torch.device = "cpu", verbose: bool = True,
                log_fn: Callable[..., None] | None = None):
    """Train an RQ-VAE on item embeddings X [n_items, d]. Returns (model, history)."""
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    Xs, _ = standardize(X.astype(np.float32))
    x_all = torch.from_numpy(Xs).to(device)
    model = RQVAE(X.shape[1], list(cfg.get("hidden", [512, 256, 128])), cfg.get("latent_dim", 32),
                  cfg.get("n_levels", 3), cfg.get("codebook_size", 256), cfg.get("beta", 0.25),
                  cfg.get("dropout", 0.0)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 0.0))
    bs = min(cfg.get("batch_size", 1024), len(Xs))
    epochs = cfg.get("epochs", 500)
    reset_every = cfg.get("reset_dead_every", 20)
    collapse_warn = cfg.get("collapse_warn_util", 0.2)
    K, L = model.quantizer.K, model.quantizer.n_levels
    g = torch.Generator(device="cpu").manual_seed(seed)

    with torch.no_grad():
        model.quantizer.kmeans_init(model.encoder(x_all), seed=seed)

    history = []
    usage = [torch.zeros(K, dtype=torch.long) for _ in range(L)]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xs), generator=g)
        tot = {"loss": 0.0, "recon": 0.0, "qloss": 0.0}
        nb = 0
        for i in range(0, len(perm), bs):
            xb = x_all[perm[i:i + bs].to(device)]
            out = model(xb)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            for k in tot:
                tot[k] += float(out[k].detach())
            nb += 1
            for l in range(L):
                usage[l] += torch.bincount(out["codes"][:, l].cpu(), minlength=K)
        row = {"epoch": ep, **{k: v / nb for k, v in tot.items()}}
        last = ep == epochs - 1
        if ep % max(1, cfg.get("log_every", 10)) == 0 or last or ep == 0:
            model.eval()
            codes = model.encode(x_all).cpu().numpy()
            row.update(codebook_stats(codes, K))
            for l in range(L):
                if row[f"util_l{l}"] < collapse_warn:
                    row[f"collapse_warning_l{l}"] = True
            history.append(row)
            if log_fn:
                log_fn(**row)
            elif verbose:
                print(row, flush=True)
        if reset_every and (ep + 1) % reset_every == 0 and not last:
            model.eval()
            with torch.no_grad():
                n_reset = model.quantizer.reset_dead_codes(model.encoder(x_all), usage, seed=seed + ep)
            if any(n_reset) and history:
                history[-1].setdefault("dead_codes_reset", n_reset)
            usage = [torch.zeros(K, dtype=torch.long) for _ in range(L)]
    model.eval()
    model.input_stats = None
    return model, history


@torch.no_grad()
def encode_items(model: RQVAE, X: np.ndarray, device="cpu") -> np.ndarray:
    Xs, _ = standardize(X.astype(np.float32))
    return model.encode(torch.from_numpy(Xs).to(device)).cpu().numpy()
