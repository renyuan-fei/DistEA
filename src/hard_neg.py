"""
hard_neg.py — hard-negative mining for the alignment
margin loss, from the GAS final-history table.

Why: the default recipe draws `neg_k=1` negative per positive
UNIFORMLY over the local table (gas.py `_gas_train_epoch_impl`,
`torch.randint(0, n_nodes, ...)`). On DBP1M a random one of 1.68M entities
is almost never confusable with the anchor, so the L1 hinge (margin 6) is
satisfied after a few epochs and stops carrying gradient; only InfoNCE's
in-batch negatives keep teaching. Training loss reaching ~0.004 while test
Hit@1 plateaus at 200-300 epochs is consistent with that (working
hypothesis). Mining the nearest wrong entities turns the hinge back on.

What: `mine_hard_negatives(emb, pairs, kg1_size, pool=m)` returns, for each
train pair (e1, e2), the `m` candidates most similar (cosine) to the anchor
on the OTHER side -- `neg_kg1[i]`: KG1 rows closest to e2 (what alignment_loss
uses to corrupt e1), `neg_kg2[i]`: KG2 rows closest to e1 -- with the pair's
own positive (and the anchor itself under side="any") excluded. No extra
forward pass: `emb` is the GAS `final_hist` that is resident on the device
anyway (1.68M x 384 fp32 at DBP1M); it is L2-normalised into a bf16 copy and
scored in anchor chunks against the candidate side (chunk x n_cand), top-m
per chunk. `sample_hard_negatives(pool_rows, k)` then draws `k` random
columns of each row per epoch (truncated sampling: not always the top-1, so
the hinge sees a spread of hard examples), and `merge_negatives` appends
them to the uniform (P, neg_k) draw -- alignment_loss accepts any
`neg_kg*.size(1)`.

Scope: pools are mined over THIS rank's table (local + halo rows), so with
N > 1 "hard" is relative to what the rank can see -- a
partition-dependent mechanism. Pure tensor code, CPU-testable
(tests/test_hard_neg.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HardNegPools:
    """Per-pair candidate pools, aligned with the `pairs` they were mined from.

    neg_kg1: (P, m) long -- rows in the KG1 range (or anywhere, side="any")
             closest to each pair's e2, excluding e1 (and e2).
    neg_kg2: (P, m) long -- rows in the KG2 range (or anywhere) closest to
             each pair's e1, excluding e2 (and e1).
    """

    neg_kg1: torch.Tensor
    neg_kg2: torch.Tensor
    side: str = "other"
    mined_at_epoch: int = -1

    @property
    def pool_size(self) -> int:
        return int(self.neg_kg1.size(1))

    def __len__(self) -> int:
        return int(self.neg_kg1.size(0))


def _topk_pool(
    anchors: torch.Tensor,        # (P, D) normalised, dtype
    cands: torch.Tensor,          # (n_cand, D) normalised, dtype
    cand_offset: int,
    exclude: torch.Tensor,        # (P, n_ex) global row ids to mask per anchor (-1 = none)
    m: int,
    chunk: int,
) -> torch.Tensor:
    """Top-m most similar candidate rows per anchor, as GLOBAL row ids."""
    P = anchors.size(0)
    n_cand = cands.size(0)
    m_eff = min(m, max(n_cand - exclude.size(1), 0))
    out = torch.empty((P, m), dtype=torch.long, device=anchors.device)
    if P == 0 or m_eff <= 0:
        return out.fill_(cand_offset) if n_cand > 0 else out.zero_()
    cands_t = cands.t().contiguous()
    neg_inf = torch.finfo(anchors.dtype).min
    for s in range(0, P, chunk):
        e = min(s + chunk, P)
        sim = anchors[s:e] @ cands_t                       # (c, n_cand)
        ex = exclude[s:e] - cand_offset                     # (c, n_ex) local cols
        valid = (ex >= 0) & (ex < n_cand)
        if bool(valid.any()):
            rows = torch.arange(e - s, device=sim.device).unsqueeze(1).expand_as(ex)
            sim[rows[valid], ex[valid]] = neg_inf
        top = torch.topk(sim, m_eff, dim=1).indices + cand_offset  # (c, m_eff)
        if m_eff < m:
            # tiny tables (tests): pad by repeating the last column
            top = torch.cat([top, top[:, -1:].expand(-1, m - m_eff)], dim=1)
        out[s:e] = top
    return out


@torch.no_grad()
def mine_hard_negatives(
    emb: torch.Tensor,
    pairs: torch.Tensor,
    kg1_size: int,
    pool: int = 50,
    side: str = "other",
    chunk: int = 1024,
    dtype: torch.dtype | None = None,
    epoch: int = -1,
) -> HardNegPools | None:
    """Mine (P, pool) hard-negative candidate pools from an embedding table.

    emb:      (N, D) table (e.g. GAS final_hist); read-only, no grad.
    pairs:    (P, 2) local index pairs, pairs[:, 0] in KG1, pairs[:, 1] in KG2.
    kg1_size: KG1 rows are [0, kg1_size), KG2 rows [kg1_size, N).
    side:     "other" = candidates for corrupting e1 come from KG1 and for e2
              from KG2 (what alignment_loss corrupts); "any" = both pools
              are mined over the whole table.
    dtype:    matmul dtype; defaults to bf16 on CUDA, fp32 elsewhere.
    Returns None when pool <= 0 or there are no pairs.
    """
    if pool <= 0 or pairs.numel() == 0:
        return None
    if side not in ("other", "any"):
        raise ValueError(f"hard_neg side must be 'other' or 'any', got {side!r}")
    if dtype is None:
        dtype = torch.bfloat16 if emb.device.type == "cuda" else torch.float32
    pairs = pairs.to(emb.device).long()
    normed = torch.nn.functional.normalize(emb.detach().to(dtype), dim=1)
    e1, e2 = pairs[:, 0], pairs[:, 1]
    none = torch.full_like(e1, -1)
    if side == "other":
        cand1, off1 = normed[:kg1_size], 0
        cand2, off2 = normed[kg1_size:], kg1_size
        ex1 = torch.stack([e1, none], dim=1)      # corrupting e1: never the true e1
        ex2 = torch.stack([e2, none], dim=1)
    else:
        cand1 = cand2 = normed
        off1 = off2 = 0
        ex1 = torch.stack([e1, e2], dim=1)        # neither the positive nor the anchor
        ex2 = torch.stack([e2, e1], dim=1)
    neg_kg1 = _topk_pool(normed[e2], cand1, off1, ex1, pool, chunk)
    neg_kg2 = _topk_pool(normed[e1], cand2, off2, ex2, pool, chunk)
    return HardNegPools(neg_kg1=neg_kg1, neg_kg2=neg_kg2, side=side, mined_at_epoch=epoch)


def sample_hard_negatives(
    pool_rows: torch.Tensor, k: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """(p, m) pool rows -> (p, k) random columns per row (with replacement)."""
    p, m = pool_rows.shape
    if p == 0 or k <= 0:
        return pool_rows.new_empty((p, 0))
    cols = torch.randint(0, m, (p, k), device=pool_rows.device, generator=generator)
    return pool_rows.gather(1, cols)


def merge_negatives(
    neg_uniform: torch.Tensor,
    pool_rows: torch.Tensor | None,
    k: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Append k sampled hard negatives per pair to the uniform (p, neg_k)
    draw. With no pool (or k <= 0) the uniform tensor is returned unchanged,
    so the default recipe is reproduced bit-for-bit when mining is off."""
    if pool_rows is None or k <= 0:
        return neg_uniform
    hard = sample_hard_negatives(pool_rows, k, generator)
    return torch.cat([neg_uniform, hard.to(neg_uniform.device)], dim=1)
