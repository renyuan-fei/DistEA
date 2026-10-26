"""
minibatch_loss.py — loss selection for the --minibatch training path,
mirroring DistTrainer.train_epoch's full-graph selection so the same CLI
flags mean the same thing on both paths:

  --loss nhsm                 -> nhsm_loss (Dual-AMN NHSM; negatives = every
                                 other seed entity in the batch)
  --loss margin + --use_infonce (default ON)
                              -> combined_align_loss (ELsEA w0*InfoNCE +
                                 w1*margin, learnable softmax weights)
  --loss margin + --no_use_infonce
                              -> alignment_loss (plain margin)

Using the same selection matters: the InfoNCE term is the single largest
factor in the full-graph results (ws=1 Hit@1 0.54 with InfoNCE vs 0.33
without), so a mini-batch loop that always used the plain margin loss would
not be comparable to the full-graph path.

Negatives live inside the batch on every branch (`emb` is the batch's
seed rows only; a separately sampled negative pool gave no benefit), so `kg1_size` is the KG1/KG2 boundary WITHIN the
batch's seed ordering, exactly as the caller already computes it.
"""

from __future__ import annotations

import numpy as np
import torch

from models.gcn import alignment_loss, combined_align_loss, nhsm_loss


def minibatch_loss(
    emb: torch.Tensor,
    pairs: torch.Tensor,
    kg1_size: int,
    loss_fn: str = "margin",
    use_infonce: bool = True,
    loss_weight: torch.Tensor | None = None,
    neg_k: int = 10,
    margin: float = 6.0,
    infonce_temp: float = 0.1,
    nhsm_gamma: float = 1.0,
) -> torch.Tensor:
    if pairs.numel() == 0:
        return emb.sum() * 0.0
    if loss_fn == "nhsm":
        return nhsm_loss(emb, pairs, gamma=nhsm_gamma)
    if use_infonce and loss_weight is not None:
        return combined_align_loss(
            emb, pairs, kg1_size, weight=loss_weight, neg_k=neg_k, margin=margin,
            infonce_temp=infonce_temp,
        )
    return alignment_loss(emb, pairs, kg1_size, neg_k=neg_k, margin=margin)


def describe(loss_fn: str, use_infonce: bool) -> str:
    if loss_fn == "nhsm":
        return "nhsm"
    return "combined(infonce+margin)" if use_infonce else "margin"


def extend_seeds_with_negatives(seed_ids, pool, k: int, rng) -> np.ndarray:
    """Add `k` random entities from `pool` (this rank's owned KG1+KG2
    ids) to a batch's seed set so the batch's forward_blocks also produces
    embeddings for non-seed entities, which then serve as margin / NHSM
    negatives. Why: the full-graph path samples negatives from EVERY local
    node (dist_trainer: torch.randint(0, N)), so test entities receive a
    repulsive gradient in every epoch; the mini-batch path's `emb` holds
    the batch's seed rows only, so non-seed (= test) entities never do.
    Running the exact full-graph schedule through the mini-batch machinery
    still loses ~9pp Hit@1 at iteration 1 (0.429 vs 0.517); this is the
    remaining structural difference. Returns the
    sorted unique union; k <= 0 or an empty pool returns seed_ids
    unchanged. Extras carry no pairs, so InfoNCE ignores them."""
    import numpy as np  # noqa: PLC0415

    seed_ids = np.asarray(seed_ids, dtype=np.int64)
    if k <= 0 or len(pool) == 0:
        return seed_ids
    extra = rng.choice(np.asarray(pool, dtype=np.int64), size=min(k, len(pool)), replace=False)
    return np.unique(np.concatenate([seed_ids, extra]))
