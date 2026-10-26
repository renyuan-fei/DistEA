"""
test_minibatch_loss.py — src/minibatch_loss.py: the --minibatch path's loss
selection must mirror DistTrainer's (margin / InfoNCE-combined / NHSM),
produce finite scalars with gradients into both the batch embeddings and
the learnable mix weight, and reduce to alignment_loss exactly when
InfoNCE is off. DGL-free, no GPU, <5 sec.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minibatch_loss import describe, minibatch_loss
from models.gcn import alignment_loss


def _batch(seed=0, n1=6, n2=6, dim=8):
    torch.manual_seed(seed)
    emb = torch.randn(n1 + n2, dim, requires_grad=True)
    pairs = torch.tensor([[i, n1 + i] for i in range(min(n1, n2))], dtype=torch.long)
    return emb, pairs, n1


def test_margin_matches_alignment_loss_when_infonce_off():
    emb, pairs, k1 = _batch()
    torch.manual_seed(1)
    a = minibatch_loss(emb, pairs, k1, loss_fn="margin", use_infonce=False, neg_k=3, margin=6.0)
    torch.manual_seed(1)
    b = alignment_loss(emb, pairs, k1, neg_k=3, margin=6.0)
    assert torch.allclose(a, b), (a, b)
    print("  test_margin_matches_alignment_loss_when_infonce_off PASSED")


def test_combined_uses_learnable_weight_and_backprops():
    emb, pairs, k1 = _batch()
    w = torch.nn.Parameter(torch.tensor([0.5, 0.5]))
    loss = minibatch_loss(emb, pairs, k1, loss_fn="margin", use_infonce=True, loss_weight=w, neg_k=3)
    assert torch.isfinite(loss) and loss.dim() == 0
    loss.backward()
    assert emb.grad is not None and emb.grad.abs().sum() > 0
    assert w.grad is not None and w.grad.abs().sum() > 0
    print("  test_combined_uses_learnable_weight_and_backprops PASSED")


def test_nhsm_branch_ignores_weight_and_backprops():
    emb, pairs, k1 = _batch()
    w = torch.nn.Parameter(torch.tensor([0.5, 0.5]))
    loss = minibatch_loss(emb, pairs, k1, loss_fn="nhsm", use_infonce=True, loss_weight=w, nhsm_gamma=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert emb.grad is not None and emb.grad.abs().sum() > 0
    assert w.grad is None  # NHSM does not use the mix weight
    print("  test_nhsm_branch_ignores_weight_and_backprops PASSED")


def test_empty_pairs_is_zero_with_grad_path():
    emb, _pairs, k1 = _batch()
    loss = minibatch_loss(emb, torch.zeros((0, 2), dtype=torch.long), k1)
    assert float(loss) == 0.0 and loss.requires_grad
    assert describe("nhsm", True) == "nhsm" and describe("margin", False) == "margin"
    assert describe("margin", True).startswith("combined")
    print("  test_empty_pairs_is_zero_with_grad_path PASSED")


def test_extend_seeds_with_negatives():
    import numpy as np

    from minibatch_loss import extend_seeds_with_negatives
    rng = np.random.default_rng(0)
    seeds = np.array([5, 1, 9])
    pool = np.arange(100)
    out = extend_seeds_with_negatives(seeds, pool, 10, rng)
    assert np.array_equal(out, np.unique(out)) and set(seeds) <= set(out.tolist())
    assert 10 <= len(out) <= 13
    assert np.array_equal(extend_seeds_with_negatives(seeds, pool, 0, rng), seeds)
    assert np.array_equal(extend_seeds_with_negatives(seeds, np.array([], dtype=np.int64), 5, rng), seeds)
    assert len(extend_seeds_with_negatives(seeds, np.array([1, 2]), 50, rng)) == 4  # k capped at pool
    print("  test_extend_seeds_with_negatives PASSED")


if __name__ == "__main__":
    print("minibatch_loss tests:")
    test_margin_matches_alignment_loss_when_infonce_off()
    test_combined_uses_learnable_weight_and_backprops()
    test_nhsm_branch_ignores_weight_and_backprops()
    test_empty_pairs_is_zero_with_grad_path()
    test_extend_seeds_with_negatives()
    print("All minibatch_loss tests PASSED.")
