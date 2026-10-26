"""neg_pool_frac: fixed-subset uniform negatives (multi-node loss ablation)."""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch

from gas import neg_pool_subset, sample_uniform_negatives


def main():
    tr = types.SimpleNamespace(rank=0)
    assert neg_pool_subset(tr, 1000, 1.0, "cpu") is None
    pool = neg_pool_subset(tr, 1000, 0.25, "cpu")
    assert pool.numel() == 250 and pool.unique().numel() == 250
    assert torch.equal(pool, neg_pool_subset(tr, 1000, 0.25, "cpu")), "cached per run"
    tr2 = types.SimpleNamespace(rank=1)
    assert not torch.equal(pool, neg_pool_subset(tr2, 1000, 0.25, "cpu")), "rank-seeded"
    neg = sample_uniform_negatives(1000, (64, 3), "cpu", pool)
    assert neg.shape == (64, 3) and torch.isin(neg, pool).all()
    full = sample_uniform_negatives(1000, (64, 3), "cpu", None)
    assert full.shape == (64, 3) and full.min() >= 0 and full.max() < 1000
    print("test_neg_pool_frac: OK")


if __name__ == "__main__":
    main()
