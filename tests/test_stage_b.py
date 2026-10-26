"""
test_stage_b.py — Stage B tests: GCN model and KG-boundary-aware alignment loss.

DGL-free, no GPU required.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.gcn import GCN, alignment_loss

# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------


def test_alignment_loss_basic():
    """alignment_loss returns a non-negative scalar."""
    N, D = 20, 16
    kg1_size = 12  # nodes 0–11 KG1, 12–19 KG2
    emb = torch.nn.functional.normalize(torch.randn(N, D), dim=1)
    pairs = torch.tensor([[0, 12], [1, 13], [2, 14]], dtype=torch.long)
    loss = alignment_loss(emb, pairs, kg1_size=kg1_size, neg_k=3, margin=1.0)
    assert loss.item() >= 0.0, f"loss should be >= 0, got {loss.item()}"
    print("  test_alignment_loss_basic PASSED")


def test_alignment_loss_empty_pairs():
    """Empty pairs tensor returns zero loss."""
    N, D = 20, 16
    emb = torch.nn.functional.normalize(torch.randn(N, D), dim=1)
    empty = torch.zeros(0, 2, dtype=torch.long)
    loss = alignment_loss(emb, empty, kg1_size=10)
    assert loss.item() == 0.0
    print("  test_alignment_loss_empty_pairs PASSED")


def test_alignment_loss_degenerate_partition():
    """Returns zero loss when no KG2 nodes exist (degenerate)."""
    N, D = 10, 8
    emb = torch.nn.functional.normalize(torch.randn(N, D), dim=1)
    pairs = torch.tensor([[0, 1]], dtype=torch.long)
    loss = alignment_loss(emb, pairs, kg1_size=N)
    assert loss.item() == 0.0
    print("  test_alignment_loss_degenerate_partition PASSED")


def test_alignment_loss_margin():
    """Identical anchor and counterpart embeddings produce a valid (>= 0) loss."""
    N, D = 10, 16
    kg1_size = 6
    emb = torch.nn.functional.normalize(torch.randn(N, D), dim=1)
    emb[kg1_size] = emb[0]  # make KG2 counterpart identical to KG1 anchor
    pairs = torch.tensor([[0, kg1_size]], dtype=torch.long)
    loss = alignment_loss(emb, pairs, kg1_size=kg1_size, neg_k=3, margin=3.0)
    assert not torch.isnan(loss), "loss is NaN"
    assert loss.item() >= 0
    print("  test_alignment_loss_margin PASSED")


def test_gcn_alignment_pipeline():
    """GCN forward + alignment_loss end-to-end (sparse adj, no DGL)."""
    # GCN-Align (ELsEA-faithful) has no per-layer transform, so h's width
    # must equal output_dim.
    N, D = 16, 16
    kg1_size = 10
    ids_row = torch.arange(N)
    adj = torch.sparse_coo_tensor(
        torch.stack([ids_row, ids_row]), torch.ones(N), size=(N, N), dtype=torch.float32
    )
    h = torch.randn(N, D)
    model = GCN(output_dim=D, num_layers=2)
    emb = model(adj, h)
    pairs = torch.tensor([[0, kg1_size], [1, kg1_size + 1]], dtype=torch.long)
    loss = alignment_loss(emb, pairs, kg1_size=kg1_size, neg_k=3, margin=1.0)
    assert loss.item() >= 0.0
    print("  test_gcn_alignment_pipeline PASSED")


if __name__ == "__main__":
    print("Stage B tests:")
    test_alignment_loss_basic()
    test_alignment_loss_empty_pairs()
    test_alignment_loss_degenerate_partition()
    test_alignment_loss_margin()
    test_gcn_alignment_pipeline()
    print("All Stage B tests PASSED.")
