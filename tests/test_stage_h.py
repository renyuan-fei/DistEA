"""
test_stage_h.py — Stage H tests: h_init cross-rank consistency regularizer
(--consistency_lambda).

DGL-free, no GPU required, <30 sec.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dist_trainer import DistTrainer
from models.gcn import GCN, consistency_reg_loss

# ---------------------------------------------------------------------------
#  consistency_reg_loss (pure function)
# ---------------------------------------------------------------------------


def test_consistency_reg_loss_zero_when_matching():
    h = torch.randn(10, 4)
    idx = torch.tensor([2, 5, 7], dtype=torch.long)
    target = h[idx].clone()
    loss = consistency_reg_loss(h, idx, target)
    assert torch.allclose(loss, torch.zeros(())), f"expected 0, got {loss}"
    print("  test_consistency_reg_loss_zero_when_matching PASSED")


def test_consistency_reg_loss_known_value():
    # h[idx] all zeros, target all ones (dim=4) -> squared L2 per row = 4,
    # mean over 3 rows = 4.
    h = torch.zeros(10, 4)
    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    target = torch.ones(3, 4)
    loss = consistency_reg_loss(h, idx, target)
    assert torch.allclose(loss, torch.tensor(4.0)), f"expected 4.0, got {loss}"
    print("  test_consistency_reg_loss_known_value PASSED")


def test_consistency_reg_loss_empty_idx_is_noop():
    h = torch.randn(10, 4)
    idx = torch.zeros(0, dtype=torch.long)
    target = torch.zeros(0, 4)
    loss = consistency_reg_loss(h, idx, target)
    assert loss.item() == 0.0
    print("  test_consistency_reg_loss_empty_idx_is_noop PASSED")


# ---------------------------------------------------------------------------
#  DistTrainer.train_epoch integration
# ---------------------------------------------------------------------------


def _make_trainer(consistency_lambda=0.0, seed=0):
    torch.manual_seed(seed)
    N, D_in, D_out = 16, 8, 8
    kg1_size = 10
    ids_row = torch.arange(N)
    adj = torch.sparse_coo_tensor(
        torch.stack([ids_row, ids_row]), torch.ones(N), size=(N, N), dtype=torch.float32
    )
    h_init = torch.nn.Parameter(torch.randn(N, D_in))
    model = GCN(output_dim=D_out, num_layers=2)
    optimizer = torch.optim.Adam(list(model.parameters()) + [h_init], lr=1e-3)
    trainer = DistTrainer(
        model, optimizer, rank=0, world_size=1, neg_k=2, margin=1.0,
        consistency_lambda=consistency_lambda,
    )
    train_pairs = torch.tensor([[0, kg1_size], [1, kg1_size + 1]], dtype=torch.long)
    return trainer, adj, h_init, train_pairs, kg1_size


def test_consistency_lambda_zero_unchanged():
    """consistency_lambda=0.0 (default) must behave exactly as before, even
    if consistency_idx/target are passed — no regression.

    consistency_target is built with a fixed (non-RNG-consuming) tensor —
    torch.randn between trainer construction and train_epoch would shift the
    global RNG stream and make the two calls' negative sampling genuinely
    diverge for a reason unrelated to consistency_lambda.
    """
    trainer_a, adj, h_init_a, train_pairs, kg1_size = _make_trainer(consistency_lambda=0.0, seed=1)
    loss_a = trainer_a.train_epoch(adj, h_init_a, train_pairs, kg1_size)

    trainer_b, adj_b, h_init_b, train_pairs_b, kg1_size_b = _make_trainer(consistency_lambda=0.0, seed=1)
    consistency_idx = torch.tensor([3, 4], dtype=torch.long)
    consistency_target = torch.full((2, 8), 100.0)  # deliberately far from h_init, no RNG use
    loss_b = trainer_b.train_epoch(
        adj_b, h_init_b, train_pairs_b, kg1_size_b,
        consistency_idx=consistency_idx, consistency_target=consistency_target,
    )
    assert abs(loss_a - loss_b) < 1e-6, (
        f"consistency_lambda=0.0 must be a no-op regardless of target, got {loss_a} vs {loss_b}"
    )
    print("  test_consistency_lambda_zero_unchanged PASSED")


def test_consistency_lambda_positive_increases_loss():
    """A positive lambda with a target far from h_init's current values must
    increase the reported loss relative to lambda=0.0, confirming the term
    actually reaches the training loss. Fixed (non-RNG) target — see
    test_consistency_lambda_zero_unchanged's docstring for why."""
    trainer_a, adj, h_init_a, train_pairs, kg1_size = _make_trainer(consistency_lambda=0.0, seed=2)
    loss_a = trainer_a.train_epoch(adj, h_init_a, train_pairs, kg1_size)

    trainer_b, adj_b, h_init_b, train_pairs_b, kg1_size_b = _make_trainer(consistency_lambda=1.0, seed=2)
    consistency_idx = torch.tensor([3, 4], dtype=torch.long)
    consistency_target = torch.full((2, 8), 100.0)
    loss_b = trainer_b.train_epoch(
        adj_b, h_init_b, train_pairs_b, kg1_size_b,
        consistency_idx=consistency_idx, consistency_target=consistency_target,
    )
    assert loss_b > loss_a, (
        f"expected consistency term to raise the loss, got {loss_a} (off) vs {loss_b} (on)"
    )
    print("  test_consistency_lambda_positive_increases_loss PASSED")


def test_consistency_empty_idx_with_positive_lambda_is_noop():
    """consistency_lambda > 0 but consistency_idx=None/empty must still be a
    no-op (mirrors _reset_adam_momentum_for_rows' empty-index convention)."""
    trainer_a, adj, h_init_a, train_pairs, kg1_size = _make_trainer(consistency_lambda=1.0, seed=3)
    loss_a = trainer_a.train_epoch(adj, h_init_a, train_pairs, kg1_size)

    trainer_b, adj_b, h_init_b, train_pairs_b, kg1_size_b = _make_trainer(consistency_lambda=1.0, seed=3)
    loss_b = trainer_b.train_epoch(
        adj_b, h_init_b, train_pairs_b, kg1_size_b,
        consistency_idx=torch.zeros(0, dtype=torch.long),
        consistency_target=torch.zeros(0, 8),
    )
    assert abs(loss_a - loss_b) < 1e-6, (
        f"empty consistency_idx must be a no-op even with lambda>0, got {loss_a} vs {loss_b}"
    )
    print("  test_consistency_empty_idx_with_positive_lambda_is_noop PASSED")


if __name__ == "__main__":
    print("Stage H tests:")
    test_consistency_reg_loss_zero_when_matching()
    test_consistency_reg_loss_known_value()
    test_consistency_reg_loss_empty_idx_is_noop()
    test_consistency_lambda_zero_unchanged()
    test_consistency_lambda_positive_increases_loss()
    test_consistency_empty_idx_with_positive_lambda_is_noop()
    print("All Stage H tests PASSED.")
