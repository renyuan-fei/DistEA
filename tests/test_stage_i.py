"""
test_stage_i.py — Stage I tests: --shared_h_init.

DGL-free, no GPU required, <30 sec.

Covers the differentiable gather that --shared_h_init relies on: DistTrainer.train_epoch(..., local_idx=...) selects a rank's local
subgraph rows out of a FULL shared h_init table, and gradients from that
rank's local loss must flow back into exactly those rows of the full table
(and nowhere else) via ordinary autograd indexing — the same mechanism an
embedding lookup uses.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dist_trainer import DistTrainer, _masked_row_and_count
from models.gcn import GCN


def _make_shared_trainer(seed=0):
    """
    A small "full" entity table (20 rows) and a "local" subgraph covering
    only 8 of those 20 global rows (local_idx), mimicking one rank's view
    under --shared_h_init.
    """
    torch.manual_seed(seed)
    N_global, D = 20, 8
    N_local = 8
    kg1_size = 5  # of the 8 local rows, first 5 are this rank's KG1 side

    h_init_full = torch.nn.Parameter(torch.randn(N_global, D))
    local_idx = torch.tensor([1, 3, 4, 6, 9, 11, 14, 17], dtype=torch.long)

    ids_row = torch.arange(N_local)
    adj = torch.sparse_coo_tensor(
        torch.stack([ids_row, ids_row]), torch.ones(N_local), size=(N_local, N_local),
        dtype=torch.float32,
    )
    model = GCN(output_dim=D, num_layers=2)
    optimizer = torch.optim.Adam(list(model.parameters()) + [h_init_full], lr=1e-3)
    trainer = DistTrainer(model, optimizer, rank=0, world_size=1, neg_k=2, margin=1.0)

    train_pairs = torch.tensor([[0, kg1_size], [1, kg1_size + 1]], dtype=torch.long)
    return trainer, adj, h_init_full, local_idx, train_pairs, kg1_size


def test_local_idx_gathers_correct_rows_and_gradient_is_localized():
    trainer, adj, h_init_full, local_idx, train_pairs, kg1_size = _make_shared_trainer()

    loss = trainer.train_epoch(
        adj, h_init_full, train_pairs, kg1_size, local_idx=local_idx,
    )
    assert loss >= 0.0 and loss == loss, f"expected finite non-negative loss, got {loss}"

    assert h_init_full.grad is not None, "expected gradient on the full h_init table"
    touched = set(local_idx.tolist())
    grad_norms = h_init_full.grad.norm(dim=1)
    for i in range(h_init_full.size(0)):
        if i in touched:
            continue
        assert grad_norms[i].item() == 0.0, (
            f"row {i} was not part of this rank's local_idx but has nonzero "
            f"gradient ({grad_norms[i].item()}) -- gather is leaking gradient "
            f"to untouched rows"
        )
    print("  test_local_idx_gathers_correct_rows_and_gradient_is_localized PASSED")


def test_local_idx_none_matches_pre_shared_h_init_behaviour():
    """local_idx=None (default) must behave exactly like passing h_init
    directly, as in the other DistTrainer tests (test_stage_a/b/d/h.py)."""
    torch.manual_seed(5)
    N, D_in, D_out = 16, 8, 8
    kg1_size = 10
    ids_row = torch.arange(N)
    adj = torch.sparse_coo_tensor(
        torch.stack([ids_row, ids_row]), torch.ones(N), size=(N, N), dtype=torch.float32
    )
    h_init = torch.nn.Parameter(torch.randn(N, D_in))
    model = GCN(output_dim=D_out, num_layers=2)
    optimizer = torch.optim.Adam(list(model.parameters()) + [h_init], lr=1e-3)
    trainer = DistTrainer(model, optimizer, rank=0, world_size=1, neg_k=2, margin=1.0)
    train_pairs = torch.tensor([[0, kg1_size], [1, kg1_size + 1]], dtype=torch.long)

    loss = trainer.train_epoch(adj, h_init, train_pairs, kg1_size, local_idx=None)
    assert loss >= 0.0 and loss == loss
    print("  test_local_idx_none_matches_pre_shared_h_init_behaviour PASSED")


# ---------------------------------------------------------------------------
#  _masked_row_and_count (--shared_h_init_weighted_avg)
# ---------------------------------------------------------------------------


def _simulate_weighted_allreduce(per_rank_h_init, per_rank_local_idx):
    """
    Manually simulate what dist.all_reduce(SUM) on _masked_row_and_count's
    outputs would produce across several ranks, without needing an actual
    multi-process torch.distributed group — sums the pure-function outputs
    directly in Python, exactly what the collective would do.
    """
    masked_sum = None
    count_sum = None
    for h_init, local_idx in zip(per_rank_h_init, per_rank_local_idx, strict=False):
        masked, count = _masked_row_and_count(h_init, local_idx)
        masked_sum = masked.clone() if masked_sum is None else masked_sum + masked
        count_sum = count.clone() if count_sum is None else count_sum + count
    return masked_sum / count_sum.clamp_min(1.0).unsqueeze(1)


def test_masked_average_gives_exclusive_row_full_value_not_diluted():
    """A row touched by only ONE of two simulated ranks must come out of the
    weighted average as exactly that rank's value — not halved the way plain
    dist.all_reduce(AVG) (dividing by world_size=2 unconditionally) would."""
    N, D = 5, 3
    h_a = torch.nn.Parameter(torch.zeros(N, D))
    h_b = torch.nn.Parameter(torch.zeros(N, D))
    with torch.no_grad():
        h_a[2] = torch.tensor([1.0, 2.0, 3.0])  # row 2: only rank A touches it
        h_b[2] = torch.tensor([99.0, 99.0, 99.0])  # rank B's stale, untouched copy
    local_idx_a = torch.tensor([2], dtype=torch.long)  # rank A's local subgraph
    local_idx_b = torch.tensor([4], dtype=torch.long)  # rank B never touches row 2

    result = _simulate_weighted_allreduce([h_a, h_b], [local_idx_a, local_idx_b])
    assert torch.allclose(result[2], torch.tensor([1.0, 2.0, 3.0])), (
        f"expected row 2 to come out as rank A's exact value (untouched by "
        f"rank B), got {result[2].tolist()} -- weighted average is still "
        f"diluting an exclusively-touched row"
    )
    print("  test_masked_average_gives_exclusive_row_full_value_not_diluted PASSED")


def test_masked_average_averages_shared_row_over_touching_ranks_only():
    """A row touched by BOTH ranks must come out as the plain mean of the
    two ranks' values (world_size=2 here, so this coincides with plain AVG,
    but via the touch-count divide, not an unconditional /world_size)."""
    N, D = 5, 3
    h_a = torch.nn.Parameter(torch.zeros(N, D))
    h_b = torch.nn.Parameter(torch.zeros(N, D))
    with torch.no_grad():
        h_a[1] = torch.tensor([2.0, 0.0, 0.0])
        h_b[1] = torch.tensor([4.0, 0.0, 0.0])
    local_idx_a = torch.tensor([1], dtype=torch.long)
    local_idx_b = torch.tensor([1], dtype=torch.long)

    result = _simulate_weighted_allreduce([h_a, h_b], [local_idx_a, local_idx_b])
    assert torch.allclose(result[1], torch.tensor([3.0, 0.0, 0.0])), (
        f"expected row 1 (touched by both ranks) to average to [3,0,0], "
        f"got {result[1].tolist()}"
    )
    print("  test_masked_average_averages_shared_row_over_touching_ranks_only PASSED")


def test_masked_row_and_count_empty_idx_is_all_zero():
    h = torch.nn.Parameter(torch.randn(4, 3))
    idx = torch.zeros(0, dtype=torch.long)
    masked, count = _masked_row_and_count(h, idx)
    assert torch.count_nonzero(masked) == 0
    assert torch.count_nonzero(count) == 0
    print("  test_masked_row_and_count_empty_idx_is_all_zero PASSED")


if __name__ == "__main__":
    print("Stage I tests:")
    test_local_idx_gathers_correct_rows_and_gradient_is_localized()
    test_local_idx_none_matches_pre_shared_h_init_behaviour()
    test_masked_average_gives_exclusive_row_full_value_not_diluted()
    test_masked_average_averages_shared_row_over_touching_ranks_only()
    test_masked_row_and_count_empty_idx_is_all_zero()
    print("All Stage I tests PASSED.")
