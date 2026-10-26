"""
test_stage_c.py — Stage C tests: write_rank_embeddings, refresh_halo_rows.

DGL-free, no GPU required.
"""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from framework import (
    _merge_pseudo_seeds,
    _reset_adam_momentum_for_rows,
    refresh_halo_rows,
    write_rank_embeddings,
)

# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------


def test_write_rank_embeddings():
    """Files written by write_rank_embeddings are readable and correct."""
    with tempfile.TemporaryDirectory() as tmp:
        ids = np.array([0, 1, 2], dtype=np.int64)
        emb = np.random.randn(3, 8).astype(np.float32)
        write_rank_embeddings(tmp, rank=0, ids=ids, emb=emb)

        assert os.path.exists(os.path.join(tmp, "ids_0.npy"))
        assert os.path.exists(os.path.join(tmp, "emb_0.npy"))
        np.testing.assert_array_equal(np.load(os.path.join(tmp, "ids_0.npy")), ids)
        np.testing.assert_allclose(np.load(os.path.join(tmp, "emb_0.npy")), emb)
        print("  test_write_rank_embeddings PASSED")


def test_refresh_halo_rows():
    """refresh_halo_rows copies pushed embeddings straight into h_init's halo
    rows. h_init and the pushed embeddings are both output_dim-wide (h_init is
    the propagated entity table itself, see models/gcn.py), so no padding."""
    with tempfile.TemporaryDirectory() as tmp:
        # Two ranks write embeddings (output_dim=4)
        ids0 = np.array([0, 1, 2], dtype=np.int64)
        emb0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
        write_rank_embeddings(tmp, rank=0, ids=ids0, emb=emb0)

        ids1 = np.array([3, 4], dtype=np.int64)
        emb1 = np.array([[0, 0, 0, 1], [1, 1, 0, 0]], dtype=np.float32)
        write_rank_embeddings(tmp, rank=1, ids=ids1, emb=emb1)

        # h_init output_dim=4; entities 3 and 4 are halo nodes at local idx 4,5
        h_init = torch.zeros(6, 4)
        halo_kg1_idx = np.array([4, 5], dtype=np.int64)
        halo_global_ids = np.array([3, 4], dtype=np.int64)

        h_out = refresh_halo_rows(h_init, halo_kg1_idx, halo_global_ids, tmp, world_size=2)

        np.testing.assert_allclose(h_out[4].numpy(), emb1[0], atol=1e-5)
        np.testing.assert_allclose(h_out[5].numpy(), emb1[1], atol=1e-5)
        np.testing.assert_allclose(h_out[0].numpy(), np.zeros(4), atol=1e-5)
        print("  test_refresh_halo_rows PASSED")


def test_refresh_halo_rows_missing_files():
    """refresh_halo_rows is a no-op when embedding files do not exist."""
    with tempfile.TemporaryDirectory() as tmp:
        h_init = torch.ones(4, 8)
        halo_kg1_idx = np.array([0, 1], dtype=np.int64)
        halo_global_ids = np.array([100, 101], dtype=np.int64)
        h_out = refresh_halo_rows(h_init, halo_kg1_idx, halo_global_ids, tmp, world_size=2)
        np.testing.assert_allclose(h_out.numpy(), h_init.numpy())
        print("  test_refresh_halo_rows_missing_files PASSED")


def test_refresh_halo_rows_unknown_ids():
    """Halo IDs not found in any rank's file leave corresponding rows unchanged."""
    with tempfile.TemporaryDirectory() as tmp:
        write_rank_embeddings(tmp, rank=0, ids=np.array([0], dtype=np.int64),
                              emb=np.ones((1, 4), dtype=np.float32))
        h_init = torch.zeros(3, 4)
        h_out = refresh_halo_rows(
            h_init,
            np.array([1, 2], dtype=np.int64),   # local indices
            np.array([99, 100], dtype=np.int64),  # global IDs not in any rank's file
            tmp, world_size=1,
        )
        np.testing.assert_allclose(h_out.numpy(), np.zeros((3, 4), dtype=np.float32), atol=1e-5)
        print("  test_refresh_halo_rows_unknown_ids PASSED")


def test_reset_adam_momentum_for_rows_zeros_specified_rows_only():
    """After a few Adam steps build up real momentum, resetting specific
    rows zeros exactly those rows' exp_avg/exp_avg_sq and leaves every
    other row's momentum untouched."""
    param = torch.nn.Parameter(torch.randn(5, 4))
    optimizer = torch.optim.Adam([param], lr=1e-2)
    for _ in range(3):
        optimizer.zero_grad()
        (param ** 2).sum().backward()
        optimizer.step()

    state = optimizer.state[param]
    assert state["exp_avg"].abs().sum() > 0
    assert state["exp_avg_sq"].abs().sum() > 0
    exp_avg_before = state["exp_avg"].clone()
    exp_avg_sq_before = state["exp_avg_sq"].clone()

    reset_rows = np.array([1, 3], dtype=np.int64)
    did_reset = _reset_adam_momentum_for_rows(optimizer, param, reset_rows)
    assert did_reset is True

    for row in range(5):
        if row in reset_rows:
            np.testing.assert_allclose(state["exp_avg"][row].numpy(), np.zeros(4))
            np.testing.assert_allclose(state["exp_avg_sq"][row].numpy(), np.zeros(4))
        else:
            np.testing.assert_allclose(
                state["exp_avg"][row].numpy(), exp_avg_before[row].numpy()
            )
            np.testing.assert_allclose(
                state["exp_avg_sq"][row].numpy(), exp_avg_sq_before[row].numpy()
            )
    print("  test_reset_adam_momentum_for_rows_zeros_specified_rows_only PASSED")


def test_reset_adam_momentum_for_rows_noop_before_first_step():
    """No optimizer.step() has ever run, so there's no momentum state for
    the parameter yet — resetting is a clean no-op (returns False), not a
    KeyError, since the persist_optimizer path can call this on the very
    first iteration a parameter is reused."""
    param = torch.nn.Parameter(torch.randn(5, 4))
    optimizer = torch.optim.Adam([param], lr=1e-2)
    did_reset = _reset_adam_momentum_for_rows(
        optimizer, param, np.array([0, 1], dtype=np.int64)
    )
    assert did_reset is False
    print("  test_reset_adam_momentum_for_rows_noop_before_first_step PASSED")


def test_reset_adam_momentum_for_rows_empty_idx_is_noop():
    """An empty row-index array (e.g. world_size=1, no halo rows at all)
    is a clean no-op."""
    param = torch.nn.Parameter(torch.randn(5, 4))
    optimizer = torch.optim.Adam([param], lr=1e-2)
    optimizer.zero_grad()
    (param ** 2).sum().backward()
    optimizer.step()
    did_reset = _reset_adam_momentum_for_rows(
        optimizer, param, np.array([], dtype=np.int64)
    )
    assert did_reset is False
    print("  test_reset_adam_momentum_for_rows_empty_idx_is_noop PASSED")


def test_merge_pseudo_seeds_rejects_contradictory_kg1_id():
    """Entity-level freezing: once kg1_id=1 is used
    (train_links already has (1,10)), a later pseudo-seed pairing kg1_id=1
    with a DIFFERENT kg2_id must be rejected, even though the exact tuple
    (1, 99) was never seen before -- exact-tuple dedup alone would accept
    it and create a contradiction."""
    train_links = [(1, 10), (2, 20)]
    new_seeds = [(1, 99), (3, 30)]
    updated, added, correct = _merge_pseudo_seeds(train_links, new_seeds, set())
    assert (1, 99) not in updated, "kg1_id=1 already used -> contradictory pair must be rejected"
    assert (3, 30) in updated, "kg1_id=3/kg2_id=30 both unused -> must be accepted"
    assert added == 1
    print("  test_merge_pseudo_seeds_rejects_contradictory_kg1_id PASSED")


def test_merge_pseudo_seeds_rejects_contradictory_kg2_id():
    """Same freezing, kg2_id side: kg2_id=20 already used (train_links has
    (2,20)) -> a pseudo-seed pairing a different kg1_id with kg2_id=20 must
    be rejected."""
    train_links = [(2, 20)]
    new_seeds = [(5, 20), (6, 60)]
    updated, added, correct = _merge_pseudo_seeds(train_links, new_seeds, set())
    assert (5, 20) not in updated, "kg2_id=20 already used -> contradictory pair must be rejected"
    assert (6, 60) in updated
    assert added == 1
    print("  test_merge_pseudo_seeds_rejects_contradictory_kg2_id PASSED")


def test_merge_pseudo_seeds_rejects_within_batch_contradiction():
    """Two candidates in the SAME new_seeds batch both touching kg1_id=1:
    only the first-encountered one should be accepted -- matches RREA's
    rest_set removal happening immediately, not just across calls."""
    train_links = []
    new_seeds = [(1, 10), (1, 11)]
    updated, added, correct = _merge_pseudo_seeds(train_links, new_seeds, set())
    assert (1, 10) in updated
    assert (1, 11) not in updated
    assert added == 1
    print("  test_merge_pseudo_seeds_rejects_within_batch_contradiction PASSED")


def test_merge_pseudo_seeds_excludes_test_links():
    """A candidate pair that happens to be in test_links_set must never be
    added, regardless of entity-freezing state -- prevents pseudo-seed
    augmentation from contaminating the held-out test set."""
    train_links = []
    new_seeds = [(7, 70)]
    updated, added, correct = _merge_pseudo_seeds(train_links, new_seeds, {(7, 70)})
    assert (7, 70) not in updated
    assert added == 0
    print("  test_merge_pseudo_seeds_excludes_test_links PASSED")


def test_merge_pseudo_seeds_correct_estimate_counts_known_pairs():
    """correct_estimate counts accepted pairs found in known_pairs
    (train+valid), used only for logging estimated precision -- must not
    count rejected pairs."""
    train_links = [(1, 10)]
    new_seeds = [(1, 99), (2, 20), (3, 30)]
    known_pairs = {(2, 20), (3, 30)}
    updated, added, correct = _merge_pseudo_seeds(
        train_links, new_seeds, set(), known_pairs
    )
    assert added == 2
    assert correct == 2, "both accepted pairs are in known_pairs"
    print("  test_merge_pseudo_seeds_correct_estimate_counts_known_pairs PASSED")


if __name__ == "__main__":
    print("Stage C tests:")
    test_write_rank_embeddings()
    test_refresh_halo_rows()
    test_refresh_halo_rows_missing_files()
    test_refresh_halo_rows_unknown_ids()
    test_reset_adam_momentum_for_rows_zeros_specified_rows_only()
    test_reset_adam_momentum_for_rows_noop_before_first_step()
    test_reset_adam_momentum_for_rows_empty_idx_is_noop()
    test_merge_pseudo_seeds_rejects_contradictory_kg1_id()
    test_merge_pseudo_seeds_rejects_contradictory_kg2_id()
    test_merge_pseudo_seeds_rejects_within_batch_contradiction()
    test_merge_pseudo_seeds_excludes_test_links()
    test_merge_pseudo_seeds_correct_estimate_counts_known_pairs()
    print("All Stage C tests PASSED.")
