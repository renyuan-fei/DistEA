"""
test_stage_e.py — Stage E tests: eval.py similarity/CSLS/ranking helpers.

Tests eval.py's metric logic (L1 distance, cosine similarity, CSLS
correction, Hit@k/MRR ranking) and the embedding export bundle directly
through the underlying helpers, rather than through selective_eval /
distributed_eval / generate_pseudo_seeds, which need an initialised
torch.distributed process group.

DGL-free, no GPU required, <30 sec.
"""

import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval import (
    _apply_csls,
    _compute_metrics,
    _compute_sim,
    _cosine_sim,
    _csls_thr,
    _neg_l1_dist,
    _save_export_bundle,
)


def test_neg_l1_dist_identical_is_zero():
    e = torch.tensor([[1.0, 2.0, 3.0]])
    sim = _neg_l1_dist(e, e, torch.device("cpu"))
    assert sim.shape == (1, 1)
    assert torch.allclose(sim, torch.zeros(1, 1)), f"expected 0, got {sim}"
    print("  test_neg_l1_dist_identical_is_zero PASSED")


def test_neg_l1_dist_known_value():
    e1 = torch.tensor([[0.0, 0.0]])
    e2 = torch.tensor([[3.0, 4.0]])
    sim = _neg_l1_dist(e1, e2, torch.device("cpu"))
    assert torch.allclose(sim, torch.tensor([[-7.0]])), f"expected -7, got {sim}"
    print("  test_neg_l1_dist_known_value PASSED")


def test_cosine_sim_identical_direction_is_one():
    e1 = torch.tensor([[1.0, 0.0]])
    e2 = torch.tensor([[1.0, 0.0]])
    sim = _cosine_sim(e1, e2, torch.device("cpu"))
    assert torch.allclose(sim, torch.tensor([[1.0]]), atol=1e-6), f"expected 1.0, got {sim}"
    print("  test_cosine_sim_identical_direction_is_one PASSED")


def test_cosine_sim_orthogonal_is_zero():
    e1 = torch.tensor([[1.0, 0.0]])
    e2 = torch.tensor([[0.0, 1.0]])
    sim = _cosine_sim(e1, e2, torch.device("cpu"))
    assert torch.allclose(sim, torch.tensor([[0.0]]), atol=1e-6), f"expected 0.0, got {sim}"
    print("  test_cosine_sim_orthogonal_is_zero PASSED")


def test_cosine_sim_is_scale_invariant():
    """Same direction at any magnitude → cosine 1.0 — unlike L1, which grows
    with the magnitude gap. This is precisely why RREA's CSLS_test
    L2-normalises before ranking and GCN-Align's raw-output get_hits does
    not: the two metrics can disagree (see test_l1_and_cosine_can_disagree)."""
    e1 = torch.tensor([[1.0, 0.0]])
    e2 = torch.tensor([[500.0, 0.0]])
    sim = _cosine_sim(e1, e2, torch.device("cpu"))
    assert torch.allclose(sim, torch.tensor([[1.0]]), atol=1e-5), f"expected 1.0, got {sim}"
    print("  test_cosine_sim_is_scale_invariant PASSED")


def test_compute_sim_dispatch_matches_helpers():
    e1 = torch.randn(5, 4)
    e2 = torch.randn(6, 4)
    device = torch.device("cpu")
    assert torch.allclose(_compute_sim("l1", e1, e2, device), _neg_l1_dist(e1, e2, device))
    assert torch.allclose(_compute_sim("cosine", e1, e2, device), _cosine_sim(e1, e2, device))
    print("  test_compute_sim_dispatch_matches_helpers PASSED")


def test_l1_and_cosine_can_disagree_on_ranking():
    """
    Why eval.py takes a metric parameter instead of always ranking by L1:
    a query can have a DIFFERENT nearest neighbour depending
    on which metric is used, on the exact same embeddings.

    query=[1,0]; A=[10,0] (same direction, large magnitude); B=[1,1]
    (different direction, small magnitude).
      L1:     dist(query,A)=9,     dist(query,B)=1     -> B is closer.
      cosine: cos(query,A)=1.0,    cos(query,B)=0.707  -> A is closer.
    """
    device = torch.device("cpu")
    query = torch.tensor([[1.0, 0.0]])
    candidates = torch.tensor([[10.0, 0.0], [1.0, 1.0]])  # A, B

    l1_sim = _neg_l1_dist(query, candidates, device)
    cos_sim = _cosine_sim(query, candidates, device)

    l1_best = l1_sim.argmax(dim=1).item()
    cos_best = cos_sim.argmax(dim=1).item()

    assert l1_best == 1, f"L1 should prefer B (index 1), got {l1_best}"
    assert cos_best == 0, f"cosine should prefer A (index 0), got {cos_best}"
    assert l1_best != cos_best, "the two metrics were expected to disagree here"
    print("  test_l1_and_cosine_can_disagree_on_ranking PASSED")


def test_csls_correction_formula():
    # 2x2 sim matrix, k=1 -> top-1 threshold is just each row's/col's own max.
    sim = torch.tensor([[0.9, 0.2], [0.1, 0.8]])
    device = torch.device("cpu")
    row_thr = _csls_thr(sim, k=1, device=device)
    col_thr = _csls_thr(sim.t(), k=1, device=device)
    assert torch.allclose(row_thr, torch.tensor([0.9, 0.8]))
    assert torch.allclose(col_thr, torch.tensor([0.9, 0.8]))  # col maxes: [0.9, 0.8]

    csls = _apply_csls(sim, row_thr, col_thr, device)
    # csls_ij = 2*sim_ij - row_thr_i - col_thr_j
    expected = torch.tensor(
        [
            [2 * 0.9 - 0.9 - 0.9, 2 * 0.2 - 0.9 - 0.8],
            [2 * 0.1 - 0.8 - 0.9, 2 * 0.8 - 0.8 - 0.8],
        ]
    )
    assert torch.allclose(csls, expected, atol=1e-6), f"expected {expected}, got {csls}"
    print("  test_csls_correction_formula PASSED")


def test_compute_metrics_perfect_alignment():
    # Identity-like sim: row i is highest at column i -> Hit@1=1, MRR=1.
    sim = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ]
    )
    test_links = [(0, 0), (1, 1), (2, 2)]
    metrics = _compute_metrics(sim, test_links, torch.device("cpu"), top_k=(1, 10))
    assert metrics["Hit@1"] == 1.0, metrics
    assert metrics["MRR"] == 1.0, metrics
    print("  test_compute_metrics_perfect_alignment PASSED")


def test_compute_metrics_worst_case_rank():
    # True target is always the LOWEST-similarity column -> Hit@1=0, low MRR.
    sim = torch.tensor(
        [
            [5.0, 4.0, 0.0],
            [5.0, 4.0, 0.0],
        ]
    )
    test_links = [(0, 2), (1, 2)]  # true target = column 2, ranked last (rank 3)
    metrics = _compute_metrics(sim, test_links, torch.device("cpu"), top_k=(1, 10))
    assert metrics["Hit@1"] == 0.0, metrics
    assert abs(metrics["MRR"] - (1.0 / 3.0)) < 1e-6, metrics
    print("  test_compute_metrics_worst_case_rank PASSED")


def test_compute_metrics_empty_links():
    sim = torch.zeros(3, 3)
    metrics = _compute_metrics(sim, [], torch.device("cpu"), top_k=(1, 10))
    assert metrics == {"Hit@1": 0.0, "Hit@10": 0.0, "MRR": 0.0}, metrics
    print("  test_compute_metrics_empty_links PASSED")


def test_compute_metrics_top_k_1_5_10_50_reports_all_four_hits():
    # The top_k tuple used by framework.py's test-eval call site -- true target sits at rank 3 for every row, so Hit@1/@5
    # should both miss, Hit@10/@50 should both hit.
    n = 60
    sim = torch.zeros(2, n)
    sim[0, 2] = 5.0  # rank 1 for row 0 -> true target (col 0) is rank 2
    sim[0, 0] = 4.0
    sim[1, 2] = 5.0
    sim[1, 1] = 4.0
    test_links = [(0, 0), (1, 1)]
    metrics = _compute_metrics(sim, test_links, torch.device("cpu"), top_k=(1, 5, 10, 50))
    assert set(metrics.keys()) == {"Hit@1", "Hit@5", "Hit@10", "Hit@50", "MRR"}, metrics
    assert metrics["Hit@1"] == 0.0, metrics
    assert metrics["Hit@5"] == 1.0, metrics
    assert metrics["Hit@10"] == 1.0, metrics
    assert metrics["Hit@50"] == 1.0, metrics
    print("  test_compute_metrics_top_k_1_5_10_50_reports_all_four_hits PASSED")


def test_save_export_bundle_writes_unified_eval_contract():
    # scripts/unified_eval.py's contract: embeddings.npy row order is
    # [kg1 rows..., kg2 rows...], ids.npy/kg_labels.npy aligned to that same
    # order, test_links.json is the raw global-id pair list (not row
    # indices) so unified_eval.py can independently re-derive row lookups
    # under whatever {metric, use_csls} protocol it's asked to score.
    kg1_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    kg2_emb = torch.tensor([[1.0, 0.1], [0.5, 0.5], [0.0, 0.9]])
    kg1_ids = np.array([100, 101], dtype=np.int64)
    kg2_ids = np.array([200, 201, 202], dtype=np.int64)
    test_links = [(100, 200), (101, 202)]

    with tempfile.TemporaryDirectory() as tmp:
        export_dir = os.path.join(tmp, "export", "distea", "demo")
        _save_export_bundle(kg1_emb, kg1_ids, kg2_emb, kg2_ids, test_links, export_dir)

        emb = np.load(os.path.join(export_dir, "embeddings.npy"))
        ids = np.load(os.path.join(export_dir, "ids.npy"))
        kg_labels = np.load(os.path.join(export_dir, "kg_labels.npy"))
        with open(os.path.join(export_dir, "test_links.json")) as f:
            loaded_links = json.load(f)

        assert emb.shape == (5, 2), emb.shape
        assert ids.tolist() == [100, 101, 200, 201, 202], ids.tolist()
        assert kg_labels.tolist() == [0, 0, 1, 1, 1], kg_labels.tolist()
        assert loaded_links == [[100, 200], [101, 202]], loaded_links
        assert np.allclose(emb[0], [1.0, 0.0]), emb[0]
        assert np.allclose(emb[2], [1.0, 0.1]), emb[2]
    print("  test_save_export_bundle_writes_unified_eval_contract PASSED")


if __name__ == "__main__":
    print("Stage E tests:")
    test_neg_l1_dist_identical_is_zero()
    test_neg_l1_dist_known_value()
    test_cosine_sim_identical_direction_is_one()
    test_cosine_sim_orthogonal_is_zero()
    test_cosine_sim_is_scale_invariant()
    test_compute_sim_dispatch_matches_helpers()
    test_l1_and_cosine_can_disagree_on_ranking()
    test_csls_correction_formula()
    test_compute_metrics_perfect_alignment()
    test_compute_metrics_worst_case_rank()
    test_compute_metrics_empty_links()
    test_compute_metrics_top_k_1_5_10_50_reports_all_four_hits()
    test_save_export_bundle_writes_unified_eval_contract()
    print("All Stage E tests PASSED.")
