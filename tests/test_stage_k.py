"""
test_stage_k.py — Stage K tests: dist_vbip.py against RREAGraph.

Checks that VBIP scoring (the --use_mm path) works when the graph is an
RREAGraph (models/rrea.py, dist_graph.py::build_rrea_graph) rather than a DGL
graph or a raw sparse tensor: compute_and_gather_vbip must return real pair
weights instead of silently falling back to uniform weights. Also covers the
ELsEA-faithful VBIP graph and score implementation (impl="elsea").

DGL-free, no GPU required, <30 sec.
"""

import os
import sys

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dist_vbip import (
    build_elsea_vbip_graph,
    compute_and_gather_vbip,
    compute_vbip_scores,
)
from models.rrea import RREAGraph


class _FakeData:
    """Minimal stand-in for OpenEAData -- just the fields
    build_elsea_vbip_graph() reads (kg1_triples, kg2_triples,
    edge_weights1, edge_weights2)."""

    def __init__(self, kg1_triples, kg2_triples, edge_weights1, edge_weights2):
        self.kg1_triples = kg1_triples
        self.kg2_triples = kg2_triples
        self.edge_weights1 = edge_weights1
        self.edge_weights2 = edge_weights2


def make_chain_rrea_graph(n=4, device=None):
    """0-1-2-3 undirected chain, symmetric-normalised with self-loop."""
    device = device or torch.device("cpu")
    edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
    eye = list(range(n))
    src = [e[0] for e in edges] + eye
    dst = [e[1] for e in edges] + eye
    val = [1.0] * len(edges) + [1.0] * n
    idx = torch.tensor([src, dst], dtype=torch.long, device=device)
    a_hat = torch.sparse_coo_tensor(idx, torch.tensor(val, device=device), (n, n)).coalesce()
    deg = torch.sparse.sum(a_hat, dim=1).to_dense().clamp(min=1e-8)
    d_inv_sqrt = deg.pow(-0.5)
    ii = a_hat.indices()
    vals = a_hat.values() * d_inv_sqrt[ii[0]] * d_inv_sqrt[ii[1]]
    adj_self_norm = torch.sparse_coo_tensor(ii, vals, (n, n), device=device).coalesce()

    edge_index = torch.tensor(edges, dtype=torch.long, device=device)
    edge_rel = torch.zeros(len(edges), dtype=torch.long, device=device)
    edge_val = torch.ones(len(edges), dtype=torch.float32, device=device)
    rel_profile_norm = torch.zeros(n, 2, device=device).to_sparse()

    return RREAGraph(adj_self_norm, rel_profile_norm, edge_index, edge_rel, edge_val, rel_num=1)


def test_compute_vbip_scores_accepts_rrea_graph_without_error():
    g = make_chain_rrea_graph()
    scores = compute_vbip_scores(
        g, anchor_local_indices=np.array([0], dtype=np.int64),
        num_nodes=4, device=torch.device("cpu"),
    )
    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()


def test_compute_vbip_scores_decays_with_graph_distance_from_anchor():
    g = make_chain_rrea_graph()
    scores = compute_vbip_scores(
        g, anchor_local_indices=np.array([0], dtype=np.int64),
        num_nodes=4, device=torch.device("cpu"),
    )
    # Node 0 (anchor) should score highest, monotonically decaying along the chain.
    assert scores[0] >= scores[1] >= scores[2] >= scores[3]
    assert scores[0] > scores[3]


def test_compute_and_gather_vbip_does_not_fall_back_to_uniform_for_rrea_graph():
    g = make_chain_rrea_graph()
    all_global_ids = np.array([100, 101, 102, 103])
    global2local = {100: 0, 101: 1, 102: 2, 103: 3}
    local_kg1_idx = np.array([0, 1])  # entities 0,1 are KG1; 2,3 are KG2
    train_links = [(100, 102)]  # kg1 global id 100 <-> kg2 global id 102
    local_train_pairs = torch.tensor([[0, 2]], dtype=torch.long)

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        pair_weights, global_scores = compute_and_gather_vbip(
            g, train_links, global2local, all_global_ids, local_kg1_idx,
            local_train_pairs, rank=0, world_size=1, device=torch.device("cpu"),
        )
    finally:
        dist.destroy_process_group()

    # (None, {}) would mean the RREAGraph input hit the uniform-weight
    # fallback; the real scoring path must succeed.
    assert pair_weights is not None
    assert global_scores != {}


# ---------------------------------------------------------------------------
#  impl="elsea" (faithful port) tests
# ---------------------------------------------------------------------------


def _make_fake_data_for_elsea_graph():
    # KG1: e0=100, e1=101, triple (100, r=0, 101). KG2: f0=102, f1=103,
    # triple (102, r=0, 103). Distinct r2f/r2if per KG so forward != reverse
    # weight and KG1 != KG2, making every direction/KG combination
    # independently checkable.
    kg1_triples = [(100, 0, 101)]
    kg2_triples = [(102, 0, 103)]
    edge_weights1 = {0: (0.5, 0.8)}  # (r2f, r2if)
    edge_weights2 = {0: (0.2, 0.9)}
    return _FakeData(kg1_triples, kg2_triples, edge_weights1, edge_weights2)


def test_build_elsea_vbip_graph_forward_reverse_weights_and_self_loop():
    data = _make_fake_data_for_elsea_graph()
    global2local = {100: 0, 101: 1, 102: 2, 103: 3}
    all_global_ids = np.array([100, 101, 102, 103])

    g = build_elsea_vbip_graph(data, global2local, all_global_ids, torch.device("cpu"))

    # Non-DGL environment (this test suite is DGL-free) -> sparse COO tensor,
    # row=dst, col=src, matching DGL's u_mul_e/sum orientation.
    g = g.coalesce()
    idx = g.indices()
    vals = g.values()
    weight_by_edge = {
        (int(idx[1, i]), int(idx[0, i])): float(vals[i]) for i in range(idx.shape[1])
    }

    # KG1 triple (100->101): w_fwd=max(r2if,0.3)=max(0.8,0.3)=0.8,
    #                         w_bwd=max(r2f,0.3)=max(0.5,0.3)=0.5
    assert abs(weight_by_edge[(0, 1)] - 0.8) < 1e-6  # src=0(100) -> dst=1(101)
    assert abs(weight_by_edge[(1, 0)] - 0.5) < 1e-6  # src=1(101) -> dst=0(100)

    # KG2 triple (102->103): w_fwd=max(r2if,0.3)=max(0.9,0.3)=0.9,
    #                         w_bwd=max(r2f,0.3)=max(0.2,0.3)=0.3 (floored)
    assert abs(weight_by_edge[(2, 3)] - 0.9) < 1e-6
    assert abs(weight_by_edge[(3, 2)] - 0.3) < 1e-6

    # Self-loops: max(all edge weights) + 0.5 = 0.9 + 0.5 = 1.4, on every node.
    for i in range(4):
        assert abs(weight_by_edge[(i, i)] - 1.4) < 1e-6


def test_build_elsea_vbip_graph_excludes_entities_outside_local_subgraph():
    data = _make_fake_data_for_elsea_graph()
    # Only entities 100, 102 are "local" -- 101/103 are not in all_global_ids,
    # so the (100,101) and (102,103) triples must be dropped entirely (both
    # endpoints have to be present, matching build_rrea_graph's convention).
    global2local = {100: 0, 102: 1}
    all_global_ids = np.array([100, 102])

    g = build_elsea_vbip_graph(data, global2local, all_global_ids, torch.device("cpu"))
    g = g.coalesce()
    idx = g.indices()
    # Only the 2 self-loops should remain (no cross-entity edges survive).
    assert idx.shape[1] == 2
    assert set(zip(idx[0].tolist(), idx[1].tolist(), strict=False)) == {(0, 0), (1, 1)}


def _make_weighted_chain_adj(n, device=None):
    """n-node chain (0-1-...-n-1), symmetric-normalised with self-loop --
    same construction as make_chain_rrea_graph but parametrised over n."""
    device = device or torch.device("cpu")
    edges = []
    for i in range(n - 1):
        edges += [(i, i + 1), (i + 1, i)]
    eye = list(range(n))
    src = [e[0] for e in edges] + eye
    dst = [e[1] for e in edges] + eye
    val = [1.0] * len(edges) + [1.0] * n
    idx = torch.tensor([src, dst], dtype=torch.long, device=device)
    a_hat = torch.sparse_coo_tensor(idx, torch.tensor(val, device=device), (n, n)).coalesce()
    deg = torch.sparse.sum(a_hat, dim=1).to_dense().clamp(min=1e-8)
    d_inv_sqrt = deg.pow(-0.5)
    ii = a_hat.indices()
    vals = a_hat.values() * d_inv_sqrt[ii[0]] * d_inv_sqrt[ii[1]]
    return torch.sparse_coo_tensor(ii, vals, (n, n), device=device).coalesce()


def test_compute_vbip_scores_elsea_impl_saturates_within_reach_and_zero_beyond_it():
    # Unlike the "simplified" model (a damped low-pass filter that decays
    # smoothly with distance), ELsEA's real formula's ratio-normalisation
    # step (x/(x+eps)) collapses ANY nonzero propagated value to ~1 before
    # squashing -- so within layer_num*2 hops (2 explicit passes) of an
    # anchor, every node saturates to the SAME score; beyond that hop
    # radius, the score is exactly 0 (genuinely unreached, not just small).
    # A 6-node chain with layer_num=2 (default) has exactly this contrast:
    # nodes 0-4 (distance <=4) are reachable, node 5 (distance 5) isn't.
    g = _make_weighted_chain_adj(6)
    scores = compute_vbip_scores(
        g, anchor_local_indices=np.array([0], dtype=np.int64),
        num_nodes=6, device=torch.device("cpu"), impl="elsea", layer_num=2,
    )
    assert scores.shape == (6,)
    assert torch.isfinite(scores).all()
    assert scores[0] > 0
    saturated = scores[0].item()
    for i in range(5):
        assert abs(scores[i].item() - saturated) < 1e-4, scores
    assert scores[5].item() == 0.0, scores


def test_compute_and_gather_vbip_elsea_impl_requires_data():
    g = make_chain_rrea_graph()
    all_global_ids = np.array([100, 101, 102, 103])
    global2local = {100: 0, 101: 1, 102: 2, 103: 3}
    local_kg1_idx = np.array([0, 1])
    train_links = [(100, 102)]
    local_train_pairs = torch.tensor([[0, 2]], dtype=torch.long)

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29502")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        # No `data` passed -> impl="elsea" must fall back gracefully (same
        # try/except contract as any other VBIP failure), not crash the caller.
        pair_weights, global_scores = compute_and_gather_vbip(
            g, train_links, global2local, all_global_ids, local_kg1_idx,
            local_train_pairs, rank=0, world_size=1, device=torch.device("cpu"),
            impl="elsea",
        )
        assert pair_weights is None
        assert global_scores == {}
    finally:
        dist.destroy_process_group()


def test_compute_and_gather_vbip_elsea_impl_succeeds_with_data():
    data = _make_fake_data_for_elsea_graph()
    all_global_ids = np.array([100, 101, 102, 103])
    global2local = {100: 0, 101: 1, 102: 2, 103: 3}
    local_kg1_idx = np.array([0, 1])
    train_links = [(100, 102)]
    local_train_pairs = torch.tensor([[0, 2]], dtype=torch.long)

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29503")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        pair_weights, global_scores = compute_and_gather_vbip(
            None, train_links, global2local, all_global_ids, local_kg1_idx,
            local_train_pairs, rank=0, world_size=1, device=torch.device("cpu"),
            impl="elsea", data=data,
        )
    finally:
        dist.destroy_process_group()

    assert pair_weights is not None
    assert global_scores != {}


if __name__ == "__main__":
    print("Stage K tests:")
    test_compute_vbip_scores_accepts_rrea_graph_without_error()
    print("  test_compute_vbip_scores_accepts_rrea_graph_without_error PASSED")
    test_compute_vbip_scores_decays_with_graph_distance_from_anchor()
    print("  test_compute_vbip_scores_decays_with_graph_distance_from_anchor PASSED")
    test_compute_and_gather_vbip_does_not_fall_back_to_uniform_for_rrea_graph()
    print("  test_compute_and_gather_vbip_does_not_fall_back_to_uniform_for_rrea_graph PASSED")
    test_build_elsea_vbip_graph_forward_reverse_weights_and_self_loop()
    print("  test_build_elsea_vbip_graph_forward_reverse_weights_and_self_loop PASSED")
    test_build_elsea_vbip_graph_excludes_entities_outside_local_subgraph()
    print("  test_build_elsea_vbip_graph_excludes_entities_outside_local_subgraph PASSED")
    test_compute_vbip_scores_elsea_impl_saturates_within_reach_and_zero_beyond_it()
    print("  test_compute_vbip_scores_elsea_impl_saturates_within_reach_and_zero_beyond_it PASSED")
    test_compute_and_gather_vbip_elsea_impl_requires_data()
    print("  test_compute_and_gather_vbip_elsea_impl_requires_data PASSED")
    test_compute_and_gather_vbip_elsea_impl_succeeds_with_data()
    print("  test_compute_and_gather_vbip_elsea_impl_succeeds_with_data PASSED")
    print("All Stage K tests PASSED.")
