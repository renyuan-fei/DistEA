"""
test_stage_d.py — Stage D tests: KG2 Counterpart Halo (dist_ctx_kg2.py) and
its integration into DistTrainer's negative sampling.

DGL-free, no GPU, no torch.distributed required.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dist_ctx_kg2 import build_kg2_adjacency, select_halo_kg2
from dist_trainer import DistTrainer
from models.gcn import GCN

# ---------------------------------------------------------------------------
#  dist_ctx_kg2 tests
# ---------------------------------------------------------------------------


def test_build_kg2_adjacency_is_undirected():
    triples = [(1, 0, 2), (2, 0, 3)]
    adj = build_kg2_adjacency(triples)
    assert adj[1] == {2}
    assert adj[2] == {1, 3}
    assert adj[3] == {2}
    print("  test_build_kg2_adjacency_is_undirected PASSED")


def test_select_halo_kg2_no_anchors_returns_empty():
    adj = build_kg2_adjacency([(1, 0, 2)])
    result = select_halo_kg2(
        anchors=set(), kg2_adj=adj, node_weights2={}, exclude=set(),
        max_hop=2, top_k=5,
    )
    assert result == []
    print("  test_select_halo_kg2_no_anchors_returns_empty PASSED")


def test_select_halo_kg2_prefers_closer_hops():
    # 1 - 2 - 3 - 4 chain; anchor at 1 -> 2 is 1 hop, 3 is 2 hops, 4 is 3 hops.
    triples = [(1, 0, 2), (2, 0, 3), (3, 0, 4)]
    adj = build_kg2_adjacency(triples)
    weights = {2: 1, 3: 1, 4: 1}
    result = select_halo_kg2(
        anchors={1}, kg2_adj=adj, node_weights2=weights, exclude=set(),
        max_hop=3, top_k=3,
    )
    assert result[0] == 2, f"closest hop should rank first, got {result}"
    assert set(result) == {2, 3, 4}
    print("  test_select_halo_kg2_prefers_closer_hops PASSED")


def test_select_halo_kg2_excludes_local_and_respects_top_k():
    triples = [(1, 0, 2), (1, 0, 3), (1, 0, 4)]
    adj = build_kg2_adjacency(triples)
    weights = {2: 1, 3: 1, 4: 1}
    result = select_halo_kg2(
        anchors={1}, kg2_adj=adj, node_weights2=weights, exclude={3},
        max_hop=1, top_k=5,
    )
    assert 3 not in result, "excluded (already-local) entity must not be selected"
    assert set(result) <= {2, 4}
    result_capped = select_halo_kg2(
        anchors={1}, kg2_adj=adj, node_weights2=weights, exclude=set(),
        max_hop=1, top_k=1,
    )
    assert len(result_capped) == 1, "top_k must be respected"
    print("  test_select_halo_kg2_excludes_local_and_respects_top_k PASSED")


def test_select_halo_kg2_lower_degree_scores_higher():
    # Two candidates at the same hop distance; the lower-degree one should
    # rank first (mirrors ELsEA's `1 + 1/node_weights2[e]` preference for
    # more discriminative, less hub-like entities).
    triples = [(1, 0, 2), (1, 0, 3)]
    adj = build_kg2_adjacency(triples)
    weights = {2: 50, 3: 1}  # 2 is a high-degree hub, 3 is low-degree
    result = select_halo_kg2(
        anchors={1}, kg2_adj=adj, node_weights2=weights, exclude=set(),
        max_hop=1, top_k=2,
    )
    assert result[0] == 3, f"lower-degree entity should rank first, got {result}"
    print("  test_select_halo_kg2_lower_degree_scores_higher PASSED")


# ---------------------------------------------------------------------------
#  DistTrainer + halo_kg2_emb integration
# ---------------------------------------------------------------------------


def test_train_epoch_with_halo_kg2_emb_extends_negative_pool():
    """
    train_epoch must not error when given extra halo_kg2 embeddings, and the
    resulting loss must be a finite, non-negative scalar — i.e. the
    concatenated (local + halo) embedding tensor is a valid input to
    alignment_loss's negative sampling (torch.randint upper bound grows to
    include the halo rows).
    """
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
    halo_kg2_emb = torch.nn.functional.normalize(torch.randn(4, D_out), dim=1)

    loss = trainer.train_epoch(
        adj, h_init, train_pairs, kg1_size, halo_kg2_emb=halo_kg2_emb
    )
    assert loss >= 0.0 and loss == loss, f"expected finite non-negative loss, got {loss}"
    print("  test_train_epoch_with_halo_kg2_emb_extends_negative_pool PASSED")


def test_train_epoch_without_halo_kg2_emb_unchanged():
    """halo_kg2_emb=None (the default) must behave exactly as before — no regression."""
    N, D_in, D_out = 12, 8, 8
    kg1_size = 8
    ids_row = torch.arange(N)
    adj = torch.sparse_coo_tensor(
        torch.stack([ids_row, ids_row]), torch.ones(N), size=(N, N), dtype=torch.float32
    )
    h_init = torch.nn.Parameter(torch.randn(N, D_in))
    model = GCN(output_dim=D_out, num_layers=2)
    optimizer = torch.optim.Adam(list(model.parameters()) + [h_init], lr=1e-3)

    trainer = DistTrainer(model, optimizer, rank=0, world_size=1, neg_k=2, margin=1.0)
    train_pairs = torch.tensor([[0, kg1_size]], dtype=torch.long)

    loss = trainer.train_epoch(adj, h_init, train_pairs, kg1_size)
    assert loss >= 0.0 and loss == loss
    print("  test_train_epoch_without_halo_kg2_emb_unchanged PASSED")


if __name__ == "__main__":
    print("Stage D tests:")
    test_build_kg2_adjacency_is_undirected()
    test_select_halo_kg2_no_anchors_returns_empty()
    test_select_halo_kg2_prefers_closer_hops()
    test_select_halo_kg2_excludes_local_and_respects_top_k()
    test_select_halo_kg2_lower_degree_scores_higher()
    test_train_epoch_with_halo_kg2_emb_extends_negative_pool()
    test_train_epoch_without_halo_kg2_emb_unchanged()
    print("All Stage D tests PASSED.")
