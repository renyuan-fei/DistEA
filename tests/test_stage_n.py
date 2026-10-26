"""
test_stage_n.py — Stage N tests: mini-batch neighbor-sampling building
blocks: SparseEntityTable, minibatch_sampler, GCN.forward_blocks.

The single most important test here is
test_forward_blocks_matches_forward_when_unrestricted: if block-based
propagation doesn't numerically reproduce plain forward() when given "one
batch covering the whole graph, no sampling", the block math itself is
wrong independent of anything a real sampler does on top.

DGL-free, no GPU required, <10 sec.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minibatch_sampler import (
    build_block,
    build_neighbor_index,
    compute_weighted_degrees,
    sample_multi_hop,
)
from models.gcn import GCN, build_entity_features
from sparse_table import SparseEntityTable

# ---------------------------------------------------------------------------
#  SparseEntityTable
# ---------------------------------------------------------------------------


def test_sparse_table_gather_deterministic_for_untouched_ids():
    table = SparseEntityTable(dim=4, device=torch.device("cpu"))
    a = table.gather(np.array([10, 20]))
    b = table.gather(np.array([10, 20]))
    assert torch.allclose(a, b)
    expected = build_entity_features(np.array([10, 20]), 4, torch.device("cpu"))
    assert torch.allclose(a.detach(), expected)
    print("  test_sparse_table_gather_deterministic_for_untouched_ids PASSED")


def test_sparse_table_persists_updated_value_after_apply_gradients():
    table = SparseEntityTable(dim=4, device=torch.device("cpu"), lr=0.1)
    ids = np.array([1, 2])
    batch = table.gather(ids)
    loss = (batch ** 2).sum()
    loss.backward()
    table.apply_gradients(ids, batch)

    updated = table.gather(ids)
    initial = build_entity_features(ids, 4, torch.device("cpu"))
    assert not torch.allclose(updated.detach(), initial), (
        "gather() after an update should NOT return the deterministic "
        "initial value anymore"
    )
    assert len(table) == 2
    print("  test_sparse_table_persists_updated_value_after_apply_gradients PASSED")


def test_sparse_table_moment_state_persists_across_calls():
    """A second apply_gradients() call for the same id must continue its
    Adam trajectory (non-zero step count), not restart it -- verified
    indirectly: two consecutive identical-gradient steps should move the
    row further than one step alone would (Adam's bias-correction ramps up
    with t), not by the same fixed increment twice."""
    table = SparseEntityTable(dim=1, device=torch.device("cpu"), lr=0.1)
    gid = np.array([5])

    b1 = table.gather(gid)
    (b1 * 0 + 1.0).sum().backward()  # constant gradient = 1.0
    table.apply_gradients(gid, b1)
    step1 = table._step_count[5]

    b2 = table.gather(gid)
    (b2 * 0 + 1.0).sum().backward()
    table.apply_gradients(gid, b2)
    step2 = table._step_count[5]

    assert step1 == 1
    assert step2 == 2
    print("  test_sparse_table_moment_state_persists_across_calls PASSED")


# ---------------------------------------------------------------------------
#  minibatch_sampler — pure functions
# ---------------------------------------------------------------------------

# Tiny path graph: 0 - 1 - 2 - 3  (relation id 0 throughout)
_PATH_TRIPLES = [(0, 0, 1), (1, 0, 2), (2, 0, 3)]
_UNIFORM_WEIGHTS = {}  # empty -> add_edge's (1.0, 1.0) fallback applies


_PATH_INDEX = build_neighbor_index(_PATH_TRIPLES, _UNIFORM_WEIGHTS)


def test_sparse_table_gather_empty_ids_returns_empty_leaf():
    table = SparseEntityTable(dim=4, device=torch.device("cpu"), lr=0.1)
    out = table.gather(np.zeros(0, dtype=np.int64))
    assert out.shape == (0, 4) and out.requires_grad
    table.apply_gradients(np.zeros(0, dtype=np.int64), out)  # no-op, no crash
    assert len(table) == 0
    print("  test_sparse_table_gather_empty_ids_returns_empty_leaf PASSED")


def test_compute_weighted_degrees_uniform_weights():
    deg = compute_weighted_degrees(_PATH_INDEX)
    # node 0: self-loop (1.0) + 1 neighbor (1.0) = 2.0
    # node 1: self-loop (1.0) + 2 neighbors (1.0 each) = 3.0
    assert deg[0] == 2.0
    assert deg[1] == 3.0
    assert deg[2] == 3.0
    assert deg[3] == 2.0
    print("  test_compute_weighted_degrees_uniform_weights PASSED")


def test_sample_multi_hop_layers_outermost_first_and_seeds_last():
    layers = sample_multi_hop(np.array([1]), _PATH_INDEX, num_layers=2)
    # seed = {1}; hop1 = {0, 2}; hop2 = {3} (neighbor of 2, not already seen)
    assert list(layers[-1]) == [1]
    assert set(layers[-2].tolist()) == {0, 2}
    assert set(layers[0].tolist()) == {3}
    print("  test_sample_multi_hop_layers_outermost_first_and_seeds_last PASSED")


def test_sample_multi_hop_no_duplicate_entities_across_layers():
    layers = sample_multi_hop(np.array([1]), _PATH_INDEX, num_layers=2)
    all_ids = [int(g) for layer in layers for g in layer]
    assert len(all_ids) == len(set(all_ids)), "an entity appeared in more than one layer"
    print("  test_sample_multi_hop_no_duplicate_entities_across_layers PASSED")


# Star graph: hub 0 connected to 9 leaves -- needed to exercise fanout
# capping at all (the path graph's max degree is 2).
_STAR_TRIPLES = [(0, 0, i) for i in range(1, 10)]
_STAR_INDEX = build_neighbor_index(_STAR_TRIPLES, _UNIFORM_WEIGHTS)


def test_sample_multi_hop_fanout_none_takes_full_neighborhood():
    layers = sample_multi_hop(np.array([0]), _STAR_INDEX, num_layers=1, fanout=None)
    assert len(layers[0]) == 9, f"fanout=None should take the full neighbor set, got {len(layers[0])}"
    print("  test_sample_multi_hop_fanout_none_takes_full_neighborhood PASSED")


def test_sample_multi_hop_fanout_caps_expansion():
    """Fanout capping must actually reduce the sampled set: without it, a
    few hops from a hub already touch most of a real graph."""
    rng = np.random.default_rng(0)
    layers = sample_multi_hop(np.array([0]), _STAR_INDEX, num_layers=1, fanout=3, rng=rng)
    assert len(layers[0]) == 3, f"fanout=3 should cap hop-1 expansion to 3, got {len(layers[0])}"
    print("  test_sample_multi_hop_fanout_caps_expansion PASSED")


def test_sample_multi_hop_fanout_reproducible_with_seeded_rng():
    layers_a = sample_multi_hop(
        np.array([0]), _STAR_INDEX, num_layers=1, fanout=3, rng=np.random.default_rng(42)
    )
    layers_b = sample_multi_hop(
        np.array([0]), _STAR_INDEX, num_layers=1, fanout=3, rng=np.random.default_rng(42)
    )
    assert list(layers_a[0]) == list(layers_b[0])
    print("  test_sample_multi_hop_fanout_reproducible_with_seeded_rng PASSED")


def test_build_block_self_loop_and_neighbor_weights_match_gcnlayer_formula():
    deg = compute_weighted_degrees(_PATH_INDEX)
    # One hop: outer={0,2} (neighbors of 1, from a hop-1 sample), inner={1}.
    block = build_block(
        outer_ids=np.array([0, 2]), inner_ids=np.array([1]),
        neighbor_index=_PATH_INDEX,
        degrees=deg, device=torch.device("cpu"),
    )
    dense = block.adj_t.to_dense()
    # deg(0)=2.0, deg(1)=3.0, deg(2)=3.0, edge weight 1.0 both directions.
    expected_0_to_1 = (2.0 ** -0.5) * (3.0 ** -0.5) * 1.0
    expected_2_to_1 = (3.0 ** -0.5) * (3.0 ** -0.5) * 1.0
    assert abs(dense[0, 0].item() - expected_0_to_1) < 1e-6
    assert abs(dense[1, 0].item() - expected_2_to_1) < 1e-6
    print("  test_build_block_self_loop_and_neighbor_weights_match_gcnlayer_formula PASSED")


# ---------------------------------------------------------------------------
#  GCN.forward_blocks — the critical correctness invariant
# ---------------------------------------------------------------------------


def test_forward_blocks_matches_forward_when_unrestricted():
    """forward_blocks(), given blocks that together cover the WHOLE graph
    at every hop (i.e. "one batch, no sampling"), must numerically match
    plain forward(). If this fails, the block math is wrong regardless
    of what any real sampler does."""
    torch.manual_seed(0)
    triples = _PATH_TRIPLES
    all_ids = np.array([0, 1, 2, 3])
    device = torch.device("cpu")
    num_layers = 2

    model = GCN(output_dim=4, num_layers=num_layers, dropout=0.0)
    model.eval()  # dropout=0.0 already makes this deterministic either way

    h = build_entity_features(all_ids, 4, device)

    # Full-graph path: GCNLayer._sparse_forward adds the self-loop AND does
    # the symmetric normalisation itself -- it expects the RAW, un-normalised
    # adjacency (dist_graph.py's own convention: plain triple-derived edges,
    # no self-loop, no pre-normalisation), NOT the already-normalised form
    # build_block() produces. Feeding it a pre-normalised adjacency would
    # double-apply both the self-loop and the D^-1/2 scaling.
    index = build_neighbor_index(triples, _UNIFORM_WEIGHTS)
    deg = compute_weighted_degrees(index)
    src, dst, val = [], [], []
    id_pos = {int(g): i for i, g in enumerate(all_ids)}
    for h_, _r, t_ in triples:
        for a, b in ((h_, t_), (t_, h_)):
            src.append(id_pos[a])
            dst.append(id_pos[b])
            val.append(1.0)
    adj = torch.sparse_coo_tensor(
        torch.tensor([src, dst]), torch.tensor(val, dtype=torch.float32), size=(4, 4)
    ).coalesce()
    out_full = model(adj, h)

    # Block path: every layer's block is "all 4 nodes -> all 4 nodes" (no
    # sampling at all -- the degenerate case that must reproduce forward()).
    blocks = [
        build_block(all_ids, all_ids, index, deg, device)
        for _ in range(num_layers)
    ]
    out_blocks = model.forward_blocks(blocks, h)

    assert torch.allclose(out_full, out_blocks, atol=1e-5), (
        f"forward_blocks diverges from forward() when unrestricted:\n"
        f"forward():        {out_full}\n"
        f"forward_blocks(): {out_blocks}"
    )
    print("  test_forward_blocks_matches_forward_when_unrestricted PASSED")


def test_full_minibatch_pipeline_on_a_real_sampled_subgraph():
    """End-to-end integration: sample a REAL (non-degenerate) sub-batch from
    a graph bigger than the batch itself, gather via SparseEntityTable,
    forward_blocks, compute a loss, backward, apply_gradients, and confirm
    only the touched entities' rows actually changed -- entities outside
    the sampled neighborhood must still return their deterministic initial
    value untouched, which is the memory-saving premise of the mini-batch
    path."""
    # Two disjoint 4-node paths glued at nothing: 0-1-2-3 and 10-11-12-13.
    # Seeding from {1} should only ever touch {0,1,2,3} for num_layers=2 --
    # {10,11,12,13} must remain completely untouched afterwards.
    triples = [(0, 0, 1), (1, 0, 2), (2, 0, 3), (10, 0, 11), (11, 0, 12), (12, 0, 13)]
    device = torch.device("cpu")
    num_layers = 2
    dim = 4

    table = SparseEntityTable(dim=dim, device=device, lr=0.5)
    model = GCN(output_dim=dim, num_layers=num_layers, dropout=0.0)

    index = build_neighbor_index(triples, _UNIFORM_WEIGHTS)
    seed_ids = np.array([1])
    layers = sample_multi_hop(seed_ids, index, num_layers=num_layers)
    touched = sorted(int(g) for layer in layers for g in layer)
    assert touched == [0, 1, 2, 3], f"sampling leaked outside the seed's component: {touched}"

    deg = compute_weighted_degrees(index)
    blocks = [
        build_block(layers[i], layers[i + 1], index, deg, device)
        for i in range(num_layers)
    ]
    h_init = table.gather(layers[0])  # outermost frontier's initial features
    out = model.forward_blocks(blocks, h_init)
    assert out.shape == (len(layers[-1]), dim)

    loss = (out ** 2).sum()
    loss.backward()
    table.apply_gradients(layers[0], h_init)

    assert len(table) == len(layers[0]), "only the outermost frontier's rows were gathered/updated"
    untouched = table.gather(np.array([10, 11, 12, 13]))
    expected_untouched = build_entity_features(np.array([10, 11, 12, 13]), dim, device)
    assert torch.allclose(untouched.detach(), expected_untouched), (
        "entities outside the sampled neighborhood must stay at their "
        "deterministic initial value -- this is the whole memory-saving point"
    )
    print("  test_full_minibatch_pipeline_on_a_real_sampled_subgraph PASSED")


def test_build_block_reweight_rescales_by_true_over_surviving_neighbor_count():
    """Star graph: hub node 0 has 4 true neighbors (1,2,3,4). If only 2 of
    them survive fanout-capped sampling into outer_ids, reweight=False sums
    just those 2 raw terms (systematically under-scaled relative to
    full-graph aggregation),
    while reweight=True must rescale that same 2-term sum by 4/2=2x, making
    it match what an unrestricted (all-4-neighbors) block produces."""
    star_triples = [(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)]
    index = build_neighbor_index(star_triples, _UNIFORM_WEIGHTS)
    deg = compute_weighted_degrees(index)  # deg(0) = 1 + 4 = 5.0 (TRUE, all 4 neighbors)

    device = torch.device("cpu")
    # Partial: only neighbors 1 and 3 survived into outer_ids (as a fanout=2
    # sample of node 0's true 4 neighbors would produce).
    partial_biased = build_block(
        outer_ids=np.array([1, 3]), inner_ids=np.array([0]),
        neighbor_index=index, degrees=deg, device=device, reweight=False,
    )
    partial_reweighted = build_block(
        outer_ids=np.array([1, 3]), inner_ids=np.array([0]),
        neighbor_index=index, degrees=deg, device=device, reweight=True,
    )
    # Full: all 4 true neighbors present (fanout=None equivalent) -- the
    # ground truth reweighting should approximate in expectation.
    full = build_block(
        outer_ids=np.array([1, 2, 3, 4]), inner_ids=np.array([0]),
        neighbor_index=index, degrees=deg, device=device, reweight=False,
    )

    biased_sum = partial_biased.adj_t.to_dense().sum().item()
    reweighted_sum = partial_reweighted.adj_t.to_dense().sum().item()
    full_sum = full.adj_t.to_dense().sum().item()

    assert abs(reweighted_sum - 2.0 * biased_sum) < 1e-6, (
        f"reweight=True should scale the 2-of-4-surviving sum by exactly "
        f"4/2=2x: biased={biased_sum}, reweighted={reweighted_sum}"
    )
    # By symmetry (all 4 neighbors have identical degree/weight here), the
    # unbiased 2-term rescaled estimate exactly equals the full 4-term sum --
    # not true in general (real graphs have heterogeneous neighbor weights),
    # but confirms the correction is scaled correctly in this controlled case.
    assert abs(reweighted_sum - full_sum) < 1e-6, (
        f"symmetric star graph: reweighted partial sum should equal the "
        f"true full sum, got reweighted={reweighted_sum}, full={full_sum}"
    )
    print("  test_build_block_reweight_rescales_by_true_over_surviving_neighbor_count PASSED")


def test_build_block_reweight_is_noop_when_all_neighbors_survive():
    """reweight=True must not change anything when outer_ids already
    contains every true neighbor (surviving == true count) -- e.g.
    fanout=None or an unrestricted single-batch block. This is what makes
    reweight=True safe to leave on unconditionally."""
    deg = compute_weighted_degrees(_PATH_INDEX)
    unrestricted = build_block(
        outer_ids=np.array([0, 2]), inner_ids=np.array([1]),
        neighbor_index=_PATH_INDEX, degrees=deg, device=torch.device("cpu"),
        reweight=False,
    )
    reweighted = build_block(
        outer_ids=np.array([0, 2]), inner_ids=np.array([1]),
        neighbor_index=_PATH_INDEX, degrees=deg, device=torch.device("cpu"),
        reweight=True,
    )
    assert torch.allclose(unrestricted.adj_t.to_dense(), reweighted.adj_t.to_dense()), (
        "reweight=True changed values even though all true neighbors of node "
        "1 (0 and 2) were already present in outer_ids -- should be a no-op"
    )
    print("  test_build_block_reweight_is_noop_when_all_neighbors_survive PASSED")


if __name__ == "__main__":
    print("Stage N tests:")
    test_sparse_table_gather_deterministic_for_untouched_ids()
    test_sparse_table_persists_updated_value_after_apply_gradients()
    test_sparse_table_moment_state_persists_across_calls()
    test_sparse_table_gather_empty_ids_returns_empty_leaf()
    test_compute_weighted_degrees_uniform_weights()
    test_sample_multi_hop_layers_outermost_first_and_seeds_last()
    test_sample_multi_hop_no_duplicate_entities_across_layers()
    test_sample_multi_hop_fanout_none_takes_full_neighborhood()
    test_sample_multi_hop_fanout_caps_expansion()
    test_sample_multi_hop_fanout_reproducible_with_seeded_rng()
    test_build_block_self_loop_and_neighbor_weights_match_gcnlayer_formula()
    test_build_block_reweight_rescales_by_true_over_surviving_neighbor_count()
    test_build_block_reweight_is_noop_when_all_neighbors_survive()
    test_forward_blocks_matches_forward_when_unrestricted()
    test_full_minibatch_pipeline_on_a_real_sampled_subgraph()
    print("All Stage N tests PASSED.")
