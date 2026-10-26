"""
test_stage_p.py — Stage P tests: RREA mini-batch block support
(rrea_minibatch.py, RREA.forward_blocks), the RREA counterpart of the GCN
block tests in test_stage_n.py.

The single most important test here is
test_rrea_forward_blocks_matches_forward_when_unrestricted — mirrors
test_stage_n.py's own GCN parity test exactly: if RREA's block-based
propagation doesn't numerically reproduce plain RREA.forward() when given
"blocks that cover the whole graph, no sampling", the block math itself is
wrong regardless of what any real sampler does on top.

DGL-free, no GPU, no torch.distributed required (dist_graph.build_rrea_graph
is a plain function with no dist calls). <15 sec.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dist_graph import build_rrea_graph
from models.gcn import build_entity_features
from models.rrea import RREA
from rrea_minibatch import (
    RREAMinibatchBatch,
    RREAMinibatchIndex,
    build_rrea_attention_block,
    build_rrea_relation_index,
    build_rrea_self_block,
    compute_rrea_self_degrees,
    nested_active_sets,
    prepare_rrea_batch,
)
from sparse_table import SparseEntityTable

# Tiny path graph: 0 - 1 - 2 - 3  (relation id 0 throughout), same fixture
# test_stage_n.py uses for GCN's own parity test.
_PATH_TRIPLES = [(0, 0, 1), (1, 0, 2), (2, 0, 3)]
_REL_NUM = 1


class _StubData:
    """Minimal stand-in for dataload.Data: build_rrea_graph only reads
    .kg1_triples/.kg2_triples (already-global-id triples)."""

    def __init__(self, kg1_triples, kg2_triples=()):
        self.kg1_triples = kg1_triples
        self.kg2_triples = list(kg2_triples)


# ---------------------------------------------------------------------------
#  rrea_minibatch — pure structure-index functions
# ---------------------------------------------------------------------------


def test_build_rrea_relation_index_is_symmetric_with_directional_rel_ids():
    index = build_rrea_relation_index(_PATH_TRIPLES, _REL_NUM)
    # entity 1: forward edge to 2 (rel 0), reverse edge to 0 (rel 0+rel_num=1)
    nbrs_1 = {(nbr, rel) for nbr, rel, _val in index[1]}
    assert (2, 0) in nbrs_1, nbrs_1
    assert (0, 1) in nbrs_1, nbrs_1  # reverse of (0,0,1): rel_num-offset id
    print("  test_build_rrea_relation_index_is_symmetric_with_directional_rel_ids PASSED")


def test_compute_rrea_self_degrees_matches_structural_count():
    index = build_rrea_relation_index(_PATH_TRIPLES, _REL_NUM)
    deg = compute_rrea_self_degrees(index)
    # node 0: self-loop (1.0) + 1 incident relation-arc (to node 1) = 2.0
    # node 1: self-loop (1.0) + 2 incident relation-arcs (to 0 and 2) = 3.0
    assert deg[0] == 2.0
    assert deg[1] == 3.0
    print("  test_compute_rrea_self_degrees_matches_structural_count PASSED")


def test_nested_active_sets_shrinks_monotonically_ending_at_seeds():
    layers = [np.array([3]), np.array([0, 2]), np.array([1])]  # outermost..seeds
    active = nested_active_sets(layers)
    assert list(active[0]) == [0, 1, 2, 3]
    assert list(active[1]) == [0, 1, 2]
    assert list(active[2]) == [1]
    assert list(active[-1]) == list(layers[-1])
    print("  test_nested_active_sets_shrinks_monotonically_ending_at_seeds PASSED")


def test_build_rrea_attention_block_requires_inner_subset_of_outer():
    index = build_rrea_relation_index(_PATH_TRIPLES, _REL_NUM)
    # outer={0,1,2}, inner={1} (inner IS a subset of outer, as
    # nested_active_sets guarantees in real use).
    e_outer, e_inner, e_rel, e_val = build_rrea_attention_block(
        outer_ids=np.array([0, 1, 2]), inner_ids=np.array([1]), relation_index=index,
    )
    # node 1's neighbors are 0 and 2, both present in outer -> 2 edges.
    assert len(e_outer) == 2, (e_outer, e_inner, e_rel, e_val)
    print("  test_build_rrea_attention_block_requires_inner_subset_of_outer PASSED")


# ---------------------------------------------------------------------------
#  RREA.forward_blocks — the critical correctness invariant
# ---------------------------------------------------------------------------


def _build_unrestricted_batch(all_ids: np.ndarray, num_layers: int, index: RREAMinibatchIndex, device):
    """Bypasses prepare_rrea_batch's sample_multi_hop-driven path (which
    cannot produce "everyone present at every hop" when seeded with the
    WHOLE entity set -- sample_multi_hop dedupes against its own seed set,
    so starting from all_ids makes every further hop empty) and instead
    manually builds the degenerate "outer=inner=all_ids at every hop"
    batch directly -- the same bypass test_stage_n.py's own GCN parity
    test uses (`build_block(all_ids, all_ids, ...)` called directly,
    without going through sample_multi_hop)."""
    pre_block = build_rrea_self_block(all_ids, all_ids, index.relation_index, index.self_degrees, device)
    active = [all_ids] * (num_layers + 2)
    attention_edges = [
        build_rrea_attention_block(all_ids, all_ids, index.relation_index)
        for _ in range(num_layers)
    ]
    return RREAMinibatchBatch(
        outer_ids=all_ids, pre_block=pre_block, active=active,
        attention_edges=attention_edges, seed_ids=all_ids,
    )


def test_rrea_forward_blocks_matches_forward_when_unrestricted():
    """forward_blocks(), given a batch that covers the WHOLE graph at every
    hop (i.e. "one batch, no sampling"), must numerically match plain
    forward() -- if this fails, the block math itself is wrong,
    independent of anything a real sampler does on top."""
    torch.manual_seed(0)
    all_ids = np.array([0, 1, 2, 3])
    device = torch.device("cpu")
    num_layers = 2
    output_dim = 4

    model = RREA(rel_num=_REL_NUM, output_dim=output_dim, num_layers=num_layers,
                 dropout=0.0, attn_heads=1)
    model.eval()

    h_init = build_entity_features(all_ids, output_dim, device)

    global2local = {int(g): i for i, g in enumerate(all_ids)}
    data = _StubData(_PATH_TRIPLES)
    g = build_rrea_graph(data, global2local, all_ids, _REL_NUM, device)
    out_full = model(g, h_init)

    index = RREAMinibatchIndex.build(_PATH_TRIPLES, _REL_NUM)
    batch = _build_unrestricted_batch(all_ids, num_layers, index, device)
    out_blocks = model.forward_blocks(batch, index, h_init)

    assert out_full.shape == out_blocks.shape, (out_full.shape, out_blocks.shape)
    assert torch.allclose(out_full, out_blocks, atol=1e-5), (
        f"forward_blocks diverges from forward() when unrestricted:\n"
        f"forward():        {out_full}\n"
        f"forward_blocks(): {out_blocks}"
    )
    print("  test_rrea_forward_blocks_matches_forward_when_unrestricted PASSED")


def test_rrea_full_minibatch_pipeline_on_a_real_sampled_subgraph():
    """End-to-end integration on a REAL (non-degenerate) sampled sub-batch,
    smaller than the full graph: sample via prepare_rrea_batch, gather via
    SparseEntityTable, forward_blocks, backward, apply_gradients, and
    confirm entities outside the sampled neighborhood stay untouched --
    mirrors test_stage_n.py's own GCN integration test."""
    # Two disjoint 4-node paths: 0-1-2-3 and 10-11-12-13. Seeding from {1}
    # with num_layers=2 (needs num_layers+1=3 raw hops) should only ever
    # touch {0,1,2,3} -- {10,11,12,13} must remain completely untouched.
    triples = [(0, 0, 1), (1, 0, 2), (2, 0, 3), (10, 0, 11), (11, 0, 12), (12, 0, 13)]
    device = torch.device("cpu")
    num_layers = 2
    dim = 4

    table = SparseEntityTable(dim=dim, device=device, lr=0.5)
    model = RREA(rel_num=_REL_NUM, output_dim=dim, num_layers=num_layers, dropout=0.0)

    index = RREAMinibatchIndex.build(triples, _REL_NUM)
    seed_ids = np.array([1])
    rng = np.random.default_rng(0)
    batch = prepare_rrea_batch(seed_ids, index, num_layers=num_layers, fanout=None, rng=rng, device=device)

    touched = sorted(int(g) for g in batch.outer_ids)
    assert touched == [0, 1, 2, 3], f"sampling leaked outside the seed's component: {touched}"
    assert list(batch.seed_ids) == [1]

    h_init_outer = table.gather(batch.outer_ids)
    out = model.forward_blocks(batch, index, h_init_outer)
    assert out.shape == (1, model.output_width)

    loss = (out ** 2).sum()
    loss.backward()
    table.apply_gradients(batch.outer_ids, h_init_outer)

    assert len(table) == len(batch.outer_ids)
    untouched = table.gather(np.array([10, 11, 12, 13]))
    expected_untouched = build_entity_features(np.array([10, 11, 12, 13]), dim, device)
    assert torch.allclose(untouched.detach(), expected_untouched), (
        "entities outside the sampled neighborhood must stay at their "
        "deterministic initial value"
    )
    print("  test_rrea_full_minibatch_pipeline_on_a_real_sampled_subgraph PASSED")


def test_rrea_forward_blocks_output_width_matches_forward():
    """forward_blocks' output width must equal model.output_width
    (output_dim * 2 * (num_layers+1)), the same convention forward()
    documents and callers (e.g. halo_kg2_emb concatenation) rely on."""
    all_ids = np.array([0, 1, 2, 3])
    device = torch.device("cpu")
    num_layers = 2
    output_dim = 4

    model = RREA(rel_num=_REL_NUM, output_dim=output_dim, num_layers=num_layers, dropout=0.0)
    index = RREAMinibatchIndex.build(_PATH_TRIPLES, _REL_NUM)
    batch = _build_unrestricted_batch(all_ids, num_layers, index, device)
    h_init = build_entity_features(all_ids, output_dim, device)

    out = model.forward_blocks(batch, index, h_init)
    assert out.shape[1] == model.output_width == output_dim * 2 * (num_layers + 1)
    print("  test_rrea_forward_blocks_output_width_matches_forward PASSED")


if __name__ == "__main__":
    print("Stage P tests:")
    test_build_rrea_relation_index_is_symmetric_with_directional_rel_ids()
    test_compute_rrea_self_degrees_matches_structural_count()
    test_nested_active_sets_shrinks_monotonically_ending_at_seeds()
    test_build_rrea_attention_block_requires_inner_subset_of_outer()
    test_rrea_forward_blocks_matches_forward_when_unrestricted()
    test_rrea_full_minibatch_pipeline_on_a_real_sampled_subgraph()
    test_rrea_forward_blocks_output_width_matches_forward()
    print("All Stage P tests PASSED.")
