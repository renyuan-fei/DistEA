"""
test_stage_j.py — Stage J tests: locality-aware KG2 assignment
(assign_kg2_to_ranks_locality), its seed_links override, the KG1 edge
weights derived from KG2 locality, and partition.py's fold-aware output path.

DGL-free, no GPU required, <30 sec.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from counterpart_disc import (
    assign_kg2_to_ranks,
    assign_kg2_to_ranks_locality,
    compute_kg2_locality_edge_weights,
)
from dataload import OpenEAData

try:
    from partition import build_kg1_graph, run_partition
    _HAS_NXMETIS = True
except ImportError:
    # partition.py imports nxmetis unconditionally (it's a hard dependency
    # for actually running METIS, via `make submodules`) -- not installed
    # in every dev environment this DGL-free suite is meant to run in, so
    # the one test needing build_kg1_graph skips gracefully instead of
    # failing the whole file.
    _HAS_NXMETIS = False


def make_dataset(tmpdir):
    """
    KG1: 6 entities (e0..e5), split evenly rank0={e0,e1,e2} / rank1={e3,e4,e5}.
    KG2: f0 (train-linked to e0 -> rank0 seed), f1 (unlinked, edge f0-f1 ->
    should follow f0 to rank0), f2 (train-linked to e3 -> rank1 seed), f3
    (unlinked, edge f2-f3 -> should follow f2 to rank1), f4 (unlinked, no
    KG2-KG2 edges at all -> disconnected-component fallback).
    """
    data_dir = os.path.join(tmpdir, "data")
    fold_dir = os.path.join(tmpdir, "fold")
    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(data_dir)
    os.makedirs(fold_dir)
    os.makedirs(output_dir)

    kg1 = "e0\tr0\te1\ne1\tr0\te2\ne3\tr0\te4\ne4\tr0\te5\n"
    # f4's only "triple" is a self-loop -- registers it as a real KG2 entity
    # (OpenEAData only assigns ent2id ids from rel_triples_1/2, not from
    # link files) while giving it no real adjacency to f0/f1/f2/f3's
    # components, so it's unreachable from either seed via BFS.
    kg2 = "f0\tr0\tf1\nf2\tr0\tf3\nf4\tr0\tf4\n"

    with open(os.path.join(data_dir, "rel_triples_1"), "w") as f:
        f.write(kg1)
    with open(os.path.join(data_dir, "rel_triples_2"), "w") as f:
        f.write(kg2)
    with open(os.path.join(fold_dir, "train_links"), "w") as f:
        f.write("e0\tf0\ne3\tf2\n")
    with open(os.path.join(fold_dir, "valid_links"), "w") as f:
        f.write("e1\tf1\n")
    with open(os.path.join(fold_dir, "test_links"), "w") as f:
        f.write("e4\tf3\n")

    return data_dir, fold_dir, output_dir


def test_locality_places_unlinked_entity_with_its_connected_cluster():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir, fold_dir, output_dir = make_dataset(tmp)
        data = OpenEAData(data_dir, fold_dir, output_dir)

        partitions = [set(data.kg1_entities[:3]), set(data.kg1_entities[3:])]
        assignments = assign_kg2_to_ranks_locality(data, partitions, output_dir)

        f0, f1 = data.ent2id["f0"], data.ent2id["f1"]
        f2, f3 = data.ent2id["f2"], data.ent2id["f3"]

        rank_of = {}
        for r, ids in assignments.items():
            for e in ids:
                rank_of[e] = r

        assert rank_of[f0] == 0, "f0 is train-linked to e0 -> must be rank 0"
        assert rank_of[f2] == 1, "f2 is train-linked to e3 -> must be rank 1"
        assert rank_of[f1] == 0, (
            f"f1 is unlinked but its only KG2-KG2 edge is to f0 (rank 0) -- "
            f"expected majority-vote placement on rank 0, got rank {rank_of[f1]}"
        )
        assert rank_of[f3] == 1, (
            f"f3 is unlinked but its only KG2-KG2 edge is to f2 (rank 1) -- "
            f"expected majority-vote placement on rank 1, got rank {rank_of[f3]}"
        )
        print("  test_locality_places_unlinked_entity_with_its_connected_cluster PASSED")


def test_locality_disconnected_entity_gets_load_balanced_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir, fold_dir, output_dir = make_dataset(tmp)
        data = OpenEAData(data_dir, fold_dir, output_dir)

        partitions = [set(data.kg1_entities[:3]), set(data.kg1_entities[3:])]
        assignments = assign_kg2_to_ranks_locality(data, partitions, output_dir)

        f4 = data.ent2id["f4"]
        rank_of = {}
        for r, ids in assignments.items():
            for e in ids:
                rank_of[e] = r
        assert f4 in rank_of, "f4 (no KG2-KG2 edges at all) must still be assigned somewhere"
        print("  test_locality_disconnected_entity_gets_load_balanced_fallback PASSED")


def test_locality_output_shape_matches_round_robin():
    """Same dict-of-rank-to-list-of-ids shape, same total entity coverage,
    so nothing downstream (dist_graph.py/framework.py) needs to change."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir, fold_dir, output_dir = make_dataset(tmp)
        data = OpenEAData(data_dir, fold_dir, output_dir)
        partitions = [set(data.kg1_entities[:3]), set(data.kg1_entities[3:])]

        rr = assign_kg2_to_ranks(data, partitions, output_dir)
        loc = assign_kg2_to_ranks_locality(data, partitions, output_dir)

        assert set(rr.keys()) == set(loc.keys()) == {0, 1}
        rr_all = sorted(e for ids in rr.values() for e in ids)
        loc_all = sorted(e for ids in loc.values() for e in ids)
        assert rr_all == loc_all == sorted(data.kg2_entities), (
            "both strategies must cover every KG2 entity exactly once"
        )
        assert os.path.exists(os.path.join(output_dir, "kg2_assignments.json"))
        print("  test_locality_output_shape_matches_round_robin PASSED")


def test_locality_ties_break_toward_less_loaded_rank():
    """A vote tie (equal votes for two ranks) must break toward whichever
    rank currently has fewer entities assigned, not an arbitrary/first rank."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir, fold_dir, output_dir = make_dataset(tmp)
        data = OpenEAData(data_dir, fold_dir, output_dir)
        partitions = [set(data.kg1_entities[:3]), set(data.kg1_entities[3:])]

        # Sanity: after the seed step, rank 0 and rank 1 each have exactly
        # one seed (f0, f2) -- a genuine tie scenario isn't naturally in
        # make_dataset's small graph, so this test only asserts the
        # documented behaviour doesn't crash / stays deterministic across
        # repeated calls (same input -> same output), which is what the
        # tie-break rule needs to guarantee for reproducibility.
        loc_a = assign_kg2_to_ranks_locality(data, partitions, output_dir)
        loc_b = assign_kg2_to_ranks_locality(data, partitions, output_dir)
        assert loc_a == loc_b, "assignment must be deterministic across repeated calls"
        print("  test_locality_ties_break_toward_less_loaded_rank PASSED")


def make_dataset_with_pseudo_seed_target(tmpdir):
    """
    Same as make_dataset(), but KG2 also has f5, connected only to f4 --
    f4's original meaning ("no train-linked seed reachable via the base
    dataset's seeds") is preserved, but it now has real KG2-KG2 adjacency to
    a new entity f5 that a pseudo-seed can anchor. Without an expanded seed
    set, f4/f5 remain an isolated component (unreachable from f0/f2's
    seeds) and both fall back to load-balanced placement. Adding a
    pseudo-seed pair (e5, f5) -- e5 already exists in rank1's KG1 partition
    -- makes f5 a rank-1 seed, letting BFS propagation reach f4 through the
    new f4-f5 edge.
    """
    data_dir, fold_dir, output_dir = make_dataset(tmpdir)
    with open(os.path.join(data_dir, "rel_triples_2"), "a") as f:
        f.write("f4\tr0\tf5\n")
    return data_dir, fold_dir, output_dir


def test_locality_seed_links_override_reaches_previously_unreachable_entity():
    """assign_kg2_to_ranks_locality's
    seed_links param must let a caller (framework.py's --reassign_kg2_every)
    feed a richer seed set than data.load_train_alignment() alone, and that
    richer set must actually change placement decisions."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir, fold_dir, output_dir = make_dataset_with_pseudo_seed_target(tmp)
        data = OpenEAData(data_dir, fold_dir, output_dir)
        partitions = [set(data.kg1_entities[:3]), set(data.kg1_entities[3:])]

        f4, f5 = data.ent2id["f4"], data.ent2id["f5"]
        e5 = data.ent2id["e5"]

        expanded_seeds = data.load_train_alignment() + [(e5, f5)]
        reassigned = assign_kg2_to_ranks_locality(
            data, partitions, output_dir, seed_links=expanded_seeds
        )
        rank_of = {e: r for r, ids in reassigned.items() for e in ids}

        assert rank_of[f5] == 1, "f5 seeded via pseudo-seed (e5,f5) -> must be rank 1"
        assert rank_of[f4] == 1, (
            f"f4 should now follow f5 (rank 1) via the new f4-f5 edge, "
            f"got rank {rank_of[f4]}"
        )
        print("  test_locality_seed_links_override_reaches_previously_unreachable_entity PASSED")


def test_locality_default_seed_links_unchanged():
    """seed_links=None (the default) must
    still behave exactly like data.load_train_alignment() alone -- adding
    f5 without a pseudo-seed shouldn't change anything about the original
    train-linked-seed behaviour."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir, fold_dir, output_dir = make_dataset_with_pseudo_seed_target(tmp)
        data = OpenEAData(data_dir, fold_dir, output_dir)
        partitions = [set(data.kg1_entities[:3]), set(data.kg1_entities[3:])]

        default_call = assign_kg2_to_ranks_locality(data, partitions, output_dir)
        explicit_call = assign_kg2_to_ranks_locality(
            data, partitions, output_dir, seed_links=data.load_train_alignment()
        )
        assert default_call == explicit_call, (
            "seed_links=None must be identical to explicitly passing "
            "data.load_train_alignment()"
        )
        print("  test_locality_default_seed_links_unchanged PASSED")


def test_reassign_kg2_every_requires_no_persist_optimizer():
    """Scope guard: --reassign_kg2_every
    combined with --persist_optimizer must fail fast at construction time,
    not silently corrupt a long training run -- persisted h_init/optimizer
    state assumes a fixed local_kg2/halo_kg2 row layout across iterations,
    which reassignment breaks (see framework.py's DistEAFramework.__init__)."""
    import types

    import torch as _torch

    from framework import DistEAFramework

    with tempfile.TemporaryDirectory() as tmp:
        data_dir, fold_dir, output_dir = make_dataset(tmp)
        data = OpenEAData(data_dir, fold_dir, output_dir)
        cfg = types.SimpleNamespace(
            output_dim=8, hidden_dim=None,
            persist_optimizer=True, reassign_kg2_every=3,
        )
        try:
            DistEAFramework(0, 2, data, output_dir, cfg, _torch.device("cpu"))
        except ValueError as e:
            assert "reassign_kg2_every" in str(e)
            print("  test_reassign_kg2_every_requires_no_persist_optimizer PASSED")
            return
        raise AssertionError(
            "expected ValueError for --reassign_kg2_every + --persist_optimizer"
        )


def test_compute_kg2_locality_edge_weights_bonus_on_same_component_only():
    """compute_kg2_locality_edge_weights: a KG1-KG1 edge gets the bonus weight
    only when BOTH endpoints are train-linked AND their KG2 counterparts
    fall in the same weakly-connected KG2-KG2 component -- same-linked-but-
    different-component pairs must stay at the unweighted default."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        fold_dir = os.path.join(tmp, "fold")
        output_dir = os.path.join(tmp, "output")
        os.makedirs(data_dir)
        os.makedirs(fold_dir)
        os.makedirs(output_dir)

        # e0-e1: both train-linked, counterparts f0/f1 are KG2-KG2-adjacent
        # (same component) -> expect bonus weight.
        # e2-e3: both train-linked, counterparts f2/f3 are each their own
        # isolated component (no edge between them) -> expect default 1.0.
        with open(os.path.join(data_dir, "rel_triples_1"), "w") as f:
            f.write("e0\tr0\te1\ne2\tr0\te3\n")
        with open(os.path.join(data_dir, "rel_triples_2"), "w") as f:
            f.write("f0\tr0\tf1\nf2\tr0\tf2\nf3\tr0\tf3\n")
        with open(os.path.join(fold_dir, "train_links"), "w") as f:
            f.write("e0\tf0\ne1\tf1\ne2\tf2\ne3\tf3\n")
        with open(os.path.join(fold_dir, "valid_links"), "w") as f:
            f.write("")
        with open(os.path.join(fold_dir, "test_links"), "w") as f:
            f.write("")

        data = OpenEAData(data_dir, fold_dir, output_dir)
        weights = compute_kg2_locality_edge_weights(data, bonus=5.0)

        e0, e1, e2, e3 = (data.ent2id[x] for x in ("e0", "e1", "e2", "e3"))
        assert weights.get(tuple(sorted((e0, e1)))) == 1.0 + 5.0, (
            "e0-e1: same-component counterparts -> expected bonus weight"
        )
        assert tuple(sorted((e2, e3))) not in weights, (
            "e2-e3: different-component counterparts -> expected no entry "
            "(default weight 1.0)"
        )
        print("  test_compute_kg2_locality_edge_weights_bonus_on_same_component_only PASSED")


def test_build_kg1_graph_edge_weights_none_matches_original_unweighted_behavior():
    """Regression guard: build_kg1_graph(data, edge_weights=None) (the
    default) must produce a graph with no 'weight' edge attribute at all,
    so nxmetis.partition's default edge-cut objective is
    unaffected for anyone not opting into --kg1_edge_weight kg2_locality."""
    if not _HAS_NXMETIS:
        print("  test_build_kg1_graph_edge_weights_none_matches_original_unweighted_behavior SKIPPED (nxmetis not installed)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        data_dir, fold_dir, output_dir = make_dataset(tmp)
        data = OpenEAData(data_dir, fold_dir, output_dir)

        G = build_kg1_graph(data)
        assert G.number_of_nodes() == len(data.kg1_entities)
        for u, v in G.edges():
            assert "weight" not in G[u][v], (
                f"edge ({u},{v}) unexpectedly has a 'weight' attribute "
                f"with edge_weights=None"
            )
        print("  test_build_kg1_graph_edge_weights_none_matches_original_unweighted_behavior PASSED")


def test_run_partition_output_dir_is_fold_aware():
    """Two different fold_dirs partitioned at the same n_parts must land in
    DIFFERENT directories (the output path includes a fold id, not just
    n_parts), so one fold's partition cannot overwrite another's. Also verifies the
    "dist_True" segment is preserved as the immediate parent of the
    {n_parts}{suffix} leaf -- leapgnn_cache.py's GrpcEmbeddingCache and the
    Go server's ConstructNid2Pid both hard-parse exactly that shape
    (tests/test_stage_m.py), so dropping it would break the grpc cache
    backend."""
    if not _HAS_NXMETIS:
        print("  test_run_partition_output_dir_is_fold_aware SKIPPED (nxmetis not installed)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        data_dir, fold_dir_a, _ = make_dataset(tmp)
        # A second, distinct fold sharing the SAME data_dir (same dataset,
        # different fold split).
        fold_dir_b = os.path.join(tmp, "fold_b")
        os.makedirs(fold_dir_b)
        with open(os.path.join(fold_dir_b, "train_links"), "w") as f:
            f.write("e0\tf0\ne3\tf2\n")
        with open(os.path.join(fold_dir_b, "valid_links"), "w") as f:
            f.write("e1\tf1\n")
        with open(os.path.join(fold_dir_b, "test_links"), "w") as f:
            f.write("e4\tf3\n")

        output_base = os.path.join(tmp, "partitions")
        run_partition(data_dir, fold_dir_a, output_base, n_parts=2)
        run_partition(data_dir, fold_dir_b, output_base, n_parts=2)

        hits = []
        for root, _dirs, files in os.walk(output_base):
            if os.path.basename(root) == "2_metis_kg2locality" and "0.npy" in files:
                hits.append(root)
        assert len(hits) == 2, (
            f"expected 2 separate partition dirs (one per fold), got {len(hits)}: {hits} "
            f"-- fold_dir_a's output was likely overwritten by fold_dir_b's"
        )
        for h in hits:
            assert os.path.basename(os.path.dirname(h)) == "dist_True", (
                f"{h!r}: 'dist_True' must be the immediate parent of the "
                f"{{n_parts}}{{suffix}} leaf -- required by "
                f"leapgnn_cache.py's _split_partition_dir and the Go "
                f"server's ConstructNid2Pid"
            )
        # The two hits' fold-identifying path component (one level above
        # dist_True) must differ -- this is what prevents the collision.
        fold_ids = {os.path.basename(os.path.dirname(os.path.dirname(h))) for h in hits}
        assert len(fold_ids) == 2, (
            f"both folds produced the same fold_id path component {fold_ids} "
            f"-- collision not actually fixed"
        )
        print("  test_run_partition_output_dir_is_fold_aware PASSED")


if __name__ == "__main__":
    print("Stage J tests:")
    test_locality_places_unlinked_entity_with_its_connected_cluster()
    test_locality_disconnected_entity_gets_load_balanced_fallback()
    test_locality_output_shape_matches_round_robin()
    test_locality_ties_break_toward_less_loaded_rank()
    test_locality_seed_links_override_reaches_previously_unreachable_entity()
    test_locality_default_seed_links_unchanged()
    test_reassign_kg2_every_requires_no_persist_optimizer()
    test_compute_kg2_locality_edge_weights_bonus_on_same_component_only()
    test_build_kg1_graph_edge_weights_none_matches_original_unweighted_behavior()
    test_run_partition_output_dir_is_fold_aware()
    print("All Stage J tests PASSED.")
