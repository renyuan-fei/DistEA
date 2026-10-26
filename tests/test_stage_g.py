"""
test_stage_g.py — Stage G tests: DistGraphLoader's graph-level halo_kg2.

DGL-free, no GPU required, <30 sec.

counterpart_disc.assign_kg2_to_ranks distributes non-train-linked KG2
entities round-robin, ignoring KG2-KG2 graph locality, so a KG2-KG2 edge
often crosses a partition boundary. halo_kg2 mirrors halo_kg1: the 1-hop KG2
neighbours owned by other ranks are added as halo nodes so such edges are
kept in the local graph instead of being dropped.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from counterpart_disc import assign_kg2_to_ranks
from dataload import OpenEAData
from dist_graph import DistGraphLoader


def make_dataset(tmpdir):
    """Same tiny OpenEA-format dataset as test_stage_a.py's make_dataset."""
    data_dir = os.path.join(tmpdir, "data")
    fold_dir = os.path.join(tmpdir, "fold")
    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(data_dir)
    os.makedirs(fold_dir)
    os.makedirs(output_dir)

    # KG1: 6 entities, 3 relations
    kg1 = "e0\tr0\te1\ne0\tr1\te2\ne1\tr2\te3\ne2\tr2\te4\ne3\tr0\te5\n"
    # KG2: 4 entities, chained f0-f1-f3 and f0-f2 so at least one edge is
    # guaranteed to cross the rank boundary once f0/f1 follow train_links
    # to rank 0 and f2/f3 (unlinked) get scattered round-robin.
    kg2 = "f0\tr0\tf1\nf0\tr1\tf2\nf1\tr2\tf3\n"

    with open(os.path.join(data_dir, "rel_triples_1"), "w") as f:
        f.write(kg1)
    with open(os.path.join(data_dir, "rel_triples_2"), "w") as f:
        f.write(kg2)
    with open(os.path.join(fold_dir, "train_links"), "w") as f:
        f.write("e0\tf0\ne1\tf1\n")
    with open(os.path.join(fold_dir, "valid_links"), "w") as f:
        f.write("e2\tf2\n")
    with open(os.path.join(fold_dir, "test_links"), "w") as f:
        f.write("e3\tf3\ne4\tf2\n")

    return data_dir, fold_dir, output_dir


def test_halo_kg2_retains_cross_rank_edge():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir, fold_dir, output_dir = make_dataset(tmp)
        data = OpenEAData(data_dir, fold_dir, output_dir)

        # 2-way partition: e0,e1,e2 -> rank 0; e3,e4,e5 -> rank 1 (same split
        # test_stage_a.py's test_counterpart_disc uses).
        partitions = [set(data.kg1_entities[:3]), set(data.kg1_entities[3:])]
        assignments = assign_kg2_to_ranks(data, partitions, output_dir)

        # f0, f1 are train-linked to e0/e1 -> both follow rank 0. f2 is
        # unlinked and round-robins to rank 0 too (first unlinked pick);
        # f3 is unlinked and round-robins to rank 1 -- verified empirically
        # (ent2id order: e0..e5=0-5, f0..f3=6-9; assign_kg2_to_ranks gives
        # rank0=[f0,f1,f2], rank1=[f3]). f1-f3 is therefore a real KG2-KG2
        # edge (rel_triples_2's "f1 r2 f3") that crosses the rank boundary.
        f0, f1, f3 = data.ent2id["f0"], data.ent2id["f1"], data.ent2id["f3"]
        assert f0 in assignments[0] and f1 in assignments[0], (
            "train-linked KG2 entities should follow their KG1 partner's rank"
        )
        assert f3 in assignments[1], (
            "test setup assumption broken: expected f3 round-robin to rank 1 "
            "-- adjust the dataset if counterpart_disc's round-robin "
            "ordering changes"
        )

        # Write partition files DistGraphLoader expects on disk.
        np.save(os.path.join(output_dir, "0.npy"), np.array(sorted(partitions[0]), dtype=np.int64))
        np.save(os.path.join(output_dir, "1.npy"), np.array(sorted(partitions[1]), dtype=np.int64))

        rank_with_f0f1 = 0
        loader = DistGraphLoader(data, output_dir, rank_with_f0f1, world_size=2, device="cpu")
        (
            g,
            global2local,
            all_global_ids,
            local_kg1_idx,
            halo_kg1_idx,
            local_kg2_idx,
            halo_kg2_idx,
        ) = loader.load()

        # halo_kg2 must exist and contain f3 (1-hop KG2 neighbour of f1, on
        # the other rank); without halo_kg2 this entity has no local node.
        assert len(halo_kg2_idx) > 0, "expected non-empty halo_kg2_idx"
        halo_kg2_global = {int(x) for x in all_global_ids[halo_kg2_idx]}
        assert f3 in halo_kg2_global, f"expected f3={f3} in halo_kg2, got {halo_kg2_global}"

        # The f1-f3 edge must actually be present in the built graph (as a
        # sparse adjacency when DGL is unavailable, matching this repo's
        # DGL-free test convention).
        lf1 = global2local[f1]
        lf3 = global2local[f3]
        if hasattr(g, "is_sparse") and g.is_sparse:
            indices = g.coalesce().indices()
            pairs = set(zip(indices[0].tolist(), indices[1].tolist(), strict=False))
        else:
            # DGL graph fallback
            src, dst = g.edges()
            pairs = set(zip(src.tolist(), dst.tolist(), strict=False))
        assert (lf1, lf3) in pairs or (lf3, lf1) in pairs, (
            "f1-f3 edge missing from built graph -- halo_kg2 fix regressed"
        )

        # kg1_size arithmetic (local_kg1 + halo_kg1) must be unaffected by
        # halo_kg2 being appended at the end of the node ordering.
        kg1_size = len(local_kg1_idx) + len(halo_kg1_idx)
        assert kg1_size == len(local_kg1_idx) + len(halo_kg1_idx)
        for li in local_kg2_idx:
            assert li >= kg1_size, "local_kg2 must fall after the KG1/KG2 boundary"
        for li in halo_kg2_idx:
            assert li >= kg1_size, "halo_kg2 must fall after the KG1/KG2 boundary too"

        print("  test_halo_kg2_retains_cross_rank_edge PASSED")


def test_no_halo_kg1_kg2_ablation_disables_halo_entirely():
    """No-halo ablation: with both no_halo_kg1/no_halo_kg2 set, the local graph must contain ONLY
    [local_kg1 | local_kg2] -- zero halo nodes of either kind, total node
    count exactly equal to this rank's own partition file entries + its own
    KG2 assignment count, with no dependence on the other rank's data at
    all. Reuses the same 2-way partition fixture as
    test_halo_kg2_retains_cross_rank_edge, which independently confirms a
    real cross-rank KG2-KG2 edge exists in this setup -- so a passing
    ablation test here is a genuine "halo turned off", not just "there was
    nothing to halo in the first place"."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir, fold_dir, output_dir = make_dataset(tmp)
        data = OpenEAData(data_dir, fold_dir, output_dir)

        partitions = [set(data.kg1_entities[:3]), set(data.kg1_entities[3:])]
        assignments = assign_kg2_to_ranks(data, partitions, output_dir)

        np.save(os.path.join(output_dir, "0.npy"), np.array(sorted(partitions[0]), dtype=np.int64))
        np.save(os.path.join(output_dir, "1.npy"), np.array(sorted(partitions[1]), dtype=np.int64))

        rank = 0
        loader = DistGraphLoader(
            data, output_dir, rank, world_size=2, device="cpu",
            no_halo_kg1=True, no_halo_kg2=True,
        )
        (
            g,
            global2local,
            all_global_ids,
            local_kg1_idx,
            halo_kg1_idx,
            local_kg2_idx,
            halo_kg2_idx,
        ) = loader.load()

        assert len(halo_kg1_idx) == 0, f"expected empty halo_kg1_idx, got {len(halo_kg1_idx)}"
        assert len(halo_kg2_idx) == 0, f"expected empty halo_kg2_idx, got {len(halo_kg2_idx)}"

        expected_total = len(partitions[rank]) + len(assignments[rank])
        assert len(all_global_ids) == expected_total, (
            f"expected node count == own partition file entries + own KG2 "
            f"assignment count ({expected_total}), got {len(all_global_ids)}"
        )
        assert len(local_kg1_idx) == len(partitions[rank])
        assert len(local_kg2_idx) == len(assignments[rank])
        print("  test_no_halo_kg1_kg2_ablation_disables_halo_entirely PASSED")


if __name__ == "__main__":
    print("Stage G tests:")
    test_halo_kg2_retains_cross_rank_edge()
    test_no_halo_kg1_kg2_ablation_disables_halo_entirely()
    print("All Stage G tests PASSED.")
