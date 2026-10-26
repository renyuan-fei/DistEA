"""
test_stage_q.py — Stage Q tests: minibatch_ocd.py's pure pieces (mini-batch
cross-rank on-demand fetch).

As with dist_ocd.py in test_stage_f.py, request_and_fulfill_halo itself calls dist.all_gather_object/
isend/irecv unconditionally (even with an empty remote-ids array), so it
needs a real multi-rank process group and cannot be exercised here.
gather_minibatch_features (the function that calls it) is therefore NOT
covered by this file — only its two pure pieces are:
  - split_outer_ids_by_ownership: the ownership-partitioning logic,
    factored out so it can be tested without a process group.
  - _SparseTableServeAdapter: the `local_embeddings` lazy-dict adapter
    passed to request_and_fulfill_halo.

DGL-free, no GPU, no torch.distributed required. <10 sec.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import tempfile

from minibatch_ocd import (
    _SparseTableServeAdapter,
    build_full_owner_map,
    split_outer_ids_by_ownership,
)
from models.gcn import build_entity_features
from sparse_table import SparseEntityTable


def test_split_outer_ids_by_ownership_separates_local_and_remote():
    owner_map = {1: 0, 2: 0, 3: 1, 4: 1}
    outer_ids = np.array([1, 2, 3, 4])
    local_ids, remote_ids = split_outer_ids_by_ownership(outer_ids, owner_map, rank=0)
    assert list(local_ids) == [1, 2]
    assert list(remote_ids) == [3, 4]
    print("  test_split_outer_ids_by_ownership_separates_local_and_remote PASSED")


def test_split_outer_ids_by_ownership_unowned_id_defaults_to_local():
    """An id absent from owner_map (e.g. never assigned by partition.py)
    falls back to `rank` in owner_map.get(gid, rank) -- matches
    dist_ocd.compute_needed_ids's own fallback exactly, so an id with no
    known owner is never spuriously requested from a peer."""
    owner_map = {1: 0}
    outer_ids = np.array([1, 99])
    local_ids, remote_ids = split_outer_ids_by_ownership(outer_ids, owner_map, rank=0)
    assert list(local_ids) == [1, 99]
    assert list(remote_ids) == []
    print("  test_split_outer_ids_by_ownership_unowned_id_defaults_to_local PASSED")


def test_split_outer_ids_by_ownership_empty_input():
    local_ids, remote_ids = split_outer_ids_by_ownership(np.array([], dtype=np.int64), {}, rank=0)
    assert len(local_ids) == 0
    assert len(remote_ids) == 0
    print("  test_split_outer_ids_by_ownership_empty_input PASSED")


def test_build_full_owner_map_merges_kg1_and_kg2():
    with tempfile.TemporaryDirectory() as tmp:
        np.save(os.path.join(tmp, "0.npy"), np.array([1, 2]))
        np.save(os.path.join(tmp, "1.npy"), np.array([3]))
        with open(os.path.join(tmp, "kg2_assignments.json"), "w") as f:
            json.dump({"0": [10], "1": [11, 12]}, f)
        owner_map = build_full_owner_map(tmp, world_size=2)
        assert owner_map == {1: 0, 2: 0, 3: 1, 10: 0, 11: 1, 12: 1}
        print("  test_build_full_owner_map_merges_kg1_and_kg2 PASSED")


def test_build_full_owner_map_missing_kg2_file_is_kg1_only():
    with tempfile.TemporaryDirectory() as tmp:
        np.save(os.path.join(tmp, "0.npy"), np.array([1]))
        owner_map = build_full_owner_map(tmp, world_size=1)
        assert owner_map == {1: 0}
        print("  test_build_full_owner_map_missing_kg2_file_is_kg1_only PASSED")


def test_serve_adapter_returns_deterministic_value_for_untouched_id():
    table = SparseEntityTable(dim=4, device=torch.device("cpu"))
    adapter = _SparseTableServeAdapter(table)
    row = adapter.get(10)
    expected = build_entity_features(np.array([10]), 4, torch.device("cpu"))[0].numpy()
    assert np.allclose(row, expected)
    print("  test_serve_adapter_returns_deterministic_value_for_untouched_id PASSED")


def test_serve_adapter_returns_updated_value_after_apply_gradients():
    """A peer requesting an entity this rank has already trained must get
    the TRAINED value, not the deterministic initial one -- the entire
    point of routing serve-side lookups through the table instead of a
    pre-built dict."""
    table = SparseEntityTable(dim=4, device=torch.device("cpu"), lr=0.5)
    ids = np.array([7])
    batch = table.gather(ids)
    (batch ** 2).sum().backward()
    table.apply_gradients(ids, batch)

    adapter = _SparseTableServeAdapter(table)
    served = adapter.get(7)
    initial = build_entity_features(ids, 4, torch.device("cpu"))[0].numpy()
    assert not np.allclose(served, initial), (
        "adapter should serve the TRAINED value, not the deterministic initial one"
    )
    assert np.allclose(served, table.gather(ids).detach().numpy()[0])
    print("  test_serve_adapter_returns_updated_value_after_apply_gradients PASSED")


def test_serve_table_snapshot_is_served_instead_of_live_table():
    """--minibatch_halo_snapshot: peers must be served the FROZEN previous
    table's values while the live table trains on (minibatch_ocd's
    gather_minibatch_features takes serve_table for exactly this)."""
    from minibatch_ocd import _SparseTableServeAdapter
    from sparse_table import SparseEntityTable
    dev = torch.device("cpu")
    frozen = SparseEntityTable(dim=4, device=dev, lr=0.5)
    rows = frozen.gather(np.array([7]))
    (rows ** 2).sum().backward()
    frozen.apply_gradients(np.array([7]), rows)          # frozen now holds a trained row 7
    live = SparseEntityTable(dim=4, device=dev, lr=0.5)  # fresh table, row 7 = deterministic init
    served = _SparseTableServeAdapter(frozen).get(7)
    fresh = _SparseTableServeAdapter(live).get(7)
    assert not np.allclose(served, fresh), "snapshot must differ from the fresh init"
    assert np.allclose(served, frozen.gather(np.array([7])).detach().numpy()[0])
    print("  test_serve_table_snapshot_is_served_instead_of_live_table PASSED")


if __name__ == "__main__":
    print("Stage Q tests:")
    test_split_outer_ids_by_ownership_separates_local_and_remote()
    test_split_outer_ids_by_ownership_unowned_id_defaults_to_local()
    test_split_outer_ids_by_ownership_empty_input()
    test_build_full_owner_map_merges_kg1_and_kg2()
    test_build_full_owner_map_missing_kg2_file_is_kg1_only()
    test_serve_adapter_returns_deterministic_value_for_untouched_id()
    test_serve_adapter_returns_updated_value_after_apply_gradients()
    test_serve_table_snapshot_is_served_instead_of_live_table()
    print("All Stage Q tests PASSED.")
