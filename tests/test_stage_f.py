"""
test_stage_f.py — Stage F tests: dist_ocd.py's pure grouping helpers.

These helpers decide which halo entities a rank requests from which owner
and which it serves to which peer; they need no live process group. The P2P
transport itself (request_and_fulfill_halo) can only be exercised by a real
multi-rank run.

DGL-free, no GPU required, <30 sec.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dist_ocd import (
    build_owner_map,
    compute_needed_ids,
    group_requests_by_owner,
    group_serving_by_peer,
)


def test_build_owner_map():
    with tempfile.TemporaryDirectory() as tmp:
        np.save(os.path.join(tmp, "0.npy"), np.array([1, 2, 3]))
        np.save(os.path.join(tmp, "1.npy"), np.array([4, 5]))
        owner_map = build_owner_map(tmp, world_size=2)
        assert owner_map == {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
        print("  test_build_owner_map PASSED")


def test_compute_needed_ids_excludes_own_entities():
    owner_map = {1: 0, 2: 0, 3: 1, 4: 1}
    halo_ids = np.array([1, 2, 3, 4])
    needed = compute_needed_ids(halo_ids, owner_map, rank=0)
    assert needed == [3, 4], f"expected only rank-1-owned ids, got {needed}"
    print("  test_compute_needed_ids_excludes_own_entities PASSED")


def test_compute_needed_ids_preserves_order():
    owner_map = {5: 1, 3: 1, 8: 1, 1: 0}
    halo_ids = np.array([5, 1, 3, 8])  # deliberately not sorted
    needed = compute_needed_ids(halo_ids, owner_map, rank=0)
    assert needed == [5, 3, 8], f"order must follow halo_ids, got {needed}"
    print("  test_compute_needed_ids_preserves_order PASSED")


def test_group_requests_by_owner():
    owner_map = {1: 1, 2: 1, 3: 2, 4: 2}
    needed = [1, 2, 3, 4]
    grouped = group_requests_by_owner(needed, owner_map, rank=0)
    assert grouped == {1: [1, 2], 2: [3, 4]}, grouped
    print("  test_group_requests_by_owner PASSED")


def test_group_serving_by_peer():
    owner_map = {1: 0, 2: 0, 3: 1}
    # rank 1 asked for [1, 3] (3 is its own — compute_needed_ids would have
    # excluded it already, but group_serving_by_peer is defensive too since
    # it re-checks ownership), rank 2 asked for [2]
    needed_by_rank = {0: [], 1: [1, 3], 2: [2]}
    to_serve = group_serving_by_peer(needed_by_rank, owner_map, my_rank=0)
    assert to_serve == {1: [1], 2: [2]}, to_serve
    print("  test_group_serving_by_peer PASSED")


def test_serving_and_requesting_sequences_match_for_isend_irecv():
    """
    The property request_and_fulfill_halo's isend/irecv pairing depends on:
    for any two ranks A (requester) and B (owner), the sequence B computes
    in group_serving_by_peer(...)[A] must be IDENTICAL (same ids, same
    order) to what A independently computes in
    group_requests_by_owner(A's own needed, owner_map, A)[B] — no handshake
    reconciles them, so a mismatch here would desync isend/irecv buffer
    shapes.
    """
    owner_map = {10: 0, 11: 0, 12: 1, 13: 1, 14: 1, 15: 2}
    halo_ids_by_rank = {
        0: np.array([12, 15, 13]),  # rank 0 needs these, in this order
        1: np.array([10, 15]),
        2: np.array([11]),
    }
    needed_by_rank = {
        r: compute_needed_ids(ids, owner_map, r)
        for r, ids in halo_ids_by_rank.items()
    }

    for requester in range(3):
        for owner in range(3):
            if requester == owner:
                continue
            requester_view = group_requests_by_owner(
                needed_by_rank[requester], owner_map, requester
            ).get(owner, [])
            owner_view = group_serving_by_peer(
                needed_by_rank, owner_map, owner
            ).get(requester, [])
            assert requester_view == owner_view, (
                f"requester {requester} expects {requester_view} from "
                f"{owner}, but {owner} would serve {owner_view}"
            )
    print("  test_serving_and_requesting_sequences_match_for_isend_irecv PASSED")


if __name__ == "__main__":
    print("Stage F tests:")
    test_build_owner_map()
    test_compute_needed_ids_excludes_own_entities()
    test_compute_needed_ids_preserves_order()
    test_group_requests_by_owner()
    test_group_serving_by_peer()
    test_serving_and_requesting_sequences_match_for_isend_irecv()
    print("All Stage F tests PASSED.")
