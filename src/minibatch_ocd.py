"""
minibatch_ocd.py — cross-rank on-demand fetch for mini-batch training.

SparseEntityTable.gather() alone is correct only for entities THIS rank
owns (its own local_kg1/local_kg2 partition, per the static partition
files dist_ocd.build_owner_map already reads) — an entity in a sampled
batch's outer_ids that belongs to ANOTHER rank needs THAT rank's current
trained value, not gather()'s deterministic-for-never-touched-ids
fallback (which would silently substitute a stale/wrong initial value for
an entity another rank has been training all along).

Reuses dist_ocd.request_and_fulfill_halo (the point-to-point fetch of the
full-graph halo path), called with just THIS BATCH's remote ids instead of
the whole halo set once per outer iteration. That function's
`local_embeddings` parameter is only ever read via `.get(gid)` (never
iterated as a whole dict), so _SparseTableServeAdapter below satisfies that
contract by computing each requested row lazily instead of requiring the
caller to pre-populate a dict for every locally-owned entity up front —
doing that eagerly would defeat mini-batch training's entire memory-saving
premise (SparseEntityTable's whole point is NOT holding every owned
entity's row in memory at once).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from comm_stats import CommStats
from dist_ocd import build_owner_map, request_and_fulfill_halo
from sparse_table import SparseEntityTable


def build_full_owner_map(partition_dir: str, world_size: int) -> dict[int, int]:
    """{global_id: owner_rank} covering BOTH KG1 (dist_ocd.build_owner_map,
    from {r}.npy METIS partition files) and KG2 (kg2_assignments.json,
    written by counterpart_disc.py — {str(rank): [kg2_global_id, ...]}).
    dist_ocd.build_owner_map alone is KG1-only (see its own docstring/every
    caller's comment to that effect in framework.py) — mini-batch training
    needs ownership for BOTH sides, since a training pair's KG2 endpoint
    can be owned by a different rank than its KG1 endpoint, and sampled
    neighborhoods routinely cross into KG2 entities generally."""
    owner_map = build_owner_map(partition_dir, world_size)
    kg2_path = os.path.join(partition_dir, "kg2_assignments.json")
    if os.path.exists(kg2_path):
        with open(kg2_path) as f:
            kg2_assignments = json.load(f)
        for rank_str, ids in kg2_assignments.items():
            r = int(rank_str)
            for gid in ids:
                owner_map[int(gid)] = r
    return owner_map


def split_outer_ids_by_ownership(
    outer_ids: np.ndarray, owner_map: dict[int, int], rank: int
):
    """(local_ids, remote_ids): pure ownership split, factored out of
    gather_minibatch_features so it's unit-testable without a live process
    group — mirrors dist_ocd.py's own pattern of keeping grouping logic
    separate from the one function that actually needs torch.distributed
    (compute_needed_ids/group_requests_by_owner/group_serving_by_peer vs.
    request_and_fulfill_halo)."""
    is_remote = np.array(
        [owner_map.get(int(g), rank) != rank for g in outer_ids], dtype=bool
    )
    return outer_ids[~is_remote], outer_ids[is_remote]


class _SparseTableServeAdapter:
    """dict-like `.get(gid, default)` view over a SparseEntityTable, for
    request_and_fulfill_halo's `local_embeddings` parameter. Computes each
    requested row on demand (SparseEntityTable.gather() already returns
    the materialized-if-touched-else-deterministic value for any id, which
    is exactly what a peer asking for one of this rank's own entities
    should receive) instead of eagerly building a dict for every entity
    this rank owns."""

    def __init__(self, table: SparseEntityTable):
        self._table = table

    def get(self, gid: int, default=None):
        row = self._table.gather(np.array([gid]))
        return row.detach().cpu().numpy()[0]


@dataclass
class MinibatchGatherResult:
    """Return value of gather_minibatch_features: `rows` is ready to feed
    straight into RREA.forward_blocks/GCN.forward_blocks (same row order
    as the `outer_ids` passed in); `local_ids`/`remote_ids` split
    `outer_ids` by ownership so the caller can scope
    SparseEntityTable.apply_gradients() to `local_ids` ONLY. Applying
    gradients to remote_ids' rows would be pointless (this wrapper always
    re-fetches fresh values for them on every call, so any locally-Adam-
    updated value sitting in this rank's own table._rows for a remote id
    is immediately overwritten and never actually used as an input again)
    and would incorrectly grow this rank's SparseEntityTable with rows for
    entities it doesn't own, corrupting `len(table)`'s "entities this rank
    has materialized" meaning."""

    rows: torch.Tensor
    local_ids: np.ndarray
    remote_ids: np.ndarray


def gather_minibatch_features(
    outer_ids: np.ndarray,
    table: SparseEntityTable,
    owner_map: dict[int, int],
    rank: int,
    world_size: int,
    device: torch.device,
    dim: int,
    comm_stats: CommStats | None = None,
    group=None,
    serve_table: SparseEntityTable | None = None,
) -> MinibatchGatherResult:
    """
    Gather h_init for one mini-batch's outermost sampled frontier
    (`outer_ids`), fetching entities owned by other ranks via one
    collective OCD round instead of `table.gather()`'s local-only
    semantics.

    Collective: every rank must call this together once per mini-batch —
    request_and_fulfill_halo's all_gather_object round requires it, same
    as the existing halo-refresh call site. If a real training loop skips
    calling this for some ranks on some batches (e.g. an empty local batch
    on one rank while others still have data), those ranks must still
    participate with an empty `outer_ids` array rather than not calling at
    all, or the collective hangs.

    `owner_map`: from dist_ocd.build_owner_map(partition_dir, world_size),
    built once per run (partitioning is static) and reused across every
    batch — matches the existing halo-refresh call site's own caching
    pattern.

    `serve_table` (--minibatch_halo_snapshot, framework.py): the table THIS
    rank serves to peers' requests. Default None = `table` (peers see this
    rank's live, currently-training rows). Passing the previous iteration's
    frozen table reproduces the full-graph path's halo semantics -- remote
    rows are the owner's end-of-previous-iteration values, fixed for the
    whole iteration, while each rank re-learns its own rows from scratch.
    """
    cs = comm_stats or CommStats()

    local_ids, remote_ids = split_outer_ids_by_ownership(outer_ids, owner_map, rank)

    fetched = request_and_fulfill_halo(
        remote_ids, owner_map, _SparseTableServeAdapter(serve_table or table), dim,
        rank, world_size, device, cs, group,
    )

    rows = table.gather(outer_ids)  # local-or-deterministic baseline for every id, right shape/order
    if fetched:
        with torch.no_grad():
            for i, gid in enumerate(outer_ids):
                if int(gid) in fetched:
                    rows[i] = torch.from_numpy(fetched[int(gid)]).to(device)

    return MinibatchGatherResult(rows=rows, local_ids=local_ids, remote_ids=remote_ids)
