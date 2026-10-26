"""
dist_ocd.py — Owner-Compute Dispatch (OCD) for halo-entity refresh.

Replaces framework.py's shared-file halo refresh with direct point-to-point
tensor exchange: each rank asks only the ranks that actually own its needed
halo entities, and each owner replies with exactly those rows in one batched
message per peer — "one request, one packaged response" — instead of every
rank reading every other rank's whole embedding file every iteration.

Protocol (request_and_fulfill_halo):
  1. Every rank knows which entities it needs (halo_global_ids) and, via the
     static owner_map, who owns each one (compute_needed_ids).
  2. All ranks all_gather_object their needed-ID lists, so every rank learns
     exactly what's being asked of it.
  3. Point-to-point fulfillment: each owner issues one batched isend of the
     requested rows per asking peer (group_serving_by_peer); each requester
     issues one matching irecv per owner it asked (group_requests_by_owner).

Plain torch.distributed point-to-point (isend/irecv) was chosen over a
separate RPC subsystem (e.g. torch.distributed.rpc) or the dormant Go
cache_server/leapgnn gRPC stack: it needs no second init/shutdown lifecycle
or port, reuses the exact primitive family already proven in this codebase
(all_reduce, all_gather, broadcast), and makes communication volume/call
count trivial to instrument at the send/recv boundary.

v1's "owner compute" is a plain local lookup — parity with the file-based
refresh path, just delivered point-to-point instead of via shared files.
Scoring/filtering requested candidates (e.g. via VBIP) is a documented
extension point: # TODO(OCD-v2).

The grouping helpers below are pure functions (no torch.distributed calls),
so they are unit-testable without a live process group; only
request_and_fulfill_halo itself needs a real multi-rank run.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.distributed as dist

from comm_stats import CommStats


def build_owner_map(partition_dir: str, world_size: int) -> dict[int, int]:
    """
    Read every rank's {r}.npy partition file once and return a global
    {kg1_global_id: owner_rank} map.

    Partitioning is static for the whole run, so callers should build this
    once (e.g. cached on the framework instance) rather than per iteration.
    """
    owner_map: dict[int, int] = {}
    for r in range(world_size):
        part_file = os.path.join(partition_dir, f"{r}.npy")
        if not os.path.exists(part_file):
            continue
        for gid in np.load(part_file).astype(np.int64):
            owner_map[int(gid)] = r
    return owner_map


def compute_needed_ids(
    halo_global_ids: np.ndarray, owner_map: dict[int, int], rank: int
) -> list[int]:
    """IDs this rank needs from a peer (excludes IDs this rank owns itself)."""
    return [int(g) for g in halo_global_ids if owner_map.get(int(g), rank) != rank]


def group_requests_by_owner(
    needed_ids: list[int], owner_map: dict[int, int], rank: int
) -> dict[int, list[int]]:
    """Requester side: {owner_rank: [gid, ...]} for this rank's own needed IDs."""
    grouped: dict[int, list[int]] = {}
    for gid in needed_ids:
        owner = owner_map.get(gid)
        if owner is not None and owner != rank:
            grouped.setdefault(owner, []).append(gid)
    return grouped


def group_serving_by_peer(
    needed_by_rank: dict[int, list[int]], owner_map: dict[int, int], my_rank: int
) -> dict[int, list[int]]:
    """
    Owner side: {peer_rank: [gid, ...]} this rank must serve, given every
    rank's gathered needed-ID list (needed_by_rank[peer] == peer's own
    compute_needed_ids() output). Iterates each peer's list in its original
    order so the resulting per-peer sequence matches what that peer computes
    independently in group_requests_by_owner — required for isend/irecv
    shapes to line up without any extra handshake.
    """
    to_serve: dict[int, list[int]] = {}
    for peer, ids in needed_by_rank.items():
        if peer == my_rank:
            continue
        for gid in ids:
            if owner_map.get(gid) == my_rank:
                to_serve.setdefault(peer, []).append(gid)
    return to_serve


def request_and_fulfill_halo(
    halo_global_ids: np.ndarray,
    owner_map: dict[int, int],
    local_embeddings: dict[int, np.ndarray],
    embedding_dim: int,
    rank: int,
    world_size: int,
    device: torch.device,
    comm_stats: CommStats | None = None,
    group=None,
) -> dict[int, np.ndarray]:
    """
    Fetch current embeddings for halo_global_ids via point-to-point dispatch.

    Args:
        halo_global_ids:  global IDs this rank needs fresh embeddings for.
        owner_map:        {global_id: owner_rank}, from build_owner_map().
        local_embeddings: this rank's own {global_id: embedding} — the rows
                           it can serve to peers that request them (framework.py
                           reuses the same dict SACA tracks for write-skipping).
        embedding_dim:    row width; must match cfg.output_dim on every rank
                          so isend/irecv tensor shapes agree.
        comm_stats:       optional CommStats to record timing/volume into.
        group:            process group to issue all_gather_object/isend/irecv
                          on. Must be a CPU-capable (e.g. gloo) group — see the
                          isend/irecv comment below for why. Defaults to the
                          world default group (only correct when that group's
                          backend is itself gloo/CPU-capable).

    Returns:
        {global_id: embedding} received from peers (entities this rank owns
        itself are excluded — the caller already has those locally).
    """
    cs = comm_stats or CommStats()

    needed = compute_needed_ids(halo_global_ids, owner_map, rank)

    gathered: list[object] = [None] * world_size
    with cs.timed("ocd_request_gather"):
        dist.all_gather_object(gathered, needed, group=group)
    cs.record("ocd_request_gather", calls=1)
    needed_by_rank: dict[int, list[int]] = {r: gathered[r] for r in range(world_size)}

    my_requests = group_requests_by_owner(needed, owner_map, rank)
    to_serve = group_serving_by_peer(needed_by_rank, owner_map, rank)

    reqs = []

    # isend/irecv buffers stay on CPU: point-to-point ops need a CPU-capable
    # (e.g. gloo) process group -- pairing them with CUDA tensors under gloo
    # crashes (gloo::IoException "Bad address" / "Connection closed by
    # peer").
    #
    # This is also why `group` must be a dedicated gloo group when the
    # world's default backend is nccl (multi-node, one GPU per rank):
    # nccl's isend/irecv lazily creates a SEPARATE point-to-point
    # sub-communicator the first time a given rank pair exchanges directly
    # (distinct from the collective communicator all_reduce/all_gather
    # use), and that bootstrap path fails across physical nodes
    # (`ncclInternalError`, "ncclSocketInit: connecting to address  with
    # family 1 is neither AF_INET(2) nor AF_INET6(10)") -- an
    # empty-address NCCL P2P bootstrap bug, not something fixable by
    # choosing a tensor device. framework.py therefore routes OCD through
    # its own `dist.new_group(backend="gloo")`.
    send_buffers: dict[int, torch.Tensor] = {}
    for peer, gids in to_serve.items():
        buf = np.zeros((len(gids), embedding_dim), dtype=np.float32)
        for i, gid in enumerate(gids):
            row = local_embeddings.get(gid)
            if row is not None:
                w = min(embedding_dim, len(row))
                buf[i, :w] = row[:w]
        send_buffers[peer] = torch.from_numpy(buf)
        reqs.append(dist.isend(send_buffers[peer], dst=peer, group=group))
        cs.record("ocd_send", num_bytes=send_buffers[peer].numel() * 4, calls=1)

    recv_buffers: dict[int, torch.Tensor] = {}
    for owner, gids in my_requests.items():
        recv_buffers[owner] = torch.zeros(len(gids), embedding_dim)
        reqs.append(dist.irecv(recv_buffers[owner], src=owner, group=group))

    with cs.timed("ocd_fulfill"):
        for r in reqs:
            r.wait()

    result: dict[int, np.ndarray] = {}
    for owner, gids in my_requests.items():
        rows = recv_buffers[owner].cpu().numpy()
        for i, gid in enumerate(gids):
            result[gid] = rows[i]
        cs.record("ocd_recv", num_bytes=rows.nbytes, calls=1)

    return result
