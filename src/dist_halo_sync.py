"""
dist_halo_sync.py — Halo embedding refresh between training iterations.

Problem: DistGraphLoader loads 1-hop cross-partition KG1 nodes (halo nodes)
into each rank's local graph.  Their features in h_init come from the *previous*
iteration's GCN output — they are stale after one training run.

Solution:
  After each iteration's training, each rank writes its local KG1 GCN-output
  embeddings to the Go gRPC cache server.  Before building h_init for the next
  iteration, each rank fetches the halo node embeddings from the Go server and
  replaces the corresponding rows in h_init.

Server lifecycle:
  The Go server is restarted between iterations. The abstract
  HaloSync.refresh() interface hides this detail so it could be replaced by
  a hot-reload RPC without changing framework.py.

Embedding dimension bridging:
  The Go server caches GCN output embeddings (output_dim = 128 by default).
  h_init rows are hidden_dim (256 by default).  We zero-pad output_dim → hidden_dim
  so the encoder's first layer sees a valid-dimension tensor.

Usage in framework.py:
    halo_sync = make_halo_sync(cfg, rank, world_size, partition_dir)
    # ... after each iteration:
    halo_sync.push(local_kg1_ids, local_kg1_emb, rank)
    halo_sync.barrier()
    # ... at start of next iteration, before model forward:
    h_init = halo_sync.refresh(h_init, halo_kg1_idx,
                                halo_global_ids, hidden_dim)
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
#  Abstract interface
# ---------------------------------------------------------------------------


class HaloSync(ABC):
    """Abstract halo embedding synchronisation layer."""

    @abstractmethod
    def push(self, global_ids: np.ndarray, embeddings: np.ndarray, rank: int) -> None:
        """Write this rank's KG1 output embeddings to the cache store."""

    @abstractmethod
    def barrier(self) -> None:
        """Wait until all ranks have finished push()."""

    @abstractmethod
    def refresh(
        self,
        h_init: torch.Tensor,
        halo_kg1_idx: np.ndarray,
        halo_global_ids: np.ndarray,
        hidden_dim: int,
    ) -> torch.Tensor:
        """
        Replace halo rows in h_init with fresh embeddings from the cache.

        Args:
            h_init:          (N_local, hidden_dim) initial feature matrix
            halo_kg1_idx:    local indices of halo nodes
            halo_global_ids: global entity IDs of halo nodes (same order)
            hidden_dim:      h_init feature dimension (may be > output_dim)

        Returns:
            Updated h_init (in-place modification + return for convenience).
        """

    @abstractmethod
    def start(self) -> None:
        """Start / prepare the underlying cache infrastructure."""

    @abstractmethod
    def stop(self) -> None:
        """Tear down / clean up the cache infrastructure."""

    @abstractmethod
    def fetch(self, global_ids: np.ndarray) -> dict[int, np.ndarray]:
        """{global_id: embedding} for whichever of global_ids are found —
        unlike refresh() (which mutates a specific h_init tensor's rows
        in-place, including the hidden_dim padding), this returns
        a plain dict so callers can feed it into the SAME
        framework.py::_patch_halo_rows path halo_kg1/halo_kg2/consistency-
        regularizer building already use uniformly, regardless of which
        HaloSync backend produced it."""


# ---------------------------------------------------------------------------
#  No-op implementation (when cache_backend == 'none')
# ---------------------------------------------------------------------------


class NoopHaloSync(HaloSync):
    """Halo sync that does nothing — halo embeddings remain stale."""

    def push(self, global_ids, embeddings, rank):
        pass

    def barrier(self):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def refresh(self, h_init, halo_kg1_idx, halo_global_ids, hidden_dim):
        return h_init  # leave stale — same as no sync

    def start(self):
        pass

    def stop(self):
        pass

    def fetch(self, global_ids):
        return {}


# ---------------------------------------------------------------------------
#  File-based halo sync (single-machine, no Go server)
# ---------------------------------------------------------------------------


class FileHaloSync(HaloSync):
    """
    Halo sync backed by per-rank .npy files on a shared filesystem.

    Each rank pushes emb_{rank}.npy + ids_{rank}.npy to cache_dir.
    refresh() loads all rank files and patches halo rows in h_init.

    No Go server required — ideal for single-node dev and debugging.
    """

    def __init__(self, cache_dir: str, world_size: int):
        self.cache_dir = cache_dir
        self.world_size = world_size
        self._id_to_emb: dict[int, np.ndarray] = {}
        self._loaded = False

    def push(self, global_ids: np.ndarray, embeddings: np.ndarray, rank: int) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        np.save(
            os.path.join(self.cache_dir, f"ids_{rank}.npy"), global_ids.astype(np.int64)
        )
        np.save(
            os.path.join(self.cache_dir, f"emb_{rank}.npy"),
            embeddings.astype(np.float32),
        )
        self._loaded = False  # invalidate cache on next refresh
        logging.debug(f"[FileHaloSync rank {rank}] pushed {len(global_ids)} embeddings")

    def barrier(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _load_all(self) -> None:
        self._id_to_emb = {}
        for r in range(self.world_size):
            ids_path = os.path.join(self.cache_dir, f"ids_{r}.npy")
            emb_path = os.path.join(self.cache_dir, f"emb_{r}.npy")
            if not os.path.exists(ids_path):
                continue
            ids = np.load(ids_path)
            emb = np.load(emb_path)
            for i, gid in enumerate(ids):
                self._id_to_emb[int(gid)] = emb[i]
        self._loaded = True
        logging.info(
            f"[FileHaloSync] loaded {len(self._id_to_emb)} embeddings from {self.cache_dir}"
        )

    def refresh(
        self,
        h_init: torch.Tensor,
        halo_kg1_idx: np.ndarray,
        halo_global_ids: np.ndarray,
        hidden_dim: int,
    ) -> torch.Tensor:
        if not self._loaded:
            self._load_all()

        if len(halo_kg1_idx) == 0:
            return h_init

        refreshed = 0
        for li, gid in zip(halo_kg1_idx, halo_global_ids, strict=False):
            emb = self._id_to_emb.get(int(gid))
            if emb is not None:
                # Zero-pad output_dim → hidden_dim if needed
                padded = np.zeros(hidden_dim, dtype=np.float32)
                padded[: len(emb)] = emb[:hidden_dim]
                h_init[li] = torch.from_numpy(padded).to(h_init.device)
                refreshed += 1

        logging.debug(
            f"[FileHaloSync] refreshed {refreshed}/{len(halo_kg1_idx)} halo rows"
        )
        return h_init

    def start(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)

    def stop(self) -> None:
        pass

    def fetch(self, global_ids: np.ndarray) -> dict[int, np.ndarray]:
        if not self._loaded:
            self._load_all()
        return {
            int(gid): self._id_to_emb[int(gid)]
            for gid in global_ids
            if int(gid) in self._id_to_emb
        }


# ---------------------------------------------------------------------------
#  gRPC-based halo sync (wraps the DistEA Go cache server)
# ---------------------------------------------------------------------------


class GrpcHaloSync(HaloSync):
    """
    Halo sync backed by the DistEA Go gRPC cache server (bin/distea_server).

    Server lifecycle (restart per iteration):
      1. All ranks call push() → write .npy files to emb_dir.
      2. All ranks call barrier().
      3. Rank 0 starts one Go server per rank (via subprocess).
      4. Rank 0 waits for all ready_file signals.
      5. All ranks call refresh() → fetch halo embeddings via gRPC.
      6. At end of iteration: rank 0 kills all Go servers.

    Args:
        emb_dir:       directory for emb_{rank}.npy / ids_{rank}.npy
        partition_dir: directory for {rank}.npy partition files
        grpc_addrs:    list of "host:port" per rank (len == world_size)
        world_size:    number of ranks
        rank:          this process's rank
        server_bin:    path to distea_server binary (bin/distea_server)
        ready_timeout: seconds to wait for each server ready_file
        node_ip:       this process's own node IP (e.g. bond0's IPv4),
                        matched against grpc_addrs' host parts to decide
                        which servers THIS node should spawn. None (default)
                        uses a single spawner (global rank 0 spawns every
                        server, locally) -- only correct when every
                        grpc_addrs entry is reachable from rank 0's own
                        machine (single physical node, e.g. all
                        "127.0.0.1:*"). subprocess.Popen can only ever start
                        a process on the machine calling it -- it can bind
                        that process to "0.0.0.0" (all of THIS node's own
                        interfaces) but can never make it reachable at a
                        DIFFERENT physical node's address. Set this (with
                        local_rank) for genuine multi-physical-node
                        deployment, where each node needs its own local
                        spawner for its own slice of grpc_addrs.
        local_rank:    this process's rank WITHIN its physical node
                        (torchrun sets LOCAL_RANK; see main.py's own
                        local_rank derivation). Only consulted when
                        node_ip is set -- exactly one process per node
                        (local_rank == 0) spawns that node's servers.
    """

    def __init__(
        self,
        emb_dir: str,
        partition_dir: str,
        grpc_addrs: list[str],
        world_size: int,
        rank: int,
        server_bin: str = "bin/distea_server",
        ready_timeout: float = 120.0,
        node_ip: str | None = None,
        local_rank: int | None = None,
    ):
        self.emb_dir = emb_dir
        self.partition_dir = partition_dir
        self.grpc_addrs = grpc_addrs
        self.world_size = world_size
        self.rank = rank
        self.server_bin = server_bin
        self.ready_timeout = ready_timeout
        self.node_ip = node_ip
        self.local_rank = local_rank if local_rank is not None else rank
        self._procs: list[subprocess.Popen] = []
        # Delegates actual push()/fetch() I/O to leapgnn_cache.GrpcEmbeddingCache
        # rather than duplicating it -- that class's partition_dir handling
        # (_split_partition_dir) parses an already-fully-qualified
        # partition_dir and any "{N}_metis*" suffix (e.g. --kg2_assignment
        # locality's "{N}_metis_kg2locality"); see tests/test_stage_m.py.
        from leapgnn_cache import GrpcEmbeddingCache

        self._cache = GrpcEmbeddingCache(
            cache_dir=emb_dir, world_size=world_size, grpc_addrs=grpc_addrs,
            partition_dir=partition_dir,
        )

    # ── push / barrier (identical to FileHaloSync) ─────────────────────────

    def push(self, global_ids: np.ndarray, embeddings: np.ndarray, rank: int) -> None:
        self._cache.push(global_ids, embeddings, rank)
        self._cache._client = None  # invalidate cached client
        logging.debug(f"[GrpcHaloSync rank {rank}] pushed {len(global_ids)} embeddings")

    def barrier(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    # ── server lifecycle ───────────────────────────────────────────────────

    def _is_spawner(self) -> bool:
        """True iff this process is responsible for spawning (a slice of)
        the Go servers. Single-node (node_ip unset): only global rank 0. Multi-node: one spawner per physical node, its own
        local_rank == 0."""
        if self.node_ip is None:
            return self.rank == 0
        return self.local_rank == 0

    def _my_server_indices(self) -> list[int]:
        """Which grpc_addrs indices THIS process should spawn. Single-node:
        all of them (rank 0 is the sole spawner). Multi-node:
        only entries whose host matches this node's own IP -- see node_ip's
        docstring above for why a locally-spawned process can never serve
        on a different physical node's address."""
        if self.node_ip is None:
            return list(range(len(self.grpc_addrs)))
        return [
            r for r, addr in enumerate(self.grpc_addrs)
            if addr.rsplit(":", 1)[0] == self.node_ip
        ]

    def start(self) -> None:
        """Start this node's slice of Go cache server processes (single-
        node: rank 0 starts all of them; multi-node: each node's local
        rank 0 starts only its own node's servers -- see _is_spawner/
        _my_server_indices). Every spawner waits for ALL world_size ready
        files regardless of which node created them, since emb_dir is
        shared-filesystem-visible cluster-wide -- downstream fetch() calls
        (from any rank, on any node) need every server up, not just this
        node's own slice."""
        if not self._is_spawner():
            return
        self._procs = []
        cache_group = ",".join(self.grpc_addrs)
        for r in self._my_server_indices():
            addr = self.grpc_addrs[r]
            host, port = addr.rsplit(":", 1)
            ready_file = os.path.join(self.emb_dir, f"ready_{r}.txt")
            # Remove stale ready file
            if os.path.exists(ready_file):
                os.remove(ready_file)
            cmd = [
                self.server_bin,
                "--config",
                "cache_server/conf/distea.yaml",
                "--partition_dir",
                self.partition_dir,
                "--emb_dir",
                self.emb_dir,
                "--cachegroup",
                cache_group,
                "--rpcport",
                port,
                "--distea_rank",
                str(r),
                "--bind_addr",
                "0.0.0.0",
                "--ready_file",
                ready_file,
                "--cachetype",
                "static",
            ]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self._procs.append(proc)
            logging.info(
                f"[GrpcHaloSync] started Go server rank {r} (pid {proc.pid}) on {addr}"
            )

        # Wait for all servers to signal readiness
        for r in range(self.world_size):
            ready_file = os.path.join(self.emb_dir, f"ready_{r}.txt")
            deadline = time.time() + self.ready_timeout
            while not os.path.exists(ready_file):
                if time.time() > deadline:
                    raise TimeoutError(
                        f"[GrpcHaloSync] Go server rank {r} did not create "
                        f"{ready_file} within {self.ready_timeout}s"
                    )
                time.sleep(0.5)
            logging.info(f"[GrpcHaloSync] Go server rank {r} ready.")

    def stop(self) -> None:
        """Terminate this node's slice of Go server processes (see start()/
        _is_spawner()). self._procs only ever holds what THIS process
        itself spawned.

        The barrier below is load-bearing: every call site runs
        fetch() -> stop(), and stop() is called by ALL ranks (the actual
        kill is gated by _is_spawner()), but without a barrier the
        spawner rank kills the servers as soon as ITS OWN fetch returns
        -- slower ranks still mid-fetch get StatusCode.UNAVAILABLE
        "Socket closed" (reliably so once ranks fall out of lockstep, e.g.
        several ranks sharing one GPU). Collective-safe precisely because
        all ranks call stop()."""
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()
        if not self._is_spawner():
            return
        for proc in self._procs:
            proc.terminate()
        for proc in self._procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._procs = []
        self._cache._client = None
        logging.info("[GrpcHaloSync] All Go servers stopped.")

    # ── refresh / fetch (delegate to the fixed, tested GrpcEmbeddingCache) ──

    def refresh(
        self,
        h_init: torch.Tensor,
        halo_kg1_idx: np.ndarray,
        halo_global_ids: np.ndarray,
        hidden_dim: int,
    ) -> torch.Tensor:
        if len(halo_kg1_idx) == 0:
            return h_init

        fetched = self._cache.fetch(halo_global_ids)
        feat_dim = fetched.shape[1]

        for i, li in enumerate(halo_kg1_idx):
            padded = np.zeros(hidden_dim, dtype=np.float32)
            padded[:feat_dim] = fetched[i, :hidden_dim]
            h_init[li] = torch.from_numpy(padded).to(h_init.device)

        logging.debug(
            f"[GrpcHaloSync] refreshed {len(halo_kg1_idx)} halo rows via gRPC"
        )
        return h_init

    def fetch(self, global_ids: np.ndarray) -> dict[int, np.ndarray]:
        """Note: unlike FileHaloSync.fetch (which omits ids not found on
        disk), GrpcEmbeddingCache.fetch returns a zero vector for any id
        the Go server doesn't recognise (its own documented behaviour) --
        every requested id is included here, some possibly zero-valued.
        Acceptable for halo refresh specifically: every entity in a real
        run is owned by some rank and was pushed this iteration, so a
        genuinely-unknown id shouldn't occur in practice; not re-engineered
        here since the Go server's wire protocol doesn't distinguish
        "legitimately zero" from "not found" at all."""
        if len(global_ids) == 0:
            return {}
        fetched = self._cache.fetch(global_ids)
        return {int(gid): fetched[i] for i, gid in enumerate(global_ids)}


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------


def make_halo_sync(cfg, rank: int, world_size: int, partition_dir: str) -> HaloSync:
    """
    Create the appropriate HaloSync from training config.

    Args:
        cfg:           argparse.Namespace with cache_backend, cache_dir, grpc_addrs
        rank:          this process's rank
        world_size:    total number of ranks
        partition_dir: directory with {rank}.npy partition files

    Returns:
        HaloSync instance (NoopHaloSync, FileHaloSync, or GrpcHaloSync)
    """
    backend = getattr(cfg, "cache_backend", "none").lower()

    if backend == "none":
        return NoopHaloSync()

    if backend == "file":
        return FileHaloSync(
            cache_dir=cfg.cache_dir,
            world_size=world_size,
        )

    if backend == "grpc":
        raw = getattr(cfg, "grpc_addrs", "") or ""
        grpc_addrs = [a.strip() for a in raw.split(",") if a.strip()]
        if not grpc_addrs:
            logging.warning(
                "[HaloSync] grpc_addrs empty — falling back to FileHaloSync"
            )
            return FileHaloSync(cache_dir=cfg.cache_dir, world_size=world_size)

        server_bin = getattr(cfg, "grpc_server_bin", "bin/distea_server")
        # DISTEA_NODE_IP: set by multi-physical-node launchers (this node's
        # own IPv4, resolved the same way the launcher resolves MASTER_ADDR)
        # so GrpcHaloSync knows which grpc_addrs entries are its own to
        # spawn. Unset in single-node deployments (and in the tests), which
        # keeps the single spawner at global rank 0 -- see its node_ip
        # docstring.
        node_ip = os.environ.get("DISTEA_NODE_IP") or None
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        return GrpcHaloSync(
            emb_dir=cfg.cache_dir,
            partition_dir=partition_dir,
            grpc_addrs=grpc_addrs,
            world_size=world_size,
            rank=rank,
            server_bin=server_bin,
            node_ip=node_ip,
            local_rank=local_rank,
        )

    raise ValueError(f"Unknown cache_backend {backend!r}")
