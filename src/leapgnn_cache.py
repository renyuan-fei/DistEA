"""
Embedding cache abstraction for DistEA (final-embedding push/fetch).

Provides two backends for persisting and retrieving entity embeddings:
  - FileEmbeddingCache: writes per-rank .npy files; rank 0 reads all for fetch.
    No external infrastructure required. Good for single-node and debugging.
  - GrpcEmbeddingCache: wraps LeapGNN's read-only DistCacheClient.
    push() writes .npy files that the Go server loads at startup.
    fetch() routes through gRPC for low-latency distributed access.

Usage in framework (after final training iteration):
    cache = make_embedding_cache(backend, cache_dir, world_size)
    if cache is not None:
        cache.push(local_kg1_ids, local_kg1_emb, rank)
        cache.barrier()      # all ranks must have pushed before rank 0 reads
        if rank == 0:
            emb = cache.fetch(query_ids)

Go server setup (GrpcEmbeddingCache only — server in cache_server/, built by
scripts/build_grpc_server.sh):
    1. All ranks call push() → writes ids_{rank}.npy + emb_{rank}.npy to cache_dir
    2. All ranks call barrier()
    3. Start LeapGNN Go server pointing at cache_dir
    4. Call fetch() — routes through grpc.insecure_channel to Go server
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch.distributed as dist

# ---------------------------------------------------------------------------
#  Abstract interface
# ---------------------------------------------------------------------------


class EmbeddingCache(ABC):
    """Persist and retrieve entity embeddings across ranks or runs."""

    @abstractmethod
    def push(self, global_ids: np.ndarray, embeddings: np.ndarray, rank: int) -> None:
        """Write this rank's embeddings to storage. Called by each rank."""

    @abstractmethod
    def barrier(self) -> None:
        """Synchronise — wait until all ranks have finished push."""

    @abstractmethod
    def fetch(self, global_ids: np.ndarray) -> np.ndarray:
        """Return (len(global_ids), D) float32 embeddings for requested IDs.

        Unknown IDs are returned as zero vectors.
        Typically called only on rank 0 after barrier().
        """


# ---------------------------------------------------------------------------
#  File-based backend (no external infrastructure)
# ---------------------------------------------------------------------------


class FileEmbeddingCache(EmbeddingCache):
    """
    Each rank writes:
        {cache_dir}/ids_{rank}.npy  — (N_rank,) int64 global entity IDs
        {cache_dir}/emb_{rank}.npy  — (N_rank, D) float32 embeddings

    fetch() loads all rank files on rank 0 and assembles a lookup table.
    """

    def __init__(self, cache_dir: str, world_size: int):
        self.cache_dir = cache_dir
        self.world_size = world_size
        self._id_to_row: dict[int, int] = {}
        self._all_emb: np.ndarray | None = None

    def push(self, global_ids: np.ndarray, embeddings: np.ndarray, rank: int) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        np.save(os.path.join(self.cache_dir, f"ids_{rank}.npy"), global_ids.astype(np.int64))
        np.save(
            os.path.join(self.cache_dir, f"emb_{rank}.npy"),
            embeddings.astype(np.float32),
        )
        logging.debug(
            f"[FileCache rank {rank}] pushed {len(global_ids)} embeddings " f"to {self.cache_dir}"
        )

    def barrier(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def fetch(self, global_ids: np.ndarray) -> np.ndarray:
        if self._all_emb is None:
            self._load_all()
        D = self._all_emb.shape[1] if self._all_emb.size else 0
        result = np.zeros((len(global_ids), D), dtype=np.float32)
        for i, gid in enumerate(global_ids):
            row = self._id_to_row.get(int(gid))
            if row is not None:
                result[i] = self._all_emb[row]
        return result

    def _load_all(self) -> None:
        ids_list, emb_list = [], []
        for r in range(self.world_size):
            ids_path = os.path.join(self.cache_dir, f"ids_{r}.npy")
            emb_path = os.path.join(self.cache_dir, f"emb_{r}.npy")
            ids_list.append(np.load(ids_path))
            emb_list.append(np.load(emb_path))
        all_ids = np.concatenate(ids_list)
        self._all_emb = np.concatenate(emb_list, axis=0).astype(np.float32)
        self._id_to_row = {int(gid): i for i, gid in enumerate(all_ids)}
        logging.info(f"[FileCache] loaded {len(all_ids)} embeddings from {self.cache_dir}")


# ---------------------------------------------------------------------------
#  gRPC backend (wraps LeapGNN's DistCacheClient — read-only protocol)
# ---------------------------------------------------------------------------


class GrpcEmbeddingCache(EmbeddingCache):
    """
    Embedding cache backed by LeapGNN's Go gRPC server.

    The Go server implements a **read-only** protocol — it loads .npy files at
    startup and has no PUT operation.  Therefore:
      - push() writes ids_{rank}.npy + emb_{rank}.npy to cache_dir (same as
        FileEmbeddingCache).  The Go server must be started *after* barrier().
      - fetch() connects to the running Go server via DistCacheClient.

    LeapGNN's ConstructNid2Pid() reads partition assignment files from
    partition_dir (the same dir used by DistGraphLoader: contains 0.npy … N.npy).
    This is separate from cache_dir which only holds the embedding shards.

    See cache_server/README.md for Go server startup instructions.
    """

    def __init__(
        self,
        cache_dir: str,
        world_size: int,
        grpc_addrs: list[str],
        gpu: int = 0,
        partition_dir: str | None = None,
    ):
        """
        Args:
            cache_dir:      directory where push() writes ids_{rank}.npy /
                            emb_{rank}.npy (the embedding shards).
            world_size:     number of ranks (= number of .npy shards).
            grpc_addrs:     list of "host:port" for Go cache server shards.
                            The first address is used for single-channel access.
            gpu:            GPU index passed to DistCacheClient.
            partition_dir:  directory containing 0.npy … (N-1).npy partition
                            files (output of partition.py).  Used by
                            ConstructNid2Pid() to build the Nid→Pid map.
                            If None, falls back to cache_dir (legacy behaviour,
                            but will fail unless partition files are also there).
        """
        self.cache_dir = cache_dir
        self.world_size = world_size
        self.grpc_addrs = grpc_addrs
        self.gpu = gpu
        self.partition_dir = partition_dir or cache_dir
        self._client: Any | None = None  # lazy-initialised DistCacheClient

    def push(self, global_ids: np.ndarray, embeddings: np.ndarray, rank: int) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        np.save(os.path.join(self.cache_dir, f"ids_{rank}.npy"), global_ids.astype(np.int64))
        np.save(
            os.path.join(self.cache_dir, f"emb_{rank}.npy"),
            embeddings.astype(np.float32),
        )
        logging.debug(
            f"[GrpcCache rank {rank}] wrote shard to {self.cache_dir} "
            f"(Go server must be started after all ranks finish push + barrier)"
        )

    def barrier(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def fetch(self, global_ids: np.ndarray) -> np.ndarray:
        client = self._get_client()
        raw = client.get_feats_from_server(global_ids.tolist())
        feat_dim = client.feat_dim
        return np.frombuffer(raw, dtype=np.float32).reshape(len(global_ids), feat_dim).copy()

    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        from leapgnn.storage.storage_dist import DistCacheClient

        addr = self.grpc_addrs[0] if self.grpc_addrs else "localhost:18112"
        logging.info(f"[GrpcCache] connecting to Go server at {addr}")
        client = DistCacheClient(grpc_port=addr, gpu=self.gpu, log=False)

        total_nodes = self._infer_total_nodes()
        # ConstructNid2Pid reads {rank}.npy from a path it reconstructs itself
        # as f"{datasetpath}/dist_True/{partition_num}_{partition_type}/" --
        # it expects a dataset ROOT, not the already-fully-qualified partition
        # dir DistEA's partition.py produces (e.g.
        # outputs/partitions/EN_DE_15K_V1/dist_True/2_metis_kg2locality).
        # Passing self.partition_dir directly double-appends dist_True/... and
        # 404s; reverse-engineer (dataset_root, partition_type) from it instead.
        dataset_root, partition_type = self._split_partition_dir()
        client.ConstructNid2Pid(dataset_root, self.world_size, partition_type, total_nodes)
        self._client = client
        return client

    def _split_partition_dir(self) -> tuple[str, str]:
        norm = os.path.normpath(self.partition_dir)
        parent, leaf = os.path.split(norm)
        dataset_root, dist_true = os.path.split(parent)
        if dist_true != "dist_True":
            raise ValueError(
                f"partition_dir {self.partition_dir!r} does not match the "
                f"'.../dist_True/{{N}}_{{type}}' layout ConstructNid2Pid requires"
            )
        prefix = f"{self.world_size}_"
        if not leaf.startswith(prefix):
            raise ValueError(
                f"partition_dir leaf {leaf!r} does not start with {prefix!r} "
                f"(world_size={self.world_size})"
            )
        return dataset_root, leaf[len(prefix):]

    def _infer_total_nodes(self) -> int:
        max_id = 0
        for r in range(self.world_size):
            ids = np.load(os.path.join(self.cache_dir, f"ids_{r}.npy"))
            if len(ids):
                max_id = max(max_id, int(ids.max()) + 1)
        return max_id


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------


def make_embedding_cache(
    backend: str,
    cache_dir: str,
    world_size: int,
    grpc_addrs: list[str] | None = None,
    gpu: int = 0,
    partition_dir: str | None = None,
) -> EmbeddingCache | None:
    """
    Create an EmbeddingCache from a string backend name.

    Args:
        backend:       'none' | 'file' | 'grpc'
        cache_dir:     directory for embedding .npy shards
        world_size:    number of distributed ranks
        grpc_addrs:    list of "host:port" (required for 'grpc' backend)
        gpu:           GPU index for gRPC data transfer
        partition_dir: directory with partition files 0.npy … (N-1).npy
                       (required for 'grpc' backend so ConstructNid2Pid
                       can build the Nid→server map; ignored for 'file')

    Returns:
        EmbeddingCache instance, or None if backend == 'none'.
    """
    backend = (backend or "none").lower()
    if backend == "none":
        return None
    if backend == "file":
        return FileEmbeddingCache(cache_dir=cache_dir, world_size=world_size)
    if backend == "grpc":
        return GrpcEmbeddingCache(
            cache_dir=cache_dir,
            world_size=world_size,
            grpc_addrs=grpc_addrs or [],
            gpu=gpu,
            partition_dir=partition_dir,
        )
    raise ValueError(f"Unknown cache backend {backend!r}. Choose: none, file, grpc")
