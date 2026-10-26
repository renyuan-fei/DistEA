"""
test_stage_s.py — Stage S tests: dist_halo_sync.py's fetch() additions and
make_halo_sync factory: halo refresh goes through the backend selected by
--cache_backend (none / file / grpc).

GrpcHaloSync's push()/fetch() I/O delegates to leapgnn_cache.GrpcEmbeddingCache,
which test_stage_m.py covers (including the end-to-end Go-server roundtrip
when bin/distea_server is built). This file covers fetch()'s dict-return
contract on Noop/File, that GrpcHaloSync wires its internal
GrpcEmbeddingCache with the right constructor args (no live server needed
for construction itself), and multi-node spawner selection.

DGL-free, no GPU, no live Go server required. <5 sec.
"""

import os
import sys
import tempfile
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dist_halo_sync import FileHaloSync, GrpcHaloSync, NoopHaloSync, make_halo_sync


def test_noop_halo_sync_fetch_is_always_empty():
    sync = NoopHaloSync()
    assert sync.fetch(np.array([1, 2, 3])) == {}
    print("  test_noop_halo_sync_fetch_is_always_empty PASSED")


def test_file_halo_sync_fetch_returns_pushed_values_omits_unknown():
    with tempfile.TemporaryDirectory() as tmp:
        sync = FileHaloSync(cache_dir=tmp, world_size=1)
        sync.push(np.array([10, 11]), np.array([[1.0, 2.0], [3.0, 4.0]]), rank=0)
        fetched = sync.fetch(np.array([10, 11, 999]))
        assert set(fetched.keys()) == {10, 11}, fetched
        assert np.allclose(fetched[10], [1.0, 2.0])
        assert np.allclose(fetched[11], [3.0, 4.0])
        print("  test_file_halo_sync_fetch_returns_pushed_values_omits_unknown PASSED")


def test_file_halo_sync_fetch_reflects_latest_push():
    """push() invalidates the loaded cache -- a second push (e.g. next
    iteration's fresher embeddings) must be visible on the next fetch,
    not a stale value from the first _load_all()."""
    with tempfile.TemporaryDirectory() as tmp:
        sync = FileHaloSync(cache_dir=tmp, world_size=1)
        sync.push(np.array([10]), np.array([[1.0]]), rank=0)
        assert np.allclose(sync.fetch(np.array([10]))[10], [1.0])
        sync.push(np.array([10]), np.array([[9.0]]), rank=0)
        assert np.allclose(sync.fetch(np.array([10]))[10], [9.0])
        print("  test_file_halo_sync_fetch_reflects_latest_push PASSED")


def test_grpc_halo_sync_wires_internal_embedding_cache_correctly():
    with tempfile.TemporaryDirectory() as tmp:
        sync = GrpcHaloSync(
            emb_dir=tmp,
            partition_dir=os.path.join(tmp, "outputs", "dist_True", "2_metis"),
            grpc_addrs=["127.0.0.1:18312", "127.0.0.1:18313"],
            world_size=2,
            rank=0,
        )
        assert sync._cache.cache_dir == tmp
        assert sync._cache.world_size == 2
        assert sync._cache.grpc_addrs == ["127.0.0.1:18312", "127.0.0.1:18313"]
        assert sync._cache.partition_dir == os.path.join(tmp, "outputs", "dist_True", "2_metis")
        print("  test_grpc_halo_sync_wires_internal_embedding_cache_correctly PASSED")


def test_make_halo_sync_selects_correct_backend():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_none = types.SimpleNamespace(cache_backend="none")
        assert isinstance(make_halo_sync(cfg_none, 0, 1, tmp), NoopHaloSync)

        cfg_file = types.SimpleNamespace(cache_backend="file", cache_dir=tmp)
        assert isinstance(make_halo_sync(cfg_file, 0, 1, tmp), FileHaloSync)

        cfg_grpc = types.SimpleNamespace(
            cache_backend="grpc", cache_dir=tmp,
            grpc_addrs="127.0.0.1:18312,127.0.0.1:18313",
        )
        sync = make_halo_sync(cfg_grpc, 0, 2, tmp)
        assert isinstance(sync, GrpcHaloSync)
        assert sync.grpc_addrs == ["127.0.0.1:18312", "127.0.0.1:18313"]
        print("  test_make_halo_sync_selects_correct_backend PASSED")


def test_make_halo_sync_grpc_without_addrs_falls_back_to_file():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = types.SimpleNamespace(cache_backend="grpc", cache_dir=tmp, grpc_addrs="")
        sync = make_halo_sync(cfg, 0, 1, tmp)
        assert isinstance(sync, FileHaloSync)
        print("  test_make_halo_sync_grpc_without_addrs_falls_back_to_file PASSED")


# ---------------------------------------------------------------------------
# Multi-physical-node spawner selection (node_ip/local_rank). subprocess.
# Popen from a single global rank can never start a process actually
# listening on a DIFFERENT physical node's address, so each node needs its
# own local spawner for its own slice of grpc_addrs -- these tests cover the
# selection logic in isolation (no live process spawning, no Go server).
# ---------------------------------------------------------------------------


def test_grpc_halo_sync_backward_compatible_when_node_ip_unset():
    """node_ip=None (the default) keeps single-node behaviour: only global
    rank 0 is the
    spawner, and it claims every grpc_addrs index -- not just a filtered
    subset."""
    with tempfile.TemporaryDirectory() as tmp:
        sync_rank0 = GrpcHaloSync(
            emb_dir=tmp, partition_dir=tmp,
            grpc_addrs=["10.0.0.1:18112", "10.0.0.2:18112"],
            world_size=2, rank=0,
        )
        assert sync_rank0._is_spawner() is True
        assert sync_rank0._my_server_indices() == [0, 1]

        sync_rank1 = GrpcHaloSync(
            emb_dir=tmp, partition_dir=tmp,
            grpc_addrs=["10.0.0.1:18112", "10.0.0.2:18112"],
            world_size=2, rank=1,
        )
        assert sync_rank1._is_spawner() is False
        print("  test_grpc_halo_sync_backward_compatible_when_node_ip_unset PASSED")


def test_grpc_halo_sync_multinode_spawner_selection():
    """With node_ip set: the spawner on each node is the process whose own
    local_rank == 0 (not global rank 0 -- node B's local rank 0 is global
    rank 4 in a 2-node/4-ranks-per-node layout), and it claims only the
    grpc_addrs entries whose host matches its OWN node's IP, regardless of
    global rank."""
    with tempfile.TemporaryDirectory() as tmp:
        addrs = [f"10.0.0.1:{18112+i}" for i in range(4)] + [f"10.0.0.2:{18112+i}" for i in range(4)]

        # Global rank 4 == node B's local rank 0 -- IS a spawner, and must
        # claim node B's slice (indices 4-7), not node A's (0-3) and not
        # "all of them" the way the backward-compat path would.
        node_b_spawner = GrpcHaloSync(
            emb_dir=tmp, partition_dir=tmp, grpc_addrs=addrs, world_size=8,
            rank=4, node_ip="10.0.0.2", local_rank=0,
        )
        assert node_b_spawner._is_spawner() is True
        assert node_b_spawner._my_server_indices() == [4, 5, 6, 7]

        # Global rank 0 == node A's local rank 0 -- claims node A's slice.
        node_a_spawner = GrpcHaloSync(
            emb_dir=tmp, partition_dir=tmp, grpc_addrs=addrs, world_size=8,
            rank=0, node_ip="10.0.0.1", local_rank=0,
        )
        assert node_a_spawner._is_spawner() is True
        assert node_a_spawner._my_server_indices() == [0, 1, 2, 3]

        # Global rank 5 (node B, but local_rank=1, not the node's spawner)
        # -- must NOT spawn anything, same as any non-zero local_rank.
        non_spawner = GrpcHaloSync(
            emb_dir=tmp, partition_dir=tmp, grpc_addrs=addrs, world_size=8,
            rank=5, node_ip="10.0.0.2", local_rank=1,
        )
        assert non_spawner._is_spawner() is False
        print("  test_grpc_halo_sync_multinode_spawner_selection PASSED")


def test_make_halo_sync_reads_node_ip_and_local_rank_from_env():
    # This suite runs as a plain script (python tests/test_stage_s.py), not
    # under pytest -- no monkeypatch fixture available.
    # Save/restore os.environ by hand instead.
    saved = {k: os.environ.get(k) for k in ("DISTEA_NODE_IP", "LOCAL_RANK")}
    try:
        os.environ["DISTEA_NODE_IP"] = "10.0.0.2"
        os.environ["LOCAL_RANK"] = "0"
        with tempfile.TemporaryDirectory() as tmp:
            cfg = types.SimpleNamespace(
                cache_backend="grpc", cache_dir=tmp,
                grpc_addrs="10.0.0.1:18112,10.0.0.2:18112",
            )
            sync = make_halo_sync(cfg, 1, 2, tmp)
            assert sync.node_ip == "10.0.0.2"
            assert sync.local_rank == 0
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("  test_make_halo_sync_reads_node_ip_and_local_rank_from_env PASSED")


def test_make_halo_sync_node_ip_absent_from_env_stays_none():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = types.SimpleNamespace(
            cache_backend="grpc", cache_dir=tmp,
            grpc_addrs="127.0.0.1:18112,127.0.0.1:18113",
        )
        sync = make_halo_sync(cfg, 0, 2, tmp)
        assert sync.node_ip is None
        print("  test_make_halo_sync_node_ip_absent_from_env_stays_none PASSED")


if __name__ == "__main__":
    print("Stage S tests:")
    test_noop_halo_sync_fetch_is_always_empty()
    test_file_halo_sync_fetch_returns_pushed_values_omits_unknown()
    test_file_halo_sync_fetch_reflects_latest_push()
    test_grpc_halo_sync_wires_internal_embedding_cache_correctly()
    test_make_halo_sync_selects_correct_backend()
    test_make_halo_sync_grpc_without_addrs_falls_back_to_file()
    test_grpc_halo_sync_backward_compatible_when_node_ip_unset()
    test_grpc_halo_sync_multinode_spawner_selection()
    test_make_halo_sync_reads_node_ip_and_local_rank_from_env()
    test_make_halo_sync_node_ip_absent_from_env_stays_none()
    print("All Stage S tests PASSED.")
