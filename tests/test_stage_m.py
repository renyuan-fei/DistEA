"""
test_stage_m.py — Stage M tests: Stage C embedding-cache push wiring
(framework.py::push_final_embeddings, main.py's --cache_backend/--grpc_addrs).

Checks that the final embeddings are pushed through leapgnn_cache.py's
EmbeddingCache for each --cache_backend and can be fetched back.

The grpc tests check three properties of the gRPC cache path:
  1. GrpcEmbeddingCache._split_partition_dir() splits partition_dir into
     (dataset_root, partition_type) so that ConstructNid2Pid(), which
     re-joins "{path}/dist_True/{N}_{type}/" itself, reconstructs the
     original directory -- for both "{N}_metis" and "{N}_metis_kg2locality".
  2/3. cache_server/services/distcache.go's routing table and
     static_cache.go's local cache read {emb_dir}/ids_{rank}.npy (the
     order-matched KG1+KG2 id list push() writes) rather than
     {partition_dir}/{rank}.npy (KG1 entities only), so KG2 entities are
     routed and stored too.

DGL-free, no GPU required, <10 sec for everything except
test_grpc_roundtrip_via_real_go_server (~5-10s, skipped when bin/distea_server
hasn't been built -- see `make go-build`).
"""

import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from framework import DistEAFramework, push_final_embeddings

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
SERVER_BIN = os.path.join(REPO_ROOT, "bin", "distea_server")
SERVER_CONFIG = os.path.join(REPO_ROOT, "cache_server", "conf", "distea.yaml")


def test_push_final_embeddings_none_backend_is_noop():
    n = push_final_embeddings(
        "none", "", "", 1, 0,
        np.array([0, 1]), np.array([0]), np.array([1]),
        torch.zeros(2, 4),
    )
    assert n == 0


def test_push_final_embeddings_file_backend_writes_ids_and_emb():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = os.path.join(tmp, "cache")
        all_global_ids = np.array([100, 101, 102, 103])
        local_kg1_idx = np.array([0, 1])
        local_kg2_idx = np.array([2, 3])
        h_init_for_eval = torch.arange(16, dtype=torch.float32).reshape(4, 4)

        n = push_final_embeddings(
            "file", cache_dir, "", 1, 0,
            all_global_ids, local_kg1_idx, local_kg2_idx, h_init_for_eval,
        )
        assert n == 4

        ids = np.load(os.path.join(cache_dir, "ids_0.npy"))
        emb = np.load(os.path.join(cache_dir, "emb_0.npy"))
        # local_kg1 (100,101) then local_kg2 (102,103) -- push order matters
        # for downstream fetch() row alignment.
        assert list(ids) == [100, 101, 102, 103]
        assert emb.shape == (4, 4)
        assert np.allclose(emb[0], h_init_for_eval[0].numpy())
        assert np.allclose(emb[3], h_init_for_eval[3].numpy())
        print("  test_push_final_embeddings_file_backend_writes_ids_and_emb PASSED")


def test_push_final_embeddings_fetch_roundtrip_via_file_cache():
    """End-to-end: push via 'file' backend, then fetch() through the same
    EmbeddingCache abstraction returns the pushed values back correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = os.path.join(tmp, "cache")
        all_global_ids = np.array([100, 101, 102, 103])
        local_kg1_idx = np.array([0, 1])
        local_kg2_idx = np.array([2, 3])
        h_init_for_eval = torch.arange(16, dtype=torch.float32).reshape(4, 4)

        push_final_embeddings(
            "file", cache_dir, "", 1, 0,
            all_global_ids, local_kg1_idx, local_kg2_idx, h_init_for_eval,
        )

        from leapgnn_cache import make_embedding_cache

        cache = make_embedding_cache("file", cache_dir, world_size=1)
        fetched = cache.fetch(np.array([102, 100, 999]))  # 999 = unknown id
        assert np.allclose(fetched[0], h_init_for_eval[2].numpy())
        assert np.allclose(fetched[1], h_init_for_eval[0].numpy())
        assert np.allclose(fetched[2], np.zeros(4))  # unknown -> zero vector
        print("  test_push_final_embeddings_fetch_roundtrip_via_file_cache PASSED")


def test_cache_backend_requires_cache_dir():
    """--cache_backend != 'none' without --cache_dir must fail fast at
    construction time (framework.py's DistEAFramework.__init__), matching
    the project's existing fail-fast-on-flag-incompatibility convention
    (see the --reassign_kg2_every + --persist_optimizer guard)."""
    import types
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
    from test_stage_j import make_dataset

    with tempfile.TemporaryDirectory() as tmp:
        from dataload import OpenEAData

        data_dir, fold_dir, output_dir = make_dataset(tmp)
        data = OpenEAData(data_dir, fold_dir, output_dir)
        cfg = types.SimpleNamespace(
            output_dim=8, hidden_dim=None,
            cache_backend="file", cache_dir="",
        )
        try:
            DistEAFramework(0, 2, data, output_dir, cfg, torch.device("cpu"))
        except ValueError as e:
            assert "cache_backend" in str(e)
            print("  test_cache_backend_requires_cache_dir PASSED")
            return
        raise AssertionError("expected ValueError for --cache_backend without --cache_dir")


def test_grpc_split_partition_dir_reverses_construct_nid2pid_join():
    """_split_partition_dir() must produce (dataset_root, partition_type) such
    that os.path.join(dataset_root, f"dist_True/{N}_{partition_type}/")
    reconstructs partition_dir exactly -- this is the exact join
    ConstructNid2Pid() performs internally, so partition_dir must be
    pre-split rather than passed straight through."""
    from leapgnn_cache import GrpcEmbeddingCache

    cache = GrpcEmbeddingCache(
        cache_dir="x", world_size=2, grpc_addrs=[],
        partition_dir="outputs/partitions/EN_DE_15K_V1/dist_True/2_metis_kg2locality",
    )
    dataset_root, partition_type = cache._split_partition_dir()
    assert dataset_root == "outputs/partitions/EN_DE_15K_V1"
    assert partition_type == "metis_kg2locality"
    rejoined = os.path.join(dataset_root, f"dist_True/{2}_{partition_type}/")
    assert os.path.normpath(rejoined) == os.path.normpath(cache.partition_dir)
    print("  test_grpc_split_partition_dir_reverses_construct_nid2pid_join PASSED")


def test_grpc_split_partition_dir_plain_metis_suffix():
    """Same logic for a partition_dir with no --kg2_assignment suffix at
    all (plain 'metis'), to make sure the fix isn't accidentally coupled to
    the locality-assignment naming specifically."""
    from leapgnn_cache import GrpcEmbeddingCache

    cache = GrpcEmbeddingCache(
        cache_dir="x", world_size=4, grpc_addrs=[],
        partition_dir="outputs/partitions/EN_DE_15K_V1/dist_True/4_metis",
    )
    dataset_root, partition_type = cache._split_partition_dir()
    assert dataset_root == "outputs/partitions/EN_DE_15K_V1"
    assert partition_type == "metis"
    print("  test_grpc_split_partition_dir_plain_metis_suffix PASSED")


def test_grpc_split_partition_dir_rejects_missing_dist_true():
    from leapgnn_cache import GrpcEmbeddingCache

    cache = GrpcEmbeddingCache(
        cache_dir="x", world_size=2, grpc_addrs=[],
        partition_dir="outputs/partitions/EN_DE_15K_V1/2_metis",
    )
    try:
        cache._split_partition_dir()
    except ValueError as e:
        assert "dist_True" in str(e)
        print("  test_grpc_split_partition_dir_rejects_missing_dist_true PASSED")
        return
    raise AssertionError("expected ValueError for missing dist_True segment")


def test_grpc_split_partition_dir_rejects_world_size_mismatch():
    """world_size passed to the cache must match the {N}_ prefix actually in
    the directory name -- e.g. constructing with world_size=4 but pointing at
    a 2_metis directory (copy-paste error) must fail loudly, not silently
    derive the wrong partition_type."""
    from leapgnn_cache import GrpcEmbeddingCache

    cache = GrpcEmbeddingCache(
        cache_dir="x", world_size=4, grpc_addrs=[],
        partition_dir="outputs/partitions/EN_DE_15K_V1/dist_True/2_metis",
    )
    try:
        cache._split_partition_dir()
    except ValueError as e:
        assert "world_size" in str(e)
        print("  test_grpc_split_partition_dir_rejects_world_size_mismatch PASSED")
        return
    raise AssertionError("expected ValueError for world_size/prefix mismatch")


def test_grpc_roundtrip_via_real_go_server():
    """End-to-end check of properties 1-3 (see module docstring): push via
    the 'file' backend (byte-identical format to what GrpcEmbeddingCache.push
    writes), start two REAL bin/distea_server processes on localhost
    (simulating 2 ranks on one node), then fetch() through the actual
    DistCacheClient/gRPC path and compare against ground truth. Covers a KG1
    id, a KG2 id (properties 2/3 -- KG2 embeddings must not be dropped), a
    cross-rank id (peer-forwarding), and an unknown id
    (zero-vector fallback).

    Skips gracefully if bin/distea_server hasn't been built (`make go-build`)
    or isn't runnable on this machine -- e.g. a Linux binary copied to a
    macOS checkout exists but cannot run; a plain os.path.isfile() check
    doesn't catch that, only an actual exec attempt does.
    """
    if not os.path.isfile(SERVER_BIN):
        print("  test_grpc_roundtrip_via_real_go_server SKIPPED (bin/distea_server not built -- run `make go-build`)")
        return
    try:
        subprocess.run([SERVER_BIN, "--help"], capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"  test_grpc_roundtrip_via_real_go_server SKIPPED (bin/distea_server not executable on this platform: {e})")
        return

    from leapgnn_cache import GrpcEmbeddingCache

    with tempfile.TemporaryDirectory() as tmp:
        emb_dir = os.path.join(tmp, "emb")
        partition_dir = os.path.join(tmp, "partitions", "dist_True", "2_metis")
        os.makedirs(partition_dir, exist_ok=True)

        # Rank 0: 2 KG1 entities (0, 1) + 2 KG2 entities (10, 11).
        # Rank 1: 2 KG1 entities (2, 3) + 2 KG2 entities (12, 13).
        # partition.py's own {rank}.npy lists KG1 IDs ONLY -- deliberately
        # shorter than ids_{rank}.npy (KG1+KG2), so a positional-zip mismatch
        # would show up if the Go server regresses to reading
        # {partition_dir}/{rank}.npy instead of {emb_dir}/ids_{rank}.npy.
        np.save(os.path.join(partition_dir, "0.npy"), np.array([0, 1], dtype=np.int64))
        np.save(os.path.join(partition_dir, "1.npy"), np.array([2, 3], dtype=np.int64))

        dim = 4
        rng = np.random.RandomState(0)
        embeddings = {
            0: (np.array([0, 1, 10, 11]), rng.rand(4, dim).astype(np.float32)),
            1: (np.array([2, 3, 12, 13]), rng.rand(4, dim).astype(np.float32)),
        }
        os.makedirs(emb_dir, exist_ok=True)
        for rank, (ids, emb) in embeddings.items():
            np.save(os.path.join(emb_dir, f"ids_{rank}.npy"), ids)
            np.save(os.path.join(emb_dir, f"emb_{rank}.npy"), emb)

        cachegroup = "127.0.0.1:18312,127.0.0.1:18313"
        procs = []
        logs = []
        try:
            for rank in (0, 1):
                port = 18312 + rank
                ready_file = os.path.join(emb_dir, f"ready_{rank}.txt")
                log_path = os.path.join(emb_dir, f"server_{rank}.log")
                logs.append(log_path)
                log_fh = open(log_path, "w")
                proc = subprocess.Popen(
                    [
                        SERVER_BIN,
                        "--config", SERVER_CONFIG,
                        "--partition_dir", partition_dir,
                        "--emb_dir", emb_dir,
                        "--cachegroup", cachegroup,
                        "--rpcport", str(port),
                        "--distea_rank", str(rank),
                        "--bind_addr", "0.0.0.0",
                        "--ready_file", ready_file,
                    ],
                    stdout=log_fh, stderr=subprocess.STDOUT,
                )
                procs.append((proc, log_fh))

            deadline = time.time() + 15
            ready_files = [os.path.join(emb_dir, f"ready_{r}.txt") for r in (0, 1)]
            while time.time() < deadline:
                if all(os.path.isfile(f) for f in ready_files):
                    break
                time.sleep(0.5)
            else:
                dumps = []
                for lp in logs:
                    with open(lp) as f:
                        dumps.append(f"--- {lp} ---\n{f.read()}")
                raise AssertionError("servers did not become ready in time\n" + "\n".join(dumps))

            cache = GrpcEmbeddingCache(
                cache_dir=emb_dir, world_size=2,
                grpc_addrs=["127.0.0.1:18312", "127.0.0.1:18313"],
                gpu=-1, partition_dir=partition_dir,
            )
            query_ids = np.array([0, 12, 3, 99999999])  # KG1(rank0), KG2(rank1, cross-rank), KG1(rank1), unknown
            fetched = cache.fetch(query_ids)

            expected = np.zeros((4, dim), dtype=np.float32)
            expected[0] = embeddings[0][1][0]  # id 0 -> rank0 row 0
            expected[1] = embeddings[1][1][2]  # id 12 -> rank1 row 2 (a KG2 entity)
            expected[2] = embeddings[1][1][1]  # id 3 -> rank1 row 1
            # expected[3] stays zero (unknown id)

            assert np.allclose(fetched, expected, atol=1e-5), (
                f"fetched:\n{fetched}\nexpected:\n{expected}"
            )
            print("  test_grpc_roundtrip_via_real_go_server PASSED")
        finally:
            for proc, _log_fh in procs:
                proc.terminate()
            for proc, log_fh in procs:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                log_fh.close()


if __name__ == "__main__":
    print("Stage M tests:")
    test_push_final_embeddings_none_backend_is_noop()
    print("  test_push_final_embeddings_none_backend_is_noop PASSED")
    test_push_final_embeddings_file_backend_writes_ids_and_emb()
    test_push_final_embeddings_fetch_roundtrip_via_file_cache()
    test_cache_backend_requires_cache_dir()
    test_grpc_split_partition_dir_reverses_construct_nid2pid_join()
    test_grpc_split_partition_dir_plain_metis_suffix()
    test_grpc_split_partition_dir_rejects_missing_dist_true()
    test_grpc_split_partition_dir_rejects_world_size_mismatch()
    test_grpc_roundtrip_via_real_go_server()
    print("All Stage M tests PASSED.")
