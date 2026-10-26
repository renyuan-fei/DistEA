"""
test_eval_shard.py -- sharded chunked evaluation (DISTEA_EVAL_SHARD=1) over
3 gloo ranks gives the unsharded numbers.

1. _chunked_metrics(shard=(rank, 3)) vs the same call unsharded on one
   rank: Hit@k and the per-group Hit@k / n identical, MRR within 1e-9 (only
   the float64 summation order differs); CSLS on and off; a chunk size that
   leaves ranks with uneven (and one rank with no) Pass-A rows.
2. selective_eval end to end: entities spread over the 3 ranks, the dense
   limit forced to 0 so the chunked path runs; DISTEA_EVAL_SHARD=1 and =0
   return the same metrics on rank 0 and None elsewhere.
3. _shard_range: contiguous, chunk-aligned, covers [0, n) exactly once.
4. pseudo-seed mining: sharded _chunked_mutual_nn returns the same row
   argmax, column argmax (ties -> earliest row, via duplicated rows) and row
   scores; generate_pseudo_seeds with DISTEA_EVAL_SHARD=1 and =0 broadcasts
   the same seeds (with and without top_frac).

Run (gloo; binds a loopback port):  .venv/bin/python tests/test_eval_shard.py
"""
from __future__ import annotations

import os
import socket
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

WS = 3
N1, N2, D = 50, 40, 8


class _Identity:
    def eval(self):
        return self

    def __call__(self, g, h):
        return h


def _data():
    g = torch.Generator().manual_seed(7)
    base = torch.randn(N2, D, generator=g)
    e1 = torch.cat([base + 0.6 * torch.randn(N2, D, generator=g), torch.randn(N1 - N2, D, generator=g)])
    e2 = base + 0.6 * torch.randn(N2, D, generator=g)
    links = [(i, i) for i in range(N2)] + [(N2 + i, i) for i in range(5)]  # 45 pairs
    groups = ["same_rank" if i % 3 else "cross_rank" for i in range(len(links))]
    return e1, e2, links, groups


def _close(a: dict, b: dict):
    assert a.keys() == b.keys(), (sorted(a), sorted(b))
    for k in a:
        if k.endswith("MRR"):
            assert abs(a[k] - b[k]) < 1e-9, (k, a[k], b[k])
        else:
            assert a[k] == b[k], (k, a[k], b[k])


def _worker(rank, port, out):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=WS)
    cpu = torch.device("cpu")
    try:
        from eval import _chunked_metrics, selective_eval

        e1, e2, links, groups = _data()
        for csls in (True, False):
            for chunk in (4, 7, 64):
                ref = _chunked_metrics("cosine", e1, e2, links, cpu, use_csls=csls, csls_k=3,
                                       top_k=(1, 5), chunk=chunk, groups=groups)
                got = _chunked_metrics("cosine", e1, e2, links, cpu, use_csls=csls, csls_k=3,
                                       top_k=(1, 5), chunk=chunk, groups=groups, shard=(rank, WS))
                _close(ref, got)

        # selective_eval: global ids 0..N1-1 are KG1, N1..N1+N2-1 KG2; rank r owns ids % WS == r
        h = torch.cat([e1, e2])
        gids = np.arange(N1 + N2)
        mine = gids[gids % WS == rank]
        test_links = [(a, N1 + b) for a, b in links]
        os.environ["DISTEA_EVAL_DENSE_LIMIT_GB"] = "0"
        res = {}
        for flag in ("0", "1"):
            os.environ["DISTEA_EVAL_SHARD"] = flag
            res[flag] = selective_eval(
                _Identity(), None, h[mine], mine, test_links, rank, WS, cpu,
                use_csls=True, csls_k=3, top_k=(1, 5, 10), metric="cosine",
                owned_local=np.arange(len(mine)),
                link_groups=groups,
            )
        if rank == 0:
            _close(res["0"], res["1"])
            assert res["1"]["Hit@1"] > 0.2, res["1"]
            out.put(res["1"])
        else:
            assert res["0"] is None and res["1"] is None, res

        # mining: duplicated KG1 rows create exact column ties
        from eval import _chunked_mutual_nn, generate_pseudo_seeds

        e1d = torch.cat([e1, e1[:6]])
        for csls in (True, False):
            for chunk in (4, 7):
                ref = _chunked_mutual_nn("cosine", e1d, e2, cpu, use_csls=csls, csls_k=3,
                                         chunk=chunk, return_row_best=True)
                got = _chunked_mutual_nn("cosine", e1d, e2, cpu, use_csls=csls, csls_k=3,
                                         chunk=chunk, return_row_best=True, shard=(rank, WS))
                assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1]), (csls, chunk)
                assert torch.equal(ref[2], got[2]), (csls, chunk)
        seeds = {}
        for flag in ("0", "1"):
            os.environ["DISTEA_EVAL_SHARD"] = flag
            for tf in (1.0, 0.5):
                seeds[(flag, tf)] = generate_pseudo_seeds(
                    _Identity(), None, h[mine], mine,
                    np.nonzero(mine < N1)[0], np.nonzero(mine >= N1)[0],
                    rank, WS, cpu, metric="cosine", use_csls=True, csls_k=3,
                    sim_on_device=True, top_frac=tf,
                )
        for tf in (1.0, 0.5):
            assert seeds[("0", tf)] == seeds[("1", tf)], tf
        assert len(seeds[("1", 1.0)]) > 10 and len(seeds[("1", 0.5)]) < len(seeds[("1", 1.0)])
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_shard_range():
    from eval import _shard_range

    for n in (0, 1, 7, 45, 512, 1000):
        for chunk in (1, 4, 512):
            for ws in (1, 2, 3, 8):
                spans = [_shard_range(n, chunk, r, ws) for r in range(ws)]
                assert spans[0][0] == 0 and spans[-1][1] == n, (n, chunk, ws, spans)
                for (_a0, a1), (b0, _b1) in zip(spans, spans[1:], strict=False):
                    assert a1 == b0, spans
                for lo, hi in spans:
                    assert lo <= hi and lo % chunk == 0 and (hi % chunk == 0 or hi == n), spans
    print("  [ok] _shard_range: chunk-aligned, contiguous, complete")


def main():
    test_shard_range()
    q = mp.get_context("spawn").SimpleQueue()
    mp.spawn(_worker, args=(_free_port(), q), nprocs=WS, join=True)
    print(f"  [ok] sharded _chunked_metrics and selective_eval match the unsharded result: {q.get()}")
    print("  [ok] sharded mutual-NN (incl. column ties) and generate_pseudo_seeds match the unsharded result")
    print("test_eval_shard: ALL PASSED")


if __name__ == "__main__":
    main()
