"""
test_table_consensus.py -- table_consensus.TableConsensus over 3 gloo ranks.

Rank 0 owns {0,1,2} and holds halo {5}; rank 1 owns {3,4,5}, halo {2};
rank 2 owns {6}, halo {2,5}. Shared rows: 2 and 5, three copies each.
Every copy starts at base[id] = id; rank r then adds (r+1)*(id+1)*0.01.
  mean : shared rows -> base + 2*(id+1)*0.01 on every rank (mean of 1,2,3)
  delta: shared rows -> base + 6*(id+1)*0.01 on every rank (sum of 1,2,3)
  rows held by one rank are untouched.
A second delta sync with no local change since must be a no-op.
  delta_dc: copies are f*base + change_r with f = 0.9 (decay ran on every
           copy); the merge must give f*base + sum of changes, i.e. the
           decay counted once: base*0.9 + 6*(id+1)*0.01.
adamw_decay_factor: (1 - lr*wd)**steps from a real AdamW, 1.0 for Adam.

Run (gloo; binds a loopback port):  .venv/bin/python tests/test_table_consensus.py
"""
from __future__ import annotations

import os
import socket
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

IDS = {0: [0, 1, 2, 5], 1: [3, 4, 5, 2], 2: [6, 2, 5]}
D = 4


F_DC = 0.9


def _table(rank, mode="mean"):
    ids = IDS[rank]
    base = torch.tensor([[float(i)] * D for i in ids])
    delta = torch.tensor([[(rank + 1) * (i + 1) * 0.01] * D for i in ids])
    if mode == "delta_dc":
        return base, F_DC * base + delta
    return base, base + delta


def _worker(rank, port, mode):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=3)
    try:
        from table_consensus import TableConsensus

        base, trained = _table(rank, mode)
        tc = TableConsensus(IDS[rank], "cpu", mode=mode)
        assert tc.n_shared == 2, tc.n_shared
        assert torch.equal(tc.counts, torch.tensor([3.0, 3.0])), tc.counts
        tab = base.clone()
        tc.sync(tab)  # copies equal: no-op, sets the delta base
        assert torch.allclose(tab, base)
        tab = trained.clone()
        tc.sync(tab, decay_factor=F_DC if mode == "delta_dc" else 1.0)
        factor = 2.0 if mode == "mean" else 6.0
        scale = F_DC if mode == "delta_dc" else 1.0
        for row, gid in enumerate(IDS[rank]):
            if gid in (2, 5):
                want = torch.full((D,), scale * gid + factor * (gid + 1) * 0.01)
            else:
                want = trained[row]
            assert torch.allclose(tab[row], want, atol=1e-6), (mode, rank, gid, tab[row], want)
        if mode in ("delta", "delta_dc"):
            again = tab.clone()
            tc.sync(again)
            assert torch.allclose(again, tab, atol=1e-6), "delta sync without change must be a no-op"
        s = tc.summary()
        assert s["syncs"] == (3 if mode in ("delta", "delta_dc") else 2) and s["shared_rows"] == 2
    finally:
        dist.destroy_process_group()


def _worker_dup(rank, port):
    """Rank 0 holds global id 1 in two rows; rank 1 holds it once: 3 copies."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        from table_consensus import TableConsensus

        ids = {0: [0, 1, 1], 1: [1, 2]}[rank]
        tc = TableConsensus(ids, "cpu", mode="mean")
        assert tc.n_shared == 1 and torch.equal(tc.counts, torch.tensor([3.0])), (tc.n_shared, tc.counts)
        assert tc.in_rank_duplicates == (1 if rank == 0 else 0)
        tab = torch.tensor([[float(i) + (rank + 1) * 0.3] * D for i in ids])
        tc.sync(tab)
        want = 1.0 + (0.3 + 0.3 + 0.6) / 3
        for row, gid in enumerate(ids):
            if gid == 1:
                assert torch.allclose(tab[row], torch.full((D,), want), atol=1e-6), (rank, row, tab[row])
            else:
                assert torch.allclose(tab[row], torch.full((D,), gid + (rank + 1) * 0.3)), (rank, row)
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_adamw_decay_factor():
    from table_consensus import adamw_decay_factor

    w = torch.nn.Parameter(torch.ones(3, 2))
    opt = torch.optim.AdamW([w], lr=0.1, weight_decay=0.5)
    f0, s0 = adamw_decay_factor(opt, w, 0.0)
    assert f0 == 1.0 and s0 == 0.0
    for _ in range(3):
        opt.zero_grad()
        (w ** 2).sum().backward()
        opt.step()
    f, s3 = adamw_decay_factor(opt, w, s0)
    assert s3 == 3.0 and abs(f - (1 - 0.05) ** 3) < 1e-12, (f, s3)
    f1, _ = adamw_decay_factor(opt, w, 2.0)
    assert abs(f1 - 0.95) < 1e-12
    a = torch.nn.Parameter(torch.ones(2))
    adam = torch.optim.Adam([a], lr=0.1)
    adam.zero_grad()
    a.sum().backward()
    adam.step()
    assert adamw_decay_factor(adam, a, 0.0)[0] == 1.0
    print("  [ok] adamw_decay_factor")


def main():
    test_adamw_decay_factor()
    for mode in ("mean", "delta", "delta_dc"):
        mp.spawn(_worker, args=(_free_port(), mode), nprocs=3, join=True)
        print(f"  [ok] {mode}: shared rows merged over 3 copies, private rows untouched")
    mp.spawn(_worker_dup, args=(_free_port(),), nprocs=2, join=True)
    print("  [ok] in-rank duplicate rows are merged as extra copies")
    print("test_table_consensus: ALL PASSED")


if __name__ == "__main__":
    main()
