"""
test_nhsm_compact.py — --nhsm_negs compact: NHSM on the
compact table (uniform pool + pair rows) instead of the whole (N, width) view.

1. Pool = every row -> the compact NHSM value and its gradient w.r.t. the
   batch embeddings equal the full-table NHSM (column order is irrelevant to
   the per-row mean/std/logsumexp; allclose).
2. A smaller pool still returns a finite loss with gradient flowing into
   the batch rows, and the compact table has exactly |pool ∪ pair rows|
   rows.
3. GAS epoch with --loss nhsm: nhsm_negs="full" is unchanged by the new
   kwargs (bit-identical to the plain call), "compact" runs finite.

Run (gloo; binds a loopback port):  .venv/bin/python tests/test_nhsm_compact.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from test_gas_softmax_history import N_HUB, _hub_graph, _model_and_h, _smh_history  # noqa: E402

from gas import (  # noqa: E402
    _compact_loss_rows,
    build_gas_clusters,
    gas_forward_batch,
    gas_train_epoch,
)
from models.gcn import nhsm_loss  # noqa: E402

CPU = torch.device("cpu")


def _case(seed=0, n=60, width=12, kg1=30, n_batch=15, p=9):
    gen = torch.Generator().manual_seed(seed)
    final_hist = torch.randn(n, width, generator=gen)
    batch = torch.sort(torch.randperm(n, generator=gen)[:n_batch]).values
    emb_batch = torch.randn(n_batch, width, generator=gen)
    in_batch = torch.zeros(n, dtype=torch.bool)
    in_batch[batch] = True
    kg1_idx = torch.randperm(kg1, generator=gen)[:p]
    kg2_idx = torch.randperm(n - kg1, generator=gen)[:p] + kg1
    pairs = torch.stack([kg1_idx, kg2_idx], dim=1)
    return final_hist, batch, emb_batch, in_batch, pairs, kg1


def test_pool_all_rows_equals_full():
    final_hist, batch, emb_batch0, in_batch, pairs, kg1 = _case()
    n = final_hist.size(0)
    # full
    eb_full = emb_batch0.clone().requires_grad_(True)
    emb_full = final_hist.detach().index_copy(0, batch, eb_full)
    l_full = nhsm_loss(emb_full, pairs, gamma=1.0)
    l_full.backward()
    # compact with the whole table as pool
    eb_c = emb_batch0.clone().requires_grad_(True)
    pool = torch.arange(n)
    rows, pairs_c, _, _, _ = _compact_loss_rows(final_hist, eb_c, batch, in_batch, pairs, pool, pool[:0], kg1)
    assert rows.size(0) == n
    l_c = nhsm_loss(rows, pairs_c, gamma=1.0)
    l_c.backward()
    assert torch.allclose(l_full, l_c, rtol=1e-5, atol=1e-6), (l_full.item(), l_c.item())
    assert torch.allclose(eb_full.grad, eb_c.grad, rtol=1e-4, atol=1e-6)
    print(f"  [ok] pool = all rows: compact NHSM == full NHSM ({l_full.item():.5f}), gradients match")


def test_smaller_pool():
    final_hist, batch, emb_batch0, in_batch, pairs, kg1 = _case(seed=1)
    eb = emb_batch0.clone().requires_grad_(True)
    pool = torch.tensor([0, 1, 2, 3, 40, 41, 42, 43, 44, 1, 2])  # duplicates allowed
    rows, pairs_c, _, _, _ = _compact_loss_rows(final_hist, eb, batch, in_batch, pairs, pool, pool[:0], kg1)
    expect = torch.unique(torch.cat([pairs.reshape(-1), pool])).numel()
    assert rows.size(0) == expect
    loss = nhsm_loss(rows, pairs_c, gamma=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert eb.grad is not None and float(eb.grad.abs().sum()) > 0
    print(f"  [ok] small pool: {rows.size(0)} compact rows, finite loss {loss.item():.4f}, grad flows")


def _setup(seed=0):
    from dist_trainer import DistTrainer

    torch.manual_seed(seed)
    g = _hub_graph()
    model, h0 = _model_and_h(g, N=N_HUB)
    model.train()
    h = torch.nn.Parameter(h0.clone())
    opt = torch.optim.Adam(list(model.parameters()) + [h], lr=1e-2)
    trainer = DistTrainer(model, opt, rank=0, world_size=1, neg_k=1, margin=6.0,
                          loss_fn="nhsm", nhsm_gamma=1.0)
    hist = _smh_history(g, model)
    warm = build_gas_clusters(g, 1)
    with torch.no_grad():
        gas_forward_batch(model, warm[0], h, hist, CPU)
    cl = build_gas_clusters(g, 4, method="bfs", softmax_history=True)
    final = torch.zeros(N_HUB, model.output_width)
    return trainer, cl, hist, final, h


def test_gas_epoch_nhsm():
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29629")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    pairs = torch.tensor([[1, 31], [2, 32], [5, 35], [7, 40], [9, 44]])
    res = []
    for kw in ({}, {"nhsm_negs": "full", "nhsm_pool": 7}):
        trainer, cl, hist, final, h = _setup()
        torch.manual_seed(3)
        loss = gas_train_epoch(trainer, cl, hist, final, h, pairs, 30, CPU, **kw)
        res.append((loss, final.clone()))
    assert abs(res[0][0] - res[1][0]) < 1e-12 and torch.equal(res[0][1], res[1][1])
    trainer, cl, hist, final, h = _setup()
    torch.manual_seed(3)
    loss_c = gas_train_epoch(trainer, cl, hist, final, h, pairs, 30, CPU, nhsm_negs="compact", nhsm_pool=12)
    assert np.isfinite(loss_c), loss_c
    print(f"  [ok] GAS nhsm: full unchanged ({res[0][0]:.4f}); compact pool 12 finite ({loss_c:.4f})")


if __name__ == "__main__":
    test_pool_all_rows_equals_full()
    test_smaller_pool()
    test_gas_epoch_nhsm()
    print("test_nhsm_compact: ALL PASSED")
