"""
test_gas_closure_cap.py — --gas_closure_cap and
--gas_history_on_device (gas.py build_gas_clusters(closure_cap=...),
_cap_closure, _sparse_softmax_by_dst_weighted, GASHistory(device=...)).

The exact GAS closure is dominated by hub destinations whose complete
in-edge set lands in every cluster (DBP1M/N=4: one cluster = 3.8M of 7.04M
local edges, re-processed K times per epoch). closure_cap keeps every
batch-src edge (the only ones that deliver messages) and at most `cap`
other in-edges per dst, weighted n_other/n_kept so the softmax denominator
is an unbiased estimate. This file checks:

1. cap=0 and cap >= max in-degree are IDENTICAL to the exact path (same
   edge/needed tensors, e_w None / all-ones) and the multi-cluster forward
   with exact history still matches RREA.forward() (test_stage_w's gate).
2. On a hub graph a small cap shrinks every cluster's closure, keeps every
   batch-src edge, weights the hub's kept others by n_other/n_kept, and the
   forward/backward still run with grads into the tables.
3. The weighted denominator is unbiased (mean over 3000 draws within 3%),
   and the weighted softmax with all-ones weights equals the plain one.
4. DISTEA_GAS_CHECKPOINT=1 reproduces the un-checkpointed capped forward
   and gradients (e_w passes through torch.utils.checkpoint).
5. GASHistory(device="cpu") round-trips like the default.

Run:  .venv/bin/python tests/test_gas_closure_cap.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from test_stage_w import _model_and_h  # noqa: E402

from gas import (  # noqa: E402
    GASHistory,
    _cap_closure,
    _sparse_softmax_by_dst_weighted,
    build_gas_clusters,
    gas_forward_batch,
)
from models.rrea import RREAGraph, _sparse_softmax_by_dst  # noqa: E402

CPU = torch.device("cpu")
N_HUB = 60
HUB = 0


def _graph_from_edges(s: torch.Tensor, d: torch.Tensor, N: int, R: int, seed: int = 0):
    torch.manual_seed(seed)
    E = s.size(0)
    edge_index = torch.stack([s, d], dim=1)
    edge_rel = torch.randint(0, 2 * R, (E,))
    eye = torch.arange(N)
    a_hat = torch.sparse_coo_tensor(
        torch.stack([torch.cat([s, eye]), torch.cat([d, eye])]),
        torch.ones(E + N), (N, N)).coalesce()
    deg = torch.sparse.sum(a_hat, dim=1).to_dense().clamp(min=1e-8)
    dis = deg.pow(-0.5)
    idx = a_hat.indices()
    vals = a_hat.values() * dis[idx[0]] * dis[idx[1]]
    adj = torch.sparse_coo_tensor(idx, vals, (N, N)).coalesce()
    rel_prof = torch.sparse_coo_tensor(
        torch.stack([s, edge_rel]), torch.full((E,), 0.2), (N, 2 * R)).coalesce()
    return RREAGraph(adj, rel_prof, edge_index, edge_rel, torch.ones(E), R)


def _hub_graph(N: int = N_HUB, R: int = 5, hub: int = HUB, seed: int = 0):
    """Ring + every node points at one hub (in-degree N-1) + random extras:
    the DBP1M pattern in miniature (a type/country entity that every
    cluster has a src pointing at, so its whole in-edge set is in every
    exact closure)."""
    gen = torch.Generator().manual_seed(seed)
    ring_s = torch.arange(N)
    ring_d = (ring_s + 1) % N
    hub_s = torch.arange(1, N)
    hub_d = torch.full((N - 1,), hub)
    ex_s = torch.randint(0, N, (40,), generator=gen)
    ex_d = torch.randint(0, N, (40,), generator=gen)
    s = torch.cat([ring_s, hub_s, ex_s])
    d = torch.cat([ring_d, hub_d, ex_d])
    keep = s != d
    return _graph_from_edges(s[keep], d[keep], N, R, seed)


def _exact_history(g, model, h):
    hist = GASHistory(g.adj_self_norm.size(0), model.output_dim, model.encoder.depth)
    warm = build_gas_clusters(g, 1)
    with torch.no_grad():
        gas_forward_batch(model, warm[0], h, hist, CPU)
    return hist


def test_cap_zero_and_large_cap_match_exact_path():
    g = _hub_graph()
    model, h = _model_and_h(g, N=N_HUB)
    with torch.no_grad():
        full = model(g, h)
    hist = _exact_history(g, model, h)
    ref = build_gas_clusters(g, 4, method="bfs")
    for cap in (0, 10_000):
        cl = build_gas_clusters(g, 4, method="bfs", closure_cap=cap)
        assert len(cl) == len(ref)
        for a, b in zip(ref, cl, strict=True):
            assert torch.equal(a.batch, b.batch) and torch.equal(a.needed, b.needed)
            assert torch.equal(a.e_dst_pos, b.e_dst_pos) and torch.equal(a.e_src_pos, b.e_src_pos)
            assert torch.equal(a.e_rel, b.e_rel) and torch.equal(a.e_val, b.e_val)
            assert torch.equal(a.e_src_batch_mask, b.e_src_batch_mask)
            assert torch.equal(a.e_src_batch_pos, b.e_src_batch_pos)
            assert a.e_w is None
            if cap == 0:
                assert b.e_w is None
            else:
                assert b.e_w is not None and bool(torch.all(b.e_w == 1.0))
        outs = torch.zeros_like(full)
        with torch.no_grad():
            for c in cl:
                outs[c.batch] = gas_forward_batch(model, c, h, hist, CPU)
        assert torch.allclose(full, outs, atol=1e-4), (cap, (full - outs).abs().max().item())
    print("  test_cap_zero_and_large_cap_match_exact_path PASSED")


def test_small_cap_shrinks_closure_keeps_batch_edges():
    g = _hub_graph()
    model, h0 = _model_and_h(g, N=N_HUB)
    cap = 3
    ref = build_gas_clusters(g, 4, method="bfs")
    cl = build_gas_clusters(g, 4, method="bfs", closure_cap=cap)
    total_ref = sum(c.e_rel.size(0) for c in ref)
    total_cap = sum(c.e_rel.size(0) for c in cl)
    assert total_cap < total_ref, (total_cap, total_ref)
    checked_hub = 0
    for a, b in zip(ref, cl, strict=True):
        assert torch.equal(a.batch, b.batch)
        # every batch-src edge survives (same count; these carry the messages)
        assert int(b.e_src_batch_mask.sum()) == int(a.e_src_batch_mask.sum())
        assert b.e_rel.size(0) <= a.e_rel.size(0)
        assert b.needed.size(0) <= a.needed.size(0)
        assert bool(torch.all(b.e_w[b.e_src_batch_mask] == 1.0))
        assert bool(torch.all(b.e_w >= 1.0))
        # hub group: kept others <= cap, each weighted n_other / n_kept
        hub_pos_b = int(np.searchsorted(b.needed.numpy(), HUB))
        hub_pos_a = int(np.searchsorted(a.needed.numpy(), HUB))
        assert int(a.needed[hub_pos_a]) == HUB  # hub is in every exact closure
        if hub_pos_b < b.needed.size(0) and int(b.needed[hub_pos_b]) == HUB:
            others_b = (b.e_dst_pos == hub_pos_b) & ~b.e_src_batch_mask
            others_a = (a.e_dst_pos == hub_pos_a) & ~a.e_src_batch_mask
            n_kept, n_other = int(others_b.sum()), int(others_a.sum())
            assert n_kept <= cap and n_kept == min(cap, n_other), (n_kept, n_other)
            if n_kept > 0:
                expect = torch.full((n_kept,), n_other / n_kept)
                assert torch.allclose(b.e_w[others_b], expect, atol=1e-6)
                checked_hub += 1
    assert checked_hub >= 1
    # forward + backward on a capped cluster: grads reach the tables
    model.train()
    h = torch.nn.Parameter(h0.clone())
    hist = GASHistory(N_HUB, model.output_dim, model.encoder.depth)
    out = gas_forward_batch(model, cl[0], h, hist, CPU)
    assert out.shape == (cl[0].batch.size(0), model.output_width)
    out.sum().backward()
    assert h.grad is not None and h.grad.abs().sum() > 0
    assert model.rel_emb.grad is not None and model.rel_emb.grad.abs().sum() > 0
    print(f"  test_small_cap_shrinks_closure_keeps_batch_edges PASSED "
          f"(closure edges {total_ref} -> {total_cap} at cap={cap})")


def test_capped_denominator_is_unbiased():
    n, n_batch, cap = 40, 5, 6
    ce_dst = np.zeros(n, dtype=np.int64)
    src_in_batch = np.zeros(n, dtype=bool)
    src_in_batch[:n_batch] = True
    z = torch.tensor(np.random.default_rng(1).normal(0.0, 1.5, n), dtype=torch.float32)
    Z = float(z.exp().sum())
    est = []
    for seed in range(3000):
        keep, w = _cap_closure(ce_dst, src_in_batch, cap, np.random.default_rng(seed))
        assert keep[:n_batch].all() and int(keep[n_batch:].sum()) == cap
        assert np.all(w[:n_batch] == 1.0)
        est.append(float((z[keep].exp() * torch.from_numpy(w[keep])).sum()))
    mean = float(np.mean(est))
    assert abs(mean - Z) / Z < 0.03, (mean, Z)
    # one draw: a batch edge's weighted-softmax att == exp(z_e) / Z_hat
    keep, w = _cap_closure(ce_dst, src_in_batch, cap, np.random.default_rng(7))
    zk, wk = z[keep], torch.from_numpy(w[keep])
    att = _sparse_softmax_by_dst_weighted(zk, torch.zeros(zk.size(0), dtype=torch.long), 1, wk)
    z_hat = float((zk.exp() * wk).sum())
    assert torch.allclose(att[:n_batch], z[:n_batch].exp() / z_hat, atol=1e-6)
    # all-ones weights == the plain segment softmax
    dst = torch.zeros(n, dtype=torch.long)
    assert torch.allclose(
        _sparse_softmax_by_dst(z, dst, 1),
        _sparse_softmax_by_dst_weighted(z, dst, 1, torch.ones(n)),
    )
    print(f"  test_capped_denominator_is_unbiased PASSED (mean {mean:.4f} vs exact {Z:.4f})")


def test_checkpoint_parity_with_cap():
    g = _hub_graph()
    model, h0 = _model_and_h(g, N=N_HUB)
    cl = build_gas_clusters(g, 4, method="bfs", closure_cap=3)
    res = {}
    for ck in ("0", "1"):
        os.environ["DISTEA_GAS_CHECKPOINT"] = ck
        model.train()
        model.zero_grad(set_to_none=True)
        h = torch.nn.Parameter(h0.clone())
        hist = GASHistory(N_HUB, model.output_dim, model.encoder.depth)
        out = gas_forward_batch(model, cl[1], h, hist, CPU)
        out.sum().backward()
        res[ck] = (out.detach().clone(), h.grad.clone(), model.rel_emb.grad.clone())
    os.environ.pop("DISTEA_GAS_CHECKPOINT", None)
    for a, b in zip(res["0"], res["1"], strict=True):
        assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max().item()
    print("  test_checkpoint_parity_with_cap PASSED")


def test_history_device_argument():
    hist_default = GASHistory(N_HUB, 16, 2)
    hist_cpu = GASHistory(N_HUB, 16, 2, device="cpu")
    assert hist_default.device == torch.device("cpu") == hist_cpu.device
    ids = torch.tensor([3, 7, 11])
    vals = torch.randn(3, 16)
    for hist in (hist_default, hist_cpu):
        hist.push("ent", 1, ids, vals)
        assert torch.equal(hist.pull("ent", 1, ids, CPU), vals)
        assert hist.tables[("ent", 1)].device == torch.device("cpu")
        assert hist.tables[("ent", 2)].abs().sum() == 0
    print("  test_history_device_argument PASSED")


if __name__ == "__main__":
    for fn in (
        test_cap_zero_and_large_cap_match_exact_path,
        test_small_cap_shrinks_closure_keeps_batch_edges,
        test_capped_denominator_is_unbiased,
        test_checkpoint_parity_with_cap,
        test_history_device_argument,
    ):
        print(f"[{fn.__name__}]")
        fn()
    print("test_gas_closure_cap: ALL PASSED")
