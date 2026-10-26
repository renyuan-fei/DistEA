"""
test_gas_src_sorted.py — build_gas_clusters(src_sorted_edges=True)
(DISTEA_GAS_SEGMENT_SOFTMAX=1 under --attn_softmax src): each cluster's
edges are reordered by src and carry the run length of every src, so the
attention head takes _SegmentSoftmax instead of the index softmax.

1. Same clusters: per cluster the same batch / needed sets and the same
   multiset of (src, dst, rel, global edge id) as the unsorted build; the
   src positions are non-decreasing and the run lengths sum to the edge
   count.
2. Parity under --attn_softmax src, with the CSR layer-0 and incidence
   gather paths on: a warm-up pass plus every cluster's forward/backward in
   sequence give the same outputs, h_init and relation gradients as the
   unsorted clusters (fp32 summation order only), and the segment path is
   the one taken.
3. The dst-grouped softmax (default recipe, denominator history) on sorted
   clusters still matches the unsorted ones: the edge permutation is
   applied consistently to every per-edge array.
4. DISTEA_GAS_SEGMENT_MP=1 on sorted clusters (src grouping): the src
   gather, the dst gather's forward and the message aggregation leave the
   incidence spmm's for row gathers and run sums, with the same outputs and
   gradients; closure clusters (some srcs outside the batch) keep the spmm
   path.

Run:  .venv/bin/python tests/test_gas_src_sorted.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from test_gas_closure_cap import N_HUB, _hub_graph  # noqa: E402
from test_stage_w import _model_and_h  # noqa: E402

import gas as gas_mod  # noqa: E402
from gas import GASHistory, build_gas_clusters, gas_forward_batch  # noqa: E402
from models.rrea import RREA  # noqa: E402

CPU = torch.device("cpu")
KW = {"method": "bfs", "softmax_history": True, "csr_spmm": True, "csr_gather": True}


def _history(g, model) -> GASHistory:
    return GASHistory(g.adj_self_norm.size(0), model.output_dim, model.encoder.depth,
                      edge_dst=g.edge_index[:, 1], attn_heads=model.encoder.attn_heads)


def _edge_rows(c):
    nd = c.needed.numpy()
    return sorted(zip(nd[c.e_src_pos.numpy()].tolist(), nd[c.e_dst_pos.numpy()].tolist(),
                      c.e_rel.numpy().tolist(), c.e_gid.numpy().tolist(), strict=False))


def test_same_clusters():
    g = _hub_graph()
    a = build_gas_clusters(g, 4, **KW)
    b = build_gas_clusters(g, 4, src_sorted_edges=True, **KW)
    assert len(a) == len(b)
    for ca, cb in zip(a, b, strict=False):
        assert torch.equal(ca.batch, cb.batch) and torch.equal(ca.needed, cb.needed)
        assert _edge_rows(ca) == _edge_rows(cb), "a different edge multiset"
        assert ca.src_seg_counts is None and cb.src_seg_counts is not None
        assert bool((cb.e_src_pos[1:] >= cb.e_src_pos[:-1]).all())
        assert int(cb.src_seg_counts.sum()) == cb.e_src_pos.numel()
        runs = torch.unique_consecutive(cb.e_src_pos, return_counts=True)[1]
        assert torch.equal(runs, cb.src_seg_counts)
    print("  [ok] sorted build: same batches, needed sets and edge multisets; runs match the src positions")


def _run(g, base, h0, axis: str, sorted_edges: bool, **build_kw):
    model = RREA(rel_num=base.rel_num, output_dim=base.output_dim, num_layers=base.encoder.depth,
                 attn_heads=base.encoder.attn_heads, attn_softmax=axis)
    model.load_state_dict(base.state_dict())
    hist = _history(g, model)
    h = torch.nn.Parameter(h0.clone())
    warm = build_gas_clusters(g, 1, src_sorted_edges=sorted_edges)
    with torch.no_grad():
        gas_forward_batch(model, warm[0], h, hist, CPU)
    outs = []
    for c in build_gas_clusters(g, 4, src_sorted_edges=sorted_edges, **{**KW, **build_kw}):
        model.zero_grad(set_to_none=True)
        h.grad = None
        out = gas_forward_batch(model, c, h, hist, CPU)
        (out * torch.arange(out.numel()).reshape(out.shape).float().div(out.numel())).sum().backward()
        outs.append((out.detach().clone(), h.grad.clone(), model.rel_emb.grad.clone()))
    return outs


def test_parity():
    g = _hub_graph()
    base, h0 = _model_and_h(g, N=N_HUB)
    calls = {"n": 0}
    real = gas_mod._SegmentSoftmax.apply

    def counting(*a):
        calls["n"] += 1
        return real(*a)

    for axis in ("src", "dst"):
        ref = _run(g, base, h0, axis, False)
        gas_mod._SegmentSoftmax.apply = counting
        try:
            calls["n"] = 0
            got = _run(g, base, h0, axis, True)
        finally:
            gas_mod._SegmentSoftmax.apply = real
        if axis == "src":
            assert calls["n"] > 0, "the segment softmax path was not taken"
        else:
            assert calls["n"] == 0
        for (oa, ha, ra), (ob, hb, rb) in zip(ref, got, strict=False):
            for x, y in ((oa, ob), (ha, hb), (ra, rb)):
                err = float((x - y).abs().max())
                assert torch.allclose(x, y, atol=1e-5, rtol=1e-4), f"{axis}: max abs diff {err:.3e}"
        print(f"  [ok] --attn_softmax {axis}: sorted clusters reproduce outputs and gradients "
              f"({len(ref)} clusters, segment-softmax calls {calls['n']})")


def test_segment_mp_parity():
    g = _hub_graph()
    base, h0 = _model_and_h(g, N=N_HUB)
    calls = {"spmm": 0, "segsum": 0}
    real_spmm, real_segsum = gas_mod._SpMMFixedT.apply, gas_mod._SegSumToRows.apply

    def spmm(*a):
        calls["spmm"] += 1
        return real_spmm(*a)

    def segsum(*a):
        calls["segsum"] += 1
        return real_segsum(*a)

    for build_kw in ({}, {"softmax_history": False}):
        runs = {}
        for flag in ("0", "1"):
            os.environ["DISTEA_GAS_SEGMENT_MP"] = flag
            gas_mod._SpMMFixedT.apply, gas_mod._SegSumToRows.apply = spmm, segsum
            calls.update(spmm=0, segsum=0)
            try:
                runs[flag] = (_run(g, base, h0, "src", True, **build_kw), dict(calls))
            finally:
                gas_mod._SpMMFixedT.apply, gas_mod._SegSumToRows.apply = real_spmm, real_segsum
                os.environ.pop("DISTEA_GAS_SEGMENT_MP", None)
        (ref, c0), (got, c1) = runs["0"], runs["1"]
        for (oa, ha, ra), (ob, hb, rb) in zip(ref, got, strict=False):
            for x, y in ((oa, ob), (ha, hb), (ra, rb)):
                assert torch.allclose(x, y, atol=1e-5, rtol=1e-4), float((x - y).abs().max())
        if build_kw:
            assert c1["segsum"] == 0 and c1["spmm"] == c0["spmm"], (c0, c1)
        else:
            assert c1["segsum"] > 0 and c1["spmm"] < c0["spmm"], (c0, c1)
        tag = "closure clusters keep the spmm path" if build_kw else "segment path taken"
        print(f"  [ok] DISTEA_GAS_SEGMENT_MP=1 reproduces outputs and gradients ({tag}; "
              f"spmm calls {c0['spmm']} -> {c1['spmm']}, run sums {c1['segsum']})")


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    test_same_clusters()
    test_parity()
    test_segment_mp_parity()
    print("test_gas_src_sorted: ALL PASSED")
