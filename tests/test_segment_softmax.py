"""
test_segment_softmax.py — gas._segment_softmax_sorted (atomics-free segment
softmax used under --attn_softmax src) equals _sparse_softmax_by_dst in
value AND gradient, including size-1 segments and one huge segment, in fp64
and fp32; and the GAS forward under src grouping gives the same h_init
gradient with the segment path and the index path.

Run:  .venv/bin/python tests/test_segment_softmax.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from test_gas_softmax_history import N_HUB, _hub_graph, _model_and_h, _smh_history  # noqa: E402

import gas as gas_mod  # noqa: E402
from gas import (  # noqa: E402
    _segment_softmax_sorted,
    _sparse_softmax_by_dst,
    build_gas_clusters,
    gas_forward_batch,
)
from models.rrea import RREA  # noqa: E402

CPU = torch.device("cpu")


def test_values_and_grads():
    gen = torch.Generator().manual_seed(0)
    counts = torch.tensor([1, 5, 1, 300, 2, 7, 1])
    seg = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    N = int(counts.numel())
    for dtype in (torch.float64, torch.float32):
        x = (torch.randn(int(counts.sum()), generator=gen) * 3).to(dtype)
        a = x.clone().requires_grad_(True)
        b = x.clone().requires_grad_(True)
        ya = _segment_softmax_sorted(a, seg)
        yb = _sparse_softmax_by_dst(b, seg, N)
        tol = 1e-12 if dtype == torch.float64 else 1e-6
        assert torch.allclose(ya, yb, atol=tol)
        w = torch.randn(x.numel(), generator=gen).to(dtype)
        (ya * w).sum().backward()
        (yb * w).sum().backward()
        assert torch.allclose(a.grad, b.grad, atol=tol * 10), (a.grad - b.grad).abs().max()
        # per-segment mass is 1
        mass = torch.zeros(N, dtype=dtype).index_add_(0, seg, ya.detach())
        assert torch.allclose(mass, torch.ones(N, dtype=dtype), atol=1e-6)
    print("  [ok] segment softmax == index softmax (values + grads, fp64/fp32, size-1 and 300-wide segments)")


def test_gas_grad_parity():
    g = _hub_graph()
    base, h0 = _model_and_h(g, N=N_HUB)
    model = RREA(rel_num=base.rel_num, output_dim=base.output_dim, num_layers=base.encoder.depth,
                 attn_heads=base.encoder.attn_heads, attn_softmax="src")
    model.load_state_dict(base.state_dict())
    grads = {}
    for use_seg in (True, False):
        gas_mod._USE_SEGMENT_SOFTMAX = use_seg
        hist = _smh_history(g, model)
        warm = build_gas_clusters(g, 1)
        h = torch.nn.Parameter(h0.clone())
        with torch.no_grad():
            gas_forward_batch(model, warm[0], h, hist, CPU)
        cl = build_gas_clusters(g, 4, method="bfs", softmax_history=True)
        model.zero_grad(set_to_none=True)
        out = gas_forward_batch(model, cl[1], h, hist, CPU)
        (out * torch.arange(out.numel()).reshape(out.shape).float()).sum().backward()
        grads[use_seg] = (out.detach().clone(), h.grad.clone(), model.rel_emb.grad.clone())
    gas_mod._USE_SEGMENT_SOFTMAX = True
    for a, b in zip(grads[True], grads[False], strict=False):
        assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()
    print("  [ok] GAS batch forward/backward identical with the segment and index softmax paths")


if __name__ == "__main__":
    test_values_and_grads()
    test_gas_grad_parity()
    print("test_segment_softmax: ALL PASSED")
