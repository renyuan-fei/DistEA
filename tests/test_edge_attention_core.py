"""
test_edge_attention_core.py — models/rrea.py edge_attention_core:
the per-edge reflection + attention logit without the (E, 3*dim)
concatenation, optionally torch.compile'd (DISTEA_GAS_COMPILE_HEAD=1).

1. Eager core == the original cat([selfs, reflected, rel]) @ kernel to fp32
   rounding, forward and gradients (selfs / neighs / rel_vec / kernel).
2. With DISTEA_GAS_COMPILE_HEAD=1 the compiled core gives the same outputs
   and gradients for two different edge counts (dynamic shapes), or falls
   back to eager with a warning when torch.compile is unavailable.

Run:  .venv/bin/python tests/test_edge_attention_core.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import models.rrea as rrea  # noqa: E402


def _reference(selfs, neighs, rel_vec, kernel):
    bias = (neighs * rel_vec).sum(dim=1, keepdim=True) * rel_vec
    nr = neighs - 2 * bias
    att_in = torch.cat([selfs, nr, rel_vec], dim=1)
    return nr, (att_in @ kernel).squeeze(-1)


def _inputs(E: int, dim: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    mk = lambda *s: torch.randn(*s, generator=gen).requires_grad_(True)  # noqa: E731
    selfs, neighs = mk(E, dim), mk(E, dim)
    rel_vec = torch.nn.functional.normalize(torch.randn(E, dim, generator=gen), dim=1).requires_grad_(True)
    kernel = mk(3 * dim, 1)
    return selfs, neighs, rel_vec, kernel


def _run(fn, E, dim, seed):
    ins = _inputs(E, dim, seed)
    nr, logits = fn(*ins)
    (nr.square().sum() + logits.square().sum()).backward()
    return nr.detach(), logits.detach(), [t.grad.clone() for t in ins]


def _assert_same(a, b, tag):
    for x, y in zip(a[:2], b[:2], strict=True):
        assert torch.allclose(x, y, atol=1e-5, rtol=1e-5), (tag, (x - y).abs().max())
    for gx, gy in zip(a[2], b[2], strict=True):
        scale = gx.abs().max().clamp_min(1.0)
        assert torch.allclose(gx, gy, atol=1e-4 * scale.item(), rtol=1e-5), (tag, ((gx - gy).abs().max() / scale))


def test_eager_matches_cat_formulation():
    os.environ.pop("DISTEA_GAS_COMPILE_HEAD", None)
    for seed, E in ((0, 257), (1, 1000)):
        _assert_same(_run(rrea.edge_attention_core, E, 16, seed), _run(_reference, E, 16, seed), f"eager E={E}")
    print("  test_eager_matches_cat_formulation PASSED")


def test_compiled_matches_eager_dynamic_shapes():
    os.environ["DISTEA_GAS_COMPILE_HEAD"] = "1"
    rrea._EDGE_CORE_COMPILED = None
    try:
        for seed, E in ((2, 300), (3, 777)):
            _assert_same(_run(rrea.edge_attention_core, E, 16, seed), _run(_reference, E, 16, seed), f"compiled E={E}")
        mode = "compiled" if rrea._EDGE_CORE_COMPILED is not rrea._edge_attention_core_eager else "eager fallback"
    finally:
        os.environ.pop("DISTEA_GAS_COMPILE_HEAD", None)
        rrea._EDGE_CORE_COMPILED = None
    print(f"  test_compiled_matches_eager_dynamic_shapes PASSED ({mode})")


if __name__ == "__main__":
    for fn in (test_eager_matches_cat_formulation, test_compiled_matches_eager_dynamic_shapes):
        print(f"[{fn.__name__}]")
        fn()
    print("test_edge_attention_core: ALL PASSED")
