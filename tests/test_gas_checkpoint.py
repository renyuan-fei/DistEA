"""
test_gas_checkpoint.py — DISTEA_GAS_CHECKPOINT=1 must not change what
gas_forward_batch computes: same outputs, same gradients into h_init and the
model parameters, same history pushes, as the un-checkpointed path. The flag
only changes WHEN the per-edge intermediates exist (recomputed in backward),
which is what lets a DBP1M/N=4 cluster whose closure is ~3.8M edges fit an
80 GB card (without it, the attention concatenation runs out of memory).

Reuses test_stage_w.py's toy graph / model builders.

Run:  uv run python tests/test_gas_checkpoint.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from test_stage_w import _model_and_h, _toy_graph  # noqa: E402

from gas import GASHistory, build_gas_clusters, gas_forward_batch  # noqa: E402

CPU = torch.device("cpu")


def _run(ckpt: bool, seed: int = 0):
    os.environ["DISTEA_GAS_CHECKPOINT"] = "1" if ckpt else "0"
    try:
        g = _toy_graph(seed=seed)
        model, h0 = _model_and_h(g, seed=seed)
        model.train()
        h = torch.nn.Parameter(h0.clone())
        hist = GASHistory(40, model.output_dim, model.encoder.depth)
        clusters = build_gas_clusters(g, 3, method="bfs")
        outs = []
        for c in clusters:
            out = gas_forward_batch(model, c, h, hist, CPU)
            outs.append(out)
        loss = sum(o.pow(2).sum() for o in outs)
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        return (
            torch.cat(outs, dim=0).detach().clone(),
            h.grad.detach().clone(),
            grads,
            {k: v.clone() for k, v in hist.tables.items()},
        )
    finally:
        os.environ.pop("DISTEA_GAS_CHECKPOINT", None)


def test_checkpoint_matches_plain_forward_backward():
    out_a, hg_a, pg_a, hist_a = _run(ckpt=False)
    out_b, hg_b, pg_b, hist_b = _run(ckpt=True)
    assert torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6), (out_a - out_b).abs().max()
    assert torch.allclose(hg_a, hg_b, atol=1e-6, rtol=1e-6), (hg_a - hg_b).abs().max()
    assert pg_a.keys() == pg_b.keys()
    for k in pg_a:
        assert torch.allclose(pg_a[k], pg_b[k], atol=1e-6, rtol=1e-6), (k, (pg_a[k] - pg_b[k]).abs().max())
    assert hist_a.keys() == hist_b.keys()
    for k in hist_a:
        assert torch.allclose(hist_a[k], hist_b[k], atol=1e-6, rtol=1e-6), k
    print("  test_checkpoint_matches_plain_forward_backward PASSED")


def test_checkpoint_is_inert_under_no_grad():
    # Eval-time forwards run under no_grad; checkpoint must simply not engage.
    os.environ["DISTEA_GAS_CHECKPOINT"] = "1"
    try:
        g = _toy_graph(seed=1)
        model, h = _model_and_h(g, seed=1)
        hist = GASHistory(40, model.output_dim, model.encoder.depth)
        clusters = build_gas_clusters(g, 1)
        with torch.no_grad():
            out = gas_forward_batch(model, clusters[0], h, hist, CPU)
        assert out.shape == (40, model.output_width) and torch.isfinite(out).all()
    finally:
        os.environ.pop("DISTEA_GAS_CHECKPOINT", None)
    print("  test_checkpoint_is_inert_under_no_grad PASSED")


if __name__ == "__main__":
    test_checkpoint_matches_plain_forward_backward()
    test_checkpoint_is_inert_under_no_grad()
    print("all gas checkpoint tests passed")
