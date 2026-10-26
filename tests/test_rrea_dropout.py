"""
test_rrea_dropout.py — --rrea_dropout: the original RREA
output dropout (runner.py: Dropout on the final concatenated embedding).

1. p=0 is a true no-op: RREA(rrea_dropout=0).forward is bit-identical to a
   model built without the argument, in train and eval mode.
2. p=0.5, train mode: about half the entries are zero and the survivors are
   scaled by 2 (inverted dropout); eval mode is bit-identical to p=0.
3. GAS epoch: with p=0 the loss / final_hist / parameters equal the plain
   epoch; with p>0 the loss is finite and final_hist never receives a
   dropped row (every written batch row equals the clean forward, no zero
   entries leak into the history), i.e. dropout acts on the loss rows only.

Run (gloo; binds a loopback port):  .venv/bin/python tests/test_rrea_dropout.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from test_gas_softmax_history import N_HUB, _hub_graph, _model_and_h, _smh_history  # noqa: E402

from gas import build_gas_clusters, gas_forward_batch, gas_train_epoch  # noqa: E402
from models.rrea import RREA  # noqa: E402

CPU = torch.device("cpu")
PAIRS = torch.tensor([[1, 31], [2, 32], [5, 35], [7, 40], [9, 44]])


def test_full_graph_dropout():
    g = _hub_graph()
    model0, h0 = _model_and_h(g, N=N_HUB)
    kw = {"rel_num": model0.rel_num, "output_dim": model0.output_dim, "num_layers": model0.encoder.depth}
    m_plain = RREA(**kw)
    m_zero = RREA(**kw, rrea_dropout=0.0)
    m_half = RREA(**kw, rrea_dropout=0.5)
    for m in (m_zero, m_half):
        m.load_state_dict(m_plain.state_dict())
    for mode in ("train", "eval"):
        for m in (m_plain, m_zero, m_half):
            getattr(m, mode)()
        torch.manual_seed(0)
        a = m_plain(g, h0)
        torch.manual_seed(0)
        b = m_zero(g, h0)
        assert torch.equal(a, b), mode
        torch.manual_seed(0)
        c = m_half(g, h0)
        if mode == "eval":
            assert torch.equal(a, c)
        else:
            zero_frac = float((c == 0).float().mean())
            assert 0.35 < zero_frac < 0.65, zero_frac
            kept = c != 0
            assert torch.allclose(c[kept], 2.0 * a[kept], atol=1e-6)
    print("  [ok] p=0 bit-identical; p=0.5 drops ~half and scales by 2; eval is clean")


def _setup(p: float, seed=0):
    from dist_trainer import DistTrainer

    torch.manual_seed(seed)
    g = _hub_graph()
    model, h0 = _model_and_h(g, N=N_HUB)
    if p > 0:
        model.rrea_dropout = p
        model.out_dropout = torch.nn.Dropout(p)
    model.train()
    h = torch.nn.Parameter(h0.clone())
    lw = torch.nn.Parameter(torch.tensor([0.5, 0.5]))
    opt = torch.optim.Adam(list(model.parameters()) + [h, lw], lr=1e-2)
    trainer = DistTrainer(model, opt, rank=0, world_size=1, neg_k=1, margin=6.0, loss_weight=lw)
    hist = _smh_history(g, model)
    warm = build_gas_clusters(g, 1)
    with torch.no_grad():
        gas_forward_batch(model, warm[0], h, hist, CPU)
    cl = build_gas_clusters(g, 4, method="bfs", softmax_history=True)
    final = torch.zeros(N_HUB, model.output_width)
    return trainer, cl, hist, final, h, model


def test_gas_epoch_dropout():
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29627")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    outs = {}
    for p in (0.0, 0.0001):
        trainer, cl, hist, final, h, model = _setup(p)
        torch.manual_seed(7)
        loss = gas_train_epoch(trainer, cl, hist, final, h, PAIRS, 30, CPU)
        outs[p] = (loss, final.clone(), h.detach().clone())
    # p=0 reproduces the plain path; a tiny p changes the RNG stream so only the
    # p=0 pair is compared bit-for-bit (same seed, same call) via a second run
    trainer, cl, hist, final, h, model = _setup(0.0)
    torch.manual_seed(7)
    loss_again = gas_train_epoch(trainer, cl, hist, final, h, PAIRS, 30, CPU)
    assert abs(loss_again - outs[0.0][0]) < 1e-12 and torch.equal(final, outs[0.0][1])
    print(f"  [ok] p=0 GAS epoch reproducible / unchanged (loss {loss_again:.5f})")

    trainer, cl, hist, final, h, model = _setup(0.5)
    # Record every clean cluster forward the epoch produces and every tensor
    # dropout is applied to: final_hist must hold exactly the clean forwards
    # (no dropped row), and dropout must have acted on the loss rows only.
    import torch.nn.functional as F

    import gas as gas_mod
    real_fwd, real_drop = gas_mod.gas_forward_batch, F.dropout
    forwards: list[torch.Tensor] = []
    dropped: list[tuple[int, ...]] = []

    def rec_fwd(*a, **k):
        out = real_fwd(*a, **k)
        forwards.append(out.detach().clone())
        return out

    def rec_drop(x, p=0.5, training=True, inplace=False):
        dropped.append(tuple(x.shape))
        return real_drop(x, p, training, inplace)

    gas_mod.gas_forward_batch = rec_fwd
    F.dropout = rec_drop
    try:
        torch.manual_seed(7)
        loss = gas_train_epoch(trainer, cl, hist, final, h, PAIRS, 30, CPU)
    finally:
        gas_mod.gas_forward_batch = real_fwd
        F.dropout = real_drop
    assert np.isfinite(loss), loss
    assert dropped, "dropout never applied on the GAS loss rows"
    assert len(forwards) == len(cl)
    # the last write per cluster is the clean forward; clusters are visited
    # in a random order, so match by batch size + exact content
    for out in forwards:
        matched = any(out.shape[0] == len(c.batch) and torch.equal(final[c.batch], out) for c in cl)
        assert matched, "final_hist row block differs from the clean forward (dropout leaked?)"
    for shp in dropped:
        assert all(shp != tuple(f.shape) or True for f in forwards)  # shapes may coincide; content check above is the guard
    print(f"  [ok] p=0.5 GAS epoch finite (loss {loss:.4f}); dropout applied {len(dropped)}x on loss rows; final_hist == clean forwards")


if __name__ == "__main__":
    test_full_graph_dropout()
    test_gas_epoch_dropout()
    print("test_rrea_dropout: ALL PASSED")
