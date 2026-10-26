"""
test_lr_schedule.py — train_opt.lr_multiplier /
build_lr_scheduler: closed-form lr curves stepped once per epoch.

1. none + no warmup -> build_lr_scheduler returns None (the default path has
   no scheduler at all); none + warmup ramps then stays at 1.
2. cosine: 1 at the start, min_ratio at the end, the half-way point at
   (1 + min_ratio) / 2; monotone non-increasing after warmup.
3. linear: straight line from 1 to min_ratio.
4. step: gamma ** floor(e / step_size), floored at min_ratio.
5. Stepping a real optimizer through LambdaLR reproduces lr_multiplier
   epoch by epoch (warmup included), and leaves the base lr intact.

Run:  .venv/bin/python tests/test_lr_schedule.py
"""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_opt import build_lr_scheduler, lr_multiplier  # noqa: E402


def test_none():
    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.Adam([p], lr=1e-3)
    assert build_lr_scheduler(opt, "none", 100) is None
    assert all(lr_multiplier(e, "none", 100) == 1.0 for e in range(100))
    ms = [lr_multiplier(e, "none", 10, warmup_epochs=4) for e in range(10)]
    assert ms[:4] == [0.25, 0.5, 0.75, 1.0] and all(m == 1.0 for m in ms[4:])
    print("  [ok] none -> no scheduler; warmup ramps then holds")


def test_cosine():
    T, r = 200, 0.1
    ms = [lr_multiplier(e, "cosine", T, min_ratio=r) for e in range(T + 1)]
    assert abs(ms[0] - 1.0) < 1e-12 and abs(ms[T] - r) < 1e-12
    assert abs(ms[T // 2] - (1 + r) / 2) < 1e-12
    assert all(a >= b - 1e-12 for a, b in zip(ms, ms[1:], strict=False))
    w = 10
    mw = [lr_multiplier(e, "cosine", T, warmup_epochs=w, min_ratio=r) for e in range(T + 1)]
    assert abs(mw[w - 1] - 1.0) < 1e-12 and abs(mw[w] - 1.0) < 1e-12 and abs(mw[T] - r) < 1e-12
    print("  [ok] cosine: 1 -> min_ratio, half-way value, monotone, warmup joins at 1")


def test_linear_and_step():
    T = 100
    ml = [lr_multiplier(e, "linear", T, min_ratio=0.2) for e in range(T + 1)]
    assert abs(ml[0] - 1.0) < 1e-12 and abs(ml[T] - 0.2) < 1e-12 and abs(ml[50] - 0.6) < 1e-12
    ms = [lr_multiplier(e, "step", T, step_size=25, gamma=0.5, min_ratio=0.2) for e in range(T + 1)]
    assert ms[0] == 1.0 and ms[24] == 1.0 and ms[25] == 0.5 and ms[50] == 0.25 and ms[75] == 0.2 and ms[100] == 0.2
    for bad in ("exp", "cos"):
        try:
            lr_multiplier(0, bad, 10)
        except ValueError:
            pass
        else:
            raise AssertionError("unknown schedule must raise")
    print("  [ok] linear and step curves; unknown schedule raises")


def test_scheduler_matches_closed_form():
    base = 2e-3
    for sched, w in (("cosine", 0), ("cosine", 5), ("linear", 3), ("step", 0), ("none", 4)):
        p = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.Adam([p], lr=base)
        sch = build_lr_scheduler(opt, sched, total_epochs=40, warmup_epochs=w, min_ratio=0.1, step_size=10, gamma=0.5)
        assert sch is not None
        seen = []
        for _e in range(40):
            seen.append(opt.param_groups[0]["lr"])
            p.grad = torch.zeros_like(p)
            opt.step()
            sch.step()
        want = [base * lr_multiplier(e, sched, 40, w, 0.1, 10, 0.5) for e in range(40)]
        assert all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-15) for a, b in zip(seen, want, strict=False)), (sched, w)
        assert opt.param_groups[0]["initial_lr"] == base
    print("  [ok] LambdaLR reproduces the closed form epoch by epoch")


if __name__ == "__main__":
    test_none()
    test_cosine()
    test_linear_and_step()
    test_scheduler_matches_closed_form()
    print("test_lr_schedule: all passed")
