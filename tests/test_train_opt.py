"""
test_train_opt.py — train_opt.build_optimizer.

1. weight_decay=0, adam -> a plain torch.optim.Adam with ONE flat param
   group in the historical order [model params..., h_init, loss_weight]
   (the default recipe, bit-identical).
2. weight_decay>0 -> AdamW with three groups: model (wd), h_init (its own
   wd, defaulting to the model's), loss_weight (0).
3. Decay is decoupled: a parameter whose gradient is all zeros shrinks by
   exactly (1 - lr * wd) per step (AdamW), while loss_weight does not move.
4. sgd / rmsprop still build; asking them for decay raises; unknown name
   raises.

Run:  .venv/bin/python tests/test_train_opt.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_opt import build_optimizer  # noqa: E402


def _params():
    torch.manual_seed(0)
    rel = torch.nn.Parameter(torch.randn(5, 4))
    ker = torch.nn.Parameter(torch.randn(3, 4))
    h = torch.nn.Parameter(torch.randn(10, 4))
    lw = torch.nn.Parameter(torch.tensor([0.5, 0.5]))
    return [rel, ker], h, lw


def test_recorded_path_is_plain_adam():
    mp, h, lw = _params()
    opt = build_optimizer(mp, h, lw, "adam", lr=1e-3)
    assert type(opt) is torch.optim.Adam
    assert len(opt.param_groups) == 1
    ps = opt.param_groups[0]["params"]
    assert [id(p) for p in ps] == [id(mp[0]), id(mp[1]), id(h), id(lw)]
    assert opt.param_groups[0]["weight_decay"] == 0
    print("  [ok] wd=0 -> plain Adam, one flat group, historical order")


def test_decay_groups():
    mp, h, lw = _params()
    opt = build_optimizer(mp, h, lw, "adam", lr=1e-3, weight_decay=0.01)
    assert isinstance(opt, torch.optim.AdamW) and len(opt.param_groups) == 3
    assert opt.param_groups[0]["weight_decay"] == 0.01
    assert opt.param_groups[1]["weight_decay"] == 0.01 and opt.param_groups[1]["params"][0] is h
    assert opt.param_groups[2]["weight_decay"] == 0.0 and opt.param_groups[2]["params"][0] is lw
    opt2 = build_optimizer(mp, h, lw, "adam", lr=1e-3, weight_decay=0.01, weight_decay_h_init=0.001)
    assert opt2.param_groups[1]["weight_decay"] == 0.001
    opt3 = build_optimizer(mp, h, lw, "adam", lr=1e-3, weight_decay=0.0, weight_decay_h_init=0.05)
    assert isinstance(opt3, torch.optim.AdamW) and opt3.param_groups[0]["weight_decay"] == 0.0
    print("  [ok] wd>0 -> AdamW with model / h_init / loss_weight groups")


def test_decoupled_shrink():
    mp, h, lw = _params()
    lr, wd = 1e-2, 0.1
    opt = build_optimizer(mp, h, lw, "adam", lr=lr, weight_decay=wd)
    before = h.detach().clone()
    lw_before = lw.detach().clone()
    for p in mp + [h, lw]:
        p.grad = torch.zeros_like(p)
    opt.step()
    assert torch.allclose(h.detach(), before * (1 - lr * wd), atol=1e-7)
    assert torch.equal(lw.detach(), lw_before)
    print("  [ok] zero-grad rows shrink by exactly (1 - lr*wd); loss_weight untouched")


def test_other_optimizers():
    mp, h, lw = _params()
    assert isinstance(build_optimizer(mp, h, lw, "sgd", lr=0.1), torch.optim.SGD)
    assert isinstance(build_optimizer(mp, h, lw, "rmsprop", lr=0.1), torch.optim.RMSprop)
    for name in ("sgd", "rmsprop"):
        try:
            build_optimizer(mp, h, lw, name, lr=0.1, weight_decay=0.1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{name} with decay must raise")
    try:
        build_optimizer(mp, h, lw, "adagrad", lr=0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown optimizer must raise")
    print("  [ok] sgd / rmsprop build; decay there and unknown names raise")


if __name__ == "__main__":
    test_recorded_path_is_plain_adam()
    test_decay_groups()
    test_decoupled_shrink()
    test_other_optimizers()
    print("test_train_opt: all passed")
