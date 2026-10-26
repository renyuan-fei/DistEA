"""
train_opt.py — optimizer construction with per-group
weight decay and per-epoch learning-rate schedules.

Why a separate module: the default optimizer is
`Adam(list(model.parameters()) + [h_init, loss_weight], lr)`; regularisation
needs (a) decoupled weight decay on the model/table parameters without
touching the loss-mix weight, and (b) an lr schedule stepped once per epoch
that the GAS and full-graph branches share. Both must leave the default
recipe bit-identical when off: with `weight_decay == 0` the optimizer
returned is that plain flat-list `Adam` (no param groups), and
`schedule="none"` returns no scheduler.

Weight decay is DECOUPLED (torch.optim.AdamW), never Adam's coupled L2:
in a GAS cluster step most of `h_init`'s rows receive a zero gradient, and
a coupled L2 term `wd * p` on those rows would be normalised by Adam's
second moment into a sign-like step of roughly `lr` regardless of `wd`.
AdamW instead shrinks every parameter by `lr * wd` per step, proportional
to its value, which is the intended mild pull towards the origin (K=32
steps/epoch x 200 epochs x lr 1e-3 x wd 0.01 -> about 6% total shrink).
"""
from __future__ import annotations

import math

import torch
import torch.optim as optim


def build_optimizer(
    model_params,
    h_init: torch.Tensor,
    loss_weight: torch.Tensor | None,
    name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    weight_decay_h_init: float | None = None,
    fused: bool = False,
) -> torch.optim.Optimizer:
    """Return the training optimizer.

    model_params: iterable of the encoder's parameters (rel_emb, kernels).
    h_init:       the entity table (its own group: `weight_decay_h_init`,
                  defaulting to `weight_decay`).
    loss_weight:  the (2,) loss-mix parameter -- never decayed.
    name:         adam | sgd | rmsprop (sgd/rmsprop ignore decay here --
                  they are legacy paths; a non-zero decay raises).
    weight_decay / weight_decay_h_init: decoupled (AdamW) coefficients.
    fused:        Adam/AdamW(fused=True) on CUDA (callers gate on device).
    """
    model_params = list(model_params)
    wd_h = weight_decay if weight_decay_h_init is None else weight_decay_h_init
    flat = model_params + [h_init] + ([loss_weight] if loss_weight is not None else [])
    any_decay = (weight_decay or 0.0) > 0.0 or (wd_h or 0.0) > 0.0

    if name == "sgd":
        if any_decay:
            raise ValueError("weight decay is only wired for the adam optimizer")
        return optim.SGD(flat, lr=lr)
    if name == "rmsprop":
        if any_decay:
            raise ValueError("weight decay is only wired for the adam optimizer")
        return optim.RMSprop(flat, lr=lr)
    if name != "adam":
        raise ValueError(f"unknown optimizer {name!r}")

    if not any_decay:
        # Default path: one flat group, fixed parameter order.
        return optim.Adam(flat, lr=lr, fused=True) if fused else optim.Adam(flat, lr=lr)

    groups = [
        {"params": model_params, "weight_decay": float(weight_decay or 0.0)},
        {"params": [h_init], "weight_decay": float(wd_h or 0.0)},
    ]
    if loss_weight is not None:
        groups.append({"params": [loss_weight], "weight_decay": 0.0})
    kw = {"lr": lr}
    if fused:
        kw["fused"] = True
    return optim.AdamW(groups, **kw)


def lr_multiplier(
    epoch: int,
    schedule: str,
    total_epochs: int,
    warmup_epochs: int = 0,
    min_ratio: float = 0.0,
    step_size: int = 50,
    gamma: float = 0.5,
) -> float:
    """Closed-form lr multiplier for 0-based `epoch` (what LambdaLR applies
    BEFORE that epoch's steps; the scheduler is stepped after each epoch).

    none:   1 everywhere (warmup still applies if > 0).
    cosine: min_ratio + (1 - min_ratio) * 0.5 * (1 + cos(pi * t)),
            t = (epoch - warmup) / (total - warmup) in [0, 1].
    linear: 1 -> min_ratio linearly over the same span.
    step:   gamma ** floor((epoch - warmup) / step_size), floored at min_ratio.
    Warmup: linear ramp (epoch + 1) / warmup over the first `warmup` epochs.
    """
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return (epoch + 1) / float(warmup_epochs)
    if schedule == "none":
        return 1.0
    span = max(total_epochs - warmup_epochs, 1)
    t = min(max(epoch - warmup_epochs, 0), span) / float(span)
    if schedule == "cosine":
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))
    if schedule == "linear":
        return min_ratio + (1.0 - min_ratio) * (1.0 - t)
    if schedule == "step":
        k = (epoch - warmup_epochs) // max(step_size, 1)
        return max(gamma ** k, min_ratio)
    raise ValueError(f"unknown lr schedule {schedule!r}")


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule: str = "none",
    total_epochs: int = 200,
    warmup_epochs: int = 0,
    min_ratio: float = 0.0,
    step_size: int = 50,
    gamma: float = 0.5,
):
    """LambdaLR over `lr_multiplier`, or None when nothing would change the
    lr (schedule none and no warmup) so the default path has no scheduler
    object at all. Call `.step()` once after every epoch."""
    if schedule == "none" and warmup_epochs <= 0:
        return None
    if schedule not in ("none", "cosine", "linear", "step"):
        raise ValueError(f"unknown lr schedule {schedule!r}")

    def fn(epoch: int) -> float:
        return lr_multiplier(epoch, schedule, total_epochs, warmup_epochs,
                             min_ratio, step_size, gamma)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)
