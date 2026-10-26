"""
dist_utils.py — small torch.distributed portability helpers.

all_reduce_avg: `dist.all_reduce(t, op=ReduceOp.AVG)` is only implemented
for gloo in recent torch releases; older ones (e.g. torch 2.4.1, as used
with DGL) raise `RuntimeError: Cannot use ReduceOp.AVG with Gloo`. SUM followed by a division by the
group size is what AVG computes anyway, so the fallback is numerically
the same operation (float summation order aside) and works on every
backend/version this project runs on (gloo for --sim, nccl otherwise).
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def all_reduce_avg(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """In-place all-reduce average of `tensor` across `group` (default:
    the world), portable across torch versions and backends."""
    try:
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG, group=group)
    except (RuntimeError, ValueError) as exc:
        if "AVG" not in str(exc):
            raise
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        tensor.div_(dist.get_world_size(group=group))
    return tensor
