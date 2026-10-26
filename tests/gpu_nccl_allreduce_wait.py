"""
gpu_nccl_allreduce_wait.py -- DistTrainer.allreduce_gpu_wait_s on real GPUs
under nccl (CPU / gloo runs never record the events). Every epoch rank 1
busy-waits on its GPU for ~DELAY_MS before the parameter all_reduce, so rank
0 sits in the all_reduce for about that long. Checks: rank 0's GPU-timed wait
is ~DELAY_MS per epoch, rank 1's is small, and rank 0's host-side timer
(what per_rank.all_reduce_time_s records) misses most of the wait.
Functional check, not a benchmark.

Run on a node with >= 2 GPUs:
    torchrun --nproc_per_node=2 tests/gpu_nccl_allreduce_wait.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DELAY_MS = 50.0
EPOCHS = 10


def _sleep_cycles_for(ms: float, dev: torch.device) -> int:
    """GPU clock cycles torch.cuda._sleep needs for about `ms` milliseconds."""
    cycles = 10_000_000
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(cycles)  # warm up
    a.record()
    torch.cuda._sleep(cycles)
    b.record()
    b.synchronize()
    return int(cycles * ms / a.elapsed_time(b))


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(dev)
    from comm_stats import CommStats
    from dist_trainer import DistTrainer

    model = torch.nn.Linear(256, 256).to(dev)
    stats = CommStats()
    trainer = DistTrainer(model, torch.optim.Adam(model.parameters()), rank, ws, comm_stats=stats)
    trainer._allreduce_sync()  # nccl communicator set-up, not timed below
    torch.cuda.synchronize(dev)
    trainer.allreduce_gpu_wait_s()
    stats.reset()
    cycles = _sleep_cycles_for(DELAY_MS, dev)
    for _ in range(EPOCHS):
        if rank == 1:
            torch.cuda._sleep(cycles)
        trainer._allreduce_sync()
    wait = trainer.allreduce_gpu_wait_s()
    host = stats.snapshot()["time_s"].get("all_reduce", 0.0)
    got = [None] * ws
    dist.all_gather_object(got, (wait, host))
    if rank == 0:
        expect = EPOCHS * DELAY_MS / 1000.0
        (w0, h0), (w1, h1) = got[0], got[1]
        print(f"injected {expect:.3f} s on rank 1; rank 0 GPU wait {w0:.3f} s, host timer {h0:.3f} s; "
              f"rank 1 GPU wait {w1:.3f} s, host timer {h1:.3f} s", flush=True)
        ok = (w0 is not None and w1 is not None and 0.6 * expect < w0 < 1.6 * expect
              and w1 < 0.3 * expect and h0 < 0.5 * w0)
        print("gpu_nccl_allreduce_wait:", "ALL PASSED" if ok else "FAILED", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
