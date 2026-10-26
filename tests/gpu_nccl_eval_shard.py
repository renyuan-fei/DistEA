"""
gpu_nccl_eval_shard.py -- DISTEA_EVAL_SHARD paths on real GPUs under NCCL
(tests/test_eval_shard.py covers gloo/CPU only). Functional check, not a
benchmark: random embeddings at a size where the chunked path runs, every
rank compares the sharded result with the unsharded one it computes itself.

Run on a node with >= 2 GPUs:
    torchrun --nproc_per_node=2 tests/gpu_nccl_eval_shard.py
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(dev)
    from eval import _chunked_metrics, _chunked_mutual_nn

    g = torch.Generator().manual_seed(0)
    n1, n2, d = 6000, 5000, 64
    base = torch.randn(n2, d, generator=g)
    e1 = torch.cat([base + 0.8 * torch.randn(n2, d, generator=g), torch.randn(n1 - n2, d, generator=g)]).to(dev)
    e2 = (base + 0.8 * torch.randn(n2, d, generator=g)).to(dev)
    links = [(i, i) for i in range(n2)]
    groups = ["a" if i % 4 else "b" for i in range(n2)]
    ok = True
    for csls in (True, False):
        t0 = time.perf_counter()
        ref = _chunked_metrics("cosine", e1, e2, links, dev, use_csls=csls, csls_k=10,
                               top_k=(1, 10), chunk=512, groups=groups)
        t1 = time.perf_counter()
        got = _chunked_metrics("cosine", e1, e2, links, dev, use_csls=csls, csls_k=10,
                               top_k=(1, 10), chunk=512, groups=groups, shard=(rank, ws))
        t2 = time.perf_counter()
        for k in ref:
            same = abs(ref[k] - got[k]) < 1e-9 if k.endswith("MRR") else ref[k] == got[k]
            ok &= same
            if not same and rank == 0:
                print("MISMATCH", csls, k, ref[k], got[k], flush=True)
        if rank == 0:
            print(f"metrics csls={csls}: Hit@1 {got['Hit@1']:.4f} unsharded {t1 - t0:.2f}s sharded {t2 - t1:.2f}s", flush=True)
        r = _chunked_mutual_nn("cosine", e1, e2, dev, use_csls=csls, csls_k=10, return_row_best=True)
        s = _chunked_mutual_nn("cosine", e1, e2, dev, use_csls=csls, csls_k=10, return_row_best=True,
                               shard=(rank, ws))
        same = torch.equal(r[0], s[0]) and torch.equal(r[1], s[1]) and torch.equal(r[2], s[2])
        ok &= same
        if rank == 0:
            print(f"mutual-NN csls={csls}: identical={same}", flush=True)
    flag = torch.tensor([int(ok)], device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("gpu_nccl_eval_shard:", "ALL PASSED" if flag.item() == 1 else "FAILED", flush=True)
    dist.destroy_process_group()
    sys.exit(0 if flag.item() == 1 else 1)


if __name__ == "__main__":
    main()
