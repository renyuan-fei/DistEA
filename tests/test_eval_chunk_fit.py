"""
test_eval_chunk_fit.py — eval._fit_chunk (row blocks of the chunked eval /
mutual-NN paths sized to the GPU memory left) and the property it relies on.

1. CPU: the block size is left as given.
2. CUDA (memory queries mocked): half of the memory left (driver-free plus
   the caching allocator's unused reserve) holds ~4 (rows x columns) fp32
   buffers; the result is a multiple of 16, at least 16 and never above the
   requested size.
3. Hit@k (CSLS on and off) and the mutual-NN pairs are identical for any
   block size and MRR differs only in fp summation order, so shrinking a
   block cannot change a result.

DGL-free, CPU, < 10 s.
Run:  .venv/bin/python tests/test_eval_chunk_fit.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import eval as ev  # noqa: E402

CPU = torch.device("cpu")


def test_cpu_unchanged():
    assert ev._fit_chunk(512, 1_000_000, CPU) == 512
    print("  [ok] CPU keeps the requested block size")


def test_cuda_sizing_mocked():
    real = (torch.cuda.mem_get_info, torch.cuda.memory_reserved, torch.cuda.memory_allocated)
    dev = torch.device("cuda")
    try:
        for free, reserved, alloc, n_cols, want in (
            (8 << 30, 0, 0, 1_350_000, 192),           # 0.5 * 8 GiB / (16 B * 1.35M) = 198 -> 192
            (6 << 30, 4 << 30, 2 << 30, 1_350_000, 192),  # 6 GiB free + 2 GiB unused reserve
            (100 << 30, 0, 0, 1_350_000, 512),         # plenty: capped at the request
            (64 << 20, 0, 0, 1_350_000, 16),           # nearly full: floor of 16 rows
        ):
            torch.cuda.mem_get_info = lambda d=None, f=free: (f, 80 << 30)
            torch.cuda.memory_reserved = lambda d=None, r=reserved: r
            torch.cuda.memory_allocated = lambda d=None, a=alloc: a
            got = ev._fit_chunk(512, n_cols, dev)
            assert got == want and got % 16 == 0, (free, reserved, alloc, got, want)
    finally:
        torch.cuda.mem_get_info, torch.cuda.memory_reserved, torch.cuda.memory_allocated = real
    print("  [ok] CUDA sizing: 192 / 192 / 512 / 16 rows for 8 GiB / 6+2 GiB / 100 GiB / 64 MiB left")


def test_results_independent_of_block_size():
    g = torch.Generator().manual_seed(0)
    n1, n2, d = 700, 650, 32
    base = torch.randn(n2, d, generator=g)
    e1 = torch.cat([base + 0.7 * torch.randn(n2, d, generator=g), torch.randn(n1 - n2, d, generator=g)])
    e2 = base + 0.7 * torch.randn(n2, d, generator=g)
    links = [(i, i) for i in range(n2)]
    for csls in (True, False):
        ref = ev._chunked_metrics("cosine", e1, e2, links, CPU, use_csls=csls, csls_k=10, top_k=(1, 10), chunk=512)
        nn_ref = ev._chunked_mutual_nn("cosine", e1, e2, CPU, use_csls=csls, csls_k=10, chunk=512)
        for chunk in (16, 48, 128):
            got = ev._chunked_metrics("cosine", e1, e2, links, CPU, use_csls=csls, csls_k=10, top_k=(1, 10), chunk=chunk)
            assert all(got[k] == ref[k] for k in ref if k.startswith("Hit@")), (csls, chunk, got, ref)
            assert abs(got["MRR"] - ref["MRR"]) < 1e-6, (csls, chunk, got, ref)  # summation order only
            nn_got = ev._chunked_mutual_nn("cosine", e1, e2, CPU, use_csls=csls, csls_k=10, chunk=chunk)
            assert all(torch.equal(a, b) for a, b in zip(nn_got, nn_ref, strict=False)), (csls, chunk)
    print(f"  [ok] Hit@k and mutual-NN pairs identical, MRR within 1e-6, for blocks of 16 / 48 / 128 / 512 rows (Hit@1 {ref['Hit@1']:.4f})")


if __name__ == "__main__":
    test_cpu_unchanged()
    test_cuda_sizing_mocked()
    test_results_independent_of_block_size()
    print("test_eval_chunk_fit: ALL PASSED")
