"""
gpu_parity_eval_device.py — GPU-side parity check for DISTEA_EVAL_DEVICE=cuda
(eval.py::_sim_device): the chunked Hit@k/MRR passes must give the same
numbers on the rank's GPU as on CPU, and the same as the dense reference.

Needs a CUDA device, so it is not part of the DGL-free test set; run it
on a GPU node:
    python tests/gpu_parity_eval_device.py
Sizes are chosen so the CPU side is still quick (~20K x 20K) while
exercising the chunk loop many times; it also prints the time on both
devices.
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval import _chunked_metrics, _chunked_mutual_nn  # noqa: E402


def main() -> None:
    assert torch.cuda.is_available(), "needs a CUDA device"
    cpu, gpu = torch.device("cpu"), torch.device("cuda", 0)
    torch.manual_seed(0)
    n1, n2, d = 20_000, 20_000, 128
    e1 = torch.randn(n1, d)
    e2 = torch.randn(n2, d)
    e2[:8000] = e1[:8000] + 0.3 * torch.randn(8000, d)
    links = [(i, i) for i in range(0, 8000)] + [(9000 + i, 12000 + i) for i in range(500)]

    for metric in ("cosine", "l1"):
        for use_csls in (False, True):
            t0 = time.perf_counter()
            m_cpu = _chunked_metrics(metric, e1, e2, links, cpu, use_csls, 10, (1, 10))
            t_cpu = time.perf_counter() - t0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            m_gpu = _chunked_metrics(metric, e1, e2, links, gpu, use_csls, 10, (1, 10))
            torch.cuda.synchronize()
            t_gpu = time.perf_counter() - t0
            for k in m_cpu:
                # fp32 matmul on GPU vs CPU can reorder sums; ranks only move
                # on near-ties, so allow a handful of pairs out of 8500.
                assert abs(m_cpu[k] - m_gpu[k]) <= 5 / len(links) + 1e-9, (
                    metric, use_csls, k, m_cpu[k], m_gpu[k])
            print(f"  {metric:6s} csls={use_csls!s:5s} cpu={m_cpu} {t_cpu:6.1f}s | "
                  f"gpu={m_gpu} {t_gpu:6.2f}s  (x{t_cpu / max(t_gpu, 1e-6):.0f})")

    r_cpu, c_cpu = _chunked_mutual_nn("cosine", e1, e2, cpu)
    r_gpu, c_gpu = _chunked_mutual_nn("cosine", e1, e2, gpu)
    diff = int((r_cpu != r_gpu.cpu()).sum()) + int((c_cpu != c_gpu.cpu()).sum())
    assert diff <= 10, f"mutual-NN rows/cols differ in {diff} positions"
    print(f"  mutual-NN cpu vs gpu: {diff} differing positions (ties) OK")
    print("GPU eval-device parity OK.")


if __name__ == "__main__":
    main()
