"""
test_chunked_eval.py — the chunked evaluation path
(eval.py::_chunked_metrics / _eval_metrics_auto) must give the same Hit@k/MRR
as the dense _compute_sim + CSLS + _compute_metrics pipeline, whose full
similarity matrix does not fit in memory at 1M scale.

DGL-free, no GPU, no torch.distributed. <10 sec.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval import (
    _apply_csls,
    _chunked_metrics,
    _chunked_mutual_nn,
    _compute_metrics,
    _compute_sim,
    _csls_thr,
    _eval_metrics_auto,
    _sim_device,
)

CPU = torch.device("cpu")


def test_sim_device_switch():
    """DISTEA_EVAL_DEVICE: unset -> CPU whatever the rank device; 'cuda'
    only takes effect when the rank device is a real, available CUDA
    device (so the switch is inert on CPU runs and on non-cuda ranks)."""
    fake_cuda = torch.device("cuda", 0)
    os.environ.pop("DISTEA_EVAL_DEVICE", None)
    assert _sim_device(CPU) == CPU
    assert _sim_device(fake_cuda) == CPU
    os.environ["DISTEA_EVAL_DEVICE"] = "cuda"
    try:
        assert _sim_device(CPU) == CPU
        got = _sim_device(fake_cuda)
        assert got == (fake_cuda if torch.cuda.is_available() else CPU), got
    finally:
        del os.environ["DISTEA_EVAL_DEVICE"]
    print("  test_sim_device_switch PASSED")


def _dense_reference(metric, e1, e2, links, use_csls, csls_k, top_k):
    sim = _compute_sim(metric, e1, e2, CPU)
    if use_csls:
        row_thr = _csls_thr(sim, csls_k, CPU)
        col_thr = _csls_thr(sim.t(), csls_k, CPU)
        sim = _apply_csls(sim, row_thr, col_thr, CPU)
    return _compute_metrics(sim, links, CPU, top_k)


def test_chunked_matches_dense_all_variants():
    torch.manual_seed(0)
    n1, n2, d = 173, 191, 16  # deliberately not multiples of the chunk size
    e1 = torch.randn(n1, d)
    e2 = torch.randn(n2, d)
    # make some pairs actually alignable so Hit@1 isn't trivially 0
    e2[:50] = e1[:50] + 0.05 * torch.randn(50, d)
    links = [(i, i) for i in range(0, 60)] + [(80, 100), (90, 3)]

    for metric in ("l1", "cosine"):
        for use_csls in (False, True):
            dense = _dense_reference(metric, e1, e2, links, use_csls, 10, (1, 10))
            chunked = _chunked_metrics(
                metric, e1, e2, links, CPU,
                use_csls=use_csls, csls_k=10, top_k=(1, 10), chunk=64,
            )
            for key in dense:
                assert abs(dense[key] - chunked[key]) < 1e-9, (
                    metric, use_csls, key, dense[key], chunked[key])
            print(f"  parity {metric} csls={use_csls}: {chunked} OK")
    print("  test_chunked_matches_dense_all_variants PASSED")


def test_auto_switch_via_env():
    torch.manual_seed(1)
    e1, e2 = torch.randn(60, 8), torch.randn(60, 8)
    links = [(i, i) for i in range(20)]
    dense = _eval_metrics_auto("cosine", e1, e2, links, CPU, True, 10, (1, 10))
    os.environ["DISTEA_EVAL_DENSE_LIMIT_GB"] = "0"
    try:
        forced = _eval_metrics_auto("cosine", e1, e2, links, CPU, True, 10, (1, 10))
    finally:
        del os.environ["DISTEA_EVAL_DENSE_LIMIT_GB"]
    for key in dense:
        assert abs(dense[key] - forced[key]) < 1e-9, (key, dense[key], forced[key])
    print("  test_auto_switch_via_env PASSED")


def test_empty_links():
    e1, e2 = torch.randn(5, 4), torch.randn(5, 4)
    m = _chunked_metrics("cosine", e1, e2, [], CPU, use_csls=True)
    assert m["Hit@1"] == 0.0 and m["MRR"] == 0.0
    print("  test_empty_links PASSED")




def test_chunked_mutual_nn_matches_dense():
    """Row/col argmax parity with the dense CSLS-adjusted matrix -- the
    pseudo-seed mutual-NN inputs must be identical either way."""
    torch.manual_seed(2)
    n1, n2, d = 157, 143, 16  # non-multiples of chunk
    e1 = torch.randn(n1, d)
    e2 = torch.randn(n2, d)
    e2[:40] = e1[:40] + 0.05 * torch.randn(40, d)

    for metric in ("l1", "cosine"):
        for use_csls in (False, True):
            sim = _compute_sim(metric, e1, e2, CPU)
            if use_csls:
                row_thr = _csls_thr(sim, 10, CPU)
                col_thr = _csls_thr(sim.t(), 10, CPU)
                sim = _apply_csls(sim, row_thr, col_thr, CPU)
            dense_r = sim.argmax(dim=1)
            dense_c = sim.argmax(dim=0)
            ch_r, ch_c = _chunked_mutual_nn(
                metric, e1, e2, CPU, use_csls=use_csls, csls_k=10, chunk=64)
            assert torch.equal(dense_r, ch_r), (metric, use_csls, "rows")
            assert torch.equal(dense_c, ch_c), (metric, use_csls, "cols")
    print("  test_chunked_mutual_nn_matches_dense PASSED")


if __name__ == "__main__":
    print("Chunked-eval tests:")
    test_sim_device_switch()
    test_chunked_matches_dense_all_variants()
    test_auto_switch_via_env()
    test_empty_links()
    test_chunked_mutual_nn_matches_dense()
    print("All chunked-eval tests PASSED.")
