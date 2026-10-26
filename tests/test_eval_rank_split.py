"""--eval_rank_split_dir: per-group (same_rank / cross_rank) Hit@k / MRR.

Checks (CPU, no dist):
  1. groups=None leaves the dense and chunked metric dicts unchanged.
  2. Each group's numbers equal the metrics of that subset evaluated alone,
     on the dense path and on the chunked path, with and without CSLS
     (CSLS thresholds come from all rows, so per-pair ranks are subset-free).
  3. Dense and chunked group numbers agree.
  4. DistEAFramework._rank_split_labels reads {r}.npy + kg2_assignments.json.
"""
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from eval import (  # noqa: E402
    _apply_csls,
    _chunked_metrics,
    _compute_metrics,
    _compute_sim,
    _csls_thr,
)

TOP_K = (1, 5, 10)


def _dense(metric, e1, e2, links, use_csls, groups=None):
    sim = _compute_sim(metric, e1, e2, torch.device("cpu"))
    if use_csls:
        rt = _csls_thr(sim, 10, torch.device("cpu"))
        ct = _csls_thr(sim.t(), 10, torch.device("cpu"))
        sim = _apply_csls(sim, rt, ct, torch.device("cpu"))
    return _compute_metrics(sim, links, torch.device("cpu"), TOP_K, batch_size=7, groups=groups)


def _chunk(metric, e1, e2, links, use_csls, groups=None):
    return _chunked_metrics(metric, e1, e2, links, torch.device("cpu"), use_csls=use_csls,
                            csls_k=10, top_k=TOP_K, chunk=9, groups=groups)


def _close(a, b, tol=1e-6):  # MRR sums float32 reciprocals in different batch groupings
    return abs(a - b) <= tol


def test_metric_groups():
    g = torch.Generator().manual_seed(0)
    e1 = torch.randn(60, 16, generator=g)
    e2 = torch.randn(70, 16, generator=g)
    links = [(i, (i * 7) % 70) for i in range(50)]
    # make some pairs easy so Hit@1 is not ~0
    for i in range(0, 50, 3):
        e2[links[i][1]] = e1[i] + 0.05 * torch.randn(16, generator=g)
    rng = np.random.default_rng(1)
    groups = rng.choice(["same_rank", "cross_rank"], size=len(links)).tolist()
    for metric in ("cosine", "l1"):
        for use_csls in (False, True):
            for fn in (_dense, _chunk):
                base = fn(metric, e1, e2, links, use_csls)
                grp = fn(metric, e1, e2, links, use_csls, groups=groups)
                for k, v in base.items():
                    assert _close(grp[k], v), (fn.__name__, metric, use_csls, k)
                assert not any("/" in k for k in base), "groups=None adds keys"
                for name in ("same_rank", "cross_rank"):
                    sub = [p for p, lab in zip(links, groups, strict=False) if lab == name]
                    ref = fn(metric, e1, e2, sub, use_csls)
                    assert grp[f"{name}/n"] == len(sub)
                    for k, v in ref.items():
                        assert _close(grp[f"{name}/{k}"], v), (fn.__name__, metric, use_csls, name, k)
            d = _dense(metric, e1, e2, links, use_csls, groups=groups)
            c = _chunk(metric, e1, e2, links, use_csls, groups=groups)
            for k in d:
                assert _close(d[k], c[k], 1e-6), ("dense vs chunked", metric, use_csls, k)
    print("  [ok] grouped metrics = per-subset metrics (dense + chunked, cosine/l1, csls on/off)")


def test_rank_split_labels():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from framework import DistEAFramework

    with tempfile.TemporaryDirectory() as tmp:
        np.save(os.path.join(tmp, "0.npy"), np.array([0, 1, 2]))
        np.save(os.path.join(tmp, "1.npy"), np.array([3, 4]))
        with open(os.path.join(tmp, "kg2_assignments.json"), "w") as f:
            json.dump({"0": [10, 11], "1": [12, 13, 14]}, f)
        fake = types.SimpleNamespace(cfg=types.SimpleNamespace(eval_rank_split_dir=tmp), rank=1)
        links = [(0, 10), (1, 12), (3, 13), (4, 11), (2, 99)]
        labels = DistEAFramework._rank_split_labels(fake, links)
        assert labels == ["same_rank", "cross_rank", "same_rank", "cross_rank", "cross_rank"], labels
        assert DistEAFramework._rank_split_labels(fake, links) is labels, "cached per list"
        off = types.SimpleNamespace(cfg=types.SimpleNamespace(eval_rank_split_dir=""), rank=0)
        assert DistEAFramework._rank_split_labels(off, links) is None
    print("  [ok] _rank_split_labels from {r}.npy + kg2_assignments.json")


if __name__ == "__main__":
    test_metric_groups()
    test_rank_split_labels()
    print("test_eval_rank_split: ALL PASSED")
