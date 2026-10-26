"""
test_feature_procs.py — DISTEA_FEAT_PROCS=P: build_entity_features
with its rows filled by P forked workers must be bit-identical to the serial
per-ID loop, for unsorted ids with gaps and large values, uneven slices, and
P larger than the table allows (capped). Also checks that the forked path
itself ran (not the serial fallback).

DGL-free, CPU, < 30 s.
Run:  .venv/bin/python tests/test_feature_procs.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import gcn  # noqa: E402


def _features(ids, procs):
    os.environ["DISTEA_FEAT_PROCS"] = str(procs)
    try:
        return gcn.build_entity_features(ids, 128, torch.device("cpu"))
    finally:
        os.environ.pop("DISTEA_FEAT_PROCS", None)


def test_forked_rows_identical():
    rng = np.random.default_rng(0)
    ids = rng.permutation(np.concatenate([np.arange(70_001), rng.integers(2**31, 2**40, 333)]))
    ref = _features(ids, 1)
    for procs in (2, 3, 7):
        out = _features(ids, procs)
        assert torch.equal(out, ref), f"procs={procs}: features differ from the serial loop"
    raw = np.zeros((len(ids), 64), dtype=np.float32)
    assert gcn._feature_rows_forked(ids, 64, raw, 3), "the forked path must succeed here"
    ser = np.zeros_like(raw)
    gcn._feature_rows(ids, 64, ser)
    assert np.array_equal(raw, ser)
    print(f"  test_forked_rows_identical PASSED ({len(ids)} ids, procs 2/3/7 (capped to 3) bit-identical)")


def test_procs_cap():
    os.environ["DISTEA_FEAT_PROCS"] = "64"
    try:
        assert gcn._feature_procs(30_000) == 1, "below 2 x 20000 rows: serial"
        assert gcn._feature_procs(10**7) <= (os.cpu_count() or 1)
    finally:
        os.environ.pop("DISTEA_FEAT_PROCS", None)
    assert gcn._feature_procs(10**7) == 1, "default is serial"
    small = np.arange(1000)
    assert torch.equal(_features(small, 8), _features(small, 1))
    print("  test_procs_cap PASSED")


if __name__ == "__main__":
    for fn in (test_forked_rows_identical, test_procs_cap):
        print(f"[{fn.__name__}]")
        fn()
    print("test_feature_procs: ALL PASSED")
