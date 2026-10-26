"""
test_halo_refresh_warm_start.py — the per-iteration halo-row refresh must
survive --warm_start's table restore.

framework.py Step 3/3a patches remote halo rows into h_init_raw (a fresh
build_entity_features() tensor); Step 3b wraps it as the Parameter; then, on
the default path (warm_start on, no --persist_optimizer),
`h_init.data = self._h_state` restores the WHOLE previous-iteration table --
which discards the patch, so the cross-rank refresh has no effect after
iteration 0.
--halo_refresh_after_warm_start re-applies the patch on top of the restored
table via framework._reapply_halo_patches.

This test (1) reproduces the discard with the same call order framework.py
uses, (2) checks the re-apply sets exactly the remote rows and nothing else,
(3) checks the no-cache / missing-id / empty-index cases.

Run:  uv run python tests/test_halo_refresh_warm_start.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from framework import _patch_halo_rows, _reapply_halo_patches  # noqa: E402


def _setup(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    n, d = 12, 4
    all_global_ids = np.arange(n) * 10 + 7  # non-trivial global ids
    halo_kg1_idx = np.array([2, 3, 5])
    halo_kg2_idx = np.array([9, 11])
    prev_table = torch.randn(n, d, generator=g)  # previous iteration's trained table
    h_raw = torch.randn(n, d, generator=g)  # this iteration's fresh init
    remote_kg1 = {
        int(all_global_ids[i]): np.random.RandomState(i).randn(d).astype(np.float32)
        for i in halo_kg1_idx
    }
    remote_kg2 = {
        int(all_global_ids[i]): np.random.RandomState(100 + i).randn(d).astype(np.float32)
        for i in halo_kg2_idx
    }
    return all_global_ids, halo_kg1_idx, halo_kg2_idx, prev_table, h_raw, remote_kg1, remote_kg2


def test_restore_discards_patch():
    """The bug: patch h_init_raw, wrap, restore -> patched rows are gone."""
    ids, k1, k2, prev, h_raw, r1, r2 = _setup()
    h_raw, n1 = _patch_halo_rows(h_raw, k1, ids[k1], r1)
    h_raw, n2 = _patch_halo_rows(h_raw, k2, ids[k2], r2)
    assert (n1, n2) == (3, 2)
    for li in k1:
        assert torch.equal(h_raw[li], torch.from_numpy(r1[int(ids[li])]))
    h = torch.nn.Parameter(h_raw)
    h.data = prev.clone()  # framework.py's warm-start restore, verbatim
    for li in np.concatenate([k1, k2]):
        assert torch.equal(h.data[li], prev[li]), "restore should have overwritten the patch"
        assert not torch.equal(h.data[li], torch.from_numpy((r1 | r2)[int(ids[li])]))
    print("  restore discards the Step 3/3a patch: reproduced")


def test_reapply_sets_exactly_the_remote_rows():
    ids, k1, k2, prev, h_raw, r1, r2 = _setup()
    h = torch.nn.Parameter(h_raw)
    h.data = prev.clone()
    n = _reapply_halo_patches(h.data, k1, k2, ids, r1, r2)
    assert n == 5, n
    for li in k1:
        assert torch.equal(h.data[li], torch.from_numpy(r1[int(ids[li])]))
    for li in k2:
        assert torch.equal(h.data[li], torch.from_numpy(r2[int(ids[li])]))
    touched = set(k1.tolist()) | set(k2.tolist())
    for li in range(len(ids)):
        if li not in touched:
            assert torch.equal(h.data[li], prev[li]), f"row {li} must stay restored"
    assert h.requires_grad and not h.data.requires_grad
    print("  re-apply after restore: 5/5 halo rows == remote, 7 local rows untouched")


def test_missing_ids_empty_dicts_empty_idx():
    ids, k1, k2, prev, h_raw, r1, r2 = _setup()
    h = prev.clone()
    assert _reapply_halo_patches(h, k1, k2, ids, {}, {}) == 0
    assert torch.equal(h, prev)
    first_gid = int(ids[k1[0]])
    partial = {first_gid: r1[first_gid]}  # only 1 of the 3 kg1 halo ids pushed
    assert _reapply_halo_patches(h, k1, k2, ids, partial, {}) == 1
    assert torch.equal(h[k1[0]], torch.from_numpy(r1[first_gid]))
    assert torch.equal(h[k1[1]], prev[k1[1]])
    empty = np.array([], dtype=np.int64)
    assert _reapply_halo_patches(h, empty, empty, ids, r1, r2) == 0
    print("  empty dicts / partial cache / empty index arrays: no-op or partial as expected")


if __name__ == "__main__":
    for fn in (
        test_restore_discards_patch,
        test_reapply_sets_exactly_the_remote_rows,
        test_missing_ids_empty_dicts_empty_idx,
    ):
        print(f"[{fn.__name__}]")
        fn()
    print("test_halo_refresh_warm_start: ALL PASSED")
