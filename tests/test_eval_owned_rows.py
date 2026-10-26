"""
test_eval_owned_rows.py — the eval gather must score each entity by its
OWNER's embedding, once (eval.py::_select_owned_rows / _dedupe_gathered).

Why: if every rank contributes its halo copies of test entities, a
`{gid: row}` map over the gathered rows picks the LAST copy (highest rank,
a stale halo row) as the target and keeps the other copies as competitors,
so multi-rank Hit@1 is scored against the wrong rows. DGL-free, no GPU, no torch.distributed.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval import _dedupe_gathered, _select_owned_rows  # noqa: E402


def _rank_graph():
    # node order [local_kg1 | halo_kg1 | local_kg2 | halo_kg2]
    all_global_ids = np.array([10, 11, 12, 20, 21, 50, 51, 60], dtype=np.int64)
    local_kg1_idx = np.arange(0, 3)
    local_kg2_idx = np.arange(5, 7)
    emb = torch.arange(8, dtype=torch.float32).unsqueeze(1).repeat(1, 4)
    return all_global_ids, local_kg1_idx, local_kg2_idx, emb


def test_without_mask_contributes_halo_copies():
    gids, _, _, emb = _rank_graph()
    ids, rows = _select_owned_rows(emb, gids, wanted_gids={11, 20, 60, 99})
    assert sorted(ids.tolist()) == [11, 20, 60], ids  # halo 20 and 60 included
    assert rows.shape == (3, 4)
    print("  test_without_mask_contributes_halo_copies PASSED")


def test_with_mask_contributes_owned_only():
    gids, l1, l2, emb = _rank_graph()
    owned = np.concatenate([l1, l2])
    ids, rows = _select_owned_rows(emb, gids, wanted_gids={11, 20, 51, 60, 99}, owned_local=owned)
    assert sorted(ids.tolist()) == [11, 51], ids
    # rows are this rank's rows for those ids
    got = {int(i): float(r[0]) for i, r in zip(ids, rows, strict=True)}
    assert got == {11: 1.0, 51: 6.0}, got
    # bool mask spelling is equivalent
    mask = np.zeros(len(gids), dtype=bool)
    mask[owned] = True
    ids_b, _ = _select_owned_rows(emb, gids, wanted_gids={11, 20, 51, 60}, owned_local=mask)
    assert sorted(ids_b.tolist()) == [11, 51]
    # empty side
    ids_e, rows_e = _select_owned_rows(emb, gids, wanted_gids={20, 60}, owned_local=owned)
    assert ids_e.size == 0 and rows_e.shape == (0, 4)
    print("  test_with_mask_contributes_owned_only PASSED")


def test_dedupe_keeps_first_copy():
    # gathered order = rank 0 rows, then rank 1 rows; entity 7 appears on both
    ids = np.array([5, 7, 9, 7, 8], dtype=np.int64)
    emb = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])
    emb2, ids2 = _dedupe_gathered(emb, ids, "kg1")
    assert ids2.tolist() == [5, 7, 9, 8], ids2
    assert emb2[:, 0].tolist() == [0.0, 1.0, 2.0, 4.0]  # rank-0 copy of 7 kept
    # no duplicates -> untouched objects
    emb3, ids3 = _dedupe_gathered(emb2, ids2, "kg1")
    assert ids3 is ids2 and emb3 is emb2
    emb4, ids4 = _dedupe_gathered(torch.zeros(0, 1), np.zeros(0, dtype=np.int64), "kg2")
    assert ids4.size == 0 and emb4.shape == (0, 1)
    print("  test_dedupe_keeps_first_copy PASSED")


if __name__ == "__main__":
    print("Eval owned-rows tests:")
    test_without_mask_contributes_halo_copies()
    test_with_mask_contributes_owned_only()
    test_dedupe_keeps_first_copy()
    print("All eval owned-rows tests PASSED.")
