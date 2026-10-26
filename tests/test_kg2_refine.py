"""kg2_refine.refine_kg2_assignment on a 2-rank toy graph.

KG1: 0-1-2-3-4-5 path, ranks {0,1,2}->0, {3,4,5}->1. Train links (0,10), (3,13).
KG2 edges: 10-11, 11-12, 12-13, 13-14, 14-15. Source assignment puts the
non-seed 15 on rank 0 although its only neighbour 14 is on rank 1.
Expected: seeds never move; 15 moves to rank 1 by label propagation; 11
(seed neighbour 10 -> partner 0 whose KG1 neighbourhood is all rank 0)
stays on 0; 12 (seed neighbour 13 -> partner 3, KG1 neighbourhood {2,3,4}
= ranks {0,1,1}) stays on 1.
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402

from kg2_refine import refine_kg2_assignment  # noqa: E402


def main():
    data = types.SimpleNamespace(
        kg1_entities=list(range(6)), kg2_entities=list(range(10, 16)),
        kg1_triples=[(i, 0, i + 1) for i in range(5)],
        kg2_triples=[(i, 0, i + 1) for i in range(10, 15)],
        load_train_alignment=lambda: [(0, 10), (3, 13)],
    )
    n = 16
    kg1_rank = np.full(n, -1)
    kg1_rank[[0, 1, 2]] = 0
    kg1_rank[[3, 4, 5]] = 1
    kg2_rank = np.full(n, -1)
    kg2_rank[[10, 11, 15]] = 0
    kg2_rank[[12, 13, 14]] = 1
    out = refine_kg2_assignment(data, kg1_rank, kg2_rank, 2, rounds=3, kg1_vote=True)
    assert out[10] == 0 and out[13] == 1, "seeds must not move"
    assert out[15] == 1, out
    assert out[11] == 0 and out[12] == 1 and out[14] == 1, out
    assert (out[:6] == -1).all(), "KG1 entries untouched"
    same = refine_kg2_assignment(data, kg1_rank, out, 2, rounds=3, kg1_vote=True)
    assert (same == out).all(), "a refined assignment is a fixed point here"
    print("test_kg2_refine: OK")


if __name__ == "__main__":
    main()
