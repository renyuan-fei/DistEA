"""
test_epoch_selector.py — epoch_select.EpochSelector and
pick_best_iteration: the pure decision logic behind --keep_best_epoch /
--epoch_patience / --select_by.

1. Strict improvement: a new best is reported only when score > best; ties
   keep the earlier epoch.
2. Patience counts consecutive non-improving evaluations, resets on a new
   best, respects min_epochs, and patience=0 never stops.
3. Epochs must be observed in increasing order.
4. pick_best_iteration: select_by="test" is the historical max test Hit@1;
   select_by="valid" uses valid_Hit@1 and falls back to test for entries
   without one.
5. summary() is JSON-serialisable and reports the chosen epoch.

Run:  .venv/bin/python tests/test_epoch_selector.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from epoch_select import (  # noqa: E402
    EpochSelector,
    parse_eval_splits,
    pick_best_iteration,
    split_prefix,
)


def test_strict_improvement_and_ties():
    s = EpochSelector(patience=0, metric="valid_Hit@1")
    assert s.observe(50, 0.10) is True
    assert s.observe(100, 0.12) is True
    assert s.observe(150, 0.12) is False      # tie -> keep epoch 100
    assert s.observe(200, 0.11) is False
    assert s.best_epoch == 100 and abs(s.best_score - 0.12) < 1e-12
    assert s.has_best
    assert s.should_stop() is False           # patience 0 never stops
    print("  [ok] strict improvement, ties keep the earlier epoch, patience=0 never stops")


def test_patience_counter():
    s = EpochSelector(patience=2, min_epochs=0)
    s.observe(10, 0.5)
    assert s.should_stop() is False
    s.observe(20, 0.4)
    assert s.bad_evals == 1 and s.should_stop() is False
    s.observe(30, 0.6)                        # new best resets the counter
    assert s.bad_evals == 0 and s.should_stop() is False
    s.observe(40, 0.55)
    s.observe(50, 0.50)
    assert s.bad_evals == 2 and s.should_stop() is True
    assert s.best_epoch == 30
    print("  [ok] patience counts consecutive misses and resets on a new best")


def test_min_epochs_floor():
    s = EpochSelector(patience=1, min_epochs=100)
    s.observe(25, 0.5)
    s.observe(50, 0.4)
    assert s.bad_evals == 1 and s.should_stop() is False   # below the floor
    s.observe(100, 0.3)
    assert s.should_stop() is True
    print("  [ok] min_epochs floor delays stopping")


def test_monotonic_epochs():
    s = EpochSelector()
    s.observe(10, 0.1)
    try:
        s.observe(10, 0.2)
    except ValueError:
        pass
    else:
        raise AssertionError("repeated epoch should raise")
    try:
        s.observe(5, 0.2)
    except ValueError:
        pass
    else:
        raise AssertionError("decreasing epoch should raise")
    print("  [ok] epochs must increase")


def test_pick_best_iteration():
    iters = [
        {"iteration": 1, "Hit@1": 0.30, "valid_Hit@1": 0.40},
        {"iteration": 2, "Hit@1": 0.35, "valid_Hit@1": 0.45},
        {"iteration": 3, "Hit@1": 0.36},                      # no valid number
        {"iteration": 4, "Hit@1": 0.33, "valid_Hit@1": 0.44},
    ]
    assert pick_best_iteration(iters, "test")["iteration"] == 3
    assert pick_best_iteration(iters, "valid")["iteration"] == 2
    # all without valid -> falls back to test
    no_valid = [{"iteration": i, "Hit@1": h} for i, h in enumerate([0.1, 0.3, 0.2], 1)]
    assert pick_best_iteration(no_valid, "valid")["iteration"] == 2
    assert pick_best_iteration([], "valid") == {}
    print("  [ok] pick_best_iteration: test / valid / fallback / empty")


def test_summary_json():
    s = EpochSelector(patience=3, metric="valid_Hit@1")
    s.observe(50, 0.2)
    s.observe(100, 0.25)
    s.observe(150, 0.24)
    d = s.summary()
    json.dumps(d)
    assert d["best_epoch"] == 100 and d["evaluated_epochs"] == 3
    assert d["stopped_early"] is False and d["select_by"] == "valid_Hit@1"
    empty = EpochSelector().summary()
    json.dumps(empty)
    assert empty["best_score"] is None and empty["best_epoch"] == 0
    print("  [ok] summary() is JSON-able")


def test_parse_eval_splits():
    assert parse_eval_splits("test") == ["test"]
    assert parse_eval_splits("test,valid,train") == ["test", "valid", "train"]
    assert parse_eval_splits(" valid , test ,valid") == ["valid", "test"]
    assert parse_eval_splits("test", select_by="valid") == ["test", "valid"]
    assert parse_eval_splits(None) == ["test"]
    assert parse_eval_splits("") == ["test"]
    try:
        parse_eval_splits("test,dev")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown split must raise")
    assert split_prefix("test") == "" and split_prefix("valid") == "valid_" and split_prefix("train") == "train_"
    print("  [ok] parse_eval_splits / split_prefix")


if __name__ == "__main__":
    test_parse_eval_splits()
    test_strict_improvement_and_ties()
    test_patience_counter()
    test_min_epochs_floor()
    test_monotonic_epochs()
    test_pick_best_iteration()
    test_summary_json()
    print("test_epoch_selector: all passed")
