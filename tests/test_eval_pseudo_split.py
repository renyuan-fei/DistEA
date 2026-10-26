"""--eval_pseudo_split labels (DistEAFramework._pseudo_split_labels / _eval_group_labels)."""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framework import DistEAFramework  # noqa: E402


def _fw(flag, train, orig, rank_dir=""):
    fw = DistEAFramework.__new__(DistEAFramework)
    fw.cfg = types.SimpleNamespace(eval_pseudo_split=flag, eval_rank_split_dir=rank_dir)
    fw.train_links = train
    fw._orig_train_links = orig
    fw.rank = 1  # no logging
    return fw


def test_labels():
    orig = [(0, 100), (1, 101)]
    pseudo = [(5, 105), (6, 999)]          # (5,105) equals a test pair; (6,999) is wrong
    fw = _fw(True, orig + pseudo, orig)
    test = [(5, 105), (6, 106), (7, 999), (8, 108)]
    assert fw._pseudo_split_labels(test) == ["ps_pair", "ps_entity", "ps_entity", "ps_clean"]
    assert fw._eval_group_labels(test) == ["ps_pair", "ps_entity", "ps_entity", "ps_clean"]
    print("  test_labels PASSED")


def test_off_or_no_pseudo_is_none():
    orig = [(0, 100)]
    assert _fw(False, orig + [(5, 105)], orig)._eval_group_labels([(5, 105)]) is None
    assert _fw(True, list(orig), orig)._eval_group_labels([(5, 105)]) is None
    print("  test_off_or_no_pseudo_is_none PASSED")


if __name__ == "__main__":
    test_labels()
    test_off_or_no_pseudo_is_none()
    print("All eval_pseudo_split tests PASSED.")
