"""
test_defer_test_eval.py — --defer_test_eval's end-of-run helper
(DistEAFramework._deferred_test_eval): it must load the recorded best
iteration's model state and h_init (not evaluate the still-loaded last
state), fill THAT iteration's metrics entry, drop its test_deferred marker
and leave the other entries alone; with no recorded best it evaluates the
loaded state into the last iteration's entry. selective_eval is replaced
by a stub whose metrics are a function of the (model, h_init) it is given.
The end-to-end property (bit-identical chosen-iteration metrics with and
without the flag) is not covered here; it needs a full training run.

DGL-free, CPU, < 5 s.
Run:  .venv/bin/python tests/test_defer_test_eval.py
"""
from __future__ import annotations

import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import framework  # noqa: E402


def _stub_eval(model, g, h, all_ids, links, rank, ws, device, **kw):
    # Depends on the model weight and the h_init it is handed.
    return {"Hit@1": float(h.sum() + model.weight.sum()), "MRR": float(h[0, 0])}


def _fake(iterations):
    fw = types.SimpleNamespace(
        cfg=types.SimpleNamespace(use_csls=True, csls_k=10, eval_metric="cosine", export_dir="",
                                  export_dtype="fp32", task_id="t", run_timestamp="r", seed=1),
        rank=0, world_size=1, device=torch.device("cpu"), results_dir=None, test_links=[(0, 1)],
        _iter_metrics=[{"iteration": i + 1, "test_deferred": True, "valid_Hit@1": 0.1 * (i + 1)}
                       for i in range(iterations)],
        _rank_split_labels=lambda links: None,
        _eval_group_labels=lambda links: None,
        _run_dir=lambda: "/nonexistent",
    )
    return fw


def test_restores_the_best_iteration():
    real = framework.selective_eval
    framework.selective_eval = _stub_eval
    try:
        model = torch.nn.Linear(3, 2, bias=False)
        h_init = torch.nn.Parameter(torch.zeros(5, 4))
        best_model = {"weight": torch.full((2, 3), 2.0)}
        best_h = torch.full((5, 4), 3.0)
        with torch.no_grad():  # the "last iteration" state still loaded
            model.weight.fill_(-1.0)
            h_init.fill_(-7.0)
        fw = _fake(3)
        framework.DistEAFramework._deferred_test_eval(
            fw, (1, best_model, best_h), model, None, h_init, None, None, None, 2)
        e = fw._iter_metrics
        want = _stub_eval(types.SimpleNamespace(weight=best_model["weight"]), None, best_h, *[None] * 5)
        assert e[1]["Hit@1"] == want["Hit@1"] == 5 * 4 * 3.0 + 6 * 2.0, e[1]
        assert e[1]["MRR"] == 3.0 and "test_deferred" not in e[1] and "eval_time_s" in e[1]
        assert "Hit@1" not in e[0] and "Hit@1" not in e[2], "only the chosen entry is filled"
        assert e[0]["test_deferred"] and e[2]["test_deferred"]
        assert torch.equal(h_init.data, best_h) and torch.equal(model.weight.data, best_model["weight"])

        # No recorded best: evaluate the loaded state into the last entry.
        fw = _fake(2)
        with torch.no_grad():
            model.weight.fill_(1.0)
            h_init.fill_(0.5)
        framework.DistEAFramework._deferred_test_eval(fw, None, model, None, h_init, None, None, None, 1)
        assert fw._iter_metrics[1]["Hit@1"] == 5 * 4 * 0.5 + 6 * 1.0
        assert "Hit@1" not in fw._iter_metrics[0]
    finally:
        framework.selective_eval = real
    print("  test_restores_the_best_iteration PASSED")


if __name__ == "__main__":
    for fn in (test_restores_the_best_iteration,):
        print(f"[{fn.__name__}]")
        fn()
    print("test_defer_test_eval: ALL PASSED")
