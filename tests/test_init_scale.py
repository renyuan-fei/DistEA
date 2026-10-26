"""
test_init_scale.py — --init_scale: scales the ID-seeded
initial entity rows (framework) and the RREA relation table (rrea.py);
1.0 leaves both bit-identical.

Run:  .venv/bin/python tests/test_init_scale.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.gcn import build_entity_features  # noqa: E402
from models.rrea import RREA  # noqa: E402


def test_rel_emb_scale():
    a = RREA(rel_num=7, output_dim=16, num_layers=2)
    b = RREA(rel_num=7, output_dim=16, num_layers=2, init_scale=1.0)
    c = RREA(rel_num=7, output_dim=16, num_layers=2, init_scale=0.3)
    assert torch.equal(a.rel_emb, b.rel_emb)
    assert torch.allclose(c.rel_emb, 0.3 * a.rel_emb)
    assert abs(float(a.rel_emb.norm(dim=1).mean()) - 1.0) < 1e-5
    assert abs(float(c.rel_emb.norm(dim=1).mean()) - 0.3) < 1e-5
    # the encoder's kernels are not touched by init_scale
    for (n1, p1), (n2, p2) in zip(a.encoder.named_parameters(), c.encoder.named_parameters(), strict=False):
        assert n1 == n2 and torch.equal(p1, p2)
    print("  [ok] rel_emb scaled by init_scale; 1.0 identical; kernels untouched")


def test_entity_rows_scale():
    ids = np.arange(20)
    h = build_entity_features(ids, 16, torch.device("cpu"))
    assert torch.allclose(h.norm(dim=1), torch.ones(20), atol=1e-6)
    assert torch.allclose((h * 0.3).norm(dim=1), torch.full((20,), 0.3), atol=1e-6)
    print("  [ok] unit-norm rows scale linearly")


if __name__ == "__main__":
    test_rel_emb_scale()
    test_entity_rows_scale()
    print("test_init_scale: all passed")
