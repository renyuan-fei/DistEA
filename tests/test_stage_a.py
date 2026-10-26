"""
test_stage_a.py — Stage A tests: graph loading, GCN, eval, KG2 assignment.

DGL-free, no GPU required, <30 sec.
"""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from counterpart_disc import assign_kg2_to_ranks
from dataload import OpenEAData
from models.gcn import GCN, alignment_loss, build_entity_features
from models.rrea import RREA

# ---------------------------------------------------------------------------
#  Minimal synthetic dataset
# ---------------------------------------------------------------------------


def make_dataset(tmpdir):
    """Create a tiny OpenEA-format dataset in tmpdir."""
    data_dir = os.path.join(tmpdir, "data")
    fold_dir = os.path.join(tmpdir, "fold")
    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(data_dir)
    os.makedirs(fold_dir)
    os.makedirs(output_dir)

    # KG1: 6 entities, 3 relations
    kg1 = "e0\tr0\te1\ne0\tr1\te2\ne1\tr2\te3\ne2\tr2\te4\ne3\tr0\te5\n"
    # KG2: 4 entities
    kg2 = "f0\tr0\tf1\nf0\tr1\tf2\nf1\tr2\tf3\n"

    with open(os.path.join(data_dir, "rel_triples_1"), "w") as f:
        f.write(kg1)
    with open(os.path.join(data_dir, "rel_triples_2"), "w") as f:
        f.write(kg2)
    with open(os.path.join(fold_dir, "train_links"), "w") as f:
        f.write("e0\tf0\ne1\tf1\n")
    with open(os.path.join(fold_dir, "valid_links"), "w") as f:
        f.write("e2\tf2\n")
    with open(os.path.join(fold_dir, "test_links"), "w") as f:
        f.write("e3\tf3\ne4\tf2\n")

    return data_dir, fold_dir, output_dir


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------


def test_dataload():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir, fold_dir, output_dir = make_dataset(tmp)
        data = OpenEAData(data_dir, fold_dir, output_dir)
        assert data.ent_num_1 == 6, f"expected 6 KG1 entities, got {data.ent_num_1}"
        assert data.ent_num_2 == 4, f"expected 4 KG2 entities, got {data.ent_num_2}"
        assert data.rel_num >= 3
        train = data.load_train_alignment()
        assert len(train) == 2
        test = data.load_test_alignment()
        assert len(test) == 2
        print("  test_dataload PASSED")


def test_counterpart_disc():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir, fold_dir, output_dir = make_dataset(tmp)
        data = OpenEAData(data_dir, fold_dir, output_dir)
        # 2-partition assignment
        partitions = [set(data.kg1_entities[:3]), set(data.kg1_entities[3:])]
        assignments = assign_kg2_to_ranks(data, partitions, output_dir)
        # All KG2 entities should be assigned to exactly one rank
        all_assigned = set(assignments[0] + assignments[1])
        assert all_assigned == set(data.kg2_entities), (
            f"Missing KG2 assignments: {set(data.kg2_entities) - all_assigned}"
        )
        # kg2_assignments.json should exist
        assert os.path.exists(os.path.join(output_dir, "kg2_assignments.json"))
        print("  test_counterpart_disc PASSED")


def test_build_entity_features():
    ids = np.array([0, 5, 100, 999], dtype=np.int64)
    feats = build_entity_features(ids, dim=64, device=torch.device("cpu"))
    assert feats.shape == (4, 64), f"expected (4,64), got {feats.shape}"
    # Same ID → same feature (deterministic)
    feats2 = build_entity_features(ids, dim=64, device=torch.device("cpu"))
    assert torch.allclose(feats, feats2), "build_entity_features is not deterministic"
    # L2 normalised
    norms = feats.norm(dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), "features not L2 normalised"
    print("  test_build_entity_features PASSED")


def test_gcn_forward():
    # GCN-Align (ELsEA-faithful) has no per-layer transform, so the model
    # can't change dimensionality — h's width must equal output_dim.
    N, D = 8, 16
    ids_row = torch.arange(N)
    adj = torch.sparse_coo_tensor(
        torch.stack([ids_row, ids_row]), torch.ones(N), size=(N, N), dtype=torch.float32
    )
    h = torch.randn(N, D)
    model = GCN(output_dim=D, num_layers=2)
    out = model(adj, h)
    assert out.shape == (N, D), f"expected ({N},{D}), got {out.shape}"
    # Raw (not L2-normalised) output — ELsEA never normalises model.outputs
    assert torch.isfinite(out).all()
    print("  test_gcn_forward PASSED")


def test_alignment_loss():
    N, D = 20, 16
    kg1_size = 10  # nodes 0–9 are KG1, 10–19 are KG2
    emb = torch.randn(N, D)
    emb = torch.nn.functional.normalize(emb, dim=1)
    pairs = torch.tensor([[0, 10], [1, 11], [2, 12]], dtype=torch.long)
    loss = alignment_loss(emb, pairs, kg1_size=kg1_size, neg_k=3, margin=1.0)
    assert loss.item() >= 0.0
    # Empty pairs → zero loss
    empty = torch.zeros(0, 2, dtype=torch.long)
    loss_empty = alignment_loss(emb, empty, kg1_size=kg1_size)
    assert loss_empty.item() == 0.0
    print("  test_alignment_loss PASSED")


def test_rrea_attn_kernels_deterministic_across_rng_states():
    """
    RREA.encoder.attn_kernels must come out identical regardless of the
    GLOBAL torch RNG's state at construction time -- every rank calls
    RREA(...) with identical arguments (framework.py never passes anything
    rank-specific), so if attn_kernels depended on the global stream (as it
    did before being seeded via a local torch.Generator), different ranks
    -- each having consumed a different, unpredictable amount of prior
    global randomness -- would start iteration 0 with genuinely different
    kernels, contradicting rel_emb's own "identical on every rank, no
    cross-rank communication needed" design intent (models/rrea.py).
    """
    torch.manual_seed(1)
    torch.rand(37)  # simulate "rank 0" having consumed some global RNG entropy
    model_a = RREA(rel_num=5, output_dim=8, num_layers=2, attn_heads=1)

    torch.manual_seed(999)
    torch.rand(4)  # simulate "rank 1" having consumed a DIFFERENT amount
    model_b = RREA(rel_num=5, output_dim=8, num_layers=2, attn_heads=1)

    kernels_a = list(model_a.encoder.attn_kernels)
    kernels_b = list(model_b.encoder.attn_kernels)
    assert len(kernels_a) == len(kernels_b)
    for ka, kb in zip(kernels_a, kernels_b, strict=False):
        assert torch.equal(ka, kb), (
            "attn_kernels differ across simulated ranks despite identical "
            "RREA(...) arguments -- kernel_seed determinism regressed"
        )
    print("  test_rrea_attn_kernels_deterministic_across_rng_states PASSED")


if __name__ == "__main__":
    print("Stage A tests:")
    test_dataload()
    test_counterpart_disc()
    test_build_entity_features()
    test_gcn_forward()
    test_alignment_loss()
    test_rrea_attn_kernels_deterministic_across_rng_states()
    print("All Stage A tests PASSED.")
