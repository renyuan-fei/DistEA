"""
test_rescore_variants.py — src/rescore.py: offline re-scoring of exported
embeddings (CSLS variants, Sinkhorn, partition blocks, export bundles).

1. Dense "bidirectional CSLS" is an identity: CSLS(S^T)^T == CSLS(S).
2. chunked_topk_both == dense CSLS matrix top-K on both axes (values and
   indices; chunk smaller than n1).
3. sparse_rank_metrics == eval._compute_metrics when every column is kept
   (K >= n2), on the dense CSLS matrix.
4. Sinkhorn: uniform marginals on a non-square block; a planted
   permutation signal on a square block is recovered by the plan's argmax.
5. Partition blocks: recall on a toy partition; the oversize fallback
   triggers with a tiny limit.
6. Export dtype round trip: fp32 bit-exact; fp16 within tolerance; meta.json
   written and read back by load_bundle.
7. Parity: a synthetic bundle's cosine_csls equals eval's online numbers
   in both the dense and the forced-chunked path; fusion runs and reports
   candidate_recall.

Run:  .venv/bin/python tests/test_rescore_variants.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval import (  # noqa: E402
    _apply_csls,
    _compute_metrics,
    _compute_sim,
    _csls_thr,
    _save_export_bundle,
)
from rescore import (  # noqa: E402
    Bundle,
    SparseScores,
    bidirectional_sparse,
    block_recall,
    blocks_from_partition,
    chunked_topk_both,
    dense_protocol_metrics,
    load_bundle,
    log_sinkhorn,
    parity_check,
    rescore,
    sinkhorn_blockwise,
    sparse_rank_metrics,
)

CPU = torch.device("cpu")


def _dense_csls(e1, e2, k=3, metric="cosine"):
    sim = _compute_sim(metric, e1, e2, CPU)
    rt = _csls_thr(sim, k, CPU)
    ct = _csls_thr(sim.t(), k, CPU)
    return _apply_csls(sim.clone(), rt, ct, CPU)


def _toy(seed=0, n1=23, n2=29, d=8, noise=0.3):
    gen = torch.Generator().manual_seed(seed)
    base = torch.randn(n2, d, generator=gen)
    e2 = base
    e1 = base[:n1] + noise * torch.randn(n1, d, generator=gen)   # row i's true partner is col i
    links = [(i, i) for i in range(n1)]
    return e1, e2, links


def test_dense_bidirectional_identity():
    e1, e2, _ = _toy()
    a = _dense_csls(e1, e2)
    b = _dense_csls(e2, e1).t()
    assert torch.allclose(a, b, atol=1e-6)
    print("  [ok] CSLS(S^T)^T == CSLS(S): dense bidirectional CSLS is an identity")


def test_topk_both_matches_dense():
    e1, e2, _ = _toy(seed=1)
    K, k = 5, 3
    rv, rc, cv, cr = chunked_topk_both("cosine", e1, e2, K, CPU, True, k, chunk=7)
    dense = _dense_csls(e1, e2, k)
    drow = dense.topk(K, dim=1)
    dcol = dense.t().topk(K, dim=1)
    assert torch.allclose(rv, drow.values, atol=1e-5) and torch.equal(rc, drow.indices)
    assert torch.allclose(cv, dcol.values, atol=1e-5) and torch.equal(cr, dcol.indices)
    rv2, rc2, _, _ = chunked_topk_both("cosine", e1, e2, K, CPU, False, k, chunk=7)
    raw = _compute_sim("cosine", e1, e2, CPU).topk(K, dim=1)
    assert torch.allclose(rv2, raw.values, atol=1e-6) and torch.equal(rc2, raw.indices)
    print("  [ok] chunked row/column top-K == dense (with and without CSLS)")


def test_sparse_rank_matches_dense():
    e1, e2, links = _toy(seed=2)
    dense = _dense_csls(e1, e2, 3)
    n1, n2 = dense.shape
    sp = SparseScores([torch.arange(n2)] * n1, [dense[i] for i in range(n1)])
    a = sparse_rank_metrics(sp, links, (1, 5, 10), k_cap=None)
    b = _compute_metrics(dense, links, CPU, (1, 5, 10))
    for key in ("Hit@1", "Hit@5", "Hit@10", "MRR"):
        assert abs(a[key] - b[key]) < 1e-12, (key, a[key], b[key])
    # bidir with K = n2 keeps every entry: recall 1, Hit@k >= dense Hit@k is not
    # guaranteed (scores change), but every target must be found
    rv, rc, cv, cr = chunked_topk_both("cosine", e1, e2, n2, CPU, True, 3)
    bd = bidirectional_sparse(rv, rc, cv, cr, n1, n2, 3, apply_csls=False)
    m = sparse_rank_metrics(bd, links, (1, 5, 10), k_cap=n2)
    assert m["candidate_recall"] == 1.0 and "MRR@K" in m
    print("  [ok] sparse ranking == dense _compute_metrics; bidir keeps all targets at K=n2")


def test_sinkhorn():
    gen = torch.Generator().manual_seed(3)
    sim = torch.randn(6, 9, generator=gen)
    plan = log_sinkhorn(sim, tau=0.5, iters=200)
    assert torch.allclose(plan.sum(1), torch.ones(6), atol=1e-5)
    # column marginals of the unnormalised plan are uniform: rows sum to 1 each
    # after our per-row scaling, so columns should sum to ~6/9 each
    assert torch.allclose(plan.sum(0), torch.full((9,), 6 / 9), atol=2e-2)
    perm = torch.randperm(12, generator=gen)
    sq = torch.full((12, 12), -1.0)
    sq[torch.arange(12), perm] = 1.0
    sq = sq + 0.05 * torch.randn(12, 12, generator=gen)
    p2 = log_sinkhorn(sq, tau=0.1, iters=100)
    assert torch.equal(p2.argmax(1), perm)
    print("  [ok] Sinkhorn: marginals right on a non-square block; permutation recovered")


def test_partition_blocks_and_fallback():
    e1, e2, links = _toy(seed=4, n1=20, n2=20)
    kg1_ids = np.arange(100, 120)
    kg2_ids = np.arange(500, 520)
    b = Bundle(e1, e2, kg1_ids, kg2_ids, links, [(int(a), int(c)) for a, c in zip(kg1_ids, kg2_ids, strict=False)])
    with tempfile.TemporaryDirectory() as d:
        # 2 blocks: block 0 = rows 0-9 / cols 0-11, block 1 = rows 10-19 / cols 12-19
        np.save(os.path.join(d, "0.npy"), kg1_ids[:10])
        np.save(os.path.join(d, "1.npy"), kg1_ids[10:])
        with open(os.path.join(d, "kg2_assignments.json"), "w") as f:
            json.dump({"0": kg2_ids[:12].tolist(), "1": kg2_ids[12:].tolist()}, f)
        b1, b2, nb = blocks_from_partition(b, d)
    assert nb == 2 and (b1[:10] == 0).all() and (b1[10:] == 1).all() and (b2[:12] == 0).all()
    rec = block_recall(b, b1, b2)
    assert abs(rec - 18 / 20) < 1e-12, rec    # rows 10,11 have their partner in block 0
    sp, info = sinkhorn_blockwise(b, b1, b2, nb, K=5, device=CPU, tau=0.2)
    assert info["blocks"] == 2 and info["fallback_blocks"] == 0
    m = sparse_rank_metrics(sp, links, (1, 5), k_cap=5)
    assert m["candidate_recall"] <= rec + 1e-12
    sp2, info2 = sinkhorn_blockwise(b, b1, b2, nb, K=5, device=CPU, max_block_elems=50)
    assert info2["fallback_blocks"] == 2
    print(f"  [ok] partition blocks: recall {rec:.2f} is the ceiling; oversize fallback triggers")


def test_export_roundtrip_and_meta():
    e1, e2, links = _toy(seed=5)
    kg1_ids = np.arange(1000, 1000 + e1.size(0))
    kg2_ids = np.arange(2000, 2000 + e2.size(0))
    tl = [(int(kg1_ids[a]), int(kg2_ids[c])) for a, c in links]
    with tempfile.TemporaryDirectory() as d:
        d32 = os.path.join(d, "fp32")
        d16 = os.path.join(d, "fp16")
        _save_export_bundle(e1, kg1_ids, e2, kg2_ids, tl, d32, dtype="fp32", meta={"metric": "cosine", "Hit@1": 0.5})
        _save_export_bundle(e1, kg1_ids, e2, kg2_ids, tl, d16, dtype="fp16")
        b32 = load_bundle(d32)
        b16 = load_bundle(d16)
        assert torch.equal(b32.kg1_emb, e1) and torch.equal(b32.kg2_emb, e2)
        assert b32.meta["dtype"] == "fp32" and b32.meta["metric"] == "cosine" and b32.meta["n_kg1"] == e1.size(0)
        assert b16.meta["dtype"] == "fp16" and (b16.kg1_emb - e1).abs().max() < 1e-2
        assert b32.link_pairs == links
    print("  [ok] export fp32 bit-exact, fp16 within tolerance, meta.json round trip")


def test_parity_and_driver():
    e1, e2, links = _toy(seed=6, n1=40, n2=55)
    kg1_ids = np.arange(40)
    kg2_ids = np.arange(100, 155)
    b = Bundle(e1, e2, kg1_ids, kg2_ids, links, [(int(kg1_ids[a]), int(kg2_ids[c])) for a, c in links])
    online = dense_protocol_metrics(b, "cosine", True, 10, CPU)
    for lim in (None, "0"):
        if lim is None:
            os.environ.pop("DISTEA_EVAL_DENSE_LIMIT_GB", None)
        else:
            os.environ["DISTEA_EVAL_DENSE_LIMIT_GB"] = lim
        res = rescore(b, ["cosine_csls", "l1_csls", "bidir", "sinkhorn", "fuse_sum", "fuse_rrf"], CPU,
                      K=10, csls_k=10, kmeans_blocks=2, online_metrics=online)
        assert res["variants"]["cosine_csls"]["parity"]["ok"], res["variants"]["cosine_csls"]["parity"]
        for v in ("l1_csls", "bidir", "sinkhorn", "fuse_sum", "fuse_rrf"):
            assert v in res["variants"], v
        assert 0.0 <= res["variants"]["sinkhorn"]["block_recall"] <= 1.0
        assert res["variants"]["fuse_sum"]["candidate_recall"] >= res["variants"]["bidir"]["candidate_recall"] - 1e-12
    os.environ.pop("DISTEA_EVAL_DENSE_LIMIT_GB", None)
    bad = parity_check(online, {**online, "Hit@1": online["Hit@1"] + 0.01})
    assert not bad["ok"]
    print("  [ok] parity holds in dense and chunked paths; all variants run; parity_check flags a mismatch")


if __name__ == "__main__":
    test_dense_bidirectional_identity()
    test_topk_both_matches_dense()
    test_sparse_rank_matches_dense()
    test_sinkhorn()
    test_partition_blocks_and_fallback()
    test_export_roundtrip_and_meta()
    test_parity_and_driver()
    print("test_rescore_variants: all passed")
