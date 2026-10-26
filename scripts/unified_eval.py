#!/usr/bin/env python
"""
unified_eval.py -- score an exported embedding bundle with ONE evaluation
implementation shared by DistEA and every baseline.

Reads:  an export bundle directory (layout below).
Writes: metrics to stdout and, with --out, a JSON file.

Every reported Hit@k / MRR comes from this script rather than from whatever
each method's own training script prints, so differences in evaluation
protocol cannot masquerade as differences in accuracy; the protocol is
explicit and swappable via the flags.

Export contract (produced by each method's own export adapter):
    export/{method}/.../
        embeddings.npy   (N, D) float32 -- row i is entity ids.npy[i]
        ids.npy          (N,)   int64   -- global entity id for row i
        kg_labels.npy    (N,)   int8    -- 0 = KG1, 1 = KG2
        test_links.json  [[kg1_id, kg2_id], ...]  ground truth
        candidates.json  OPTIONAL {kg1_id: [kg2_id, ...]} per-entity candidate
                          pool (e.g. ELsEA's per-subtask restricted set).
                          Absent => global eval (candidate set = every KG2 id
                          present in ids.npy), matching DistEA's own
                          distributed_eval/selective_eval semantics.

Protocol flags mirror src/eval.py's own _compute_sim/_apply_csls exactly (see
that file's docstring for the ELsEA-GCN-Align vs RREA-CSLS provenance of
each default) so DistEA's own exported embeddings score identically here as
they already do inside framework.py -- this script reuses those functions
rather than reimplementing the metric.

Usage:
    python scripts/unified_eval.py export/distea/EN_DE_15K_V1/721_5fold_1/ws1 \
        --metric cosine --use_csls --csls_k 10 --candidate_pool test_side \
        --out outputs/unified/EN_DE_15K_V1/distea_ws1.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.eval import (  # noqa: E402
    _apply_csls,
    _chunked_metrics,
    _compute_metrics,
    _compute_sim,
    _csls_thr,
    _dense_limit_bytes,
)


def load_export(export_dir: str):
    emb = np.load(os.path.join(export_dir, "embeddings.npy"))
    ids = np.load(os.path.join(export_dir, "ids.npy"))
    kg_labels = np.load(os.path.join(export_dir, "kg_labels.npy"))
    with open(os.path.join(export_dir, "test_links.json")) as f:
        test_links = [(int(a), int(b)) for a, b in json.load(f)]
    cand_path = os.path.join(export_dir, "candidates.json")
    candidates = None
    if os.path.exists(cand_path):
        with open(cand_path) as f:
            raw = json.load(f)
        candidates = {int(k): [int(v) for v in vs] for k, vs in raw.items()}
    return emb, ids, kg_labels, test_links, candidates


def compute_candidate_coverage(test_links, candidates) -> float:
    """Fraction of test_links whose true kg2 partner is IN that kg1 entity's
    candidate pool. A low Hit@1 under a restricted candidate set can come
    from candidate generation missing the true target entirely, not from
    weak embedding similarity -- this number separates the two failure
    modes."""
    if candidates is None:
        return 1.0  # global eval: every kg2 id in the embedding set is always a candidate
    if not test_links:
        return 0.0
    hit = sum(1 for e1, e2 in test_links if e2 in candidates.get(e1, ()))
    return hit / len(test_links)


def run_eval(
    export_dir: str,
    metric: str = "l1",
    use_csls: bool = False,
    csls_k: int = 10,
    top_k: tuple[int, ...] = (1, 5, 10, 50),
    candidate_pool: str = "all",
    device: str = "cpu",
) -> dict[str, float]:
    """candidate_pool (no candidates.json only): "all" ranks against every
    KG2 row in the export (the historical behaviour, which subtask_unified_
    eval's pool_all bundles rely on); "test_side" keeps only the KG1/KG2
    entities that appear in test_links -- what src/eval.py's selective_eval
    does and what is used for cross-method scoring.
    Exports that already contain only test-side rows score identically
    either way; full-graph exports (e.g. export_clusterea_to_unified.py)
    do not."""
    emb, ids, kg_labels, test_links, candidates = load_export(export_dir)

    kg1_mask = kg_labels == 0
    kg2_mask = kg_labels == 1
    if candidate_pool == "test_side" and candidates is None:
        test_e1 = np.array(sorted({e1 for e1, _ in test_links}), dtype=ids.dtype)
        test_e2 = np.array(sorted({e2 for _, e2 in test_links}), dtype=ids.dtype)
        kg1_mask = kg1_mask & np.isin(ids, test_e1)
        kg2_mask = kg2_mask & np.isin(ids, test_e2)
    elif candidate_pool not in ("all", "test_side"):
        raise ValueError(f"candidate_pool must be 'all' or 'test_side', got {candidate_pool!r}")
    kg1_ids = ids[kg1_mask]
    kg2_ids = ids[kg2_mask]
    kg1_id2col = {int(g): i for i, g in enumerate(kg1_ids)}
    kg2_id2col = {int(g): i for i, g in enumerate(kg2_ids)}

    emb_t = torch.from_numpy(emb).float()
    kg1_emb = emb_t[torch.from_numpy(kg1_mask)]
    kg2_emb = emb_t[torch.from_numpy(kg2_mask)]

    cpu = torch.device("cpu")

    # Above src/eval.py's dense limit (DISTEA_EVAL_DENSE_LIMIT_GB, default 4)
    # with no per-entity candidate mask: use src/eval.py's own never-
    # materialise two-pass path, the one DistEA's selective_eval takes at
    # DBP1M scale (same arithmetic as the dense path, parity-tested in
    # tests/test_stage_v.py). Below the limit nothing changes.
    n1, n2 = kg1_emb.size(0), kg2_emb.size(0)
    if candidates is None and n1 * n2 * 4 > _dense_limit_bytes():
        link_pairs, skipped = [], 0
        for e1, e2 in test_links:
            r1 = kg1_id2col.get(e1)
            r2 = kg2_id2col.get(e2)
            if r1 is not None and r2 is not None:
                link_pairs.append((r1, r2))
            else:
                skipped += 1
        dev = torch.device(device)
        print(f"  [unified_eval] chunked path: {n1} x {n2} dense would be "
              f"{n1 * n2 * 4 / (1 << 30):.1f} GB; running on {dev}")
        metrics = _chunked_metrics(
            metric, kg1_emb, kg2_emb, link_pairs, dev,
            use_csls=use_csls, csls_k=csls_k, top_k=top_k,
        )
        metrics["candidate_coverage"] = 1.0
        metrics["n_test_links"] = float(len(test_links))
        metrics["n_scored"] = float(len(link_pairs))
        metrics["n_skipped_missing_embedding"] = float(skipped)
        metrics["n_kg1_rows"] = float(n1)
        metrics["n_kg2_rows"] = float(n2)
        return metrics

    sim = _compute_sim(metric, kg1_emb, kg2_emb, cpu)

    if use_csls and sim.size(0) > 0 and sim.size(1) > 0:
        # src/eval.py's own training-time CSLS pass already clamps
        # (`cat.topk(min(csls_k, cat.size(0)), dim=0)`); _csls_thr (the
        # simpler helper this comparison script calls directly) has no
        # such guard, and this script's per-subtask pools can be far
        # smaller than csls_k=10 (on DBP1M some subtasks have only a
        # handful of valid-test KG2 entities) -- .topk(k) on a
        # dimension smaller than k raises, so clamp the same way here.
        k_eff = min(csls_k, sim.size(0), sim.size(1))
        if k_eff < csls_k:
            print(f"  [unified_eval] pool too small for csls_k={csls_k} "
                  f"(sim shape {tuple(sim.shape)}); using csls_k={k_eff}")
        row_thr = _csls_thr(sim, k_eff, cpu)
        col_thr = _csls_thr(sim.t(), k_eff, cpu)
        sim = _apply_csls(sim, row_thr, col_thr, cpu)

    # Restrict to per-entity candidate set if provided: mask out every
    # column NOT in that row's candidate pool to -inf so it can never be
    # argmax'd/ranked ahead of the true target -- this is what per-subtask
    # candidate-set evaluation (ELsEA-style) means: a method is only ever
    # compared against entities it put in play for that source entity. If
    # the true target isn't in the pool, its score is also masked to -inf,
    # which correctly produces a near-worst-case rank (a genuine candidate-
    # generation miss), not a silently dropped/excused pair.
    if candidates is not None:
        sim = sim.clone()
        neg_inf = torch.finfo(sim.dtype).min
        for e1, col1 in kg1_id2col.items():
            allowed = candidates.get(e1)
            row = sim[col1]
            masked = torch.full_like(row, neg_inf)
            if allowed:
                allowed_cols = [kg2_id2col[e2] for e2 in allowed if e2 in kg2_id2col]
                if allowed_cols:
                    idx = torch.tensor(allowed_cols, dtype=torch.long)
                    masked[idx] = row[idx]
            sim[col1] = masked

    link_pairs = []
    skipped = 0
    for e1, e2 in test_links:
        r1 = kg1_id2col.get(e1)
        r2 = kg2_id2col.get(e2)
        if r1 is not None and r2 is not None:
            link_pairs.append((r1, r2))
        else:
            skipped += 1

    metrics = _compute_metrics(sim, link_pairs, cpu, top_k)
    metrics["candidate_coverage"] = compute_candidate_coverage(test_links, candidates)
    metrics["n_test_links"] = float(len(test_links))
    metrics["n_scored"] = float(len(link_pairs))
    metrics["n_skipped_missing_embedding"] = float(skipped)
    metrics["n_kg1_rows"] = float(kg1_emb.size(0))
    metrics["n_kg2_rows"] = float(kg2_emb.size(0))
    return metrics


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "export_dir",
        help="directory with embeddings.npy/ids.npy/kg_labels.npy/test_links.json[/candidates.json]",
    )
    p.add_argument("--metric", choices=["l1", "cosine"], default="l1")
    p.add_argument("--use_csls", action="store_true")
    p.add_argument("--csls_k", type=int, default=10)
    p.add_argument("--top_k", type=int, nargs="+", default=[1, 5, 10, 50])
    p.add_argument("--out", default=None, help="optional path to write metrics as JSON")
    p.add_argument("--candidate_pool", choices=["all", "test_side"], default="all",
                   help="without candidates.json: rank against every KG2 row ('all', "
                        "historical default) or only test-side KG1/KG2 entities "
                        "('test_side', = selective_eval, used for cross-method scoring)")
    p.add_argument("--device", default="cpu",
                   help="device for the chunked path (e.g. cuda); dense path stays on CPU")
    args = p.parse_args()

    metrics = run_eval(
        args.export_dir,
        metric=args.metric,
        use_csls=args.use_csls,
        csls_k=args.csls_k,
        top_k=tuple(args.top_k),
        candidate_pool=args.candidate_pool,
        device=args.device,
    )

    print(f"[unified_eval] {args.export_dir}")
    print(f"  protocol: metric={args.metric} use_csls={args.use_csls} csls_k={args.csls_k} "
          f"candidate_pool={args.candidate_pool}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(
                {
                    "export_dir": args.export_dir,
                    "metric": args.metric,
                    "use_csls": args.use_csls,
                    "csls_k": args.csls_k,
                    "candidate_pool": args.candidate_pool,
                    **metrics,
                },
                f,
                indent=2,
            )
        print(f"[unified_eval] wrote {args.out}")


if __name__ == "__main__":
    main()
