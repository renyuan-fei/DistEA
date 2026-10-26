#!/usr/bin/env python
"""
tie_rank_check.py -- how much of a unified-eval score comes from ties.

src/eval.py's _compute_metrics ranks the gold target as 1 + #(candidates with
STRICTLY higher similarity), so a candidate that ties the gold exactly does
not push it down (optimistic tie rule). Trained embedding tables almost never
tie; label-propagation features (LightEA) can: entities with identical
neighbourhoods get identical feature rows. This script scores one export
bundle (same layout as scripts/unified_eval.py reads, test-side pool) under
three tie rules and reports how many links have at least one exact tie with
the gold:

    optimistic   rank = 1 + #(sim >  gold)            (= unified_eval)
    pessimistic  rank = #(sim >= gold)                 (gold included)
    average      rank = 1 + #(sim > gold) + #(sim == gold, other) / 2

Reads an export bundle; writes nothing (prints a JSON summary).

Usage:
    python scripts/tie_rank_check.py export/lightea/EN_DE_15K_V1/custom_30pct_fold_1 \
        [--use_csls --csls_k 10] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("export_dir")
    p.add_argument("--use_csls", action="store_true")
    p.add_argument("--csls_k", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=1024)
    args = p.parse_args()
    dev = torch.device(args.device)

    d = args.export_dir
    emb = np.load(os.path.join(d, "embeddings.npy"))
    ids = np.load(os.path.join(d, "ids.npy"))
    labels = np.load(os.path.join(d, "kg_labels.npy"))
    with open(os.path.join(d, "test_links.json")) as f:
        links = [(int(a), int(b)) for a, b in json.load(f)]
    e1 = np.array(sorted({a for a, _ in links}), dtype=ids.dtype)
    e2 = np.array(sorted({b for _, b in links}), dtype=ids.dtype)
    m1 = (labels == 0) & np.isin(ids, e1)
    m2 = (labels == 1) & np.isin(ids, e2)
    c1 = {int(g): i for i, g in enumerate(ids[m1])}
    c2 = {int(g): i for i, g in enumerate(ids[m2])}
    pairs = [(c1[a], c2[b]) for a, b in links if a in c1 and b in c2]

    x1 = torch.from_numpy(emb[m1]).float().to(dev)
    x2 = torch.from_numpy(emb[m2]).float().to(dev)
    zero1 = int((x1.norm(dim=1) == 0).sum())
    zero2 = int((x2.norm(dim=1) == 0).sum())
    # Same arithmetic as src/eval.py's _chunked_metrics (cosine, then
    # csls = 2 sim - row_thr - col_thr), never materialising (N1, N2).
    e1 = torch.nn.functional.normalize(x1, p=2, dim=1)
    e2 = torch.nn.functional.normalize(x2, p=2, dim=1)
    k = min(args.csls_k, e1.size(0), e2.size(0))
    col_thr = None
    if args.use_csls:
        running = None
        for i in range(0, e1.size(0), args.batch):
            block = e1[i:i + args.batch] @ e2.t()
            cat = block if running is None else torch.cat([running, block], dim=0)
            running = cat.topk(min(k, cat.size(0)), dim=0).values
        col_thr = running.mean(dim=0)

    r1 = torch.tensor([a for a, _ in pairs], device=dev)
    r2 = torch.tensor([b for _, b in pairs], device=dev)
    out = {"opt": [], "pes": [], "avg": []}
    n_tied = 0
    for i in range(0, len(pairs), args.batch):
        rows = e1[r1[i:i + args.batch]] @ e2.t()
        if col_thr is not None:
            row_thr = rows.topk(k, dim=1).values.mean(dim=1)
            rows = 2 * rows - row_thr.unsqueeze(1) - col_thr.unsqueeze(0)
        gold = rows.gather(1, r2[i:i + args.batch].unsqueeze(1))
        gt = (rows > gold).sum(1).float()
        eq = (rows == gold).sum(1).float() - 1  # other candidates tying the gold
        n_tied += int((eq > 0).sum())
        out["opt"].append(gt + 1)
        out["pes"].append(gt + 1 + eq)
        out["avg"].append(gt + 1 + eq / 2)
    res = {"export_dir": d, "n_links": len(pairs), "n_links_with_tie": n_tied,
           "zero_rows_kg1": zero1, "zero_rows_kg2": zero2, "use_csls": args.use_csls}
    for rule, chunks in out.items():
        r = torch.cat(chunks)
        res[rule] = {"Hit@1": float((r <= 1).float().mean()), "Hit@10": float((r <= 10).float().mean()),
                     "MRR": float((1 / r).mean())}
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
